#!/usr/bin/env python3
"""
从 HM3D val split 导出场景 RGB/深度/位姿到 habitat_val 目录，格式与 configs/data/habitat 一致，
供后续：seg → transform_pointcloud → build_instance_clip_vectors → DBSGConstruction 建图 使用。

依赖：Habitat 已安装，HM3D val 数据已下载（data/datasets/instance_imagenav, data/scene_datasets/hm3d_v0.2/val）。

用法:
  cd DBSGNavigation
  python scripts/export_habitat_val_data.py --output_dir ../DBSGConstruction/configs/data/habitat_val
  python scripts/export_habitat_val_data.py --output_dir ... --max_scenes 50   # 只导出 50 个场景
  python scripts/export_habitat_val_data.py --output_dir ... --max_scenes 2    # 测试：仅 2 个场景
  python scripts/export_habitat_val_data.py --output_dir ... --max_frames_per_scene 1000  # 每场景更多帧
  python scripts/export_habitat_val_data.py --output_dir ... --grid_res 0.2    # 更精细去重（保留更多位姿）
  
  高精度（不丢位姿）:
  python scripts/export_habitat_val_data.py --output_dir ... --no_dedupe --max_frames_per_scene 0  # 不去重、不截断
  
  要每场景都像 CrMo8WxCyVb 那样清晰（少白图）:
  python scripts/export_habitat_val_data.py --output_dir ... --poses start_only  # 仅用 episode 起点，不用目标视角
  
  FPS均匀采样（推荐用于建图）:
  python scripts/export_habitat_val_data.py --output_dir ... --max_frames_per_scene 1000 --sampling_method fps
"""

import argparse
import json
import os
import sys
import time
import math
import signal
import traceback
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import quaternion  # 添加quaternion库支持

# 项目根
NAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NAV_ROOT)

# 相机参数（与 instance_imagenav 及 habitat.yaml 一致）
CAM_W = 640  # Habitat默认是640x480
CAM_H = 480
HFOV_DEG = 90  # Habitat默认HFOV是90度，不是42度！
DEPTH_SCALE = 1000.0  # 深度存为 mm


def quat_to_rotmat(q):
    """Quaternion [x,y,z,w] or quaternion obj -> 3x3 R."""
    if hasattr(q, "w"):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    
    # 确保是单位四元数
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if abs(norm - 1.0) > 1e-6:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def pose_to_transform_matrix(position, rotation):
    """Habitat position (3,) + rotation (quat) -> 4x4 c2w list for transforms.json."""
    R = quat_to_rotmat(rotation)
    t = np.array(position, dtype=np.float64).reshape(3, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3:4] = t
    return c2w.tolist()


def get_camera_metadata():
    """计算准确的相机内参，与Habitat保持一致"""
    # 根据HFOV计算焦距
    # fx = (width/2) / tan(hfov/2)
    hfov_rad = math.radians(HFOV_DEG)
    fx = (CAM_W / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx  # 假设像素是正方形
    
    cx = (CAM_W - 1) / 2.0  # 光学中心通常在图像中心
    cy = (CAM_H - 1) / 2.0
    
    return {
        "w": CAM_W,
        "h": CAM_H,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "camera_model": "OPENCV",
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def collect_poses_from_episodes(dataset, scene_name, use_goal_view_points=True):
    """从数据集中收集指定场景的 (position, rotation) 用于采样。
    use_goal_view_points=False 时只收集 episode 起点，通常渲染更清晰。"""
    poses = []
    for ep in dataset.episodes:
        sid = os.path.basename(ep.scene_id).split(".")[0]
        if sid != scene_name:
            continue
        
        # start（起点一般在可通行处，渲染多为清晰室内图）
        pos = getattr(ep, "start_position", None)
        rot = getattr(ep, "start_rotation", None)
        if pos is not None and rot is not None:
            # 确保旋转是列表形式
            if hasattr(rot, "components"):
                rot = [rot.x, rot.y, rot.z, rot.w]
            elif hasattr(rot, "tolist"):
                rot = rot.tolist()
            poses.append((list(pos), rot))
        
        # goal view_points（可选；很多在墙内/不可见，易出白图）
        if use_goal_view_points and ep.goals and len(ep.goals) > 0:
            g = ep.goals[0]
            vps = getattr(g, "view_points", None)
            if vps:
                for v in vps:
                    astate = getattr(v, "agent_state", v)
                    p = getattr(astate, "position", None)
                    r = getattr(astate, "rotation", None)
                    if p is not None and r is not None:
                        if hasattr(r, "tolist"):
                            r = r.tolist()
                        elif hasattr(r, "components"):
                            r = [r.x, r.y, r.z, r.w]
                        poses.append((list(p), r))
    
    # 去重（基于位置）
    unique_poses = []
    seen_positions = set()
    for pos, rot in poses:
        pos_key = (round(pos[0], 2), round(pos[1], 2), round(pos[2], 2))
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_poses.append((pos, rot))
    
    return unique_poses


def is_valid_depth(depth, min_depth=0.1, max_depth=10.0):
    """检查深度图是否有效"""
    if depth is None or depth.size == 0:
        return False
    
    # 检查深度值范围
    depth_valid = depth[(depth > min_depth) & (depth < max_depth)]
    if len(depth_valid) < 0.1 * depth.size:  # 至少10%的像素有有效深度
        return False
    
    return True


def process_rgb_image(rgb):
    """安全处理RGB图像，确保输出正确的彩色图像"""
    if rgb is None:
        return None
    
    # 转换为numpy数组
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    elif hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    
    # 处理不同格式
    if rgb.ndim == 3:
        # 检查是否是 (3, H, W) 格式
        if rgb.shape[0] == 3 and rgb.shape[1] == CAM_H and rgb.shape[2] == CAM_W:
            rgb = np.transpose(rgb, (1, 2, 0))
        # 检查是否是 (H, W, 3) 格式
        elif rgb.shape[2] == 3:
            pass
        else:
            print(f"  警告：未知RGB形状 {rgb.shape}")
            return None
    
    # Habitat的RGB是uint8范围 [0, 255]
    if rgb.dtype == np.uint8:
        rgb_uint8 = rgb.copy()
    elif rgb.max() <= 1.0 and rgb.min() >= 0:
        rgb_uint8 = (rgb * 255).astype(np.uint8)
    elif rgb.max() <= 255 and rgb.min() >= 0:
        rgb_uint8 = rgb.astype(np.uint8)
    else:
        # 异常情况，归一化
        rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
    
    # 确保是RGB格式
    if len(rgb_uint8.shape) == 2:
        rgb_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_GRAY2RGB)
    elif rgb_uint8.shape[2] == 1:
        rgb_uint8 = cv2.cvtColor(rgb_uint8[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif rgb_uint8.shape[2] == 4:  # RGBA
        rgb_uint8 = rgb_uint8[:, :, :3]
    
    return rgb_uint8


def process_depth_image(depth):
    """安全处理深度图像"""
    if depth is None:
        return None
    
    # 转换为numpy数组
    if hasattr(depth, "numpy"):
        depth = depth.numpy()
    elif hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    
    # Habitat深度是float32，单位是米
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    
    # 处理不同形状
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]  # (1, H, W) -> (H, W)
    elif depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]  # (H, W, 1) -> (H, W)
    
    # 确保尺寸正确
    if depth.shape != (CAM_H, CAM_W):
        depth = cv2.resize(depth, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)
    
    return depth


def normalize_depth_to_meters_if_needed(depth_maybe_norm, sensor_max_depth_m):
    """将可能为[0,1]归一化的深度统一到米。"""
    if depth_maybe_norm is None:
        return None
    depth = depth_maybe_norm.astype(np.float32, copy=False)
    # Habitat 在 normalize_depth=True 时常返回 [0, 1]；此处做兜底恢复到米。
    if depth.size > 0 and float(np.nanmax(depth)) <= 1.5:
        depth = depth * float(sensor_max_depth_m)
    return depth


def farthest_point_sampling(poses, num_samples):
    """
    最远点采样（FPS），保证空间覆盖均匀性
    
    Args:
        poses: list of (position, rotation) tuples
        num_samples: 目标采样数量
    
    Returns:
        list of sampled poses
    """
    if len(poses) <= num_samples:
        return poses
    
    # 提取位置坐标
    positions = np.array([p[0] for p in poses])
    num_poses = len(positions)
    
    # 随机选择第一个点
    selected_indices = [np.random.randint(num_poses)]
    remaining_indices = list(range(num_poses))
    remaining_indices.remove(selected_indices[0])
    
    # 初始化距离数组（每个点到已选点集的最小距离）
    min_distances = np.full(num_poses, np.inf)
    for i in remaining_indices:
        min_distances[i] = np.linalg.norm(positions[i] - positions[selected_indices[0]])
    
    # 迭代选择最远点
    for _ in range(1, num_samples):
        # 选择距离最大的点
        if not remaining_indices:
            break
        farthest_idx = remaining_indices[np.argmax(min_distances[remaining_indices])]
        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)
        
        # 更新剩余点到已选点集的最小距离
        for i in remaining_indices:
            dist = np.linalg.norm(positions[i] - positions[farthest_idx])
            if dist < min_distances[i]:
                min_distances[i] = dist
    
    return [poses[i] for i in selected_indices]


def process_scene(args):
    """处理单个场景的函数，用于并行执行"""
    (scene_name, cfg_path, out_root, use_goal_vps, no_dedupe, 
     grid_res, max_frames, sampling_method, skip_white, white_mean_max) = args
    
    try:
        # 动态导入，避免进程间共享问题
        import habitat
        from habitat.config.default import get_config
        from habitat import make_dataset
        from vector_env.envs.instance_imagenav_env import NiceEnv
        
        print(f"[{scene_name}] 开始处理...")
        start_time = time.time()
        
        scene_dir = os.path.join(out_root, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        rgb_dir = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # 加载配置和数据集
        cfg = get_config(cfg_path)
        with habitat.config.read_write(cfg):
            cfg.habitat.dataset.split = "val"
            cfg.habitat.dataset.content_scenes = ["*"]
            # 确保传感器配置正确
            cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = CAM_H
            cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = CAM_W
            cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = int(HFOV_DEG)
            cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = CAM_H
            cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = CAM_W
            cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = int(HFOV_DEG)
            # 导出深度时统一用“米”，避免归一化深度(0~1)被误当作米。
            cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
            cfg.end2end_imagenav.split = "val"
        
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        
        # 收集位姿
        poses = collect_poses_from_episodes(dataset, scene_name, use_goal_view_points=use_goal_vps)
        
        if not poses:
            print(f"[{scene_name}] 跳过：无位姿")
            return scene_name, 0, 0, time.time() - start_time
        
        original_count = len(poses)
        
        # 更新配置为当前场景
        with habitat.config.read_write(cfg):
            cfg.habitat.dataset.content_scenes = [scene_name]
        
        sub_dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        env = NiceEnv(cfg, sub_dataset, rank=0)
        env.reset()
        
        cam_meta = get_camera_metadata()
        sensor_max_depth_m = float(cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth)
        frames_list = []
        written_count = 0
        skipped_invalid = 0
        
        # 采样
        if max_frames and max_frames > 0 and len(poses) > max_frames:
            before_sample = len(poses)
            if sampling_method == "fps":
                poses = farthest_point_sampling(poses, max_frames)
                print(f"[{scene_name}] FPS采样: {before_sample} → {len(poses)}")
            else:  # step sampling
                step = max(1, len(poses) // max_frames)
                poses = [poses[i] for i in range(0, len(poses), step)][:max_frames]
                print(f"[{scene_name}] 步长采样: {before_sample} → {len(poses)}")
        
        # 批量处理位姿
        total_poses = len(poses)
        for i, (pos, rot) in enumerate(poses):
            if i % 50 == 0:
                print(f"[{scene_name}] 进度: {i}/{total_poses}")
            
            try:
                # 确保旋转格式正确
                if isinstance(rot, list) and len(rot) == 4:
                    rot_quat = np.quaternion(rot[3], rot[0], rot[1], rot[2])  # w, x, y, z
                else:
                    rot_quat = rot
                
                obs = env.get_observation_at(np.array(pos), rot_quat)
            except Exception as e:
                print(f"[{scene_name}] 渲染失败: {e}")
                continue
            
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            
            if rgb is None or depth is None:
                continue
            
            # 处理RGB图像
            rgb_uint8 = process_rgb_image(rgb)
            if rgb_uint8 is None:
                continue
            
            # 处理深度
            depth_float = process_depth_image(depth)
            if depth_float is None:
                continue
            depth_float = normalize_depth_to_meters_if_needed(depth_float, sensor_max_depth_m)
            
            # 检查深度有效性
            if not is_valid_depth(depth_float):
                skipped_invalid += 1
                continue
            
            # 转换为mm保存
            depth_mm = (depth_float * DEPTH_SCALE).astype(np.uint16)
            
            # 保存文件
            idx = str(written_count).zfill(5)
            rgb_path = f"rgb/frame_{idx}.png"
            depth_path = f"depth/{idx}.png"
            
            # 保存RGB (BGR格式)
            cv2.imwrite(os.path.join(scene_dir, rgb_path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
            
            # 保存深度
            cv2.imwrite(os.path.join(scene_dir, depth_path), depth_mm)
            
            # 保存变换矩阵
            c2w = pose_to_transform_matrix(pos, rot)
            frames_list.append({
                "file_path": rgb_path,
                "depth_file_path": depth_path,
                "transform_matrix": c2w,
            })
            
            written_count += 1
        
        # 保存transforms.json
        transforms = {
            "frames": frames_list,
            **cam_meta,
            "aabb_scale": 16,  # 为Gaussian Splatting添加
        }
        
        with open(os.path.join(scene_dir, "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=2)
        
        # 也保存一个简洁版本
        transforms_simple = {
            "frames": [{k: v for k, v in frame.items() if k != "transform_matrix"} 
                      for frame in frames_list],
            **cam_meta
        }
        with open(os.path.join(scene_dir, "transforms_simple.json"), "w") as f:
            json.dump(transforms_simple, f, indent=2)
        
        env.close()
        
        elapsed = time.time() - start_time
        print(f"[{scene_name}] 完成: {written_count}帧 (跳过{skipped_invalid}无效), 耗时{elapsed:.1f}秒")
        return scene_name, written_count, skipped_invalid, elapsed
        
    except Exception as e:
        print(f"[{scene_name}] 错误: {e}")
        traceback.print_exc()
        return scene_name, 0, 0, 0


def signal_handler(sig, frame):
    """处理Ctrl+C"""
    print("\n用户中断，正在退出...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Export HM3D val scenes to habitat_val (rgb/depth/transforms.json).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root, e.g. ../DBSGConstruction/configs/data/habitat_val",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(NAV_ROOT, "configs", "instance_imagenav_val_nogt.yaml"),
        help="Habitat 配置（脚本内会强制 dataset.split=val）。默认用 val_nogt；若需 GT 评测配置可传 configs/instance_imagenav.yaml 等",
    )
    parser.add_argument("--max_scenes", type=int, default=None, help="Max number of scenes to export (default: all).")
    parser.add_argument(
        "--max_frames_per_scene",
        type=int,
        default=500,
        help="每场景最多帧数（去重后下采样）。0 表示不限制，保留全部位姿；默认 500",
    )
    parser.add_argument("--grid_res", type=float, default=0.5, help="位姿去重网格分辨率(m)。越小保留位姿越多")
    parser.add_argument(
        "--no_dedupe",
        action="store_true",
        help="不去重：保留数据集中每一个位姿，最精确；与 --max_frames_per_scene 0 同用可完全不丢帧",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default="all",
        choices=("all", "start_only"),
        help="位姿来源。all=起点+目标视角(易出白图)；start_only=仅 episode 起点，清晰图比例高",
    )
    parser.add_argument("--skip_white_frames", action="store_true", help="跳过几乎全白的渲染帧（坏位姿）")
    parser.add_argument(
        "--white_mean_max",
        type=float,
        default=250,
        help="与 --skip_white_frames 同用：图像均值超过此值视为白图并跳过；默认 250",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print scene list and pose counts")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="并行处理的进程数。默认1（串行），建议设置为CPU核心数",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="fps",
        choices=("step", "fps"),
        help="采样方法: step=步长采样, fps=最远点采样（更均匀，推荐用于建图）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，打印更多信息",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90,
        help="水平视场角（度），Habitat默认90度",
    )
    
    args = parser.parse_args()
    
    # 设置HFOV
    global HFOV_DEG
    HFOV_DEG = args.hfov
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    out_root = args.output_dir or os.path.join(
        os.path.dirname(NAV_ROOT), "DBSGConstruction", "configs", "data", "habitat_val"
    )
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"输出目录: {out_root}")
    print(f"采样方法: {args.sampling_method}")
    print(f"HFOV: {HFOV_DEG}度")
    print(f"图像尺寸: {CAM_W}x{CAM_H}")
    
    # 导入Habitat相关模块
    import habitat
    from habitat.config.default import get_config
    from habitat import make_dataset
    
    cfg = get_config(args.config)
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.split = "val"
        cfg.habitat.dataset.content_scenes = ["*"]
        cfg.end2end_imagenav.split = "val"
    
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    scene_ids = dataset.get_scenes_to_load(cfg.habitat.dataset)
    scene_names = sorted({os.path.basename(s).split(".")[0] for s in scene_ids})
    
    if args.max_scenes is not None:
        scene_names = scene_names[:args.max_scenes]
    
    print(f"待处理场景数: {len(scene_names)}")
    
    use_goal_vps = args.poses == "all"
    
    # Dry run模式
    if args.dry_run:
        for sn in scene_names:
            poses = collect_poses_from_episodes(dataset, sn, use_goal_view_points=use_goal_vps)
            msg = f"  {sn}: {len(poses)} poses"
            if args.max_frames_per_scene and args.max_frames_per_scene > 0 and len(poses) > args.max_frames_per_scene:
                msg += f" → {args.max_frames_per_scene} (after {args.sampling_method} sampling)"
            print(msg)
        return
    
    # 准备任务参数
    task_args = []
    for scene_name in scene_names:
        task_args.append((
            scene_name,
            args.config,
            out_root,
            use_goal_vps,
            args.no_dedupe,
            args.grid_res,
            args.max_frames_per_scene,
            args.sampling_method,
            args.skip_white_frames,
            args.white_mean_max,
        ))
    
    # 并行处理
    total_start = time.time()
    total_frames = 0
    total_skipped = 0
    scene_results = []
    
    if args.num_workers > 1 and len(scene_names) > 1:
        print(f"使用 {args.num_workers} 个进程并行处理...")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_scene, task) for task in task_args]
            
            for future in as_completed(futures):
                try:
                    scene_name, frames, skipped, elapsed = future.result(timeout=300)
                    scene_results.append((scene_name, frames, skipped, elapsed))
                    total_frames += frames
                    total_skipped += skipped
                    print(f"进度: {scene_name} 完成")
                except Exception as e:
                    print(f"任务失败: {e}")
    else:
        print("串行处理...")
        for task in task_args:
            scene_name, frames, skipped, elapsed = process_scene(task)
            scene_results.append((scene_name, frames, skipped, elapsed))
            total_frames += frames
            total_skipped += skipped
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print(f"处理完成！总耗时: {total_elapsed:.1f}秒")
    print(f"总帧数: {total_frames}")
    print(f"总跳过无效帧: {total_skipped}")
    print(f"输出目录: {out_root}")
    print(f"采样方法: {args.sampling_method}")
    print(f"HFOV: {HFOV_DEG}度")
    
    # 打印每个场景的结果
    print("\n场景详情:")
    for scene_name, frames, skipped, elapsed in scene_results:
        if frames > 0:
            print(f"  {scene_name}: {frames}帧 (跳过{skipped}), 耗时{elapsed:.1f}秒")
    
    print("="*60)
    
    # 提示下一步
    print("\n下一步:")
    print(f"1. 检查输出目录: {out_root}")
    print("2. 运行 segmentation:")
    print(f"   python scripts/run_ov_seg.py --data_dir {out_root}")
    print("3. 运行点云转换:")
    print(f"   python scripts/transform_pointcloud.py --data_dir {out_root}")
    print("4. 运行实例特征提取:")
    print(f"   python scripts/build_instance_clip_vectors.py --data_dir {out_root}")
    print("5. 运行Gaussian建图:")
    print(f"   python train.py -s {out_root} -m output/your_scene_name")


if __name__ == "__main__":
    main()