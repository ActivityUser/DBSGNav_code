#!/usr/bin/env python3
"""
根据 seg_semantic/*.npy 与 depth/*.png 生成 instance_retrieval.json
适配具体的目录结构：scenes_root/habitat_val/场景名/
深度图单位：mm
"""

import sys
import os.path as osp
import torch
import numpy as np
import cv2
from glob import glob
import os
from natsort import natsorted
from tqdm import tqdm
import skimage
import skimage.morphology
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 颜色映射（保持不变）
color_mapping = {
    56: [0.9400000000000001, 0.7818, 0.66],
    57: [0.8882000000000001, 0.9400000000000001, 0.66],
    58: [0.66, 0.9400000000000001, 0.8518000000000001],
    59: [0.7117999999999999, 0.66, 0.9400000000000001],
    61: [0.9218, 0.66, 0.9400000000000001],
    62: [0.9400000000000001, 0.66, 0.748199999999999]
}


def preprocess_mask(depth: np.ndarray, mask: Dict, conf: float = 0.5) -> Dict:
    """
    预处理掩码，与原函数完全相同的功能，但更健壮
    """
    new_mask = {}
    
    for key in mask:
        if mask[key] is None:
            new_mask[key] = None
            continue
            
        new_curr_mask_list = []
        curr_mask_list = mask[key]
        
        for curr_mask in curr_mask_list:
            if curr_mask is None or not np.any(curr_mask):
                continue
                
            try:
                # 形态学腐蚀
                selem = skimage.morphology.disk(3)
                eroded_mask = skimage.morphology.erosion(curr_mask, selem)
                
                if not np.any(eroded_mask):
                    continue
                
                # 基于深度一致性过滤
                new_depth = depth.copy()
                new_masked_depth = new_depth[eroded_mask]
                
                if len(new_masked_depth) == 0:
                    continue
                
                mean = np.mean(new_masked_depth)
                std = np.std(new_masked_depth)
                
                threshold = conf * std
                curr_mask_copy = curr_mask.copy()
                
                # 只保留深度值在合理范围内的像素
                depth_diff = np.abs(depth - mean)
                curr_mask_filtered = np.where(depth_diff > threshold, False, curr_mask)
                curr_mask_filtered = np.logical_and(curr_mask_copy, curr_mask_filtered)
                
                if np.any(curr_mask_filtered):
                    new_curr_mask_list.append(curr_mask_filtered)
                    
            except Exception as e:
                print(f"警告：预处理掩码时出错: {e}")
                continue
        
        new_mask[key] = new_curr_mask_list if len(new_curr_mask_list) > 0 else None
    
    return new_mask


def get_pointcloud(mask: Dict, depth: np.ndarray, w2c: torch.Tensor, 
                   transform_pts: bool = True, return_bbox: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    从深度图和掩码生成点云 - 针对mm单位深度图的修正版
    """
    # 获取深度图尺寸
    height, width = depth.shape
    
    # 关键修复：将深度从mm转换为米
    depth_m = depth.astype(np.float32) / 1000.0
    
    # 转换为tensor
    depth_tensor = torch.from_numpy(depth_m).cuda()
    
    # 相机内参 - Habitat模拟器的默认内参
    # 根据图像尺寸调整
    CX = width / 2.0  # 使用图像中心作为光心
    CY = height / 2.0
    
    # 假设视场角为90度，计算焦距
    fov = 90  # 度
    FX = width / (2 * np.tan(np.radians(fov) / 2))
    FY = height / (2 * np.tan(np.radians(fov) / 2))
    
    print(f"使用相机内参: FX={FX:.2f}, FY={FY:.2f}, CX={CX:.2f}, CY={CY:.2f}")
    
    # 生成像素坐标网格
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).cuda().float(),
        torch.arange(height).cuda().float(),
        indexing='xy'
    )
    
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth_tensor.reshape(-1)
    
    # 过滤无效深度值
    valid_depth = depth_z > 0.1  # 过滤掉太近的点
    xx = xx[valid_depth]
    yy = yy[valid_depth]
    depth_z = depth_z[valid_depth]
    
    # 计算相机坐标系下的点
    # Habitat坐标系: x-right, y-up, z-back
    pts_cam = torch.stack((xx * depth_z, -yy * depth_z, -depth_z), dim=-1)
    
    if transform_pts and len(pts_cam) > 0:
        # 添加齐次坐标
        pix_ones = torch.ones(len(pts_cam), 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        
        # 转换到世界坐标系
        pts = (w2c @ pts4.T).T[:, :3]
    else:
        pts = pts_cam
    
    # 初始化颜色和包围盒
    if len(pts) > 0:
        cols = torch.ones((len(pts), 3), device='cuda:0', dtype=torch.float32) * 0.9
    else:
        cols = torch.ones((0, 3), device='cuda:0', dtype=torch.float32)
    
    instance_bbox = {}
    
    # 为每个实例上色并计算包围盒
    for key in mask:
        if mask[key] is not None and len(mask[key]) > 0:
            curr_bbox_list = []
            
            for curr_mask in mask[key]:
                if curr_mask is None or not np.any(curr_mask):
                    continue
                    
                # 确保掩码是布尔类型
                curr_mask_bool = curr_mask.astype(bool)
                curr_mask_tensor = torch.from_numpy(curr_mask_bool).cuda()
                curr_mask_tensor = curr_mask_tensor.reshape(-1)
                
                # 只保留有效深度对应的掩码
                curr_mask_tensor = curr_mask_tensor[valid_depth.cpu().numpy()]
                
                if not curr_mask_tensor.any():
                    continue
                
                # 上色
                if curr_mask_tensor.any():
                    cols[curr_mask_tensor] = torch.tensor(color_mapping[key], dtype=torch.float32).cuda()
                
                # 计算包围盒
                if return_bbox:
                    masked_pts = pts[curr_mask_tensor]
                    
                    if len(masked_pts) > 0:
                        min_coords = masked_pts.min(dim=0)[0]
                        max_coords = masked_pts.max(dim=0)[0]
                        
                        curr_bbox_list.append([
                            min_coords[0].item(), max_coords[0].item(),
                            min_coords[1].item(), max_coords[1].item(),
                            min_coords[2].item(), max_coords[2].item()
                        ])
            
            instance_bbox[key] = curr_bbox_list if curr_bbox_list else None
        else:
            instance_bbox[key] = None
    
    return pts, cols, instance_bbox


def voxelize(pointcloud: torch.Tensor, voxel_size: float = 0.05) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    体素化点云
    """
    if len(pointcloud) == 0:
        return torch.tensor([]).cuda(), np.array([0,0,0]), np.array([0,0,0])
    
    xyz_min_bounds = pointcloud.min(0)[0]
    xyz_max_bounds = pointcloud.max(0)[0]
    
    # 添加小偏移以避免除零
    xyz_max_bounds = xyz_max_bounds + 1e-6
    
    grid_sizes = ((xyz_max_bounds - xyz_min_bounds) / voxel_size).long() + 1
    
    # 限制网格大小
    max_grid_size = 200
    if grid_sizes.max() > max_grid_size:
        voxel_size = (xyz_max_bounds - xyz_min_bounds).max().item() / max_grid_size
        grid_sizes = ((xyz_max_bounds - xyz_min_bounds) / voxel_size).long() + 1
    
    grid_sizes = tuple(max(1, int(s)) for s in grid_sizes.tolist())
    
    # 如果网格太大，返回空
    if grid_sizes[0] * grid_sizes[1] * grid_sizes[2] > 1e7:
        print(f"警告：体素网格过大 {grid_sizes}，跳过")
        return torch.tensor([]).cuda(), xyz_min_bounds.cpu().numpy(), xyz_max_bounds.cpu().numpy()
    
    voxel_grid = torch.zeros(grid_sizes, device='cuda:0', dtype=torch.float32)
    
    indices = ((pointcloud - xyz_min_bounds) / voxel_size).long()
    
    # 确保索引在有效范围内
    for i in range(3):
        indices[:, i] = torch.clamp(indices[:, i], 0, grid_sizes[i] - 1)
    
    # 使用唯一值来避免重复索引
    unique_indices = torch.unique(indices, dim=0)
    voxel_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = 1
    
    return voxel_grid, xyz_min_bounds.cpu().numpy(), xyz_max_bounds.cpu().numpy()


def find_navigable_viewpoints(instance_center: List[float], 
                             voxel_grid: torch.Tensor,
                             voxel_min_bounds: np.ndarray,
                             voxel_size: float,
                             camera_height: float,
                             max_radius: int = 40) -> List[List[float]]:
    """
    寻找实例周围的可导航视角点
    """
    navigable_viewpoints = []
    
    if len(voxel_grid) == 0 or voxel_grid.numel() == 0:
        return navigable_viewpoints
    
    # 处理输入中心点格式
    if len(instance_center) >= 3:
        center_x, center_y, center_z = instance_center[:3]
    else:
        center_x, center_z = instance_center
        center_y = camera_height
    
    # 计算相机高度对应的体素范围
    min_y = int((camera_height - 0.5 - voxel_min_bounds[1]) / voxel_size)
    max_y = int((camera_height + 0.5 - voxel_min_bounds[1]) / voxel_size)
    
    # 确保索引有效
    min_y = max(0, min(min_y, voxel_grid.shape[1] - 1))
    max_y = max(0, min(max_y, voxel_grid.shape[1] - 1))
    
    if min_y >= max_y:
        return navigable_viewpoints
    
    # 创建2D可通行地图
    try:
        grid_map_3d = voxel_grid[:, min_y:max_y, :]
        grid_map = torch.sum(grid_map_3d, dim=1)
        grid_map = (grid_map > 0).float().cpu().numpy()
    except:
        return navigable_viewpoints
    
    # 形态学膨胀
    selem = skimage.morphology.disk(3)
    dilated_grid_map = skimage.morphology.dilation(grid_map, selem)
    
    # 将实例中心转换到地图坐标
    map_center_x = int((center_x - voxel_min_bounds[0]) / voxel_size)
    map_center_z = int((center_z - voxel_min_bounds[2]) / voxel_size)
    
    h, w = dilated_grid_map.shape
    if not (0 <= map_center_x < h and 0 <= map_center_z < w):
        return navigable_viewpoints
    
    # 搜索可导航点
    found_points = []
    
    for r in range(3, max_radius):
        try:
            # 生成环形区域的掩码
            yy, xx = np.ogrid[:h, :w]
            distance_map = np.sqrt((xx - map_center_z)**2 + (yy - map_center_x)**2)
            
            # 找到距离在r-2到r之间的环形区域
            ring_mask = (distance_map > r-2) & (distance_map <= r)
            
            # 在环形区域内找到可通行的点
            navigable_in_ring = ring_mask & (dilated_grid_map == 1)
            
            if np.any(navigable_in_ring):
                # 找到第一个可通行的点
                y_idx, x_idx = np.where(navigable_in_ring)
                
                # 转换回世界坐标
                world_x = y_idx[0] * voxel_size + voxel_min_bounds[0]
                world_z = x_idx[0] * voxel_size + voxel_min_bounds[2]
                found_points.append([float(world_x), float(world_z)])
                break
                
        except Exception as e:
            continue
    
    return found_points


def _atomic_json_dump(path: str, data: Dict) -> None:
    """原子写入 JSON，避免中断时留下半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _load_instance_retrieval_partial(path: str) -> Optional[Dict]:
    """加载已有 instance_retrieval.json；失败返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return None
        # 统一为 int 类 id 键，与 process_scene 内一致
        out: Dict[int, List] = {}
        for k, v in raw.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                continue
            out[ik] = v if isinstance(v, list) else []
        return out
    except Exception:
        return None


def _norm_frame_id(x: Any) -> str:
    """统一帧号：00000 / 0 / \"0\" -> \"0\"，避免续跑时重复处理。"""
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    return s


def _processed_frame_ids_from_ir(data: Dict[int, List]) -> set:
    """从已有检索结果中收集已处理完的帧 id（规范化后的字符串）。"""
    done: set = set()
    for lst in data.values():
        if not isinstance(lst, list):
            continue
        for entry in lst:
            if isinstance(entry, list) and len(entry) >= 2:
                done.add(_norm_frame_id(entry[1]))
    return done


def load_transforms(transforms_path: str) -> List:
    """
    加载transforms.json文件
    """
    try:
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        # 处理不同的json格式
        if 'frames' in data:
            return data['frames']
        elif isinstance(data, list):
            return data
        else:
            print(f"警告：未知的transforms格式")
            return []
    except Exception as e:
        print(f"警告：加载transforms.json失败: {e}")
        return []


def process_scene(
    scene_path: str,
    resume: bool = True,
    checkpoint_every: int = 5,
    out_json_path: Optional[str] = None,
) -> Dict:
    """
    处理单个场景。
    resume=True 时：若已有 instance_retrieval.json，则加载并跳过其中已记录的帧，并周期性写回检查点。
    """
    json_path = out_json_path or os.path.join(scene_path, "instance_retrieval.json")
    json_dump_data: Dict[int, List] = {56: [], 57: [], 58: [], 59: [], 61: [], 62: []}
    if resume:
        loaded = _load_instance_retrieval_partial(json_path)
        if loaded is not None:
            for k, v in loaded.items():
                json_dump_data[int(k)] = v if isinstance(v, list) else []
            for k in (56, 57, 58, 59, 61, 62):
                json_dump_data.setdefault(k, [])
    processed_frames = _processed_frame_ids_from_ir(json_dump_data) if resume else set()

    # 检查必要的目录
    depth_dir = os.path.join(scene_path, 'depth')
    seg_dir = os.path.join(scene_path, 'seg_semantic')
    
    if not os.path.exists(depth_dir):
        print(f"警告：{depth_dir} 不存在")
        return json_dump_data
    
    if not os.path.exists(seg_dir):
        print(f"警告：{seg_dir} 不存在")
        return json_dump_data
    
    # 尝试加载transforms.json（可能有不同名称）
    transforms_paths = [
        os.path.join(scene_path, 'transforms.json'),
        os.path.join(scene_path, 'transforms_simple.json')
    ]
    
    transforms_path = None
    for path in transforms_paths:
        if os.path.exists(path):
            transforms_path = path
            break
    
    if transforms_path is None:
        print(f"警告：找不到transforms.json文件")
        return json_dump_data
    
    print(f"加载 {transforms_path}")
    frames = load_transforms(transforms_path)
    
    if not frames:
        print(f"警告：transforms.json中没有有效数据")
        return json_dump_data
    
    # 加载所有相机位姿
    all_w2cs = []
    for i in range(len(frames)):
        try:
            w2c = torch.tensor(frames[i]['transform_matrix'], dtype=torch.float32).cuda()
            all_w2cs.append(w2c)
        except:
            all_w2cs.append(None)
    
    # 获取所有分割文件
    seg_masks_list = glob(os.path.join(seg_dir, "*.npy"))
    seg_masks_list = natsorted(seg_masks_list)
    
    if not seg_masks_list:
        print(f"警告：{seg_dir} 中没有分割文件")
        return json_dump_data
    
    print(f"找到 {len(seg_masks_list)} 个分割文件")
    if processed_frames:
        print(f"断点续跑：将跳过已存在 instance_retrieval 中的 {len(processed_frames)} 个帧 id")

    n_seg = len(seg_masks_list)
    # 处理每一帧
    for seg_i, seg_mask_path in enumerate(
        tqdm(seg_masks_list, desc=f"处理 {os.path.basename(scene_path)}"), start=1
    ):
        try:
            # 解析帧索引
            filename = os.path.basename(seg_mask_path)
            # 支持 frame_00000.npy 或 00000.npy 格式
            index = filename.split('_')[-1].split('.')[0]

            if resume and _norm_frame_id(index) in processed_frames:
                continue

            # 加载掩码
            mask_data = np.load(seg_mask_path, allow_pickle=True).item()
            
            # 加载深度图
            depth_file_path = os.path.join(depth_dir, f'{index}.png')
            if not os.path.exists(depth_file_path):
                # 尝试其他格式
                depth_file_path = os.path.join(depth_dir, f'frame_{index}.png')
                if not os.path.exists(depth_file_path):
                    continue
            
            depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            if depth_data is None:
                continue
            
            # 获取相机位姿
            try:
                frame_idx = int(index)
                if frame_idx >= len(all_w2cs):
                    continue
                w2c = all_w2cs[frame_idx]
                if w2c is None:
                    continue
            except (IndexError, ValueError):
                continue
            
            # 预处理掩码
            mask_data = preprocess_mask(depth_data, mask_data)
            
            # 生成点云和包围盒
            pts, cols, bbox = get_pointcloud(mask_data, depth_data, w2c)
            
            # 如果没有有效点，跳过
            if len(pts) == 0:
                continue
            
            # 体素化
            voxel_size = 0.1
            voxel_grid, voxel_min_bounds, voxel_max_bounds = voxelize(pts, voxel_size)
            
            # 相机高度
            camera_height = w2c[1, 3].item()
            
            # 处理每个实例
            for key in bbox:
                if bbox[key] is not None and len(bbox[key]) > 0:
                    navigable_viewpoint = []
                    
                    for bbox_i in range(len(bbox[key])):
                        curr_bbox = bbox[key][bbox_i]
                        
                        # 计算实例中心
                        curr_center = [
                            (curr_bbox[0] + curr_bbox[1]) / 2,
                            (curr_bbox[2] + curr_bbox[3]) / 2,
                            (curr_bbox[4] + curr_bbox[5]) / 2
                        ]
                        
                        # 寻找可导航视角点
                        viewpoints = find_navigable_viewpoints(
                            curr_center, voxel_grid, voxel_min_bounds, 
                            voxel_size, camera_height
                        )
                        
                        navigable_viewpoint.append(viewpoints if viewpoints else None)
                    
                    json_dump_data[key].append([bbox[key], index, navigable_viewpoint])

            processed_frames.add(_norm_frame_id(index))
            if checkpoint_every > 0 and (
                seg_i % checkpoint_every == 0 or seg_i == n_seg
            ):
                _atomic_json_dump(json_path, json_dump_data)

        except Exception as e:
            print(f"警告：处理 {seg_mask_path} 时出错: {e}")
            continue

    # 最终落盘（覆盖检查点，保证与内存一致）
    _atomic_json_dump(json_path, json_dump_data)
    return json_dump_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 seg_semantic/*.npy 与 depth/*.png 生成 instance_retrieval.json"
    )
    parser.add_argument(
        "--scenes_root",
        type=str,
        default="../DBSGConstruction/configs/data",
        help="场景根目录，包含 habitat/ 和 habitat_val/ 子目录",
    )
    parser.add_argument(
        "--scenes_file",
        type=str,
        default=None,
        help="可选。每行一个 scene 目录名，只处理这些场景；不传则处理 scenes_root/habitat_val 下全部子目录。",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="habitat_val",
        choices=["habitat", "habitat_val", "habitat_train"],
        help="数据集类型：habitat, habitat_val, 或 habitat_train",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有 instance_retrieval.json，每个场景从头重算",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5,
        metavar="N",
        help="每处理完 N 帧写入一次检查点（默认 5；SSH 易断可改为 1，略增 IO）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 构建完整的场景根目录
    full_scenes_root = os.path.join(args.scenes_root, args.dataset_type)
    
    if not os.path.isdir(full_scenes_root):
        print(f"Error: {full_scenes_root} 不存在")
        sys.exit(1)
    
    # 获取场景列表
    if args.scenes_file and os.path.isfile(args.scenes_file):
        with open(args.scenes_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        scenes_list = [os.path.join(full_scenes_root, n) for n in names]
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"从 --scenes_file 加载 {len(names)} 个场景，有效 {len(scenes_list)} 个")
    else:
        scenes_list = glob(os.path.join(full_scenes_root, '*'))
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"在 {full_scenes_root} 中找到 {len(scenes_list)} 个场景")
    
    if not scenes_list:
        print(f"警告：没有找到任何场景")
        sys.exit(0)
    
    # 处理每个场景
    for scene in tqdm(scenes_list, desc="处理场景"):
        try:
            print(f'\n********** {os.path.basename(scene)} **********')
            
            json_file_path = os.path.join(scene, "instance_retrieval.json")
            json_dump_data = process_scene(
                scene,
                resume=not args.no_resume,
                checkpoint_every=0 if args.no_resume else max(1, args.checkpoint_every),
                out_json_path=json_file_path,
            )
            total_instances = sum(len(v) for v in json_dump_data.values())
            print(f"完成：{total_instances} 个实例，结果保存到 {json_file_path}")
            
        except Exception as e:
            print(f"错误：处理 {scene} 失败")
            traceback.print_exc()
            continue
    
    print("\n所有场景处理完成！")


if __name__ == "__main__":
    main()