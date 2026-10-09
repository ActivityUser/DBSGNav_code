#!/usr/bin/env python3
"""
从 Habitat HM3D val 场景导出建图所需数据（transforms.json + rgb/ + depth/），
供 DBSGConstruction 的 run_build_val_semantic_gaussians.py 使用。

用法（在 DBSGNavigation 目录下）:
  # 导出单个场景（需 val 数据集中有该场景的 episode）
  python scripts/export_habitat_val_for_construction.py --sequence cvZr5TUy5C5 --output_dir ../DBSGConstruction/configs/data/habitat

  # 导出数据集中所有场景（每个场景用一条 trajectory，最多 max_frames 帧）
  python scripts/export_habitat_val_for_construction.py --output_dir ../DBSGConstruction/configs/data/habitat --max_frames 300

  # 指定配置文件与最大帧数
  python scripts/export_habitat_val_for_construction.py --config configs/instance_imagenav.yaml --sequence 6s7QHgap2fw --output_dir ../DBSGConstruction/configs/data/habitat --max_frames 200

要求：
  - 已安装 habitat-lab，且 instance_imagenav 的 dataset 能正常加载（HM3D val 等）
  - 每个场景至少有一条 episode，导出时用该 episode 的 trajectory 采帧
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import quaternion

# 与 DBSGConstruction 默认配置对齐：W=640, H=480（landscape）
TARGET_W = 640
TARGET_H = 480
DEPTH_SCALE = 1000.0  # m -> mm

# 保证能导入 habitat 与项目 config
_NAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NAV_ROOT not in sys.path:
    sys.path.insert(0, _NAV_ROOT)


def _scene_name_from_scene_id(scene_id: str) -> str:
    return os.path.basename(scene_id).split(".")[0]


def _agent_pose_to_c2w(position, rotation):
    """Habitat agent position (3,) + rotation (quaternion) -> 4x4 c2w (camera-to-world)."""
    R = quaternion.as_rotation_matrix(rotation)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = np.array(position)
    return c2w.tolist()


def _get_camera_intrinsics_from_config(config):
    """从 habitat simulator 的 rgb_sensor 得到 fl_x, fl_y, cx, cy, w, h."""
    w = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width
    h = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height
    hfov_deg = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
    hfov_rad = np.deg2rad(hfov_deg)
    fl = (w / 2.0) / np.tan(hfov_rad / 2.0)
    cx = w / 2.0
    cy = h / 2.0
    return {"fl_x": fl, "fl_y": fl, "cx": cx, "cy": cy, "w": int(w), "h": int(h)}


def _process_rgb_image(rgb, target_h: int, target_w: int):
    """统一 RGB 到 uint8 且尺寸为 (target_h, target_w)."""
    if rgb is None:
        return None

    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    elif hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    rgb = np.asarray(rgb)

    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.clip(0, 255).astype(np.uint8)

    if rgb.shape[:2] != (target_h, target_w):
        import cv2
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _process_depth_image(depth, target_h: int, target_w: int):
    """统一深度为 float32(H, W)，单位暂未假设（可能米，也可能[0,1]归一化）。"""
    if depth is None:
        return None

    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    elif hasattr(depth, "numpy"):
        depth = depth.numpy()
    depth = np.asarray(depth)

    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    elif depth.ndim == 3:
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32, copy=False)

    if depth.shape[:2] != (target_h, target_w):
        import cv2
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return depth


def _normalize_depth_to_meters_if_needed(depth_maybe_norm, sensor_max_depth_m: float):
    """若深度看起来是[0,1]归一化值，则恢复到米。"""
    if depth_maybe_norm is None:
        return None
    depth = depth_maybe_norm.astype(np.float32, copy=False)
    finite = np.isfinite(depth)
    if finite.any():
        dmax = float(np.nanmax(depth[finite]))
        if dmax <= 1.5:
            depth = depth * float(sensor_max_depth_m)
    return depth


def run_export(
    config_path: str,
    output_dir: str,
    sequence: str | None,
    max_frames: int,
    skip_existing: bool,
):
    import habitat
    from habitat.config.default import get_config
    from habitat import make_dataset

    config = get_config(config_path)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = getattr(config.end2end_imagenav, "split", "val")
        # 与建图配置保持一致：统一传感器分辨率，避免 train 导出为竖屏 480x640。
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = TARGET_W
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = TARGET_H
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = TARGET_W
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = TARGET_H
        # 导出深度优先用米；若上游仍给归一化值，后续会兜底恢复。
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False

    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    scene_to_episodes = {}
    for ep in dataset.episodes:
        sn = _scene_name_from_scene_id(ep.scene_id)
        if sn not in scene_to_episodes:
            scene_to_episodes[sn] = []
        scene_to_episodes[sn].append(ep)

    if sequence:
        scenes = [s for s in [sequence.strip()] if s]
        if not scenes:
            print("Error: --sequence is empty.", file=sys.stderr)
            sys.exit(1)
        for s in scenes:
            if s not in scene_to_episodes:
                print(f"Warning: no episode for scene '{s}', skip.", file=sys.stderr)
        scenes = [s for s in scenes if s in scene_to_episodes]
    else:
        scenes = sorted(scene_to_episodes.keys())

    if not scenes:
        print("No scenes to export.", file=sys.stderr)
        sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)
    intrinsics = _get_camera_intrinsics_from_config(config)

    for scene_name in scenes:
        out_scene = os.path.join(output_dir, scene_name)
        transforms_path = os.path.join(out_scene, "transforms.json")
        if skip_existing and os.path.isfile(transforms_path):
            print(f"[skip] {scene_name} (exists: {transforms_path})")
            continue

        # 兼容不同 Habitat 版本：避免使用 reset(episode=...) 这种新旧 API 不一致的写法。
        scene_cfg = get_config(config_path)
        with habitat.config.read_write(scene_cfg):
            scene_cfg.habitat.dataset.split = getattr(scene_cfg.end2end_imagenav, "split", "val")
            scene_cfg.habitat.dataset.content_scenes = [scene_name]
            scene_cfg.end2end_imagenav.split = scene_cfg.habitat.dataset.split
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = TARGET_W
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = TARGET_H
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = TARGET_W
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = TARGET_H
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        scene_ds = make_dataset(scene_cfg.habitat.dataset.type, config=scene_cfg.habitat.dataset)
        if len(scene_ds.episodes) == 0:
            print(f"Warning: no episode for scene '{scene_name}', skip.")
            continue
        intrinsics = _get_camera_intrinsics_from_config(scene_cfg)
        env = habitat.Env(config=scene_cfg, dataset=scene_ds)
        obs = env.reset()
        rgb_dir = os.path.join(out_scene, "rgb")
        depth_dir = os.path.join(out_scene, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        frames = []
        num_frames = 0
        max_depth = float(getattr(
            scene_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor,
            "max_depth", 10.0
        ))
        import cv2

        while num_frames < max_frames:
            rgb = obs.get("rgb")
            depth = obs.get("depth")
            if rgb is None:
                break
            rgb = _process_rgb_image(rgb, intrinsics["h"], intrinsics["w"])
            depth = _process_depth_image(depth, intrinsics["h"], intrinsics["w"])
            if depth is not None:
                depth_m = _normalize_depth_to_meters_if_needed(depth, max_depth)
                depth_m = np.clip(depth_m, 0.0, max_depth)
                depth_mm = (depth_m * DEPTH_SCALE).clip(0, 65535).astype(np.uint16)
            else:
                depth_mm = np.zeros((intrinsics["h"], intrinsics["w"]), dtype=np.uint16)

            frame_id = f"{num_frames:05d}"
            rgb_path = os.path.join("rgb", f"frame_{frame_id}.png")
            depth_path = os.path.join("depth", f"{frame_id}.png")
            cv2.imwrite(os.path.join(out_scene, rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_scene, depth_path), depth_mm)

            agent_state = env.sim.get_agent_state(0)
            c2w = _agent_pose_to_c2w(agent_state.position, agent_state.rotation)
            frames.append({
                "file_path": rgb_path.replace("\\", "/"),
                "depth_file_path": depth_path.replace("\\", "/"),
                "transform_matrix": c2w,
            })
            num_frames += 1

            if num_frames >= max_frames:
                break
            # 随机前进一步或左/右转（Habitat 动作: 0=stop, 1=forward, 2=left, 3=right）
            action = np.random.choice([1, 2, 3], p=[0.7, 0.15, 0.15])
            step_result = env.step(action)
            obs = step_result[0] if isinstance(step_result, (tuple, list)) else step_result
            if isinstance(step_result, (tuple, list)) and len(step_result) > 2 and step_result[2]:
                break  # episode done

        if num_frames == 0:
            print(f"Warning: 0 frames for {scene_name}, skip.")
            continue

        meta = {
            "camera_model": "OPENCV",
            "fl_x": intrinsics["fl_x"],
            "fl_y": intrinsics["fl_y"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
            "w": intrinsics["w"],
            "h": intrinsics["h"],
            "frames": frames,
        }
        with open(transforms_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Exported {scene_name}: {num_frames} frames -> {out_scene}")
        env.close()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Export Habitat val scenes to construction data (transforms.json + rgb/ + depth/)."
    )
    parser.add_argument("--config", type=str,
                        default=os.path.join(_NAV_ROOT, "configs", "instance_imagenav.yaml"),
                        help="Path to instance_imagenav config.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_NAV_ROOT, "..", "DBSGConstruction", "configs", "data", "habitat"),
                        help="Output root: each scene will be output_dir/<scene_id>/.")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Single scene_id to export (e.g. cvZr5TUy5C5). If not set, export all scenes in dataset.")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames per scene (default 300).")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Overwrite existing transforms.json.")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    run_export(
        config_path=args.config,
        output_dir=output_dir,
        sequence=args.sequence,
        max_frames=args.max_frames,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
