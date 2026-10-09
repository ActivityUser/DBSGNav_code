#!/usr/bin/env python3
"""
从 SplaTAM / habitat_splatam 输出的 params.npz 可视化高斯。
- 默认：交互式 Open3D（与 final_recon.py 相同逻辑）
- --headless：无显示器时导出 PLY + 首帧 RGB/深度渲染 + 简单 index.html

用法:
  python viz_scripts/viz_habitat_params.py --params experiments/.../params.npz
  python viz_scripts/viz_habitat_params.py --params .../params.npz --headless --output ./viz_output
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from importlib.machinery import SourceFileLoader


def _load_viz_cfg(config_path: str | None) -> dict:
    path = config_path or os.path.join(_BASE_DIR, "configs", "habitat", "habitat_splatam.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    mod = SourceFileLoader("habitat_splatam_cfg", path).load_module()
    return dict(mod.config["viz"])


def _headless_export(params_path: str, cfg: dict, out_dir: str) -> None:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import open3d as o3d

    import final_recon as fr

    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.abspath(params_path)

    w2c, k = fr.load_camera(cfg, params_path)
    scene_data, scene_depth_data, _ = fr.load_scene_data(params_path, w2c, k)
    im, depth, _ = fr.render(w2c, k, scene_data, scene_depth_data, cfg)

    # RGB [3,H,W] -> PNG
    rgb = torch.clamp(im, 0.0, 1.0).detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    plt.imsave(os.path.join(out_dir, "render_rgb.png"), rgb)

    # Depth single channel
    dep = depth[0].detach().cpu().numpy()
    plt.imsave(os.path.join(out_dir, "render_depth.png"), dep, cmap="jet")

    # Point cloud from first-frame view
    pts, cols = fr.rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = pts
    pcd.colors = cols
    o3d.io.write_point_cloud(os.path.join(out_dir, "gaussian_points.ply"), pcd)

    # Gaussian centers as a second PLY (raw positions + colors)
    all_p = dict(np.load(params_path, allow_pickle=True))
    means = np.asarray(all_p["means3D"])
    colors = np.asarray(all_p["rgb_colors"])
    if colors.ndim == 2 and colors.shape[1] == 3:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(means.astype(np.float64))
        pcd2.colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0.0, 1.0).astype(np.float64)
        )
        o3d.io.write_point_cloud(os.path.join(out_dir, "gaussian_centers.ply"), pcd2)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>viz params</title></head>
<body>
<h1>params.npz 可视化导出</h1>
<p>来源: <code>{params_path}</code></p>
<img src="render_rgb.png" width="640"/><br/>
<img src="render_depth.png" width="640"/><br/>
<p>点云: gaussian_points.ply（视角重建） / gaussian_centers.ply（高斯中心）</p>
</body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"已写入: {os.path.abspath(out_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize habitat_splatam params.npz")
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="params.npz 绝对路径或相对 DBSGConstruction 根目录的路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="habitat_splatam 风格配置文件（用于读取 viz 子配置），默认 configs/habitat/habitat_splatam.py",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无显示器：导出 PLY 与 PNG 到 --output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./viz_output",
        help="--headless 时输出目录",
    )
    args = parser.parse_args()

    params_path = args.params
    if not os.path.isabs(params_path):
        params_path = os.path.join(_BASE_DIR, params_path)
    if not os.path.isfile(params_path):
        print(f"Error: params file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    viz_cfg = _load_viz_cfg(args.config)

    if args.headless:
        _headless_export(params_path, viz_cfg, args.output)
        return

    import final_recon as fr

    fr.visualize(params_path, viz_cfg)


if __name__ == "__main__":
    main()
