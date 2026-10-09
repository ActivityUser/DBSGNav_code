"""Visualize offline scene exploration data (RGB-D + camera trajectory).

Example:
  python scripts/viz_scene_exploration.py --scene 4ok3usBNeis
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "GaussianConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"


def load_trajectory(scene_dir: str):
    tf_path = os.path.join(scene_dir, "transforms.json")
    with open(tf_path) as f:
        data = json.load(f)
    frames = data["frames"]
    xs, zs, ys = [], [], []
    for fr in frames:
        m = np.array(fr["transform_matrix"], dtype=np.float64)
        xs.append(m[0, 3])
        ys.append(m[1, 3])
        zs.append(m[2, 3])
    return np.array(xs), np.array(ys), np.array(zs), len(frames)


def read_rgb(scene_dir: str, frame_id: int):
    path = os.path.join(scene_dir, "rgb", f"frame_{frame_id:05d}.png")
    im = cv2.imread(path)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_depth_vis(scene_dir: str, frame_id: int):
    path = os.path.join(scene_dir, "depth", f"{frame_id:05d}.png")
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    d = d.astype(np.float32)
    if d.max() > 50:
        d = d / 1000.0  # mm -> m
    d = np.clip(d, 0, 10)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cm = plt.colormaps["turbo"]
    return (cm(d_norm)[:, :, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Visualize scene exploration RGB-D + trajectory")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--out",
        default="../docs/figs/scene_exploration_explainer.png",
    )
    ap.add_argument("--n_rgb_samples", type=int, default=4)
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    xs, ys, zs, n_frames = load_trajectory(scene_dir)
    sample_ids = np.linspace(0, n_frames - 1, args.n_rgb_samples, dtype=int)

    fig = plt.figure(figsize=(13, 7), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.28, wspace=0.12)

    for i, fid in enumerate(sample_ids):
        ax = fig.add_subplot(gs[0, i])
        rgb = read_rgb(scene_dir, int(fid))
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_title(f"RGB frame {int(fid):05d}", fontsize=9)
        ax.axis("off")

    ax_d = fig.add_subplot(gs[1, 0])
    mid = int(sample_ids[len(sample_ids) // 2])
    dep = read_depth_vis(scene_dir, mid)
    if dep is not None:
        ax_d.imshow(dep)
    ax_d.set_title(f"Depth frame {mid:05d}", fontsize=9)
    ax_d.axis("off")

    ax_t = fig.add_subplot(gs[1, 1:])
    ax_t.plot(xs, zs, "-", color="#444", linewidth=0.8, alpha=0.7)
    ax_t.scatter(xs[:: max(1, len(xs) // 400)], zs[:: max(1, len(zs) // 400)],
                 c=np.arange(len(xs))[:: max(1, len(xs) // 400)], cmap="viridis",
                 s=4, linewidths=0)
    ax_t.scatter(xs[0], zs[0], c="green", s=40, zorder=5, label="start")
    ax_t.scatter(xs[-1], zs[-1], c="red", s=40, zorder=5, label="end")
    for fid in sample_ids:
        ax_t.scatter(xs[int(fid)], zs[int(fid)], s=60, facecolors="none",
                     edgecolors="orange", linewidths=1.5, zorder=6)
    ax_t.set_aspect("equal")
    ax_t.set_xlabel("x (m)")
    ax_t.set_ylabel("z (m)")
    ax_t.set_title(f"Exploration trajectory ({n_frames} viewpoints, top-down)", fontsize=10)
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.invert_yaxis()

    fig.suptitle(
        f"Scene exploration — {args.scene}  |  "
        "Habitat agent collects RGB-D along trajectories (offline, per scene)",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"scene={args.scene} frames={n_frames}")
    print("saved", out_path)

    thumb = os.path.join(ROOT, "..", "docs/figs/method_thumbs/scene_exploration.png")
    rgb0 = read_rgb(scene_dir, int(sample_ids[1]))
    if rgb0 is not None:
        cv2.imwrite(thumb, cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        print("saved", thumb)


if __name__ == "__main__":
    main()
