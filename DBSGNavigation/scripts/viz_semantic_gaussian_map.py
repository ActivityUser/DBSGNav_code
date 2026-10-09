"""Visualize Semantic Gaussian Map (params.npz) for one scene + goal image.

Outputs a 3-panel figure:
  (a) top-down RGB appearance of Gaussians
  (b) top-down CLIP similarity to goal image (semantic field)
  (c) reference goal image

Example:
  python scripts/viz_semantic_gaussian_map.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png \\
    --out ../docs/figs/semantic_gaussian_map_explainer.png
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_MAP_ROOT = os.path.join(
    ROOT, "..", "DBSGConstruction/experiments/habitat_val_final"
)
DEFAULT_SCENE = "4ok3usBNeis"


def resolve_params_path(scene_name: str, gaussian_map_root: str) -> str:
    """Same convention as navigation: <root>/<scene>_seed4_clip/params.npz."""
    if not scene_name or not gaussian_map_root:
        raise ValueError("scene and gaussian_map_root are required")
    root = os.path.abspath(gaussian_map_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"gaussian_map_root not found: {root}")
    candidates = [
        d
        for d in glob(os.path.join(root, f"{scene_name}*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "params.npz"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no params.npz under {root} for scene '{scene_name}' "
            f"(expected {scene_name}_seed4_clip/params.npz)"
        )
    candidates.sort(key=lambda d: ("_clip" not in os.path.basename(d), d))
    return os.path.join(candidates[0], "params.npz")


def load_params(path: str):
    data = dict(np.load(path, allow_pickle=True))
    means = np.asarray(data["means3D"], dtype=np.float64).reshape(-1, 3)
    clip = np.asarray(data["clip_vectors"], dtype=np.float32).reshape(-1, 512)
    norm = np.linalg.norm(clip, axis=1, keepdims=True)
    clip = clip / np.maximum(norm, 1e-8)
    rgb = None
    if "rgb_colors" in data:
        rgb = np.asarray(data["rgb_colors"], dtype=np.float32).reshape(-1, 3)
        if float(rgb.max()) > 1.5:
            rgb = np.clip(rgb / 255.0, 0, 1)
        else:
            rgb = np.clip(rgb, 0, 1)
    return means, clip, rgb


def encode_goal(path: str, device: str = "cpu"):
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    g = cv2.imread(path)
    if g is None:
        raise FileNotFoundError(path)
    g = g[:, :, ::-1]
    r = OpenVocabRetriever(torch.device(device))
    return r.encode_image(g)


def subsample_idx(n: int, max_pts: int, seed: int = 0):
    if n <= max_pts:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_pts, replace=False)


def topdown_scatter(ax, x, z, colors, title, s=0.15, cmap=None):
    if cmap is None:
        sc = ax.scatter(x, z, c=colors, s=s, linewidths=0, marker=".")
    else:
        sc = ax.scatter(x, z, c=colors, s=s, linewidths=0, marker=".", cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.invert_yaxis()
    return sc


def main():
    ap = argparse.ArgumentParser(
        description="Visualize semantic Gaussian map (params.npz) for one scene + goal image."
    )
    ap.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="HM3D scene id (resolves <gaussian_map_root>/<scene>_seed4_clip/params.npz)",
    )
    ap.add_argument(
        "--gaussian_map_root",
        default=DEFAULT_MAP_ROOT,
        help="Root dir of offline maps (default: habitat_val_final, same as val config)",
    )
    ap.add_argument(
        "--params",
        default=None,
        help="Explicit path to params.npz (overrides --scene)",
    )
    ap.add_argument("--goal_image", default="examples/viz_0.8-0.6/0/3/goal_query.png")
    ap.add_argument("--out", default="../docs/figs/semantic_gaussian_map_explainer.png")
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.params:
        params_path = os.path.abspath(args.params)
    else:
        params_path = resolve_params_path(args.scene, args.gaussian_map_root)
    goal_path = os.path.abspath(os.path.join(ROOT, args.goal_image))
    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    means, clip, rgb = load_params(params_path)
    n = means.shape[0]
    idx = subsample_idx(n, args.max_points)
    xyz = means[idx]
    x, z = xyz[:, 0], xyz[:, 2]

    print(f"loaded {n:,} Gaussians from {params_path}")
    print(f"subsampled {len(idx):,} for plot")

    v_g = encode_goal(goal_path, device=args.device)
    sims = np.dot(clip[idx], v_g.astype(np.float32))
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    if rgb is None:
        rgb_plot = np.full((len(idx), 3), 0.65)
    else:
        rgb_plot = rgb[idx]

    goal_bgr = cv2.imread(goal_path)
    goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200)
    topdown_scatter(axes[0], x, z, rgb_plot, "(a) Semantic Gaussian map (appearance)")
    sc = topdown_scatter(
        axes[1], x, z, sims, "(b) CLIP similarity to goal image", cmap="jet"
    )
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="cosine sim (norm.)")
    axes[2].imshow(goal_rgb)
    axes[2].set_title("(c) Goal image (query)")
    axes[2].axis("off")

    scene = args.scene or os.path.basename(os.path.dirname(params_path)).replace(
        "_seed4_clip", ""
    )
    fig.suptitle(
        f"Semantic Gaussian Map — scene {scene}  |  each dot = one 3D Gaussian (means3D + clip_vectors)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)

    # also refresh method thumb
    thumb_dir = os.path.join(ROOT, "..", "docs/figs/method_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    panel = cv2.imread(out_path)
    if panel is not None:
        # crop panel (a) from saved figure is hard; save dedicated top-down RGB png
        rgb_only = os.path.join(thumb_dir, "semantic_gaussian_map_topdown.png")
        fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=200)
        topdown_scatter(ax2, x, z, rgb_plot, "Semantic Gaussian map (top-down RGB)")
        fig2.tight_layout()
        fig2.savefig(rgb_only, bbox_inches="tight")
        plt.close(fig2)
        print("saved", rgb_only)


if __name__ == "__main__":
    main()
