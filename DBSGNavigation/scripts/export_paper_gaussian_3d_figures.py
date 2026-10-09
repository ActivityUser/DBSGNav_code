#!/usr/bin/env python3
"""Export paper figures: Millions of 3D Gaussians + DBSCAN cluster zoom.

Outputs PDF + EPS (vector text/lines) + PNG for each figure stem.

Usage (CWD = DBSGNavigation):
  python scripts/export_paper_gaussian_3d_figures.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png \\
    --out_dir ../docs/paperfile/figs \\
    --auto-straighten --rotate-deg 180 --align-left-edge

  # From eval/plots Ground Truth RGB (e.g. frame 0199):
  python scripts/export_paper_gaussian_3d_figures.py \\
    --scene 4ok3usBNeis \\
    --eval_plot ../DBSGConstruction/experiments/habitat_val_final/4ok3usBNeis_seed4_clip/eval/plots/0199.png \\
    --out_dir ../docs/paperfile/figs \\
    --auto-straighten --rotate-deg 180 --align-left-edge
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GN_ROOT, ".."))
sys.path.insert(0, GN_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))

from paper_matplotlib import setup_matplotlib  # noqa: E402
from scripts.export_paper_similarity_map import (  # noqa: E402
    align_to_axes,
    anchor_to_axis_origin,
    estimate_align_angle,
    flip_horizontal,
    flip_vertical,
    latex_matched_fontsize,
    rotate_points,
    save_vector,
)
from scripts.viz_semantic_gaussian_map import (  # noqa: E402
    DEFAULT_MAP_ROOT,
    DEFAULT_SCENE,
    load_params,
    resolve_params_path,
    subsample_idx,
)

DEFAULT_GOAL = "examples/viz_0.8-0.6/0/3/goal_query.png"
DEFAULT_GOAL_INFO = "examples/viz_0.8-0.6/0/3/goal_info.txt"
DATA_ROOT = os.path.join(GN_ROOT, "..", "DBSGConstruction/configs/data/habitat_val")

# eval/plots composite layout (same as export_paper_recon_plots.py)
EVAL_ROW1 = (98, 355)
EVAL_COLS = [(10, 392), (406, 788), (803, 1185)]

NAV = dict(
    sim_floor=0.20,
    top_k_cluster=200,
    eps_m=0.6,
    min_cluster_size=6,
    floor_y_band=1.5,
)

# Saturated label palette (background label gets muted gray).
LABEL_COLORS = {
    0: "#e15759",  # red
    1: "#76b7b2",  # teal
    2: "#59a14f",  # green
    3: "#bab0ac",  # walls / background (largest class)
    4: "#4e79a7",  # blue
    5: "#b07aa1",  # purple
    6: "#ff9da7",  # pink
}
CLUSTER_COLORS = ["#9467bd", "#ff7f0e", "#2ca02c"]  # A, B, C
CLUSTER_NAMES = ["Cluster A", "Cluster B", "Cluster C"]

VIEW_ELEV = 18.0
VIEW_AZIM = -52.0
SAVE_DPI = 300
BODY_PT = 10.0
LATEX_FIG_WIDTH_CM = 8.5
BG_LABEL = 3  # largest semantic class (walls / background)


def stratified_subsample(
    labels: np.ndarray,
    max_points: int,
    *,
    bg_label: int = BG_LABEL,
    bg_frac: float = 0.22,
    seed: int = 0,
) -> np.ndarray:
    """Keep semantic objects dense; downsample background walls."""
    n = labels.shape[0]
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    fg = np.where(labels != bg_label)[0]
    bg = np.where(labels == bg_label)[0]
    n_bg = min(int(max_points * bg_frac), bg.size)
    n_fg = min(max_points - n_bg, fg.size)
    pick = []
    if n_fg > 0:
        pick.append(rng.choice(fg, size=n_fg, replace=False))
    if n_bg > 0:
        pick.append(rng.choice(bg, size=n_bg, replace=False))
    if not pick:
        return subsample_idx(n, max_points, seed=seed)
    idx = np.sort(np.concatenate(pick))
    if idx.size < max_points:
        rest = np.setdiff1d(np.arange(n), idx)
        extra = min(max_points - idx.size, rest.size)
        if extra > 0:
            idx = np.sort(np.concatenate([idx, rng.choice(rest, size=extra, replace=False)]))
    return idx


def y_band_mask(xyz: np.ndarray, lo_q: float = 3.0, hi_q: float = 93.0) -> np.ndarray:
    y = xyz[:, 1]
    lo, hi = np.percentile(y, [lo_q, hi_q])
    return (y >= lo) & (y <= hi)


def extract_gt_rgb_from_eval_plot(eval_plot_path: str, *, trim_black: bool = True) -> np.ndarray:
    """Crop Ground Truth RGB (top-left panel) from eval/plots composite PNG."""
    from PIL import Image

    path = os.path.abspath(eval_plot_path)
    im = np.array(Image.open(path).convert("RGB"))
    x0, x1 = EVAL_COLS[0]
    y0, y1 = EVAL_ROW1
    panel = im[y0:y1, x0:x1].copy()
    if trim_black:
        gray = panel.mean(axis=2)
        cols = np.where(gray > 10)[0]
        if cols.size > 8:
            panel = panel[:, cols[0] : cols[-1] + 1]
    return panel


def resolve_out_stems(out_dir: str, out_prefix: str | None) -> tuple[str, str]:
    if out_prefix:
        p = out_prefix.strip().removesuffix("_")
        return (
            os.path.join(out_dir, f"{p}_millions_3d_gaussians"),
            os.path.join(out_dir, f"{p}_dbscan_clusters"),
        )
    return (
        os.path.join(out_dir, "03_millions_3d_gaussians"),
        os.path.join(out_dir, "04_dbscan_clusters"),
    )


def read_goal_info(path: str) -> dict:
    info = {}
    if not os.path.isfile(path):
        return info
    for line in open(path):
        line = line.strip()
        if "=" not in line or line.startswith("---"):
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def agent_from_scene_start(scene_dir: str) -> np.ndarray | None:
    tf = os.path.join(scene_dir, "transforms.json")
    if not os.path.isfile(tf):
        return None
    with open(tf) as f:
        m = np.array(json.load(f)["frames"][0]["transform_matrix"], dtype=np.float64)
    return m[:3, 3].astype(np.float64)


def load_labels(params_path: str) -> np.ndarray:
    data = dict(np.load(params_path, allow_pickle=True))
    if "labels" not in data:
        raise KeyError(f"labels missing in {params_path}")
    return np.asarray(data["labels"], dtype=np.int64).reshape(-1)


def label_to_rgba(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((labels.shape[0], 4), dtype=np.float64)
    for lab, hex_c in LABEL_COLORS.items():
        m = labels == lab
        if not np.any(m):
            continue
        rgb = np.array(matplotlib.colors.to_rgb(hex_c))
        out[m, :3] = rgb
        out[m, 3] = 0.55 if lab == BG_LABEL else 0.98
    unknown = ~np.isin(labels, list(LABEL_COLORS.keys()))
    out[unknown, :3] = 0.55
    out[unknown, 3] = 0.85
    return out


def apply_map_transforms(
    x: np.ndarray,
    z: np.ndarray,
    *,
    auto_straighten: bool,
    rotate_deg: float | None,
    flip_horizontal_flag: bool,
    flip_vertical_flag: bool,
    align_left_edge: bool,
    axis_inset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if auto_straighten:
        angle = -estimate_align_angle(x, z)
        print(f"  auto-straighten angle = {np.degrees(angle):.1f} deg")
        x, z = rotate_points(x, z, angle)
    if rotate_deg is not None:
        print(f"  manual rotate angle = {rotate_deg:.1f} deg")
        x, z = rotate_points(x, z, np.radians(rotate_deg))
    if flip_horizontal_flag:
        print("  flip horizontal")
        x, z = flip_horizontal(x, z)
    if flip_vertical_flag:
        print("  flip vertical")
        x, z = flip_vertical(x, z)
    if align_left_edge:
        x, z = align_to_axes(x, z)
    x, z = anchor_to_axis_origin(x, z)
    if axis_inset_m > 0:
        x += axis_inset_m
        z += axis_inset_m
    return x, z


def transform_xyz(
    means: np.ndarray,
    *,
    auto_straighten: bool,
    rotate_deg: float | None,
    flip_horizontal_flag: bool,
    flip_vertical_flag: bool,
    align_left_edge: bool,
    axis_inset_m: float,
) -> np.ndarray:
    xyz = means.astype(np.float64, copy=True)
    x, z = apply_map_transforms(
        xyz[:, 0].copy(),
        xyz[:, 2].copy(),
        auto_straighten=auto_straighten,
        rotate_deg=rotate_deg,
        flip_horizontal_flag=flip_horizontal_flag,
        flip_vertical_flag=flip_vertical_flag,
        align_left_edge=align_left_edge,
        axis_inset_m=axis_inset_m,
    )
    xyz[:, 0] = x
    xyz[:, 2] = z
    return xyz


def set_equal_3d(ax, pts: np.ndarray, pad: float = 0.08) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(spans)) * (1.0 + pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_3d_ax(ax, *, show_axes: bool = False) -> None:
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_axis_off()
    else:
        ax.set_xlabel("x", fontsize=9, labelpad=-2)
        ax.set_ylabel("y", fontsize=9, labelpad=-2)
        ax.set_zlabel("z", fontsize=9, labelpad=-2)
        ax.tick_params(labelsize=8, pad=0)


def scatter3d(ax, xyz: np.ndarray, colors: np.ndarray, *, s: float = 0.35, alpha: float | None = None) -> None:
    if colors.shape[1] == 4:
        rgba = colors
        if alpha is not None:
            rgba = rgba.copy()
            rgba[:, 3] = alpha
        fc = rgba
    else:
        fc = colors
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=fc,
        s=s,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def export_millions_figure(
    xyz: np.ndarray,
    labels: np.ndarray,
    out_stem: str,
    *,
    max_points: int,
    latex_width_cm: float,
) -> None:
    n = xyz.shape[0]
    band = y_band_mask(xyz)
    idx_pool = np.where(band)[0]
    idx = stratified_subsample(labels[idx_pool], max_points)
    idx = idx_pool[idx]
    pts = xyz[idx]
    rgba = label_to_rgba(labels[idx])

    fig_w, fig_h = 5.6, 4.4
    font_pt = latex_matched_fontsize(fig_w, latex_width_cm=latex_width_cm)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=SAVE_DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    scatter3d(ax, pts, rgba, s=0.55)
    set_equal_3d(ax, pts, pad=0.04)
    style_3d_ax(ax, show_axes=False)

    caption = f"Millions of 3D Gaussians  ({n:,} total)"
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=font_pt, color="black")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.0)
    save_vector(fig, os.path.abspath(out_stem))
    print(f"  millions: {n:,} Gaussians, plotted {len(idx):,}")


def get_dbscan_clusters(
    scene: str,
    goal_rgb: np.ndarray,
    gaussian_map_root: str,
    agent_pos: np.ndarray | None,
    device: str,
    *,
    target_k: int = 3,
):
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever
    from map_planning_utils.semantic_gaussian_goal import SemanticGaussianGoal

    r = OpenVocabRetriever(torch.device(device))
    sg = SemanticGaussianGoal(
        torch.device(device),
        gaussian_map_root=os.path.abspath(gaussian_map_root),
        clip_encode_image=r.encode_image,
    )
    if not sg.load_scene(scene):
        raise RuntimeError(sg.get_last_error())

    def _prep(eps_m: float, min_sz: int, sim_floor: float, top_k: int):
        return sg._navigable_cluster_prep(
            goal_rgb,
            agent_pos,
            sim_floor,
            top_k,
            eps_m,
            min_sz,
            NAV["floor_y_band"],
        )

    prep = _prep(NAV["eps_m"], NAV["min_cluster_size"], NAV["sim_floor"], NAV["top_k_cluster"])
    if prep is None:
        raise RuntimeError("cluster prep failed")
    clusters = sorted(prep[-1], key=lambda c: float(c["score"]), reverse=True)

    if len(clusters) < target_k:
        for eps_m, min_sz, sim_floor, top_k in [
            (0.48, 5, 0.18, 320),
            (0.42, 4, 0.15, 400),
            (0.38, 4, 0.12, 500),
        ]:
            alt = _prep(eps_m, min_sz, sim_floor, top_k)
            if alt is None:
                continue
            clusters = sorted(alt[-1], key=lambda c: float(c["score"]), reverse=True)
            if len(clusters) >= target_k:
                print(f"  dbscan fallback eps={eps_m} -> {len(clusters)} clusters")
                break

    return clusters[:target_k]


def export_dbscan_figure(
    xyz: np.ndarray,
    labels: np.ndarray,
    clusters: list,
    out_stem: str,
    *,
    max_points_scene: int,
    latex_width_cm: float,
) -> None:
    n = xyz.shape[0]
    band = y_band_mask(xyz)
    idx_pool = np.where(band)[0]
    idx_scene = stratified_subsample(labels[idx_pool], max_points_scene, bg_frac=0.28)
    idx_scene = idx_pool[idx_scene]
    scene_pts = xyz[idx_scene]
    scene_rgba = label_to_rgba(labels[idx_scene])

    cluster_blocks = []
    for i, cl in enumerate(clusters[:3]):
        gidx = np.asarray(cl["gaussian_indices"], dtype=np.int64).reshape(-1)
        if gidx.size > 12000:
            gidx = gidx[np.linspace(0, gidx.size - 1, 12000, dtype=int)]
        cluster_blocks.append((CLUSTER_NAMES[i], xyz[gidx], CLUSTER_COLORS[i]))

    fig_w, fig_h = 9.2, 4.6
    font_pt = latex_matched_fontsize(fig_w, latex_width_cm=latex_width_cm)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=SAVE_DPI, facecolor="white")

    ax_l = fig.add_axes([0.03, 0.12, 0.42, 0.78], projection="3d")
    scatter3d(ax_l, scene_pts, scene_rgba, s=0.35)
    set_equal_3d(ax_l, scene_pts, pad=0.05)
    style_3d_ax(ax_l, show_axes=False)

    ax_r = fig.add_axes([0.56, 0.14, 0.40, 0.74], projection="3d")
    all_cl_pts = []
    legend_handles = []
    for name, pts, color in cluster_blocks:
        all_cl_pts.append(pts)
        scatter3d(
            ax_r,
            pts,
            np.tile(matplotlib.colors.to_rgba(color), (pts.shape[0], 1)),
            s=1.8,
            alpha=0.95,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7, label=name)
        )
    cl_stack = np.vstack(all_cl_pts) if all_cl_pts else scene_pts
    set_equal_3d(ax_r, cl_stack, pad=0.18)
    style_3d_ax(ax_r, show_axes=True)

    # Zoom box on left scene around cluster union.
    if all_cl_pts:
        focus = np.vstack(all_cl_pts)
        c = focus.mean(axis=0)
        r = float(np.max(np.linalg.norm(focus - c, axis=1))) * 1.15
        ax_l.plot(
            [c[0] - r, c[0] + r, c[0] + r, c[0] - r, c[0] - r],
            [c[1] - r, c[1] - r, c[1] + r, c[1] + r, c[1] - r],
            [c[2] - r, c[2] - r, c[2] - r, c[2] - r, c[2] - r],
            color="black",
            lw=0.9,
            ls="--",
        )

    # Connector lines (2D figure coords).
    fig.add_artist(
        Line2D([0.46, 0.54], [0.52, 0.52], transform=fig.transFigure, color="black", lw=0.9, ls="--")
    )
    fig.add_artist(
        Line2D([0.46, 0.54], [0.30, 0.30], transform=fig.transFigure, color="black", lw=0.9, ls="--")
    )
    fig.text(0.50, 0.50, "DBSCAN\nclusters", ha="center", va="center", fontsize=font_pt, color="black")

    ax_r.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=font_pt * 0.85,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.98)
    save_vector(fig, os.path.abspath(out_stem))
    print(f"  dbscan: {len(cluster_blocks)} clusters exported")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper export: 3D Gaussians + DBSCAN clusters")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--gaussian_map_root", default=DEFAULT_MAP_ROOT)
    ap.add_argument("--goal_image", default=None, help="Goal/query RGB for DBSCAN (overridden by --eval_plot)")
    ap.add_argument("--goal_info", default=DEFAULT_GOAL_INFO)
    ap.add_argument(
        "--eval_plot",
        default=None,
        help="eval/plots/*.png composite; use top-left Ground Truth RGB as CLIP query",
    )
    ap.add_argument(
        "--out_prefix",
        default=None,
        help="Output stem prefix, e.g. 0199 -> 0199_millions_3d_gaussians / 0199_dbscan_clusters",
    )
    ap.add_argument("--save_goal_crop", default=None, help="Optional path to save extracted GT RGB PNG")
    ap.add_argument("--out_dir", default="../docs/paperfile/figs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_points", type=int, default=180000, help="Max points for millions figure")
    ap.add_argument("--max_points_scene", type=int, default=120000, help="Max points for DBSCAN left panel")
    ap.add_argument("--latex-width-cm", type=float, default=LATEX_FIG_WIDTH_CM)
    ap.add_argument("--auto-straighten", action="store_true")
    ap.add_argument("--rotate-deg", type=float, default=None)
    ap.add_argument("--flip-horizontal", action="store_true")
    ap.add_argument("--flip-vertical", action="store_true")
    ap.add_argument("--align-left-edge", action="store_true")
    ap.add_argument("--axis-inset", type=float, default=0.6)
    args = ap.parse_args()

    setup_matplotlib()

    if args.eval_plot:
        g_rgb = extract_gt_rgb_from_eval_plot(args.eval_plot)
        print(f"  goal from eval plot: {os.path.abspath(args.eval_plot)}  shape={g_rgb.shape}")
        if args.out_prefix is None:
            base = os.path.splitext(os.path.basename(args.eval_plot))[0]
            args.out_prefix = base
        if args.save_goal_crop:
            crop_path = os.path.abspath(
                os.path.join(GN_ROOT, args.save_goal_crop)
                if not os.path.isabs(args.save_goal_crop)
                else args.save_goal_crop
            )
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cv2.imwrite(crop_path, g_rgb[:, :, ::-1], [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            print(f"  saved goal crop: {crop_path}")
    else:
        goal_path = os.path.abspath(
            os.path.join(GN_ROOT, args.goal_image or DEFAULT_GOAL)
        )
        g_bgr = cv2.imread(goal_path)
        if g_bgr is None:
            raise FileNotFoundError(goal_path)
        g_rgb = g_bgr[:, :, ::-1]
        print(f"  goal image: {goal_path}")

    params_path = resolve_params_path(args.scene, args.gaussian_map_root)
    means, _clip, _rgb = load_params(params_path)
    labels = load_labels(params_path)
    print(f"loaded {means.shape[0]:,} Gaussians from {params_path}")

    xyz = transform_xyz(
        means,
        auto_straighten=args.auto_straighten,
        rotate_deg=args.rotate_deg,
        flip_horizontal_flag=args.flip_horizontal,
        flip_vertical_flag=args.flip_vertical,
        align_left_edge=args.align_left_edge,
        axis_inset_m=args.axis_inset,
    )

    out_dir = os.path.abspath(os.path.join(GN_ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stem_m, stem_d = resolve_out_stems(out_dir, args.out_prefix)

    export_millions_figure(
        xyz,
        labels,
        stem_m,
        max_points=args.max_points,
        latex_width_cm=args.latex_width_cm,
    )

    scene_dir = os.path.abspath(os.path.join(DATA_ROOT, args.scene))
    agent_pos = agent_from_scene_start(scene_dir)
    clusters = get_dbscan_clusters(
        args.scene,
        g_rgb,
        args.gaussian_map_root,
        agent_pos=None,  # match SG prep without agent floor snap for multi-cluster viz
        device=args.device,
        target_k=3,
    )

    export_dbscan_figure(
        xyz,
        labels,
        clusters,
        stem_d,
        max_points_scene=args.max_points_scene,
        latex_width_cm=args.latex_width_cm,
    )


if __name__ == "__main__":
    main()
