#!/usr/bin/env python3
"""Export paper fig 03_similarity_gaussian_map: no title, tight crop, vector labels.

Usage (CWD = DBSGNavigation):
  python scripts/export_paper_similarity_map.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png \\
    --out ../docs/paperfile/figs/03_similarity_gaussian_map \\
    --auto-straighten --rotate-deg 180 --align-left-edge
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GN_ROOT, ".."))
sys.path.insert(0, GN_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))
from paper_matplotlib import hide_top_right_spines, setup_matplotlib  # noqa: E402
from scripts.viz_semantic_gaussian_map import (  # noqa: E402
    DEFAULT_MAP_ROOT,
    DEFAULT_SCENE,
    load_params,
    resolve_params_path,
    subsample_idx,
)

DEFAULT_GOAL = "examples/viz_0.8-0.6/0/3/goal_query.png"
AXIS_PAD_M = 0.3
AXIS_INSET_M = 0.6  # 地图与 x/z 轴留白，避免散点压在轴线上
POINT_SIZE = 0.35
SAVE_DPI = 300
BODY_PT = 10.0
# cas-sc 单图常见插入宽度（\\columnwidth 或 ~0.48\\textwidth）；全文宽图用 17.5
LATEX_FIG_WIDTH_CM = 8.5
COLORBAR_LABEL = "cosine similarity and norm"


def _fmt_nonneg(v: float, _: int) -> str:
    return "" if v < -1e-6 else f"{v:g}"


def _ticks_with_zero(lo: float, hi: float, *, nbins: int = 5) -> np.ndarray:
    ticks = MaxNLocator(nbins=nbins).tick_values(lo, hi)
    ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
    if lo <= 0.0 <= hi and not np.any(np.isclose(ticks, 0.0)):
        ticks = np.sort(np.unique(np.concatenate([[0.0], ticks])))
    return ticks


def latex_matched_fontsize(fig_width_in: float, *, latex_width_cm: float = LATEX_FIG_WIDTH_CM) -> float:
    """反算 matplotlib pt，使 \\includegraphics[width=<latex_width>] 后与正文同大。"""
    latex_width_in = latex_width_cm / 2.54
    return BODY_PT * (fig_width_in / latex_width_in)


def save_vector(fig: plt.Figure, base_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.08, transparent=True)
    fig.savefig(base_path + ".pdf", **kw)
    fig.savefig(base_path + ".eps", format="eps", **kw)
    fig.savefig(base_path + ".png", dpi=SAVE_DPI, **kw)
    plt.close(fig)
    print("saved", base_path + ".pdf", base_path + ".eps", base_path + ".png")


def _transparent_axes(*axes) -> None:
    for axis in axes:
        axis.set_facecolor("none")
        patch = axis.patch
        patch.set_facecolor("none")
        patch.set_alpha(0.0)


def tight_limits(x: np.ndarray, z: np.ndarray, pad: float, q: float = 1.0) -> tuple[float, float, float, float]:
    x_lo, x_hi = float(np.percentile(x, q)), float(np.percentile(x, 100 - q))
    z_lo, z_hi = float(np.percentile(z, q)), float(np.percentile(z, 100 - q))
    return x_lo - pad, x_hi + pad, z_lo - pad, z_hi + pad


def estimate_align_angle(x: np.ndarray, z: np.ndarray) -> float:
    """用 PCA 估计主方向，返回把主轴转到水平所需的角度（弧度）。"""
    pts = np.stack([x, z], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.arctan2(major[1], major[0]))


def estimate_left_edge_vertical_correction(x: np.ndarray, z: np.ndarray, q: float = 1.0) -> float:
    """估计使外轮廓与坐标轴平行（左/右墙 ∥ z 轴）的 CCW 旋转角（弧度）。"""
    x1, x99 = float(np.percentile(x, q)), float(np.percentile(x, 100 - q))
    z1, z99 = float(np.percentile(z, q)), float(np.percentile(z, 100 - q))
    mi = (x >= x1) & (x <= x99) & (z >= z1) & (z <= z99)
    pts = np.stack([x[mi], z[mi]], axis=1).astype(np.float32)
    (_, _), (w, h), angle_deg = cv2.minAreaRect(pts)
    if w < h:
        angle_deg += 90.0
    return -np.radians(angle_deg)


def rotate_points(
    x: np.ndarray, z: np.ndarray, angle_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """绕质心旋转点云 angle_rad（正值逆时针）。"""
    cx, cz = float(x.mean()), float(z.mean())
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    dx, dz = x - cx, z - cz
    xr = cos_a * dx - sin_a * dz + cx
    zr = sin_a * dx + cos_a * dz + cz
    return xr, zr


def anchor_to_axis_origin(x: np.ndarray, z: np.ndarray, q: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """平移 inlier 包围盒，使左下角落在 (0,0)。"""
    x1, x99 = float(np.percentile(x, q)), float(np.percentile(x, 100 - q))
    z1, z99 = float(np.percentile(z, q)), float(np.percentile(z, 100 - q))
    mi = (x >= x1) & (x <= x99) & (z >= z1) & (z <= z99)
    x -= float(np.min(x[mi]))
    z -= float(np.min(z[mi]))
    return x, z


def flip_horizontal(x: np.ndarray, z: np.ndarray, q: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """沿 z 轴镜像（水平翻转 180°）。"""
    x1, x99 = float(np.percentile(x, q)), float(np.percentile(x, 100 - q))
    x_mid = 0.5 * (x1 + x99)
    return 2.0 * x_mid - x, z


def flip_vertical(x: np.ndarray, z: np.ndarray, q: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """沿 x 轴镜像（垂直翻转 180°）。"""
    z1, z99 = float(np.percentile(z, q)), float(np.percentile(z, 100 - q))
    z_mid = 0.5 * (z1 + z99)
    return x, 2.0 * z_mid - z


def align_to_axes(x: np.ndarray, z: np.ndarray, q: float = 1.0, *, max_iter: int = 3, tol_deg: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """迭代旋转，使外轮廓与 x/z 轴平行。"""
    total_deg = 0.0
    for i in range(max_iter):
        angle = estimate_left_edge_vertical_correction(x, z, q)
        step_deg = float(np.degrees(angle))
        if abs(step_deg) < tol_deg:
            break
        x, z = rotate_points(x, z, angle)
        total_deg += step_deg
        print(f"  align-to-axes iter {i + 1}: {step_deg:.2f} deg ({'CCW' if step_deg > 0 else 'CW'})")
    if abs(total_deg) >= tol_deg:
        print(f"  align-to-axes total = {total_deg:.2f} deg")
    return x, z


def export_similarity_map(
    scene: str,
    goal_image: str,
    out_stem: str,
    *,
    gaussian_map_root: str = DEFAULT_MAP_ROOT,
    device: str = "cpu",
    max_points: int = 80000,
    latex_width_cm: float = LATEX_FIG_WIDTH_CM,
    rotate_deg: float | None = None,
    auto_straighten: bool = False,
    flip_horizontal_flag: bool = False,
    flip_vertical_flag: bool = False,
    align_left_edge: bool = False,
    axis_inset_m: float = AXIS_INSET_M,
) -> None:
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    setup_matplotlib()

    goal_path = os.path.abspath(os.path.join(GN_ROOT, goal_image))
    g_bgr = cv2.imread(goal_path)
    if g_bgr is None:
        raise FileNotFoundError(goal_path)
    g_rgb = g_bgr[:, :, ::-1]

    params_path = resolve_params_path(scene, gaussian_map_root)
    means, clip, _rgb = load_params(params_path)

    retriever = OpenVocabRetriever(torch.device(device))
    v_g = retriever.encode_image(g_rgb).astype(np.float32).reshape(-1)

    idx = subsample_idx(means.shape[0], max_points)
    x, z = means[idx, 0], means[idx, 2]
    sim_plot = np.dot(clip[idx], v_g)
    sim_n = (sim_plot - sim_plot.min()) / (sim_plot.max() - sim_plot.min() + 1e-8)

    transformed = False
    if auto_straighten:
        angle = -estimate_align_angle(x, z)
        print(f"  auto-straighten angle = {np.degrees(angle):.1f} deg")
        x, z = rotate_points(x, z, angle)
        transformed = True
    if rotate_deg is not None:
        print(f"  manual rotate angle = {rotate_deg:.1f} deg ({'CCW' if rotate_deg > 0 else 'CW'})")
        x, z = rotate_points(x, z, np.radians(rotate_deg))
        transformed = True
    if flip_horizontal_flag:
        print("  flip horizontal (180° mirror about z-axis)")
        x, z = flip_horizontal(x, z)
        transformed = True
    if flip_vertical_flag:
        print("  flip vertical (180° mirror about x-axis)")
        x, z = flip_vertical(x, z)
        transformed = True
    if align_left_edge:
        x, z = align_to_axes(x, z)
        transformed = True
    if transformed:
        x, z = anchor_to_axis_origin(x, z)
        if axis_inset_m > 0:
            x += axis_inset_m
            z += axis_inset_m
            print(f"  axis inset = {axis_inset_m:.2f} m")

    q = 1.0
    x_lo, x_hi, z_lo, z_hi = tight_limits(x, z, AXIS_PAD_M, q=q)
    if transformed:
        x_lo, z_lo = 0.0, 0.0
    x_span = max(x_hi - x_lo, 1e-6)
    z_span = max(z_hi - z_lo, 1e-6)
    fig_h = 4.2
    cbar_w = 1.1
    fig_w = fig_h * (x_span / z_span) + cbar_w

    font_pt = latex_matched_fontsize(fig_w, latex_width_cm=latex_width_cm)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=SAVE_DPI)
    fig.patch.set_alpha(0.0)
    _transparent_axes(ax)
    sc = ax.scatter(
        x,
        z,
        c=sim_n,
        s=POINT_SIZE,
        linewidths=0,
        marker=".",
        cmap="viridis",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_xticks(_ticks_with_zero(x_lo, x_hi))
    ax.set_yticks(_ticks_with_zero(z_lo, z_hi))
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_nonneg))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_nonneg))
    ax.set_xlabel("x (m)", fontsize=font_pt, labelpad=8)
    ax.set_ylabel("z (m)", fontsize=font_pt, labelpad=8)
    ax.tick_params(axis="both", labelsize=font_pt, pad=4, length=4)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_clip_on(False)
    hide_top_right_spines(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.14)
    cbar = fig.colorbar(sc, cax=cax)
    _transparent_axes(cax)
    cbar.set_label(COLORBAR_LABEL, fontsize=font_pt, labelpad=8)
    cbar.ax.tick_params(labelsize=font_pt, pad=3)
    fig.subplots_adjust(left=0.17, bottom=0.20, top=0.98)
    save_vector(fig, os.path.abspath(out_stem))
    print(f"  font={font_pt:.1f}pt (→ ~{BODY_PT:.0f}pt @ {latex_width_cm}cm LaTeX width)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper export: similarity on Gaussian map")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--gaussian_map_root", default=DEFAULT_MAP_ROOT)
    ap.add_argument("--goal_image", default=DEFAULT_GOAL)
    ap.add_argument("--out", required=True, help="Output path stem (no extension)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument(
        "--latex-width-cm",
        type=float,
        default=LATEX_FIG_WIDTH_CM,
        help="Expected \\includegraphics width in cm (8.5=column, 17.5=\\textwidth)",
    )
    ap.add_argument(
        "--auto-straighten",
        action="store_true",
        help="PCA-align point cloud to horizontal (off by default; keeps Habitat x–z orientation).",
    )
    ap.add_argument(
        "--rotate-deg",
        type=float,
        default=None,
        help="Extra rotation in degrees (+CCW / −CW); stacks with --auto-straighten.",
    )
    ap.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Mirror map horizontally (180° flip about z-axis); applied before --flip-vertical.",
    )
    ap.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Mirror map vertically (180° flip about x-axis).",
    )
    ap.add_argument(
        "--align-left-edge",
        action="store_true",
        help="Rotate so the left wall boundary is parallel to the z-axis (y-axis in plot).",
    )
    ap.add_argument(
        "--axis-inset",
        type=float,
        default=AXIS_INSET_M,
        help="Gap (m) between map and x/z axes (default: 0.6).",
    )
    args = ap.parse_args()

    export_similarity_map(
        args.scene,
        args.goal_image,
        args.out,
        gaussian_map_root=args.gaussian_map_root,
        device=args.device,
        max_points=args.max_points,
        latex_width_cm=args.latex_width_cm,
        rotate_deg=args.rotate_deg,
        auto_straighten=args.auto_straighten,
        flip_horizontal_flag=args.flip_horizontal,
        flip_vertical_flag=args.flip_vertical,
        align_left_edge=args.align_left_edge,
        axis_inset_m=args.axis_inset,
    )


if __name__ == "__main__":
    main()
