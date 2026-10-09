#!/usr/bin/env python3
"""Export reconstruction overview + per-instance Gaussian point clouds for one scene.

Outputs PDF/PNG/EPS (vector text; scatter rasterized in EPS) under out_dir.

Usage (CWD = DBSGNavigation):
  python scripts/export_scene_reconstruction_pack.py \\
    --scene 4ok3usBNeis \\
    --run_dir ../DBSGConstruction/experiments/habitat_val_final/4ok3usBNeis_seed4_clip \\
    --eval_plot ../DBSGConstruction/experiments/habitat_val_final/4ok3usBNeis_seed4_clip/eval/plots/0199.png \\
    --out_dir ../docs/paperfile/figs/4ok3usBNeis_recon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GN_ROOT, ".."))
sys.path.insert(0, GN_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))

from paper_matplotlib import hide_top_right_spines, setup_matplotlib  # noqa: E402
from scripts.export_paper_gaussian_3d_figures import (  # noqa: E402
    CLUSTER_COLORS,
    CLUSTER_NAMES,
    LABEL_COLORS,
    extract_gt_rgb_from_eval_plot,
    export_dbscan_figure,
    export_millions_figure,
    get_dbscan_clusters,
    label_to_rgba,
    load_labels,
    transform_xyz,
)
from scripts.export_paper_similarity_map import save_vector  # noqa: E402
from scripts.viz_semantic_gaussian_map import (  # noqa: E402
    load_params,
    resolve_params_path,
    subsample_idx,
)
from scripts.viz_instance_database import (  # noqa: E402
    COCO_IDS,
    COCO_NAMES,
    crop_instance,
    resolve_rgb,
    resolve_seg_npy,
)

DEFAULT_MAP_ROOT = os.path.join(
    GN_ROOT, "..", "DBSGConstruction/experiments/habitat_val_final"
)
DATA_ROOT = os.path.join(GN_ROOT, "..", "DBSGConstruction/configs/data/habitat_val")
SAVE_DPI = 300
LABEL_NAMES = {
    0: "seg-0",
    1: "seg-1",
    2: "seg-2",
    3: "background",
    4: "seg-4",
    5: "seg-5",
    6: "seg-6",
}


def normalize_rgb_colors(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() <= 1.01:
        return np.clip(rgb, 0, 1)
    lo = np.percentile(rgb, 2, axis=0)
    hi = np.percentile(rgb, 98, axis=0)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def parse_bbox(entry: list) -> tuple[float, float, float, float, float, float]:
    b = entry[0][0]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])


def bbox_center(b: tuple[float, ...]) -> np.ndarray:
    return np.array(
        [
            0.5 * (b[0] + b[1]),
            0.5 * (b[2] + b[3]),
            0.5 * (b[4] + b[5]),
        ],
        dtype=np.float64,
    )


def dedupe_instance_groups(entries: list, eps: float = 1.2) -> list[list[int]]:
    centers = np.array([bbox_center(parse_bbox(e)) for e in entries], dtype=np.float64)
    used = np.zeros(len(entries), dtype=bool)
    groups: list[list[int]] = []
    for i in range(len(entries)):
        if used[i]:
            continue
        d = np.linalg.norm(centers - centers[i], axis=1)
        grp = np.where(d < eps)[0]
        used[grp] = True
        groups.append(grp.tolist())
    return groups


def merge_bbox(indices: list[int], entries: list) -> tuple[float, float, float, float, float, float]:
    boxes = [parse_bbox(entries[i]) for i in indices]
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        min(b[4] for b in boxes),
        max(b[5] for b in boxes),
    )


def mask_in_bbox(xyz: np.ndarray, bbox: tuple[float, ...], margin: float = 0.08) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox
    return (
        (xyz[:, 0] >= x0 - margin)
        & (xyz[:, 0] <= x1 + margin)
        & (xyz[:, 1] >= y0 - margin)
        & (xyz[:, 1] <= y1 + margin)
        & (xyz[:, 2] >= z0 - margin)
        & (xyz[:, 2] <= z1 + margin)
    )


def scatter3d_rgb(ax, xyz: np.ndarray, colors: np.ndarray, s: float = 0.8) -> None:
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=s,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def style_3d(ax, elev: float = 18, azim: float = -52, show_axes: bool = False) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor("white")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
    ax.grid(False)
    if not show_axes:
        ax.set_axis_off()


def set_equal_3d(ax, pts: np.ndarray, pad: float = 0.12) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) * (1.0 + pad)
    radius = max(radius, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def export_scene_overview(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    gt_rgb: np.ndarray,
    raster_rgb: np.ndarray | None,
    out_stem: str,
    n_gaussians: int,
) -> None:
    idx = subsample_idx(xyz.shape[0], 90000)
    pts = xyz[idx]
    rgb_n = normalize_rgb_colors(rgb[idx])
    lbl_rgba = label_to_rgba(labels[idx])

    fig = plt.figure(figsize=(10.5, 8.2), dpi=SAVE_DPI, facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(pts[:, 0], pts[:, 2], c=rgb_n, s=0.12, linewidths=0, rasterized=True)
    ax0.set_aspect("equal")
    ax0.set_title("(a) Reconstructed map (top-down RGB)", fontsize=10)
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("z (m)")
    hide_top_right_spines(ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(
        pts[:, 0],
        pts[:, 2],
        c=lbl_rgba,
        s=0.12,
        linewidths=0,
        rasterized=True,
    )
    ax1.set_aspect("equal")
    ax1.set_title("(b) Semantic Gaussian labels (top-down)", fontsize=10)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    hide_top_right_spines(ax1)

    ax2 = fig.add_subplot(gs[1, 0], projection="3d")
    scatter3d_rgb(ax2, pts, lbl_rgba, s=0.45)
    set_equal_3d(ax2, pts, pad=0.05)
    style_3d(ax2)
    ax2.set_title(f"(c) Millions of 3D Gaussians ({n_gaussians:,})", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(gt_rgb)
    ax3.set_title("(d) Query GT RGB (eval frame 0199)", fontsize=10)
    ax3.axis("off")
    if raster_rgb is not None:
        inset = ax3.inset_axes([0.02, 0.02, 0.42, 0.42])
        inset.imshow(raster_rgb)
        inset.set_title("Rasterized", fontsize=7)
        inset.axis("off")

    fig.suptitle("4ok3usBNeis — offline Semantic Gaussian reconstruction", fontsize=11, y=0.98)
    save_vector(fig, out_stem)


def find_coco_id_for_entry(ir: dict, ent: list) -> int | None:
    frame = str(ent[1])
    c = bbox_center(parse_bbox(ent))
    best_cid = None
    best_d = 1e9
    for cid in COCO_IDS:
        for e in ir.get(str(cid), []):
            if str(e[1]) != frame:
                continue
            d = float(np.linalg.norm(bbox_center(parse_bbox(e)) - c))
            if d < best_d:
                best_d = d
                best_cid = cid
    return best_cid if best_d < 0.05 else None


def export_semantic_label_instances(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for lab in sorted(np.unique(labels)):
        pick = np.where(labels == lab)[0]
        if pick.size < 20:
            continue
        if pick.size > 40000:
            pick = pick[subsample_idx(pick.size, 40000)]
        pts = xyz[pick]
        cols = normalize_rgb_colors(rgb[pick])

        fig = plt.figure(figsize=(5.2, 4.4), dpi=SAVE_DPI, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        scatter3d_rgb(ax, pts, cols, s=0.9)
        set_equal_3d(ax, pts)
        style_3d(ax)
        name = LABEL_NAMES.get(int(lab), f"label-{int(lab)}")
        fig.text(
            0.5,
            0.03,
            f"Semantic label {int(lab)} ({name}) — {int((labels == lab).sum()):,} Gaussians",
            ha="center",
            fontsize=9,
        )
        stem = os.path.join(out_dir, f"semantic_label_{int(lab):02d}_{name}")
        save_vector(fig, stem)


def export_class_instance_grid(
    xyz: np.ndarray,
    rgb: np.ndarray,
    ir: dict,
    out_stem: str,
    max_per_class: int = 6,
) -> None:
    ncols = 3
    nrows = 2
    fig = plt.figure(figsize=(10.5, 7.2), dpi=SAVE_DPI, facecolor="white")
    gs = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.18)
    plot_i = 0
    for cid in COCO_IDS:
        if plot_i >= nrows * ncols:
            break
        entries = ir.get(str(cid), [])
        if not entries:
            continue
        groups = dedupe_instance_groups(entries)
        groups.sort(key=lambda g: -len(g))
        ax = fig.add_subplot(gs[plot_i // ncols, plot_i % ncols], projection="3d")
        cmap = plt.colormaps["tab20"].resampled(min(len(groups), max_per_class))
        shown = 0
        for gi, grp in enumerate(groups[:max_per_class]):
            bbox = merge_bbox(grp, entries)
            m = mask_in_bbox(xyz, bbox)
            pts = xyz[m]
            if pts.shape[0] < 15:
                continue
            if pts.shape[0] > 6000:
                pts = pts[subsample_idx(pts.shape[0], 6000)]
            color = cmap(gi)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[color],
                s=1.0,
                linewidths=0,
                rasterized=True,
                alpha=0.95,
            )
            shown += 1
        set_equal_3d(ax, xyz[mask_in_bbox(xyz, merge_bbox(groups[0], entries))] if groups else xyz[:1000])
        style_3d(ax)
        ax.set_title(f"{COCO_NAMES[cid]} ({shown} instances shown)", fontsize=9)
        plot_i += 1
    fig.suptitle("Per-class instance Gaussian clusters (deduplicated 3D bboxes)", fontsize=11, y=0.98)
    save_vector(fig, out_stem)


def export_unique_instance(
    xyz: np.ndarray,
    rgb: np.ndarray,
    entry: list,
    cid: int,
    inst_id: int,
    out_stem: str,
) -> int:
    bbox = parse_bbox(entry)
    m = mask_in_bbox(xyz, bbox)
    pts = xyz[m]
    if pts.shape[0] < 12:
        return 0
    cols = normalize_rgb_colors(rgb[m])
    if pts.shape[0] > 25000:
        sel = subsample_idx(pts.shape[0], 25000)
        pts, cols = pts[sel], cols[sel]

    fig = plt.figure(figsize=(4.8, 4.0), dpi=SAVE_DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    scatter3d_rgb(ax, pts, cols, s=1.1)
    set_equal_3d(ax, pts, pad=0.15)
    style_3d(ax)
    frame_id = entry[1]
    fig.text(
        0.5,
        0.03,
        f"{COCO_NAMES[cid]} #{inst_id}  |  {pts.shape[0]:,} Gaussians  |  frame {frame_id}",
        ha="center",
        fontsize=8,
    )
    save_vector(fig, out_stem)
    return pts.shape[0]


def export_query_matched_instances(
    xyz: np.ndarray,
    rgb: np.ndarray,
    scene_dir: str,
    goal_rgb: np.ndarray,
    out_dir: str,
    top_k: int = 3,
    device: str = "cpu",
) -> None:
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    os.makedirs(out_dir, exist_ok=True)
    cv_path = os.path.join(scene_dir, "instance_clip_vectors_v2.json")
    r = OpenVocabRetriever(
        torch.device(device), clip_vectors_filename=os.path.basename(cv_path)
    )
    if not r.load_scene(scene_dir, instance_clip_vectors_path=cv_path):
        print("  [warn] query match skipped: load_scene failed")
        return
    top = r.retrieve(goal_rgb, top_k=top_k)
    fig, axes = plt.subplots(1, top_k + 1, figsize=(2.3 * (top_k + 1), 2.8), dpi=SAVE_DPI)
    if top_k == 0:
        axes = [axes]
    axes[0].imshow(goal_rgb)
    axes[0].set_title("Query GT RGB", fontsize=9)
    axes[0].axis("off")

    with open(os.path.join(scene_dir, "instance_retrieval.json")) as f:
        ir = json.load(f)

    for i, (ent, _pos, sim) in enumerate(top):
        ax = axes[i + 1]
        coco_id = find_coco_id_for_entry(ir, ent) or 57
        bbox = parse_bbox(ent)
        m = mask_in_bbox(xyz, bbox)
        pts = xyz[m]
        cols = normalize_rgb_colors(rgb[m])
        if pts.shape[0] > 8000:
            sel = subsample_idx(pts.shape[0], 8000)
            pts, cols = pts[sel], cols[sel]
        ax.scatter(pts[:, 0], pts[:, 2], c=cols, s=1.5, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"#{i+1} {COCO_NAMES.get(coco_id,'?')} sim={sim:.3f}", fontsize=8)
        ax.axis("off")
        stem = os.path.join(out_dir, f"query_match_{i+1:02d}_{COCO_NAMES.get(coco_id,'inst')}")
        fig_i = plt.figure(figsize=(4.8, 4.0), dpi=SAVE_DPI)
        ax3 = fig_i.add_subplot(111, projection="3d")
        scatter3d_rgb(ax3, pts, cols, s=1.2)
        set_equal_3d(ax3, pts)
        style_3d(ax3)
        fig_i.text(0.5, 0.03, f"CLIP match #{i+1}  sim={sim:.3f}", ha="center", fontsize=8)
        save_vector(fig_i, stem)

    fig.suptitle("Query GT RGB → CLIP Top-K instance Gaussian clusters", fontsize=10, y=1.02)
    save_vector(fig, os.path.join(out_dir, "query_topk_overview"))


def crop_eval_panel(src: str, row: int, col: int) -> np.ndarray:
    from scripts.export_paper_recon_plots import COLS, ROW1, ROW2

    im = np.array(Image.open(src).convert("RGB"))
    row_slice = ROW1 if row == 0 else ROW2
    x0, x1 = COLS[col]
    y0, y1 = row_slice
    return im[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Scene reconstruction + instance point cloud pack")
    ap.add_argument("--scene", default="4ok3usBNeis")
    ap.add_argument("--run_dir", default=None, help=".../4ok3usBNeis_seed4_clip")
    ap.add_argument("--gaussian_map_root", default=DEFAULT_MAP_ROOT)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--eval_plot", required=True)
    ap.add_argument("--out_dir", default="../docs/paperfile/figs/4ok3usBNeis_recon")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--auto-straighten", action="store_true")
    ap.add_argument("--rotate-deg", type=float, default=None)
    ap.add_argument("--align-left-edge", action="store_true")
    ap.add_argument("--export_all_instances", action="store_true", help="Export every deduped instance (many files)")
    args = ap.parse_args()

    setup_matplotlib()
    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_dir = os.path.abspath(os.path.join(GN_ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    params_path = resolve_params_path(args.scene, args.gaussian_map_root)
    means, _clip, rgb_raw = load_params(params_path)
    labels = load_labels(params_path)
    rgb = rgb_raw if rgb_raw is not None else np.full((means.shape[0], 3), 0.6)

    xyz = transform_xyz(
        means,
        auto_straighten=args.auto_straighten,
        rotate_deg=args.rotate_deg,
        flip_horizontal_flag=False,
        flip_vertical_flag=False,
        align_left_edge=args.align_left_edge,
        axis_inset_m=0.6,
    )
    n = means.shape[0]
    print(f"loaded {n:,} Gaussians from {params_path}")

    gt_rgb = extract_gt_rgb_from_eval_plot(args.eval_plot)
    eval_plot = os.path.abspath(args.eval_plot)
    raster_rgb = None
    if os.path.isfile(eval_plot):
        try:
            raster_rgb = crop_eval_panel(eval_plot, row=1, col=0)
        except Exception:
            pass

    prefix = os.path.splitext(os.path.basename(eval_plot))[0]
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_gt_rgb_crop.png"),
        gt_rgb[:, :, ::-1],
        [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
    )

    print("[1/5] scene overview")
    export_scene_overview(
        xyz,
        rgb,
        labels,
        gt_rgb,
        raster_rgb,
        os.path.join(out_dir, "scene_overview"),
        n,
    )

    print("[2/5] semantic label point clouds")
    export_semantic_label_instances(
        xyz, rgb, labels, os.path.join(out_dir, "semantic_labels")
    )

    print("[3/5] per-class + query-matched instances")
    with open(os.path.join(scene_dir, "instance_retrieval.json")) as f:
        ir = json.load(f)
    export_class_instance_grid(
        xyz, rgb, ir, os.path.join(out_dir, "instances_by_class_overview")
    )
    export_query_matched_instances(
        xyz,
        rgb,
        scene_dir,
        gt_rgb,
        os.path.join(out_dir, "query_matches"),
        top_k=3,
        device=args.device,
    )

    inst_dir = os.path.join(out_dir, "instances")
    os.makedirs(inst_dir, exist_ok=True)
    if args.export_all_instances:
        print("[4/5] all unique instances")
        for cid in COCO_IDS:
            entries = ir.get(str(cid), [])
            groups = dedupe_instance_groups(entries)
            for j, grp in enumerate(groups):
                rep = entries[grp[0]]
                stem = os.path.join(inst_dir, f"{COCO_NAMES[cid]}_{j+1:02d}")
                export_unique_instance(xyz, rgb, rep, cid, j + 1, stem)
    else:
        print("[4/5] representative instances (largest per class)")
        for cid in COCO_IDS:
            entries = ir.get(str(cid), [])
            groups = dedupe_instance_groups(entries)
            groups.sort(key=lambda g: -len(g))
            if not groups:
                continue
            rep = entries[groups[0][0]]
            stem = os.path.join(inst_dir, f"{COCO_NAMES[cid]}_rep")
            export_unique_instance(xyz, rgb, rep, cid, 1, stem)

    print("[5/5] Millions + DBSCAN for query GT RGB")
    export_millions_figure(
        xyz,
        labels,
        os.path.join(out_dir, f"{prefix}_millions_3d_gaussians"),
        max_points=180000,
        latex_width_cm=8.5,
    )
    clusters = get_dbscan_clusters(
        args.scene,
        gt_rgb,
        args.gaussian_map_root,
        agent_pos=None,
        device=args.device,
        target_k=3,
    )
    export_dbscan_figure(
        xyz,
        labels,
        clusters,
        os.path.join(out_dir, f"{prefix}_dbscan_clusters"),
        max_points_scene=120000,
        latex_width_cm=8.5,
    )

    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write(
            f"Scene reconstruction pack — {args.scene}\n"
            f"Gaussians: {n:,}\n"
            f"Query eval plot: {eval_plot}\n\n"
            "Key outputs:\n"
            "  scene_overview.pdf/png/eps\n"
            "  semantic_labels/semantic_label_*.pdf/png/eps\n"
            "  instances_by_class_overview.pdf/png/eps\n"
            "  query_matches/query_topk_overview + query_match_*.pdf/png/eps\n"
            "  instances/{class}_rep.pdf/png/eps  (or all with --export_all_instances)\n"
            f"  {prefix}_millions_3d_gaussians.pdf/png/eps\n"
            f"  {prefix}_dbscan_clusters.pdf/png/eps\n"
        )
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
