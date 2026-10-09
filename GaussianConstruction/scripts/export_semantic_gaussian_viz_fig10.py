#!/usr/bin/env python3
"""Export Fig.-style Semantic Gaussian visualization (white-bg splat renders).

Renders photorealistic 3D Gaussian splats from multiple scenes / viewpoints,
composites a 2×4 panel figure like paper Fig.10, exports PDF/PNG/EPS.

Usage (CWD = GaussianConstruction, gausshabitat env recommended):
  CUDA_VISIBLE_DEVICES=1 python scripts/export_semantic_gaussian_viz_fig10.py \\
    --map_root experiments/habitat_val_final \\
    --out ../docs/paperfile/figs/fig10_semantic_gaussian_viz \\
    --scenes 4ok3usBNeis CrMo8WxCyVb yr17PDCnDDW 5cdEh9F2hJL \\
             bCPU9suPUw9 eF36g7L6Z9M MHPLjHsuG27 QaLdnwvtxbs
"""
from __future__ import annotations

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GC_ROOT, ".."))
sys.path.insert(0, GC_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))
sys.path.insert(0, os.path.join(GC_ROOT, "viz_scripts"))

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from paper_matplotlib import setup_matplotlib  # noqa: E402
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation

DEFAULT_SCENES = [
    "4ok3usBNeis",
    "CrMo8WxCyVb",
    "yr17PDCnDDW",
    "5cdEh9F2hJL",
    "bCPU9suPUw9",
    "eF36g7L6Z9M",
    "MHPLjHsuG27",
    "QaLdnwvtxbs",
]

DEFAULT_SCENE_FRAMES = {
    "4ok3usBNeis": 199,
}


def prepare_rgb_colors(raw_rgb: torch.Tensor) -> torch.Tensor:
    """Match SplaTAM / eval: use optimized rgb_colors directly (no sigmoid)."""
    return raw_rgb.float()


def apply_tone_correct(img: np.ndarray, *, black: float = 0.02, gain: float = 1.08, gamma: float = 0.95) -> np.ndarray:
    """Mild contrast lift for paper figures; keeps colors close to eval renders."""
    out = np.clip((img - black) * gain, 0.0, 1.0)
    return np.clip(out ** gamma, 0.0, 1.0)


LABEL_PALETTE = np.array(
    [
        [0.88, 0.34, 0.34],
        [0.46, 0.72, 0.70],
        [0.35, 0.63, 0.31],
        [0.73, 0.69, 0.67],
        [0.31, 0.47, 0.65],
        [0.69, 0.48, 0.63],
        [1.00, 0.62, 0.65],
    ],
    dtype=np.float32,
)

SAVE_DPI = 200
PANEL_W = 640
PANEL_H = 480


def load_viz_cfg() -> dict:
    mod = SourceFileLoader("habitat_splatam_val", os.path.join(GC_ROOT, "configs/habitat/habitat_splatam_val.py")).load_module()
    cfg = dict(mod.config["viz"])
    cfg["viz_w"] = PANEL_W
    cfg["viz_h"] = PANEL_H
    return cfg


def resolve_params(map_root: str, scene: str) -> str:
    p = os.path.join(map_root, f"{scene}_seed4_clip", "params.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


def load_camera_from_npz(params_path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = dict(np.load(params_path, allow_pickle=True))
    org_w = float(raw["org_width"])
    org_h = float(raw["org_height"])
    w2c = np.asarray(raw["w2c"], dtype=np.float64)
    k = np.asarray(raw["intrinsics"], dtype=np.float64)[:3, :3].copy()
    k[0, :] *= cfg["viz_w"] / org_w
    k[1, :] *= cfg["viz_h"] / org_h
    return w2c, k


def load_gaussian_render_vars(params_path: str, first_w2c: np.ndarray, device: torch.device):
    raw = dict(np.load(params_path, allow_pickle=True))
    skip = {
        "org_width",
        "org_height",
        "w2c",
        "intrinsics",
        "gt_w2c_all_frames",
        "cam_unnorm_rots",
        "cam_trans",
        "keyframe_time_indices",
        "timestep",
    }
    means = torch.tensor(raw["means3D"], device=device, dtype=torch.float32)
    if "labels" in raw and raw["labels"] is not None:
        labels = torch.tensor(raw["labels"].reshape(-1), device=device, dtype=torch.long)
    else:
        labels = None

    rgb = prepare_rgb_colors(torch.tensor(raw["rgb_colors"], device=device))

    log_scales = torch.tensor(raw["log_scales"], device=device, dtype=torch.float32)
    if log_scales.shape[-1] == 1:
        log_scales = log_scales.repeat(1, 3)
    rotations = F.normalize(torch.tensor(raw["unnorm_rotations"], device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(raw["logit_opacities"], device=device, dtype=torch.float32))

    w2c_t = torch.tensor(first_w2c, device=device, dtype=torch.float32)
    rendervar = {
        "means3D": means,
        "colors_precomp": rgb,
        "rotations": rotations,
        "opacities": opacities,
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(means),
    }
    depth_rendervar = {
        "means3D": means,
        "colors_precomp": get_depth_and_silhouette(means, w2c_t),
        "rotations": rotations,
        "opacities": opacities,
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(means),
    }
    gt_w2cs = None
    if "gt_w2c_all_frames" in raw:
        gt_w2cs = np.asarray(raw["gt_w2c_all_frames"], dtype=np.float64)
    return rendervar, depth_rendervar, labels, gt_w2cs


def semantic_colors_from_labels(labels: torch.Tensor) -> torch.Tensor:
    idx = labels.clamp(0, LABEL_PALETTE.shape[0] - 1).cpu().numpy()
    cols = LABEL_PALETTE[idx]
    return torch.tensor(cols, device=labels.device, dtype=torch.float32)


def pick_frame_indices(n_frames: int, n_picks: int = 1) -> list[int]:
    if n_frames <= 0:
        return [0]
    if n_picks == 1:
        return [min(n_frames // 2, n_frames - 1)]
    edges = np.linspace(0, n_frames - 1, n_picks + 2, dtype=int)[1:-1]
    return [int(x) for x in edges]


def frame_coverage_score(img: np.ndarray, *, white_thresh: float = 0.97) -> float:
    """Higher = denser scene fill (fewer white-background pixels)."""
    non_white = np.any(img < white_thresh, axis=-1)
    return float(non_white.mean())


def pick_best_frames_near_anchor(
    gt_w2cs: np.ndarray,
    k: np.ndarray,
    rendervar: dict,
    depth_rendervar: dict,
    cfg: dict,
    *,
    anchor: int,
    n_picks: int,
    window: int = 80,
    stride: int = 2,
    min_gap: int = 10,
    semantic_labels: torch.Tensor | None = None,
    tone_correct: bool = False,
) -> list[int]:
    n_frames = gt_w2cs.shape[0]
    lo = max(0, anchor - window)
    hi = min(n_frames - 1, anchor + window)
    candidates = list(range(lo, hi + 1, stride))
    if anchor not in candidates:
        candidates.append(anchor)
    candidates = sorted(set(candidates))

    scored: list[tuple[float, int]] = []
    for fi in candidates:
        img = render_rgb(
            gt_w2cs[fi], k, rendervar, depth_rendervar, cfg,
            semantic_labels=semantic_labels, tone_correct=tone_correct,
        )
        scored.append((frame_coverage_score(img), fi))

    scored.sort(key=lambda x: (-x[0], abs(x[1] - anchor)))
    picked: list[int] = []
    for _score, fi in scored:
        if all(abs(fi - p) >= min_gap for p in picked):
            picked.append(fi)
        if len(picked) >= n_picks:
            break
    if len(picked) < n_picks:
        for _score, fi in scored:
            if fi not in picked:
                picked.append(fi)
            if len(picked) >= n_picks:
                break
    return sorted(picked)


def render_rgb(
    w2c: np.ndarray,
    k: np.ndarray,
    rendervar: dict,
    depth_rendervar: dict,
    cfg: dict,
    *,
    semantic_labels: torch.Tensor | None = None,
    tone_correct: bool = False,
) -> np.ndarray:
    device = rendervar["means3D"].device
    cam = setup_camera(cfg["viz_w"], cfg["viz_h"], k, w2c, cfg["viz_near"], cfg["viz_far"])
    white_bg = Camera(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device),
        scale_modifier=cam.scale_modifier,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=cam.sh_degree,
        campos=cam.campos,
        prefiltered=cam.prefiltered,
    )
    rv = rendervar
    if semantic_labels is not None:
        rv = dict(rendervar)
        rv["colors_precomp"] = semantic_colors_from_labels(semantic_labels)
    with torch.no_grad():
        im, _, _ = Renderer(raster_settings=white_bg)(**rv)
    im_np = im.detach().cpu().numpy().transpose(1, 2, 0)
    im_np = np.clip(im_np, 0.0, 1.0)
    if tone_correct and semantic_labels is None:
        im_np = apply_tone_correct(im_np)
    return im_np


def render_scene_panel(
    params_path: str,
    cfg: dict,
    device: torch.device,
    frame_idx: int | None,
    color_mode: str,
    *,
    tone_correct: bool = False,
) -> np.ndarray:
    w2c0, k = load_camera_from_npz(params_path, cfg)
    rendervar, depth_rendervar, labels, gt_w2cs = load_gaussian_render_vars(params_path, w2c0, device)
    if gt_w2cs is not None and gt_w2cs.shape[0] > 0:
        fi = frame_idx if frame_idx is not None else pick_frame_indices(gt_w2cs.shape[0], 1)[0]
        fi = int(np.clip(fi, 0, gt_w2cs.shape[0] - 1))
        w2c = gt_w2cs[fi]
    else:
        w2c = w2c0
    sem = labels if color_mode == "semantic" else None
    return render_rgb(
        w2c, k, rendervar, depth_rendervar, cfg, semantic_labels=sem, tone_correct=tone_correct
    )


def save_vector(fig: plt.Figure, base_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    fig.savefig(base_path + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base_path + ".eps", bbox_inches="tight", format="eps", pad_inches=0.04)
    fig.savefig(base_path + ".png", bbox_inches="tight", dpi=SAVE_DPI, pad_inches=0.04)
    plt.close(fig)
    print("  saved", base_path + ".pdf/.png/.eps")


def save_single_panel(img: np.ndarray, base_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(PANEL_W / SAVE_DPI, PANEL_H / SAVE_DPI), dpi=SAVE_DPI)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_vector(fig, base_path)


def export_fig10(
    panels: list[np.ndarray],
    out_stem: str,
    *,
    title: str = "Semantic Gaussian visualization results.",
    ncols: int = 4,
) -> None:
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig_w = 3.35 * ncols
    fig_h = 2.55 * nrows + 0.35
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=SAVE_DPI)
    axes = np.atleast_1d(axes).ravel()
    for ax, img in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02, hspace=0.02)
    save_vector(fig, out_stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Semantic Gaussian visualization figure")
    ap.add_argument("--map_root", default=os.path.join(GC_ROOT, "experiments/habitat_val_final"))
    ap.add_argument("--out", default=os.path.join(REPO, "docs/paperfile/figs/fig10_semantic_gaussian_viz"))
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument(
        "--frame_indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional per-scene frame index (same count as --scenes)",
    )
    ap.add_argument(
        "--color_mode",
        choices=["rgb", "semantic"],
        default="rgb",
        help="rgb = photorealistic splat (paper Fig.10 style); semantic = label colors",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--single_scene_views", type=int, default=0, help="If >0, render N views of first scene only")
    ap.add_argument(
        "--anchor_frame",
        type=int,
        default=None,
        help="With --single_scene_views: pick best views near this frame (e.g. 199)",
    )
    ap.add_argument("--frame_window", type=int, default=80, help="Search radius around --anchor_frame")
    ap.add_argument("--frame_stride", type=int, default=2, help="Stride when scoring candidate frames")
    ap.add_argument("--min_frame_gap", type=int, default=10, help="Min frame separation among picked views")
    ap.add_argument(
        "--tone_correct",
        action="store_true",
        help="Mild contrast/gamma lift for RGB panels (optional)",
    )
    ap.add_argument(
        "--panel_names",
        nargs="*",
        default=None,
        help="Optional stem for each exported panel (same count as panels)",
    )
    args = ap.parse_args()

    setup_matplotlib()
    cfg = load_viz_cfg()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_stem = os.path.abspath(args.out)
    panel_dir = out_stem + "_panels"
    os.makedirs(panel_dir, exist_ok=True)

    panels: list[np.ndarray] = []
    panel_names: list[str] = []
    scene_list = args.scenes

    if args.single_scene_views > 0:
        params_path = resolve_params(args.map_root, scene_list[0])
        w2c0, k = load_camera_from_npz(params_path, cfg)
        rendervar, depth_rendervar, labels, gt_w2cs = load_gaussian_render_vars(params_path, w2c0, device)
        if gt_w2cs is None:
            raise RuntimeError("gt_w2c_all_frames missing")
        sem = labels if args.color_mode == "semantic" else None
        if args.anchor_frame is not None:
            frame_ids = pick_best_frames_near_anchor(
                gt_w2cs,
                k,
                rendervar,
                depth_rendervar,
                cfg,
                anchor=args.anchor_frame,
                n_picks=args.single_scene_views,
                window=args.frame_window,
                stride=args.frame_stride,
                min_gap=args.min_frame_gap,
                semantic_labels=sem,
                tone_correct=args.tone_correct,
            )
            print(f"  anchor={args.anchor_frame} -> best frames: {frame_ids}")
        else:
            frame_ids = pick_frame_indices(gt_w2cs.shape[0], args.single_scene_views)
        for fi in frame_ids:
            img = render_rgb(
                gt_w2cs[fi], k, rendervar, depth_rendervar, cfg,
                semantic_labels=sem, tone_correct=args.tone_correct,
            )
            panels.append(img)
            stem = f"{scene_list[0]}_frame{fi:04d}"
            panel_names.append(stem)
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(panel_dir, f"{stem}.png"))
            torch.cuda.empty_cache()
    else:
        for i, scene in enumerate(scene_list):
            params_path = resolve_params(args.map_root, scene)
            fi = None
            if args.frame_indices and i < len(args.frame_indices):
                fi = args.frame_indices[i]
            elif scene in DEFAULT_SCENE_FRAMES:
                fi = DEFAULT_SCENE_FRAMES[scene]
            print(f"render {scene} frame={fi} ...")
            img = render_scene_panel(
                params_path, cfg, device, fi, args.color_mode, tone_correct=args.tone_correct
            )
            panels.append(img)
            panel_names.append(scene)
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(panel_dir, f"{scene}.png"))
            torch.cuda.empty_cache()

    if args.panel_names:
        if len(args.panel_names) != len(panels):
            raise ValueError("--panel_names count must match number of panels")
        panel_names = list(args.panel_names)

    export_fig10(panels, out_stem, title="Semantic Gaussian visualization results.")
    for i, (img, name) in enumerate(zip(panels, panel_names)):
        save_single_panel(img, os.path.join(panel_dir, f"{i+1:02d}_{name}"))
    print("done ->", out_stem)


if __name__ == "__main__":
    main()
