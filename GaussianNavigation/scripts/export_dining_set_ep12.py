#!/usr/bin/env python3
"""Export dining-table + surrounding chairs near chair ep12 (4ok3usBNeis).

Outputs under out_dir:
  chair_XX/chair_XX.{png,pdf,eps}   one full-chair image per dining chair
  dining_set_complete.{png,pdf,eps} table + all 4 chairs in one frame
  manifest.json

Example:
  cd GaussianNavigation
  python scripts/export_dining_set_ep12.py \\
    --out_dir ../../gaussnav_code_online/docs/paperfile/figs/dining_set_ep12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GN_ROOT, ".."))
sys.path.insert(0, GN_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))

from paper_matplotlib import setup_matplotlib  # noqa: E402
from scripts.export_chair_instances import (  # noqa: E402
    bbox_center,
    dedupe_instance_groups,
    parse_bbox,
)
from scripts.viz_instance_database import resolve_rgb, resolve_seg_npy  # noqa: E402

SCENE = "4ok3usBNeis"
DATA_ROOT = os.path.join(GN_ROOT, "..", "GaussianConstruction/configs/data/habitat_val")
CHAIR_ID = 56
SAVE_DPI = 600  # paper export; crops saved at native resolution (640x480 source)


def crop_chair_native(
    rgb: np.ndarray,
    seg_path: str,
    coco_id: int,
    target_uv: np.ndarray | None = None,
    *,
    pad: float = 0.18,
    min_side: int = 80,
    max_dist: float = 120.0,
) -> np.ndarray | None:
    """Full-resolution chair crop; pick seg mask nearest projected 2D center."""
    if not os.path.isfile(seg_path):
        return None
    masks = np.load(seg_path, allow_pickle=True)
    if isinstance(masks, np.ndarray) and masks.dtype == object:
        masks = masks.item()
    if not isinstance(masks, dict) or coco_id not in masks or not masks[coco_id]:
        return None

    H, W = rgb.shape[:2]
    best = None
    best_score = -1.0
    for mask in masks[coco_id]:
        if not isinstance(mask, np.ndarray):
            continue
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            continue
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if target_uv is not None:
            dist = float(np.linalg.norm(np.array([cx, cy]) - target_uv.reshape(2)))
            if dist > max_dist:
                continue
        else:
            dist = 0.0
        area = float((y2 - y1 + 1) * (x2 - x1 + 1))
        score = area / (1.0 + dist)
        if score <= best_score:
            continue
        ph, pw = int((y2 - y1) * pad), int((x2 - x1) * pad)
        yy1, yy2 = max(0, y1 - ph), min(H - 1, y2 + ph)
        xx1, xx2 = max(0, x1 - pw), min(W - 1, x2 + pw)
        crop = rgb[yy1 : yy2 + 1, xx1 : xx2 + 1]
        if min(crop.shape[:2]) < min_side:
            continue
        best_score = score
        best = crop.copy()
    return best


def save_hires_exports(crop: np.ndarray, out_stem: str, caption: str) -> None:
    """1:1 pixel PNG + PDF/EPS wrapper (photo stays raster, but full resolution)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    h, w = crop.shape[:2]

    # Native PNG — sharpest, no matplotlib resampling
    cv2.imwrite(
        out_stem + ".png",
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
    )

    # Matplotlib PDF/EPS: figure size matches pixel grid exactly at SAVE_DPI
    fig_w, fig_h = w / SAVE_DPI, h / SAVE_DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.18), dpi=SAVE_DPI, facecolor="white")
    ax.imshow(crop, interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0, right=1, bottom=0.06, top=1)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".eps", bbox_inches="tight", format="eps", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out_stem}.png ({w}x{h}) + pdf/eps")

VERIFIED_DINING_CHAIRS = ("chair_06", "chair_16", "chair_24", "chair_10")
TARGET_CHAIR = "chair_06"

# Visually verified exports: distinct frames, full chair height (legs visible).
# chair_10 = back-left chair against window wall (only partial back in dining_set_complete).
CHAIR_VIEW_PRESETS: dict[str, dict] = {
    "chair_06": {"frame": "00889", "method": "seg"},
    "chair_10": {
        "frame": "00029",
        "method": "proj",
        "anchor_chair": "chair_10",
        "size": (220, 360),
        "center_bias_y": 0.55,
    },
    "chair_16": {
        "frame": "00029",
        "method": "proj",
        "anchor_chair": "chair_16",
        "size": (220, 360),
        "center_bias_y": 0.55,
    },
    "chair_24": {
        "frame": "00894",
        "method": "proj",
        "anchor_chair": "chair_24",
        "size": (210, 340),
        "offset_uv": (80, 0),
        "center_bias_y": 0.55,
    },
}


def select_dining_chairs(manifest_instances: list[dict], target_id: str = TARGET_CHAIR) -> tuple[str, ...]:
    """Return the 4 verified dining chairs around ep12 (validated visually)."""
    by_id = {x["id"] for x in manifest_instances}
    if all(cid in by_id for cid in VERIFIED_DINING_CHAIRS):
        return VERIFIED_DINING_CHAIRS

    target = next(x for x in manifest_instances if x["id"] == target_id)
    target_xyz = np.array(target["center_xyz"], dtype=np.float64)
    cands: list[tuple[float, str]] = []
    for x in manifest_instances:
        cid = x["id"]
        if cid == target_id:
            continue
        c = np.array(x["center_xyz"], dtype=np.float64)
        dxz = float(np.linalg.norm(c[[0, 2]] - target_xyz[[0, 2]]))
        if dxz < 1.0 or dxz > 4.5:
            continue
        if abs(float(c[1] - target_xyz[1])) > 0.5:
            continue
        if abs(float(c[2] - target_xyz[2])) > 1.5:
            continue
        cands.append((dxz, cid))
    cands.sort()

    selected = [target_id]
    for _, cid in cands:
        c = np.array(next(x for x in manifest_instances if x["id"] == cid)["center_xyz"])
        if any(
            np.linalg.norm(c[[0, 2]] - np.array(next(y for y in manifest_instances if y["id"] == s)["center_xyz"])[[0, 2]])
            < 0.5
            for s in selected
        ):
            continue
        selected.append(cid)
        if len(selected) == 4:
            break
    return tuple(selected)


def load_transforms(scene_dir: str) -> dict:
    with open(os.path.join(scene_dir, "transforms.json"), encoding="utf-8") as f:
        return json.load(f)


def frame_id_from_path(file_path: str) -> str:
    fid = file_path.split("_")[-1]
    return fid.replace(".png", "")


def get_c2w(meta: dict, frame_id: str) -> np.ndarray | None:
    fid = int(str(frame_id))
    for fr in meta["frames"]:
        if int(frame_id_from_path(fr["file_path"])) == fid:
            return np.array(fr["transform_matrix"], dtype=np.float64)
    return None


def project_points(c2w: np.ndarray, K: np.ndarray, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points; return (uv Nx2, valid mask). Habitat cam: -Z forward, Y up."""
    w2c = np.linalg.inv(c2w)
    ones = np.ones((pts_w.shape[0], 1), dtype=np.float64)
    pw = np.hstack([pts_w, ones])
    pc = (w2c @ pw.T).T[:, :3]
    z = -pc[:, 2]
    valid = z > 0.05
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * (-pc[:, 1]) / z + K[1, 2]
    return np.stack([u, v], axis=1), valid


def intrinsics_from_meta(meta: dict) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = meta["fl_x"]
    K[1, 1] = meta["fl_y"]
    K[0, 2] = meta["cx"]
    K[1, 2] = meta["cy"]
    return K


def aabb_corners(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    xs = [x0, x1]
    ys = [y0, y1]
    zs = [z0, z1]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def estimate_table_aabb(chair_centers: np.ndarray) -> tuple[float, float, float, float, float, float]:
    cx = float(chair_centers[:, 0].mean())
    cz = float(chair_centers[:, 2].mean())
    hx = 0.5 * float(np.ptp(chair_centers[:, 0])) + 0.55
    hz = 0.5 * float(np.ptp(chair_centers[:, 2])) + 0.45
    y_top = float(chair_centers[:, 1].mean()) + 0.15
    y_bot = y_top - 0.75
    return cx - hx, cx + hx, y_bot, y_top, cz - hz, cz + hz


def crop_2d_bbox(
    rgb: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    *,
    pad_frac: float = 0.12,
    min_size: int = 32,
) -> np.ndarray | None:
    H, W = rgb.shape[:2]
    m = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if m.sum() < 2:
        return None
    pts = uv[m]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1 = int(max(0, math.floor(x1 - px)))
    y1 = int(max(0, math.floor(y1 - py)))
    x2 = int(min(W - 1, math.ceil(x2 + px)))
    y2 = int(min(H - 1, math.ceil(y2 + py)))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return rgb[y1 : y2 + 1, x1 : x2 + 1].copy()


def camera_azimuth(c2w: np.ndarray, target: np.ndarray) -> float:
    cam = c2w[:3, 3]
    d = target - cam
    return float(math.atan2(d[0], d[2]))


def export_panel(crop: np.ndarray, out_stem: str, caption: str) -> None:
    save_hires_exports(crop, out_stem, caption)


def select_diverse_views(
    candidates: list[tuple[str, np.ndarray, np.ndarray]],
    target_center: np.ndarray,
    meta: dict,
    max_views: int,
) -> list[tuple[str, np.ndarray, str]]:
    """Pick frames with diverse camera azimuth around target."""
    scored: list[tuple[float, str, np.ndarray, float]] = []
    for frame_id, crop, c2w in candidates:
        az = camera_azimuth(c2w, target_center)
        scored.append((az, frame_id, crop, 0.0))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    chosen: list[tuple[str, np.ndarray, str]] = []
    min_sep = math.radians(25)
    for az, frame_id, crop, _ in scored:
        if all(abs(az - camera_azimuth(get_c2w(meta, c[0]), target_center)) >= min_sep for c in chosen):
            chosen.append((frame_id, crop, f"frame {frame_id}"))
        if len(chosen) >= max_views:
            break

    # fill if too few
    if len(chosen) < max_views:
        for az, frame_id, crop, _ in scored:
            if any(frame_id == c[0] for c in chosen):
                continue
            chosen.append((frame_id, crop, f"frame {frame_id}"))
            if len(chosen) >= max_views:
                break
    return chosen


def merge_bbox_3d(indices: list[int], entries: list) -> tuple[float, float, float, float, float, float]:
    boxes = [parse_bbox(entries[i]) for i in indices]
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        min(b[4] for b in boxes),
        max(b[5] for b in boxes),
    )


def expand_bbox_3d(
    bbox: tuple[float, ...],
    *,
    margin_xz: float = 0.18,
    margin_y: float = 0.25,
) -> tuple[float, float, float, float, float, float]:
    x0, x1, y0, y1, z0, z1 = bbox
    return (
        x0 - margin_xz,
        x1 + margin_xz,
        y0 - margin_y,
        y1 + margin_y,
        z0 - margin_xz,
        z1 + margin_xz,
    )


def bbox_corners(b: tuple[float, ...]) -> np.ndarray:
    return aabb_corners(*b)


def score_full_object_crop(
    uv: np.ndarray,
    valid: np.ndarray,
    *,
    min_visible_frac: float = 0.75,
    min_side: int = 80,
) -> float:
    if valid.sum() < int(8 * min_visible_frac):
        return -1.0
    H_ok = valid
    pts = uv[H_ok]
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    if min(w, h) < min_side:
        return -1.0
    area = w * h
    vis = float(valid.mean())
    return area * vis


def crop_object_from_3d_bbox(
    rgb: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    bbox: tuple[float, ...],
    *,
    pad_frac: float = 0.10,
    min_side: int = 80,
) -> tuple[np.ndarray | None, float]:
    corners = bbox_corners(bbox)
    uv, valid = project_points(c2w, K, corners)
    score = score_full_object_crop(uv, valid, min_side=min_side)
    if score < 0:
        return None, score
    crop = crop_2d_bbox(rgb, uv, valid, pad_frac=pad_frac, min_size=min_side)
    if crop is None:
        return None, -1.0
    return crop, score


def chair_group_center(entries: list, group_indices: list[int]) -> np.ndarray:
    centers = np.array([bbox_center(parse_bbox(entries[i])) for i in group_indices])
    return centers.mean(axis=0)


def crop_projection_window(
    rgb: np.ndarray,
    uv: np.ndarray,
    *,
    width: int = 210,
    height: int = 340,
    center_bias_y: float = 0.55,
) -> np.ndarray | None:
    """Fixed-size crop around a 2D point (for chairs without reliable seg masks)."""
    H, W = rgb.shape[:2]
    u, v = float(uv[0]), float(uv[1])
    x1 = int(max(0, min(W - width, u - width / 2)))
    y1 = int(max(0, min(H - height, v - height * center_bias_y)))
    crop = rgb[y1 : y1 + height, x1 : x1 + width]
    if crop.shape[0] < 160 or crop.shape[1] < 120:
        return None
    return crop.copy()


def crop_chair_seg_scored(
    rgb: np.ndarray,
    seg_path: str,
    target_uv: np.ndarray,
    *,
    pad: float = 0.10,
    max_dist: float = 65.0,
    min_h: int = 130,
    min_aspect: float = 0.72,
    min_leg_frac: float = 0.82,
    max_fill: float = 0.42,
) -> tuple[np.ndarray, float] | None:
    """Pick seg mask closest to target; prefer tall crops with legs near frame bottom."""
    if not os.path.isfile(seg_path):
        return None
    masks = np.load(seg_path, allow_pickle=True)
    if isinstance(masks, np.ndarray) and masks.dtype == object:
        masks = masks.item()
    if not isinstance(masks, dict) or CHAIR_ID not in masks or not masks[CHAIR_ID]:
        return None

    H, W = rgb.shape[:2]
    best = None
    best_score = -1.0
    for mask in masks[CHAIR_ID]:
        if not isinstance(mask, np.ndarray):
            continue
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            continue
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        mdist = float(np.linalg.norm(np.array([cx, cy]) - target_uv.reshape(2)))
        if mdist > max_dist:
            continue
        bh, bw = y2 - y1 + 1, x2 - x1 + 1
        aspect = bh / max(bw, 1)
        leg = y2 / H
        fill = (bh * bw) / (H * W)
        if bh < min_h or aspect < min_aspect or leg < min_leg_frac or fill > max_fill:
            continue
        if y1 < 8 or x1 < 8 or x2 > W - 8:
            continue
        score = float(bh * aspect * (leg**2) / (1.0 + mdist))
        if score <= best_score:
            continue
        ph, pw = int(bh * pad), int(bw * pad)
        crop = rgb[max(0, y1 - ph) : min(H - 1, y2 + ph) + 1, max(0, x1 - pw) : min(W - 1, x2 + pw) + 1]
        if min(crop.shape[:2]) < 100:
            continue
        best_score = score
        best = (crop.copy(), score, mdist, (cx, cy))
    if best is None:
        return None
    return best


def pick_best_full_chair_view(
    scene_dir: str,
    entries: list,
    group_indices: list[int],
    chair_id: str,
    chair_centers: dict[str, np.ndarray],
    meta: dict,
    K: np.ndarray,
    *,
    used_frames: set[str] | None = None,
    used_mask_centers: list[tuple[float, float]] | None = None,
) -> tuple[str, np.ndarray, str] | None:
    """One best frame: full chair (legs visible), distinct from already exported chairs."""
    center = chair_group_center(entries, group_indices)
    used_frames = used_frames or set()
    used_mask_centers = used_mask_centers or []

    preset = CHAIR_VIEW_PRESETS.get(chair_id)
    if preset:
        frame_id = preset["frame"]
        c2w = get_c2w(meta, frame_id)
        rgb_path = resolve_rgb(scene_dir, frame_id)
        rgb = cv2.imread(rgb_path)
        if c2w is not None and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if preset.get("method") == "proj":
                anchor = preset.get("anchor_chair", chair_id)
                anchor_xyz = chair_centers.get(anchor, center)
                uv, valid = project_points(c2w, K, anchor_xyz.reshape(1, 3))
                if valid[0]:
                    w, h = preset.get("size", (210, 340))
                    offset = preset.get("offset_uv", (0, 0))
                    target_uv = uv[0] + np.array(offset, dtype=np.float64)
                    cby = float(preset.get("center_bias_y", 0.55))
                    crop = crop_projection_window(
                        rgb, target_uv, width=int(w), height=int(h), center_bias_y=cby
                    )
                    if crop is not None:
                        tag = f"frame {frame_id} | proj window"
                        return frame_id, crop, tag
            else:
                uv, valid = project_points(c2w, K, center.reshape(1, 3))
                if valid[0]:
                    picked = crop_chair_seg_scored(
                        rgb, resolve_seg_npy(scene_dir, frame_id), uv[0]
                    )
                    if picked is not None:
                        crop, _, mdist, mcenter = picked
                        tag = f"frame {frame_id} | seg mask d={mdist:.0f}"
                        return frame_id, crop, tag

    best = None
    best_score = -1.0
    for fr in meta["frames"]:
        frame_id = frame_id_from_path(fr["file_path"])
        if frame_id in used_frames:
            continue
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        uv, valid = project_points(c2w, K, center.reshape(1, 3))
        if not valid[0]:
            continue
        rgb_path = resolve_rgb(scene_dir, frame_id)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        picked = crop_chair_seg_scored(rgb, resolve_seg_npy(scene_dir, frame_id), uv[0])
        if picked is None:
            continue
        crop, score, mdist, mcenter = picked
        if any(np.linalg.norm(np.array(mcenter) - np.array(p)) < 90 for p in used_mask_centers):
            continue
        if score > best_score:
            best_score = score
            best = (frame_id, crop, f"frame {frame_id} | seg mask d={mdist:.0f}", mcenter)
    if best is None:
        return None
    return best[0], best[1], best[2]


def find_complete_dining_view(
    scene_dir: str,
    chair_centers: dict[str, np.ndarray],
    table_aabb: tuple[float, ...],
    meta: dict,
    K: np.ndarray,
) -> tuple[str, np.ndarray] | None:
    """One frame showing table + all dining chairs (centers in view)."""
    table_center = bbox_center(table_aabb)
    pts = np.stack([table_center, *chair_centers.values()], axis=0)

    best = None
    best_score = -1.0
    for fr in meta["frames"]:
        frame_id = frame_id_from_path(fr["file_path"])
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        cam = c2w[:3, 3]
        if float(np.linalg.norm(cam[[0, 2]] - table_center[[0, 2]])) > 4.0:
            continue

        uv, valid = project_points(c2w, K, pts)
        if valid.sum() < len(pts):
            continue
        H, W = 480, 640
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        if in_bounds.sum() < len(pts):
            continue

        rgb_path = resolve_rgb(scene_dir, frame_id)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        spread_x = float(x2 - x1)
        spread_y = float(y2 - y1)
        if spread_x < 200:
            continue

        # When chairs span most of the frame horizontally, keep full height for context.
        if spread_x > 350 and spread_y < 180:
            crop = rgb.copy()
        else:
            pad_x = 0.08 * spread_x
            pad_y = max(0.10 * spread_y, 40.0)
            xx1 = int(max(0, math.floor(x1 - pad_x)))
            yy1 = int(max(0, math.floor(y1 - pad_y)))
            xx2 = int(min(W - 1, math.ceil(x2 + pad_x)))
            yy2 = int(min(H - 1, math.ceil(y2 + pad_y)))
            if xx2 - xx1 < 180 or yy2 - yy1 < 120:
                continue
            crop = rgb[yy1 : yy2 + 1, xx1 : xx2 + 1].copy()

        area = crop.shape[0] * crop.shape[1]
        score = area * spread_x
        if score > best_score:
            best_score = score
            best = (frame_id, crop)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Export dining table + nearby chairs (ep12)")
    ap.add_argument(
        "--out_dir",
        default="../../gaussnav_code_online/docs/paperfile/figs/dining_set_ep12",
    )
    ap.add_argument(
        "--max_views",
        type=int,
        default=1,
        help="views per chair (default 1 = one full-chair image each)",
    )
    ap.add_argument(
        "--export_table_views",
        action="store_true",
        help="also export separate table multi-view folder",
    )
    args = ap.parse_args()

    setup_matplotlib()
    scene_dir = os.path.join(DATA_ROOT, SCENE)
    out_dir = os.path.abspath(os.path.join(GN_ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    man_path = os.path.join(
        os.path.dirname(out_dir), "chair_instances_4ok3usBNeis/manifest.json"
    )
    if not os.path.isfile(man_path):
        raise FileNotFoundError(f"chair instance manifest not found: {man_path}")
    if os.path.isfile(man_path):
        manifest_chairs = json.load(open(man_path, encoding="utf-8"))

    dining_chair_ids = select_dining_chairs(manifest_chairs["instances"])
    print(f"[info] dining chairs: {dining_chair_ids}")

    chair_meta = {x["id"]: x for x in manifest_chairs["instances"] if x["id"] in dining_chair_ids}
    chair_centers = {k: np.array(v["center_xyz"], dtype=np.float64) for k, v in chair_meta.items()}
    centers = np.stack(list(chair_centers.values()), axis=0)
    table_aabb = estimate_table_aabb(centers)

    with open(os.path.join(scene_dir, "instance_retrieval.json"), encoding="utf-8") as f:
        ir = json.load(f)
    entries = ir[str(CHAIR_ID)]
    groups = dedupe_instance_groups(entries)

    meta = load_transforms(scene_dir)
    K = intrinsics_from_meta(meta)

    manifest = {
        "scene": SCENE,
        "target_chair": TARGET_CHAIR,
        "dining_chairs": list(dining_chair_ids),
        "table_aabb_xyz": {
            "x0": table_aabb[0],
            "x1": table_aabb[1],
            "y0": table_aabb[2],
            "y1": table_aabb[3],
            "z0": table_aabb[4],
            "z1": table_aabb[5],
        },
        "exports": {},
    }

    id_to_group = {}
    for gi, grp in enumerate(groups):
        cid = f"chair_{gi + 1:02d}"
        id_to_group[cid] = grp

    # --- optional table folder ---
    if args.export_table_views and args.max_views > 0:
        table_dir = os.path.join(out_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        manifest["exports"]["table"] = []
        corners = bbox_corners(table_aabb)
        center = corners.mean(axis=0)
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for fr in meta["frames"]:
            frame_id = frame_id_from_path(fr["file_path"])
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            rgb = cv2.imread(resolve_rgb(scene_dir, frame_id))
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            crop, sc = crop_object_from_3d_bbox(
                rgb, c2w, K, expand_bbox_3d(table_aabb, margin_xz=0.08, margin_y=0.05)
            )
            if crop is not None:
                candidates.append((frame_id, crop, c2w))
        table_views = select_diverse_views(candidates, center, meta, args.max_views)
        for vi, (frame_id, crop, tag) in enumerate(table_views, start=1):
            stem = os.path.join(table_dir, f"view_{vi:02d}")
            export_panel(crop, stem, f"table | {tag}")
            manifest["exports"]["table"].append(
                {"view": vi, "frame": frame_id, "path": f"table/view_{vi:02d}.png"}
            )

    # --- one full-chair image per dining chair ---
    used_frames: set[str] = set()
    used_mask_centers: list[tuple[float, float]] = []
    for chair_id in dining_chair_ids:
        grp = id_to_group.get(chair_id)
        if grp is None:
            print(f"[warn] missing group for {chair_id}")
            continue
        picked = pick_best_full_chair_view(
            scene_dir,
            entries,
            grp,
            chair_id,
            chair_centers,
            meta,
            K,
            used_frames=used_frames,
            used_mask_centers=used_mask_centers,
        )
        obj_dir = os.path.join(out_dir, chair_id)
        os.makedirs(obj_dir, exist_ok=True)
        manifest["exports"][chair_id] = []
        if picked is None:
            print(f"[warn] no full-chair view for {chair_id}")
            continue
        frame_id, crop, tag = picked
        preset = CHAIR_VIEW_PRESETS.get(chair_id, {})
        if preset.get("frame") != frame_id or preset.get("method") == "proj":
            used_frames.add(frame_id)
        if preset.get("method") != "proj":
            c2w = get_c2w(meta, frame_id)
            if c2w is not None:
                uv, valid = project_points(c2w, K, chair_centers[chair_id].reshape(1, 3))
                if valid[0]:
                    seg_path = resolve_seg_npy(scene_dir, frame_id)
                    scored = crop_chair_seg_scored(
                        cv2.cvtColor(cv2.imread(resolve_rgb(scene_dir, frame_id)), cv2.COLOR_BGR2RGB),
                        seg_path,
                        uv[0],
                    )
                    if scored is not None:
                        _, _, _, mcenter = scored
                        used_mask_centers.append((float(mcenter[0]), float(mcenter[1])))
        star = " *TARGET*" if chair_id == TARGET_CHAIR else ""
        stem = os.path.join(obj_dir, chair_id)
        cap = f"{chair_id}{star} | {tag}"
        export_panel(crop, stem, cap)
        manifest["exports"][chair_id].append(
            {"frame": frame_id, "path": f"{chair_id}/{chair_id}.png", "size": [int(crop.shape[1]), int(crop.shape[0])]}
        )

    # --- complete dining set: table + 4 chairs ---
    complete = find_complete_dining_view(scene_dir, chair_centers, table_aabb, meta, K)
    if complete:
        frame_id, crop = complete
        stem = os.path.join(out_dir, "dining_set_complete")
        cap = f"dining set (table + chairs) | frame {frame_id}"
        export_panel(crop, stem, cap)
        manifest["exports"]["dining_set_complete"] = {
            "frame": frame_id,
            "path": "dining_set_complete.png",
        }

    man_out = os.path.join(out_dir, "manifest.json")
    with open(man_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] dining set exports -> {out_dir}")
    print(f"[done] manifest -> {man_out}")


if __name__ == "__main__":
    main()
