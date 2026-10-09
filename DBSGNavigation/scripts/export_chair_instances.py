#!/usr/bin/env python3
"""Export deduplicated chair instance RGB crops for one HM3D scene (PNG + PDF + EPS).

Each unique chair (3D bbox clustering on instance_retrieval.json, coco id 56) is saved as:
  chair_XX.png / .pdf / .eps

Example (CWD = DBSGNavigation):
  python scripts/export_chair_instances.py \\
    --scene 4ok3usBNeis \\
    --out_dir ../../gaussnav_code_online/docs/paperfile/figs/chair_instances_4ok3usBNeis \\
    --goal_meta ../../gaussnav_code_online/docs/paperfile/figs/chair_ep12_goal.json
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

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GN_ROOT, ".."))
sys.path.insert(0, GN_ROOT)
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))

from paper_matplotlib import setup_matplotlib  # noqa: E402
from scripts.export_paper_similarity_map import save_vector  # noqa: E402
from scripts.viz_instance_database import (  # noqa: E402
    crop_instance,
    resolve_rgb,
    resolve_seg_npy,
)

DEFAULT_DATA_ROOT = os.path.join(
    GN_ROOT, "..", "DBSGConstruction/configs/data/habitat_val"
)
CHAIR_COCO_ID = 56
SAVE_DPI = 300


def parse_bbox(entry: list) -> tuple[float, float, float, float, float, float]:
    b = entry[0][0]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])


def bbox_center(b: tuple[float, ...]) -> np.ndarray:
    return np.array(
        [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])],
        dtype=np.float64,
    )


def dedupe_instance_groups(entries: list, eps: float = 1.2) -> list[list[int]]:
    centers = np.array([bbox_center(parse_bbox(e)) for e in entries], dtype=np.float64)
    used = np.zeros(len(entries), dtype=bool)
    groups: list[list[int]] = []
    for i in range(len(entries)):
        if used[i]:
            continue
        dist = np.linalg.norm(centers - centers[i], axis=1)
        grp = np.where(dist < eps)[0]
        used[grp] = True
        groups.append(grp.tolist())
    return groups


def pick_representative(entries: list, indices: list[int]) -> tuple[int, list]:
    centers = np.array([bbox_center(parse_bbox(entries[i])) for i in indices])
    mean_c = centers.mean(axis=0)
    best_i = min(indices, key=lambda i: float(np.linalg.norm(bbox_center(parse_bbox(entries[i])) - mean_c)))
    return best_i, entries[best_i]


def load_crop(scene_dir: str, entry: list, *, pad: float = 0.25, max_side: int = 512) -> np.ndarray | None:
    frame_id = entry[1]
    rgb_path = resolve_rgb(scene_dir, frame_id)
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    crop = crop_instance(rgb, resolve_seg_npy(scene_dir, frame_id), CHAIR_COCO_ID, pad=pad)
    h, w = crop.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def export_single(crop: np.ndarray, out_stem: str, caption: str) -> None:
    h, w = crop.shape[:2]
    fig_w = max(2.4, w / SAVE_DPI + 0.35)
    fig_h = max(2.0, h / SAVE_DPI + 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=SAVE_DPI, facecolor="white")
    ax.imshow(crop)
    ax.axis("off")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.98)
    save_vector(fig, out_stem)


def export_overview(crops: list[tuple[str, np.ndarray, str]], out_stem: str, ncols: int = 6) -> None:
    if not crops:
        return
    n = len(crops)
    nrows = int(np.ceil(n / ncols))
    cell = 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows), dpi=SAVE_DPI)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (label, crop, tag) in enumerate(crops):
        axes[i].imshow(crop)
        axes[i].set_title(f"{label}\n{tag}", fontsize=7)
        axes[i].axis("off")
    fig.suptitle("Chair instances — 4ok3usBNeis (deduplicated)", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_vector(fig, out_stem)


def load_goal_center(goal_meta: str | None) -> np.ndarray | None:
    if not goal_meta or not os.path.isfile(goal_meta):
        return None
    meta = json.load(open(goal_meta, encoding="utf-8"))
    scene_dir = os.path.join(DEFAULT_DATA_ROOT, "4ok3usBNeis")
    gi = os.path.join(
        GN_ROOT,
        "examples/case_study_ov/full/chair/goal_info.txt",
    )
    if os.path.isfile(gi):
        info = {}
        for line in open(gi, encoding="utf-8"):
            if "=" in line and not line.startswith("---"):
                k, v = line.strip().split("=", 1)
                info[k.strip()] = v.strip()
        if "gt_viewpoint_goal_pos" in info:
            pos = json.loads(info["gt_viewpoint_goal_pos"])
            return np.array(pos, dtype=np.float64)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export chair instance crops (PNG/PDF/EPS)")
    ap.add_argument("--scene", default="4ok3usBNeis")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--out_dir",
        default="../docs/paperfile/figs/chair_instances_4ok3usBNeis",
    )
    ap.add_argument("--goal_meta", default=None, help="Optional chair_ep12_goal.json to mark target")
    ap.add_argument("--dedupe_eps", type=float, default=1.2)
    ap.add_argument("--max_side", type=int, default=512)
    args = ap.parse_args()

    setup_matplotlib()
    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_dir = os.path.abspath(os.path.join(GN_ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    ir_path = os.path.join(scene_dir, "instance_retrieval.json")
    with open(ir_path, encoding="utf-8") as f:
        ir = json.load(f)
    entries = ir.get(str(CHAIR_COCO_ID), [])
    if not entries:
        raise RuntimeError(f"no chair entries in {ir_path}")

    groups = dedupe_instance_groups(entries, eps=args.dedupe_eps)
    goal_center = load_goal_center(args.goal_meta)

    group_centers = []
    for grp in groups:
        c = np.mean([bbox_center(parse_bbox(entries[i])) for i in grp], axis=0)
        group_centers.append(c)
    group_centers_arr = np.array(group_centers)

    target_idx = None
    if goal_center is not None:
        dists = np.linalg.norm(group_centers_arr - goal_center, axis=1)
        target_idx = int(np.argmin(dists))

    overview: list[tuple[str, np.ndarray, str]] = []
    manifest = {
        "scene": args.scene,
        "n_raw_entries": len(entries),
        "n_deduplicated": len(groups),
        "target_chair_id": None,
        "instances": [],
    }

    for gi, grp in enumerate(groups):
        inst_id = gi + 1
        rep_i, rep = pick_representative(entries, grp)
        center = group_centers[gi]
        crop = load_crop(scene_dir, rep, max_side=args.max_side)
        if crop is None:
            print(f"[skip] chair_{inst_id:02d}: missing RGB/seg for frame {rep[1]}")
            continue

        is_target = target_idx == gi
        tag = "TARGET" if is_target else f"frame {rep[1]}"
        label = f"chair_{inst_id:02d}"
        if is_target:
            label += " *"
            manifest["target_chair_id"] = label

        stem = os.path.join(out_dir, f"chair_{inst_id:02d}")
        caption = (
            f"{label}  |  center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})  |  {tag}"
        )
        export_single(crop, stem, caption)
        overview.append((label, crop, tag))
        manifest["instances"].append(
            {
                "id": f"chair_{inst_id:02d}",
                "is_target": is_target,
                "frame": rep[1],
                "center_xyz": center.tolist(),
                "n_merged_detections": len(grp),
            }
        )

    export_overview(overview, os.path.join(out_dir, "chair_overview"))
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] exported {len(overview)} chairs -> {out_dir}")
    print(f"[done] manifest -> {manifest_path}")
    if manifest["target_chair_id"]:
        print(f"[info] ep12 target chair: {manifest['target_chair_id']}")


if __name__ == "__main__":
    main()
