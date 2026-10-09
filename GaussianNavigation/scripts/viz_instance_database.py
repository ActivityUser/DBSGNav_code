"""Visualize Instance Database (instance_retrieval + CLIP crops + optional Top-K).

Example:
  python scripts/viz_instance_database.py --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "GaussianConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"
COCO_IDS = [56, 57, 58, 59, 61, 62]
COCO_NAMES = {
    56: "chair",
    57: "sofa",
    58: "plant",
    59: "bed",
    61: "toilet",
    62: "tv",
}


def resolve_rgb(scene_dir, frame_id):
    fid = str(frame_id)
    if not fid.startswith("frame_"):
        fid = f"frame_{int(fid):05d}"
    if not fid.endswith(".png"):
        fid += ".png"
    return os.path.join(scene_dir, "rgb", fid)


def resolve_seg_npy(scene_dir, frame_id):
    fid = str(frame_id)
    if fid.startswith("frame_"):
        fid = fid[6:]
    if fid.endswith(".png"):
        fid = fid[:-4]
    return os.path.join(scene_dir, "seg_semantic", f"frame_{int(fid):05d}.npy")


def crop_instance(rgb, seg_path, coco_id, pad=0.25):
    if not os.path.isfile(seg_path):
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (160, 120))
    masks = np.load(seg_path, allow_pickle=True)
    if isinstance(masks, np.ndarray) and masks.dtype == object:
        masks = masks.item()
    if not isinstance(masks, dict) or coco_id not in masks:
        return cv2.resize(rgb, (160, 120))
    mlist = masks[coco_id]
    if not mlist:
        return cv2.resize(rgb, (160, 120))
    mask = mlist[0]
    if not isinstance(mask, np.ndarray):
        return cv2.resize(rgb, (160, 120))
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return cv2.resize(rgb, (160, 120))
    H, W = rgb.shape[:2]
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y2 - y1) * pad), int((x2 - x1) * pad)
    y1, y2 = max(0, y1 - ph), min(H - 1, y2 + ph)
    x1, x2 = max(0, x1 - pw), min(W - 1, x2 + pw)
    crop = rgb[y1 : y2 + 1, x1 : x2 + 1]
    return cv2.resize(crop, (160, 120))


def sample_entries_per_class(ir, n_per_class=2, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for cid in COCO_IDS:
        key = str(cid)
        entries = ir.get(key, [])
        if not entries:
            continue
        if len(entries) <= n_per_class:
            idx = list(range(len(entries)))
        else:
            idx = rng.choice(len(entries), size=n_per_class, replace=False)
        out[cid] = [entries[i] for i in idx]
    return out


def load_retrieval_topk(scene_dir, goal_path, top_k=5, device="cpu"):
    cv_path = os.path.join(scene_dir, "instance_clip_vectors_v2.json")
    if not os.path.isfile(cv_path):
        cv_path = os.path.join(scene_dir, "instance_clip_vectors.json")
    if not os.path.isfile(cv_path):
        return None
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    g = cv2.imread(goal_path)
    if g is None:
        return None
    r = OpenVocabRetriever(
        torch.device(device), clip_vectors_filename=os.path.basename(cv_path)
    )
    if not r.load_scene(scene_dir, instance_clip_vectors_path=cv_path):
        return None
    return r.retrieve(g[:, :, ::-1], top_k=top_k)


def main():
    ap = argparse.ArgumentParser(description="Visualize instance database for one scene")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--goal_image", default=None, help="optional: show CLIP Top-K retrieval")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", default="../docs/figs/instance_database_explainer.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ir_path = os.path.join(scene_dir, "instance_retrieval.json")
    with open(ir_path) as f:
        ir = json.load(f)

    counts = [len(ir.get(str(c), [])) for c in COCO_IDS]
    total = sum(counts)

    has_goal = args.goal_image is not None
    n_rows = 3 if has_goal else 2
    fig = plt.figure(figsize=(13.5, 8.5 if has_goal else 6), dpi=200)
    gs = fig.add_gridspec(n_rows, 6, height_ratios=[0.9, 1.4, 1.0] if has_goal else [0.9, 1.5],
                         hspace=0.35, wspace=0.15)

    ax_bar = fig.add_subplot(gs[0, :])
    bars = ax_bar.bar([COCO_NAMES[c] for c in COCO_IDS], counts, color="#4a7ab8")
    ax_bar.set_ylabel("# instances")
    ax_bar.set_title(f"(a) Instance inventory — {total:,} entries in instance_retrieval.json", fontsize=10)
    for b, v in zip(bars, counts):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v),
                    ha="center", va="bottom", fontsize=8)

    samples = sample_entries_per_class(ir, n_per_class=2)
    ax_grid_title = fig.add_subplot(gs[1, :])
    ax_grid_title.axis("off")
    ax_grid_title.set_title(
        "(b) Sample instance records (RGB crop + 3D bbox + navigable viewpoint + CLIP vector in .json)",
        fontsize=10, loc="left",
    )

    plot_idx = 0
    for cid in COCO_IDS:
        if cid not in samples:
            continue
        for ent in samples[cid]:
            row, col = divmod(plot_idx, 6)
            ax = fig.add_subplot(gs[1, col])
            frame_id = ent[1]
            rgb_path = resolve_rgb(scene_dir, frame_id)
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            crop = crop_instance(rgb, resolve_seg_npy(scene_dir, frame_id), cid)
            ax.imshow(crop)
            ax.set_title(f"{COCO_NAMES[cid]}\nframe {frame_id}", fontsize=7)
            ax.axis("off")
            plot_idx += 1
            if plot_idx >= 6:
                break
        if plot_idx >= 6:
            break

    if has_goal:
        goal_path = os.path.abspath(os.path.join(ROOT, args.goal_image))
        top = load_retrieval_topk(scene_dir, goal_path, args.top_k, args.device)
        ax_top = fig.add_subplot(gs[2, :])
        ax_top.axis("off")
        if top:
            panels = []
            titles = []
            g = cv2.imread(goal_path)
            g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
            panels.append(cv2.resize(g, (160, 120)))
            titles.append("Goal")
            for ent, pos, sim in top:
                frame_id = ent[1]
                rgb_path = resolve_rgb(scene_dir, frame_id)
                rgb = cv2.imread(rgb_path)
                if rgb is None:
                    continue
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cid_guess = 57
                for cid in COCO_IDS:
                    if ent in ir.get(str(cid), []):
                        cid_guess = cid
                        break
                crop = crop_instance(rgb, resolve_seg_npy(scene_dir, frame_id), cid_guess)
                crop = cv2.resize(crop, (160, 120))
                panels.append(crop)
                titles.append(f"#{len(panels)-1} sim={sim:.3f}\nframe {frame_id}")
            comb = np.hstack(panels)
            ax_top.imshow(comb)
            ax_top.set_title(
                f"(c) CLIP retrieval Top-{len(top)} (instance_clip_vectors_v2.json) — online OV branch entry",
                fontsize=10, loc="left",
            )
            for i, t in enumerate(titles):
                ax_top.text(i * 160 + 70, -8, t, ha="center", fontsize=7, transform=ax_top.transData)
        else:
            ax_top.text(0.5, 0.5, "CLIP vectors missing — skip Top-K panel",
                        ha="center", transform=ax_top.transAxes)

    fig.suptitle(
        f"Instance Database — scene {args.scene}  |  "
        "per-scene JSON: 3D bbox + viewpoint + frame id + 512-d CLIP embedding",
        fontsize=11, y=0.99,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"scene={args.scene} total_instances={total}")
    print("saved", out_path)

    # method thumb montage
    thumb_dir = os.path.join(ROOT, "..", "docs/figs/method_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    thumbs = []
    for cid in [57, 56, 59, 62]:
        if cid not in samples:
            continue
        ent = samples[cid][0]
        rgb = cv2.imread(resolve_rgb(scene_dir, ent[1]))
        if rgb is None:
            continue
        thumbs.append(crop_instance(rgb, resolve_seg_npy(scene_dir, ent[1]), cid))
    if thumbs:
        row = cv2.hconcat(thumbs[:4])
        cv2.imwrite(
            os.path.join(thumb_dir, "instance_database.png"),
            cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        cv2.imwrite(
            os.path.join(thumb_dir, "instance_retrieval.png"),
            cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )


if __name__ == "__main__":
    main()
