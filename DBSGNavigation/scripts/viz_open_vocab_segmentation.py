"""Export open-vocabulary segmentation viz aligned with scene_exploration thumb.

Uses the same exploration frame as ``viz_scene_exploration.py`` (2nd linspace sample).

Example:
  python scripts/viz_open_vocab_segmentation.py --scene 4ok3usBNeis
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "DBSGConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"


def exploration_sample_frame_id(scene_dir: str, sample_index: int = 1) -> int:
    """Match viz_scene_exploration thumb frame (default: 2nd of 4 linspace samples)."""
    with open(os.path.join(scene_dir, "transforms.json")) as f:
        n = len(json.load(f)["frames"])
    ids = np.linspace(0, n - 1, 4, dtype=int)
    return int(ids[sample_index])


def read_bgr(path: str):
    im = cv2.imread(path)
    return im


def main():
    ap = argparse.ArgumentParser(
        description="Open-vocabulary segmentation thumb (Grounding DINO + SAM viz)"
    )
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="override frame id; default = same as scene_exploration thumb",
    )
    ap.add_argument(
        "--out",
        default="../docs/figs/open_vocab_segmentation_explainer.png",
    )
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    fid = args.frame if args.frame is not None else exploration_sample_frame_id(scene_dir)
    frame_name = f"frame_{fid:05d}"

    rgb_path = os.path.join(scene_dir, "rgb", f"{frame_name}.png")
    seg_path = os.path.join(scene_dir, "seg_semantic", f"{frame_name}.png")
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(
            f"Missing {seg_path}. Run seg_train_imgs.py on this scene first."
        )

    rgb_bgr = read_bgr(rgb_path)
    seg_bgr = read_bgr(seg_path)
    if rgb_bgr is None or seg_bgr is None:
        raise FileNotFoundError(f"Could not read RGB/seg for {frame_name}")

    # --- method thumb (same resolution as scene_exploration.png) ---
    thumb_dir = os.path.join(ROOT, "..", "docs/figs/method_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, "ov_segmentation.png")
    cv2.imwrite(thumb_path, seg_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # --- optional explainer: RGB | OV overlay ---
    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    axes[0].imshow(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"(a) Exploration RGB — {frame_name}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("(b) Open-vocabulary segmentation (DINO + SAM)", fontsize=10)
    axes[1].axis("off")
    fig.suptitle(
        f"Open-vocabulary segmentation — {args.scene}  |  "
        "text prompts → detect → SAM masks (6 IIN classes + viz)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"scene={args.scene} frame={frame_name}")
    print("saved", thumb_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
