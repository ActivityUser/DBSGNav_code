#!/usr/bin/env python3
"""
为分割质量评测重生成 DINO+SAM 掩码（抑制 plant/tv 过检）。

改动相对原版 seg_train_imgs：
  1) strict_coco6：仅 6 类文本，去掉 wall/floor/lamp→plant/tv 的宽松短语映射
  2) plant(58)/tv(62) 提高 box 阈值（默认 0.50，可用 CLI 覆盖）
  3) 输出到 seg_semantic_tight/，不覆盖生产用 seg_semantic/

评测：
  python eval_seg_quality.py --scenes ... --dino_subdir seg_semantic_tight \\
      --out ../docs/seg_quality_tight.md

依赖：与 seg_train_imgs 相同（GroundingDINO + SAM + CUDA）。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from glob import glob
from natsort import natsorted

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GAUSS_UTILS = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../../GaussianNavigation/gaussian_utils")
)
if _GAUSS_UTILS not in sys.path:
    sys.path.insert(0, _GAUSS_UTILS)

from open_vocab_segmenter import (  # noqa: E402
    DEFAULT_CATEGORY_THRESHOLDS,
    STRICT_COCO6_TEXT_PROMPT,
    init_segmenter,
    segment_frame,
)

FRAMES_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../configs/data/habitat_val")
)
DEFAULT_SCENES = [
    "4ok3usBNeis",
    "5cdEh9F2hJL",
    "6s7QHgap2fW",
    "a8BtkwhxdRV",
    "CrMo8WxCyVb",
]


def parse_category_thresholds(plant_box: float, tv_box: float, default_box: float):
    thr = {k: (default_box, 0.25) for k in (56, 57, 59, 61)}
    thr[58] = (plant_box, DEFAULT_CATEGORY_THRESHOLDS[58][1])
    thr[62] = (tv_box, DEFAULT_CATEGORY_THRESHOLDS[62][1])
    return thr


def frame_ids_from_gt(scene_dir: str, stride: int):
    gt_dir = os.path.join(scene_dir, "seg_gt")
    if not os.path.isdir(gt_dir):
        return []
    files = natsorted(glob(os.path.join(gt_dir, "frame_*.npy")))
    if stride > 1:
        files = files[::stride]
    return [os.path.basename(f).replace(".npy", "") for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--frames_root", default=FRAMES_ROOT)
    ap.add_argument("--out_subdir", default="seg_semantic_tight")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="对 seg_gt 帧列表下采样；须与 GT 覆盖一致，论文评测用 1（勿用 10）",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grounding_config", required=True)
    ap.add_argument("--grounding_checkpoint", required=True)
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--box_threshold", type=float, default=0.35, help="chair/couch/bed/toilet 等")
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--plant_box_threshold", type=float, default=0.50)
    ap.add_argument(
        "--tv_box_threshold",
        type=float,
        default=0.42,
        help="tv 略低于 plant，避免 0.5 全漏（可在 0.40–0.50 扫参）",
    )
    ap.add_argument("--nms_threshold", type=float, default=0.8)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    cat_thr = parse_category_thresholds(
        args.plant_box_threshold, args.tv_box_threshold, args.box_threshold
    )
    print("[tight] strict_coco6=True")
    print("[tight] prompt:", STRICT_COCO6_TEXT_PROMPT)
    print("[tight] category_thresholds:", cat_thr)
    print("[tight] out_subdir:", args.out_subdir)

    segmenter = init_segmenter(
        device=args.device,
        sam_checkpoint_path=args.sam_checkpoint,
        grounding_config_path=args.grounding_config,
        grounding_checkpoint_path=args.grounding_checkpoint,
    )

    for scene in args.scenes:
        scene_dir = os.path.join(args.frames_root, scene)
        rgb_dir = os.path.join(scene_dir, "rgb")
        out_dir = os.path.join(scene_dir, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)

        fids = frame_ids_from_gt(scene_dir, args.stride)
        if not fids:
            fids = [
                os.path.splitext(os.path.basename(p))[0]
                for p in natsorted(glob(os.path.join(rgb_dir, "*.png")))
            ]
            if args.stride > 1:
                fids = fids[:: args.stride]

        for fid in fids:
            rgb_path = os.path.join(rgb_dir, fid + ".png")
            if not os.path.isfile(rgb_path):
                rgb_path = os.path.join(rgb_dir, fid.replace("frame_", "") + ".png")
            if not os.path.isfile(rgb_path):
                print("skip missing rgb", scene, fid)
                continue

            npy_out = os.path.join(out_dir, fid + ".npy")
            if not args.no_resume and os.path.isfile(npy_out):
                continue

            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            rgb = rgb[:, :, ::-1]

            save_mask, _vis = segment_frame(
                segmenter,
                rgb,
                text_prompt=STRICT_COCO6_TEXT_PROMPT,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_threshold=args.nms_threshold,
                strict_coco6=True,
                category_thresholds=cat_thr,
            )
            np.save(npy_out, save_mask)
            print("wrote", npy_out)

    print("done.")


if __name__ == "__main__":
    main()
