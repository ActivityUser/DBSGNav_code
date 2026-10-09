#!/usr/bin/env python3
"""
对已有 seg_semantic/*.npy 做 plant(58)/tv(62) 后处理，抑制宽松 prompt 过检。

策略（仅改 58/62，其它类不动）：
  1) 连通域分析：丢弃面积 < min_area 的碎片
  2) 每类最多保留 max_components 个大连通域（按面积降序）
  3) 单帧该类像素占比 > max_area_frac 时，只保留前 K 个域直至占比达标

输出到 seg_semantic_refined/（或 --inplace 写回，慎用）

用法：
  python refine_seg_plant_tv.py --scenes 4ok3usBNeis ... \\
      --max_area_frac 0.08 --max_components 2 --min_area 400
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "../configs/data/habitat_val")
)
REFINE_CLASSES = (58, 62)


def refine_binary_mask(
    mask: np.ndarray,
    min_area: int,
    max_components: int,
    max_area_frac: float,
) -> np.ndarray:
    h, w = mask.shape
    cap = int(max_area_frac * h * w)
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comps.append((area, i))
    comps.sort(reverse=True)
    out = np.zeros((h, w), dtype=bool)
    kept = 0
    total = 0
    for area, lab in comps:
        if kept >= max_components:
            break
        if total + area > cap and kept > 0:
            break
        out |= labels == lab
        total += area
        kept += 1
    return out


def refine_save_mask(
    save_mask: dict,
    min_area: int,
    max_components: int,
    max_area_frac: float,
) -> dict:
    out = dict(save_mask)
    for cid in REFINE_CLASSES:
        masks = out.get(cid)
        if not masks:
            continue
        acc = np.zeros_like(np.asarray(masks[0]), dtype=bool)
        for m in masks:
            acc |= np.asarray(m, dtype=bool)
        if not acc.any():
            out[cid] = None
            continue
        refined = refine_binary_mask(
            acc, min_area, max_components, max_area_frac
        )
        out[cid] = [refined] if refined.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--frames_root", default=FRAMES_ROOT)
    ap.add_argument("--in_subdir", default="seg_semantic")
    ap.add_argument("--out_subdir", default="seg_semantic_refined")
    ap.add_argument("--min_area", type=int, default=400)
    ap.add_argument("--max_components", type=int, default=2)
    ap.add_argument("--max_area_frac", type=float, default=0.08,
                    help="单帧 plant/tv 像素上限占图比例")
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    if args.inplace:
        args.out_subdir = args.in_subdir

    for scene in args.scenes:
        sd = os.path.join(args.frames_root, scene)
        ind = os.path.join(sd, args.in_subdir)
        outd = os.path.join(sd, args.out_subdir)
        os.makedirs(outd, exist_ok=True)
        files = sorted(glob(os.path.join(ind, "*.npy")))
        for path in files:
            data = np.load(path, allow_pickle=True).item()
            refined = refine_save_mask(
                data,
                args.min_area,
                args.max_components,
                args.max_area_frac,
            )
            out_path = os.path.join(outd, os.path.basename(path))
            np.save(out_path, refined)
        print(f"{scene}: refined {len(files)} -> {outd}")


if __name__ == "__main__":
    main()
