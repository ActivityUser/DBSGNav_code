"""
批量生成 seg_semantic/*.npy 与可视化 .png，使用开放词汇分割（Grounding DINO + SAM）。
【本脚本为 seg_train_imgs.py 的别名】seg_train_imgs.py 已改为使用 Grounding DINO + SAM，
直接使用 seg_train_imgs.py 即可。本脚本保留以兼容旧调用。
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from natsort import natsorted
from tqdm import tqdm

import cv2
import numpy as np
import torch

# 保证从任意 cwd 运行都能找到同目录的 open_vocab_segmenter
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from open_vocab_segmenter import init_segmenter, segment_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-vocab 分割：遍历场景 rgb/*.png，写 seg_semantic/*.npy 与 *.png"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选 YAML 配置文件路径，其中的路径与参数会被命令行覆盖",
    )
    parser.add_argument(
        "--scenes_root",
        type=str,
        default=None,
        help="场景根目录，其下每子目录为 scene，含 rgb/ 与将生成的 seg_semantic/",
    )
    parser.add_argument(
        "--grounding_config",
        type=str,
        default=None,
        help="Grounding DINO 配置文件路径（如 .py）",
    )
    parser.add_argument(
        "--grounding_checkpoint",
        type=str,
        default=None,
        help="Grounding DINO 权重路径",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="SAM 权重路径（如 sam_vit_h_4b8939.pth）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备，如 cuda:0",
    )
    parser.add_argument(
        "--scene_index_start",
        type=int,
        default=None,
        help="从第几个场景开始（含），用于断点续跑",
    )
    parser.add_argument(
        "--scene_index_end",
        type=int,
        default=None,
        help="到第几个场景结束（不含）；默认处理全部",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=None,
        help="Grounding DINO 框置信度阈值",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=None,
        help="Grounding DINO 文本阈值",
    )
    args = parser.parse_args()

    # 从 YAML 加载默认值（若存在）
    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if value is not None and getattr(args, key, None) is None:
                setattr(args, key, value)

    # 必填项
    if not args.scenes_root:
        raise ValueError("请指定 --scenes_root 或通过 --config 在 YAML 中设置 scenes_root")
    if not args.grounding_config:
        raise ValueError("请指定 --grounding_config 或通过 --config 在 YAML 中设置")
    if not args.grounding_checkpoint:
        raise ValueError("请指定 --grounding_checkpoint 或通过 --config 在 YAML 中设置")
    if not args.sam_checkpoint:
        raise ValueError("请指定 --sam_checkpoint 或通过 --config 在 YAML 中设置")

    # 默认值
    if args.device is None:
        args.device = "cuda:0"
    if args.scene_index_start is None:
        args.scene_index_start = 0
    if args.box_threshold is None:
        args.box_threshold = 0.35
    if args.text_threshold is None:
        args.text_threshold = 0.25
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    scenes_list = natsorted(glob(os.path.join(args.scenes_root, "*")))
    scenes_list = [d for d in scenes_list if os.path.isdir(d)]
    if args.scene_index_end is not None:
        scenes_list = scenes_list[args.scene_index_start : args.scene_index_end]
    else:
        scenes_list = scenes_list[args.scene_index_start :]

    segmenter = init_segmenter(
        device=device,
        sam_checkpoint_path=args.sam_checkpoint,
        grounding_config_path=args.grounding_config,
        grounding_checkpoint_path=args.grounding_checkpoint,
    )

    for scene in tqdm(scenes_list, desc="scenes"):
        rgbs_list = glob(os.path.join(scene, "rgb", "*.png"))
        rgbs_list = natsorted(rgbs_list)
        save_dir = os.path.join(scene, "seg_semantic")
        os.makedirs(save_dir, exist_ok=True)
        for rgb_path in tqdm(rgbs_list, desc=os.path.basename(scene), leave=False):
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue
            rgb = rgb[:, :, ::-1]  # BGR -> RGB
            save_file_name = os.path.splitext(os.path.basename(rgb_path))[0]
            save_mask, save_bgr_vis = segment_frame(
                segmenter,
                rgb,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            save_flag = any(save_mask[k] is not None for k in save_mask)
            if save_flag:
                np.save(os.path.join(save_dir, save_file_name + ".npy"), save_mask)
                cv2.imwrite(os.path.join(save_dir, save_file_name + ".png"), save_bgr_vis)


if __name__ == "__main__":
    main()
