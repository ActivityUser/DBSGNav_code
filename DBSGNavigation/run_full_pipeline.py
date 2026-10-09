#!/usr/bin/env python3
"""
一键运行完整流程：预处理 → 建图（可选）→ 导航评测

步骤：
  1. seg_train_imgs（创新点1：开放词汇分割）
  2. transform_pointcloud（生成 instance_retrieval.json）
  3. build_instance_clip_vectors（生成 instance_clip_vectors.json）
  4. run_ablation（导航评测，默认 full 预设）

前置：DBSGConstruction 建图（params.npz）若尚未完成，需先运行：
  cd DBSGConstruction && python scripts/run_build_val_semantic_gaussians.py ...

用法：
  cd DBSGNavigation
  python run_full_pipeline.py
  python run_full_pipeline.py --preset baseline
  python run_full_pipeline.py --skip-seg   # 36 个场景已跑过 seg 时跳过创新点1，节省时间
  python run_full_pipeline.py --skip-seg --skip-transform --skip-clip   # 仅运行导航
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# 使用 HuggingFace 镜像，避免 CLIP 下载时 huggingface.co 不可达
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_NAV_ROOT = os.path.dirname(os.path.abspath(__file__))
_SCENES_ROOT = os.path.normpath(os.path.join(_NAV_ROOT, "../DBSGConstruction/configs/data/habitat"))


def _run(cmd: list[str], step_name: str, cwd: str | None = None) -> bool:
    """执行命令，失败时返回 False。"""
    cwd = cwd or _NAV_ROOT
    print(f"\n{'='*60}\n[步骤] {step_name}\n{'='*60}")
    print(" ".join(cmd))
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        print(f"\n[失败] {step_name} 返回码 {ret.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="一键运行：seg → transform → clip → 导航")
    parser.add_argument(
        "--preset",
        default="full",
        choices=["full", "baseline", "innov2", "innov3", "innov4", "innov5", "innov6"],
        help="导航消融预设，默认 full",
    )
    parser.add_argument("--skip-seg", action="store_true", help="跳过 seg_train_imgs")
    parser.add_argument("--skip-transform", action="store_true", help="跳过 transform_pointcloud")
    parser.add_argument("--skip-clip", action="store_true", help="跳过 build_instance_clip_vectors")
    parser.add_argument("--skip-nav", action="store_true", help="跳过导航评测（仅跑预处理）")
    parser.add_argument(
        "--scenes-root",
        default=_SCENES_ROOT,
        help="场景根目录，默认 ../DBSGConstruction/configs/data/habitat",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.scenes_root):
        print(f"Error: 场景目录不存在: {args.scenes_root}")
        sys.exit(1)

    steps_ok = True

    # 1. seg_train_imgs（若所有场景已有 seg_semantic/*.npy 则自动跳过）
    run_seg = not args.skip_seg
    if run_seg and os.path.isdir(args.scenes_root):
        from glob import glob
        scene_dirs = [d for d in glob(os.path.join(args.scenes_root, "*")) if os.path.isdir(d)]
        seg_ok = 0
        for d in scene_dirs:
            seg_dir = os.path.join(d, "seg_semantic")
            if os.path.isdir(seg_dir) and len(glob(os.path.join(seg_dir, "*.npy"))) > 0:
                seg_ok += 1
        if scene_dirs and seg_ok == len(scene_dirs):
            print("\n[跳过] 所有场景已有 seg_semantic/*.npy，跳过创新点1（如需重跑请先删 seg_semantic 或不加 --skip-seg 时手动跑 seg_train_imgs）")
            run_seg = False
    if run_seg:
        cfg = os.path.join(_NAV_ROOT, "configs", "seg_open_vocab.yaml")
        if not os.path.isfile(cfg):
            print(f"Warning: 未找到 {cfg}，请手动指定 seg 配置")
        else:
            steps_ok = _run(
                [sys.executable, "gaussian_utils/seg_train_imgs.py", "--config", cfg],
                "创新点1：开放词汇分割 (seg_train_imgs)",
            ) and steps_ok

    # 2. transform_pointcloud
    if not args.skip_transform and steps_ok:
        steps_ok = _run(
            [sys.executable, "gaussian_utils/transform_pointcloud.py", "--scenes_root", args.scenes_root],
            "生成 instance_retrieval.json (transform_pointcloud)",
        ) and steps_ok

    # 3. build_instance_clip_vectors
    if not args.skip_clip and steps_ok:
        steps_ok = _run(
            [sys.executable, "gaussian_utils/build_instance_clip_vectors.py", "--scenes_root", args.scenes_root],
            "生成 instance_clip_vectors.json",
        ) and steps_ok

    # 4. 导航评测
    if not args.skip_nav and steps_ok:
        steps_ok = _run(
            [sys.executable, "run_ablation.py", "--preset", args.preset],
            f"导航评测 (preset={args.preset})",
        ) and steps_ok

    if not steps_ok:
        sys.exit(1)
    print("\n" + "=" * 60)
    print("全部步骤完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
