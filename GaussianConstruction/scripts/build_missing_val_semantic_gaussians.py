#!/usr/bin/env python3
"""
一键补全 HM3D val 中缺失的 Semantic Gaussian（params.npz）。
要求：configs/data/habitat_val（或 --data_dir）下已有各场景的 transforms.json + rgb/depth（建图数据已存在则直接建图）。

用法:
  cd /path/to/GaussianConstruction
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python scripts/build_missing_val_semantic_gaussians.py
  # 若显存紧张，可限制帧数：
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python scripts/build_missing_val_semantic_gaussians.py --max_frames 100 --stride 2
"""

import argparse
import os
import subprocess
import sys
from glob import glob

# 与 check_missing_semantic_gaussian_scenes.py 一致
HM3D_VAL_SCENE_IDS = [
    "TEEsavR23oF", "wcojb4TFT35", "k1cupFYWXJ6", "y9hTuugGdiq",
    "CrMo8WxCyVb", "svBbv1Pavdk", "p53SfW6mjZe", "h1zeeAwLh9Z",
    "mL8ThkuaVTM", "eF36g7L6Z9M", "7MXmsvcQjpJ", "Dd4bFSTQ8gi",
    "BAbdmeyTvMZ", "QaLdnwvtxbs", "yr17PDCnDDW", "qyAac8rV8Zk",
    "q3zU7Yy5E5s", "zt1RVoi7PcG", "DYehNKdT76V", "q5QZSEeHe5g",
    "bCPU9suPUw9", "ziup5kvtCCR", "a8BtkwhxdRV", "5cdEh9F2hJL",
    "GLAQ4DNUx5U", "LT9Jq6dN3Ea", "MHPLjHsuG27", "VBzV5z6i1WS",
    "bxsVRursffK", "mv2HUxq3B53", "4ok3usBNeis", "XB4GS9ShBRE",
    "Nfvxx8J5NCo", "6s7QHgap2fW", "cvZr5TUy5C5", "HY1NcmCgn3n",
]


def main():
    parser = argparse.ArgumentParser(description="Build missing Semantic Gaussians for HM3D val scenes.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="建图数据根目录，默认 configs/data/habitat_val")
    parser.add_argument("--group_name", type=str, default="habitat_val_final",
                        help="输出目录 experiments/<group_name>")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="每场景最大帧数，减轻显存（如 100）")
    parser.add_argument("--stride", type=int, default=None,
                        help="帧步长，如 2 表示每隔一帧（减轻显存）")
    parser.add_argument("--dry_run", action="store_true", help="只打印将要执行的命令，不执行")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = args.data_dir or os.path.join(base_dir, "configs", "data", "habitat_val")
    workdir = os.path.join(base_dir, "experiments", args.group_name)
    os.makedirs(workdir, exist_ok=True)

    def has_params(sid):
        for d in glob(os.path.join(workdir, f"{sid}*")):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "params.npz")):
                return True
        return False

    missing = [s for s in HM3D_VAL_SCENE_IDS if not has_params(s)]
    if not missing:
        print("所有 val 场景均已有 params.npz，无需补全。")
        return 0

    print(f"待建图场景数: {len(missing)} / {len(HM3D_VAL_SCENE_IDS)}")
    print("缺失:", ", ".join(sorted(missing)))

    # 写入临时场景列表，只建缺失的
    scenes_file = os.path.join(workdir, "_missing_val_scenes.txt")
    with open(scenes_file, "w") as f:
        for s in missing:
            f.write(s + "\n")

    cmd = [
        sys.executable,
        os.path.join(script_dir, "run_build_val_semantic_gaussians.py"),
        "--scenes_file", scenes_file,
        "--group_name", args.group_name,
        "--no_skip_existing",
    ]
    if args.max_frames is not None:
        cmd += ["--max_frames", str(args.max_frames)]
    if args.stride is not None:
        cmd += ["--stride", str(args.stride)]

    print("执行:", " ".join(cmd))
    if args.dry_run:
        return 0
    ret = subprocess.run(cmd, cwd=base_dir)
    if ret.returncode == 0:
        try:
            os.remove(scenes_file)
        except Exception:
            pass
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
