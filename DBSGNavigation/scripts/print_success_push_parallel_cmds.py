#!/usr/bin/env python3
"""按 GPU 列表自动切分 success_push 试验，并打印并行命令。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def split_ranges(total: int, n_parts: int) -> List[Tuple[int, int]]:
    base = total // n_parts
    rem = total % n_parts
    out: List[Tuple[int, int]] = []
    start = 1
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        end = start + size - 1
        out.append((start, end))
        start = end + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="打印 success_push 四卡并行命令。")
    parser.add_argument("--resume_dir", required=True, help="调参目录，如 experiments/tune_goal_params/20260417_021210")
    parser.add_argument("--gpus", default="2,4,5,6", help="GPU 列表，逗号分隔，如 2,4,5,6")
    parser.add_argument("--max_episodes", type=int, default=1000, help="每个 trial 的 episode 数。")
    parser.add_argument("--base_config", default="configs/instance_imagenav_val_nogt.yaml")
    parser.add_argument("--summary_name", default="summary_1000ep_success_push.tsv")
    parser.add_argument("--log_dirname", default="logs_1000ep_success_push")
    args = parser.parse_args()

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("Empty --gpus")
    total_trials = 14  # make_trials_success_push 当前固定为 14 组
    ranges = split_ranges(total_trials, len(gpus))

    print("# 在 DBSGNavigation 根目录执行；每段放到一个独立终端。")
    print(f"RESUME_DIR={args.resume_dir}")
    print()
    for idx, (gpu, (lo, hi)) in enumerate(zip(gpus, ranges), 1):
        if lo > hi:
            continue
        print(f"# Terminal {idx} | GPU {gpu} | trials {lo}-{hi}")
        print(
            "python3 scripts/tune_goal_params.py "
            f"--suite success_push --max_episodes {args.max_episodes} "
            f"--base_config {args.base_config} --resume_dir \"$RESUME_DIR\" "
            f"--summary_name {args.summary_name} --log_dirname {args.log_dirname} "
            f"--only_trials {lo}-{hi} --cuda_visible_devices {gpu} --no_session_log"
        )
        print()

    out_path = Path(args.resume_dir) / args.summary_name
    print("# 全部终端跑完后，汇总在：")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

