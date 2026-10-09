#!/usr/bin/env python3
"""读取 ablation_dyn_sweep_8gpu 各 stem 日志，按 Success 排序打印（对比 wo_dyn 目标）。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NAV = Path(__file__).resolve().parents[4]
_DEFAULT_LOG_DIR = (
    _NAV
    / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_dyn_sweep_8gpu_Agroup_g6_train"
)

STEMS = [
    "s01_alpha005",
    "s02_alpha015",
    "s03_int1",
    "s04_int3",
    "s05_cons025",
    "s06_cons045",
    "s07_plthr024",
    "s08_plmin4",
]

PAT = re.compile(
    r"succ\s*/\s*spl\s*/\s*dist.*?:\s*"
    r"(?P<succ>[0-9.]+)/(?P<spl>[0-9.]+)/(?P<dist>[0-9.]+)/"
    r"(?P<noncov>[0-9.]+)/(?P<gtfb>[0-9.]+)\((?P<n>\d+)\)"
)


def last_n(path: Path, want_n: int) -> tuple[float, float, float] | None:
    got = None
    with path.open("r", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m or int(m.group("n")) != want_n:
                continue
            got = (float(m.group("succ")), float(m.group("spl")), float(m.group("dist")))
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    ap.add_argument("--ep", type=int, default=1000)
    ap.add_argument(
        "--target-succ",
        type=float,
        default=0.689,
        help="wo_dyn 等基线的 Success，用于打印 gap",
    )
    ap.add_argument(
        "--target-spl",
        type=float,
        default=0.619,
        help="wo_dyn 等基线的 SPL",
    )
    args = ap.parse_args()

    rows: list[tuple[str, float, float, float, str]] = []
    for stem in STEMS:
        p = args.log_dir / f"{stem}_{args.ep}ep.log"
        if not p.is_file():
            print(f"[warn] missing {p}", file=sys.stderr)
            continue
        m = last_n(p, args.ep)
        if m is None:
            print(f"[warn] no n={args.ep} metrics in {p}", file=sys.stderr)
            continue
        s, sp, d = m
        rows.append((stem, s, sp, d, str(p)))

    rows.sort(key=lambda x: (-x[1], -x[2]))

    print(f"target wo_dyn  approx: succ={args.target_succ:.3f} spl={args.target_spl:.3f}")
    print("")
    print(f"{'stem':<16} {'succ':>8} {'spl':>8} {'dist':>8} {'Δsucc':>8} {'Δspl':>8}")
    print("-" * 64)
    for stem, s, sp, d, _ in rows:
        print(
            f"{stem:<16} {s:8.3f} {sp:8.3f} {d:8.3f} "
            f"{s - args.target_succ:+8.3f} {sp - args.target_spl:+8.3f}"
        )

    if rows:
        best = rows[0]
        print("")
        print(f"best: {best[0]} succ={best[1]:.3f} spl={best[2]:.3f}")


if __name__ == "__main__":
    main()
