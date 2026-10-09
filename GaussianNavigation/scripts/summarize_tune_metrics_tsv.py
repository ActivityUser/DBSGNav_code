#!/usr/bin/env python3
"""
从调参/评测目录下的 *.log 解析最后一行「succ / spl / dist / nonGT_cov / GT_fb_rate : ... (N)」
汇总为 TSV（默认 summary_1000ep.tsv，与 tune_goal_params 的 summary.tsv 用途类似）。

示例：
  cd GaussianNavigation
  python3 scripts/summarize_tune_metrics_tsv.py \\
    experiments/tune_goal_params/20260417_021210/logs_1000ep

  python3 scripts/summarize_tune_metrics_tsv.py \\
    experiments/tune_goal_params/20260417_021210/logs_1000ep \\
    --output experiments/tune_goal_params/20260417_021210/summary_1000ep.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRIC_RE = re.compile(
    r"succ / spl / dist / nonGT_cov / GT_fb_rate : "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\(([0-9]+)\),"
)


def parse_last_metrics(text: str) -> Optional[Dict[str, float]]:
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "succ": float(m.group(1)),
        "spl": float(m.group(2)),
        "dist": float(m.group(3)),
        "nonGT_cov": float(m.group(4)),
        "GT_fb_rate": float(m.group(5)),
        "count": int(m.group(6)),
    }


def collect_rows(log_dir: Path) -> List[Tuple[str, Optional[Dict], bool, Path, Path]]:
    """返回 (trial_name, metrics_or_none, completed, log_path, cfg_path_guess)。"""
    rows: List[Tuple[str, Optional[Dict], bool, Path, Path]] = []
    for log_path in sorted(log_dir.glob("*.log")):
        trial = log_path.stem
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        metrics = parse_last_metrics(text)
        completed = "'all episode over'" in text
        cfg_guess = log_path.parent.parent / "configs" / f"{trial}.yaml"
        rows.append((trial, metrics, completed, log_path.resolve(), cfg_guess.resolve()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 logs 目录解析 succ/spl 等指标行，写出 summary TSV。"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="含 *.log 的目录，例如 experiments/tune_goal_params/<ts>/logs_1000ep",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出 TSV 路径；默认 <log_dir 的父目录>/summary_1000ep.tsv",
    )
    args = parser.parse_args()

    nav_root = Path(__file__).resolve().parent.parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = nav_root / log_dir
    if not log_dir.is_dir():
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = nav_root / out_path
    else:
        out_path = log_dir.parent / "summary_1000ep.tsv"

    rows = collect_rows(log_dir)
    if not rows:
        print(f"No *.log under {log_dir}", file=sys.stderr)
        return 1

    header = (
        "trial\tsucc\tspl\tdist\tnonGT_cov\tGT_fb_rate\tcount\tcompleted\t"
        "cfg_path\tlog_path\n"
    )
    lines = [header]
    for trial, m, completed, log_p, cfg_p in rows:
        if m is None:
            lines.append(
                f"{trial}\t\t\t\t\t\t\t{int(completed)}\t{cfg_p}\t{log_p}\n"
            )
        else:
            lines.append(
                f"{trial}\t{m['succ']}\t{m['spl']}\t{m['dist']}\t"
                f"{m['nonGT_cov']}\t{m['GT_fb_rate']}\t{m['count']}\t"
                f"{int(completed)}\t{cfg_p}\t{log_p}\n"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
