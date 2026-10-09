#!/usr/bin/env python3
"""创新点5（use_open_vocab_benchmark）价值实验汇总。"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PAT = re.compile(
    r"succ\s*/\s*spl\s*/\s*dist.*?:\s*"
    r"(?P<succ>[0-9.]+)/(?P<spl>[0-9.]+)/(?P<dist>[0-9.]+)/"
    r"(?P<noncov>[0-9.]+)/(?P<gtfb>[0-9.]+)\((?P<n>\d+)\)"
)

NAV = Path(__file__).resolve().parents[4]
DEFAULT_LOG_DIR = NAV / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innov5_benchmark_value"


def last_n_metrics(path: Path, want_n: int) -> tuple[float, float, float, int] | None:
    got = None
    with path.open("r", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            n = int(m.group("n"))
            if n != want_n:
                continue
            got = (float(m.group("succ")), float(m.group("spl")), float(m.group("dist")), n)
    return got


def pct_delta(new: float, base: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    return (new - base) / base * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--ep", type=int, default=1000)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    off_p = args.log_dir / f"bench_off_{args.ep}ep.log"
    on_p = args.log_dir / f"bench_on_{args.ep}ep.log"

    if not off_p.is_file() or not on_p.is_file():
        print("[err] 缺少日志文件：", file=sys.stderr)
        print(f"  - {off_p}", file=sys.stderr)
        print(f"  - {on_p}", file=sys.stderr)
        sys.exit(1)

    off = last_n_metrics(off_p, args.ep)
    on = last_n_metrics(on_p, args.ep)
    if off is None or on is None:
        print("[err] 未在日志中找到 n=ep 的最终指标行", file=sys.stderr)
        sys.exit(1)

    off_s, off_spl, off_d, _ = off
    on_s, on_spl, on_d, _ = on

    ds = on_s - off_s
    dspl = on_spl - off_spl
    dd = on_d - off_d

    lines = []
    lines.append(f"# 创新点5价值对比（ep={args.ep}）")
    lines.append("")
    lines.append("同一模型其余开关固定，仅切换 `use_open_vocab_benchmark`。")
    lines.append("")
    lines.append("| config | use_open_vocab_benchmark | Success | SPL | dist |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(f"| bench_off | false | {off_s:.5f} | {off_spl:.5f} | {off_d:.5f} |")
    lines.append(f"| bench_on | true | {on_s:.5f} | {on_spl:.5f} | {on_d:.5f} |")
    lines.append("")
    lines.append("## 差值（bench_on - bench_off）")
    lines.append("")
    lines.append(f"- Success: `{ds:+.5f}` ({pct_delta(on_s, off_s):+.2f}%)")
    lines.append(f"- SPL: `{dspl:+.5f}` ({pct_delta(on_spl, off_spl):+.2f}%)")
    lines.append(f"- dist: `{dd:+.5f}` ({pct_delta(on_d, off_d):+.2f}%)")
    lines.append("")
    lines.append("## 日志")
    lines.append(f"- `bench_off`: `{off_p}`")
    lines.append(f"- `bench_on`: `{on_p}`")
    lines.append("")
    lines.append(f"_generated_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

    content = "\n".join(lines) + "\n"
    out_path = args.output or (args.log_dir / f"summary_benchmark_value_{args.ep}ep.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(content)
    print(f"[ok] wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
