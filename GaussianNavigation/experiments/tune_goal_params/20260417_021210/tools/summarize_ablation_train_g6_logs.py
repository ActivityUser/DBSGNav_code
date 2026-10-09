#!/usr/bin/env python3
"""从 train 消融日志中提取最后一次 Instance ImageGoal Navigation Metrics 行。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PAT = re.compile(
    r"succ\s*/\s*spl\s*/\s*dist.*?:\s*"
    r"(?P<succ>[0-9.]+)/(?P<spl>[0-9.]+)/(?P<dist>[0-9.]+)/"
    r"(?P<noncov>[0-9.]+)/(?P<gtfb>[0-9.]+)\((?P<n>\d+)\)"
)


def last_metrics(
    path: Path, want_n: int
) -> tuple[tuple[float, float, float, int] | None, tuple[float, float, float, int] | None]:
    last_any: tuple[float, float, float, int] | None = None
    last_want: tuple[float, float, float, int] | None = None
    with path.open("r", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            n = int(m.group("n"))
            succ = float(m.group("succ"))
            spl = float(m.group("spl"))
            dist = float(m.group("dist"))
            tup = (succ, spl, dist, n)
            last_any = tup
            if n == want_n:
                last_want = tup
    return last_want, last_any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="日志文件 .log 或目录（默认同 ablation_train_g6 目录）",
    )
    ap.add_argument("--ep", type=int, default=1000, help="期望 episode 数（优先取 n 等于该值的最后一行）")
    args = ap.parse_args()
    default_dir = Path(__file__).resolve().parents[1] / "logs_1000ep_rerun" / "ablation_train_g6"
    if args.paths:
        log_files: list[Path] = []
        for p in args.paths:
            if p.is_dir():
                log_files.extend(sorted(p.glob("*.log")))
            elif p.is_file():
                log_files.append(p)
            else:
                print(f"[warn] skip missing: {p}")
    else:
        log_files = sorted(default_dir.glob("*.log")) if default_dir.is_dir() else []

    if not log_files:
        print(f"[err] no .log files (default dir: {default_dir})")
        raise SystemExit(1)

    rows: list[tuple[str, tuple[float, float, float, int] | None]] = []
    for p in log_files:
        exact, last = last_metrics(p, args.ep)
        rows.append((p.name, exact, last))

    print(f"{'log':<52} {'succ':>8} {'spl':>8} {'dist':>8} {'n':>6}  note")
    for name, exact, last in rows:
        pick = exact if exact is not None else last
        if pick is None:
            print(f"{name:<52} {'(no metrics)':>8}")
            continue
        succ, spl, dist, n = pick
        note = "" if exact is not None else f"(incomplete, last n={n})"
        print(f"{name:<52} {succ:8.5f} {spl:8.5f} {dist:8.5f} {n:6d}  {note}")


if __name__ == "__main__":
    main()
