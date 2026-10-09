#!/usr/bin/env python3
"""
Failure decomposition from run log.

Reads [episode_goal_summary] lines and prints:
1) Overall success / timeout
2) goal_source breakdown
3) Top failing scenes (supporting "fix by failure cluster")
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"\[episode_goal_summary\]\s+scene=(?P<scene>\S+)\s+ep=(?P<ep>\d+)\s+"
    r"goal_source=(?P<src>\S+)\s+dist_pred_to_gt=(?P<dist>[0-9.]+)\s+"
    r"success=(?P<succ>[01])\s+spl=(?P<spl>[0-9.]+)\s+dist_to_goal=(?P<dtg>[0-9.]+)\s+"
    r"timesteps=(?P<ts>\d+)"
)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="run log path")
    parser.add_argument("--top_scenes", type=int, default=15, help="top failing scenes")
    args = parser.parse_args()

    p = Path(args.log).resolve()
    txt = p.read_text(encoding="utf-8", errors="ignore")

    rows = []
    for m in LINE_RE.finditer(txt):
        d = m.groupdict()
        rows.append(
            {
                "scene": d["scene"],
                "src": d["src"],
                "dist": float(d["dist"]),
                "succ": int(d["succ"]),
                "spl": float(d["spl"]),
                "dtg": float(d["dtg"]),
                "ts": int(d["ts"]),
            }
        )

    if not rows:
        raise SystemExit("No episode_goal_summary lines found.")

    n = len(rows)
    succ = sum(r["succ"] for r in rows)
    timeouts = sum(1 for r in rows if r["ts"] >= 1000)
    print(f"[log] {p}")
    print(f"[episodes] {n}  [succ] {succ}/{n} ({pct(succ/n)})  [timeouts] {timeouts} ({pct(timeouts/n)})")
    print("")

    by_src = defaultdict(list)
    by_scene = defaultdict(list)
    for r in rows:
        by_src[r["src"]].append(r)
        by_scene[r["scene"]].append(r)

    print("== goal_source breakdown ==")
    print(f"{'source':26s} {'n':>6s} {'succ':>8s} {'avg_spl':>10s} {'avg_dist_pred_to_gt':>20s} {'avg_steps':>10s}")
    for src, rs in sorted(by_src.items(), key=lambda kv: len(kv[1]), reverse=True):
        nn = len(rs)
        sr = sum(r["succ"] for r in rs) / nn
        sp = sum(r["spl"] for r in rs) / nn
        ad = sum(r["dist"] for r in rs) / nn
        at = sum(r["ts"] for r in rs) / nn
        print(f"{src:26s} {nn:6d} {sr:8.3f} {sp:10.3f} {ad:20.3f} {at:10.1f}")
    print("")

    print("== top failing scenes ==")
    print(f"{'scene':18s} {'n':>6s} {'succ':>8s} {'avg_spl':>10s} {'avg_dist_pred_to_gt':>20s} {'timeouts':>10s}")
    ranked = []
    for scene, rs in by_scene.items():
        nn = len(rs)
        sr = sum(r["succ"] for r in rs) / nn
        sp = sum(r["spl"] for r in rs) / nn
        ad = sum(r["dist"] for r in rs) / nn
        to = sum(1 for r in rs if r["ts"] >= 1000)
        ranked.append((sr, nn, scene, sp, ad, to))
    ranked.sort(key=lambda x: (x[0], -x[1]))  # low succ first, then many episodes
    for sr, nn, scene, sp, ad, to in ranked[: args.top_scenes]:
        print(f"{scene:18s} {nn:6d} {sr:8.3f} {sp:10.3f} {ad:20.3f} {to:10d}")


if __name__ == "__main__":
    main()

