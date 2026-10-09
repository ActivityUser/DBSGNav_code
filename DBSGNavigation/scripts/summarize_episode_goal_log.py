#!/usr/bin/env python3
"""
从评测 tee 日志中解析 [episode_goal_summary] 行，统计失败与 dist_pred_to_gt 分布。

用法:
  python scripts/summarize_episode_goal_log.py experiments/push_metrics_xxx.log
  python scripts/summarize_episode_goal_log.py experiments/*.log

依赖: 配置 end2end_imagenav.log_episode_goal_summary: True 后重新跑评测生成带摘要的日志。

对 ``python run.py --log_file experiments/foo.log``：``[episode_goal_summary]`` 在子进程内写入
``experiments/foo_episode_goal_summary.log``（与主 log 同目录的侧车文件）。可对该侧车直接本脚本；若传入主
``.log`` 而解析 0 条，本脚本会尝试自动读取 ``<同名>_episode_goal_summary.log``。

如何对照 B/C/D: 看「按 goal_source」表（各分支 succ%、vs 全体 Δsucc、失败里 dist<1m vs >=1m）。
  semantic_gaussian 明显高于 open_vocab*，而失败/分歧多来自 disagree → 疑 C/D；
  open_vocab* succ 低且该分支失败中 dist>=1m 多 → 疑 B/E；
  仅某一类 open_vocab_disagree* 极低且 n 大 → 疑 D。
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

ROW_RE = re.compile(
    r"\[episode_goal_summary\] scene=(?P<scene>\S+) ep=(?P<ep>\d+) goal_source=(?P<gs>\S+) "
    r"dist_pred_to_gt=(?P<dpg>[\d.]+) success=(?P<succ>\d) spl=(?P<spl>[\d.]+) "
    r"dist_to_goal=(?P<dist>[\d.]+) timesteps=(?P<ts>\d+) forced=(?P<forced>\d)"
)


def parse_rows(text: str) -> List[dict]:
    rows = []
    for m in ROW_RE.finditer(text):
        rows.append(
            {
                "scene": m.group("scene"),
                "ep": int(m.group("ep")),
                "goal_source": m.group("gs"),
                "dist_pred_to_gt": float(m.group("dpg")),
                "success": int(m.group("succ")),
                "spl": float(m.group("spl")),
                "dist_to_goal": float(m.group("dist")),
                "timesteps": int(m.group("ts")),
                "forced": int(m.group("forced")),
            }
        )
    return rows


def bucket(d: float) -> str:
    if d < 1.0:
        return "<1m"
    if d < 2.0:
        return "1-2m"
    if d < 5.0:
        return "2-5m"
    return ">=5m"


def _summarize_by_goal_source(rows: List[dict], base_succ_rate: float) -> None:
    """各分支的 success 与失败中 dist 分布，用于对照 B/C/D。"""
    by: dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["goal_source"]].append(r)
    print("\n-- 按 goal_source（Success 与失败中 dist 分布；对照 B/C/D）--")
    order = sorted(by.keys(), key=lambda g: (-len(by[g]), g))
    for gs in order:
        g = by[gs]
        ni = len(g)
        s_ok = sum(1 for r in g if r["success"] == 1)
        rate = 100.0 * s_ok / ni if ni else 0.0
        delta = rate - base_succ_rate
        dmean = sum(r["dist_pred_to_gt"] for r in g) / ni if ni else 0.0
        line = f"  {gs}: n={ni}  succ={s_ok} ({rate:.1f}%)  mean_dist={dmean:.2f}m  vs 全体Δsucc={delta:+.1f}pt"
        print(line)
        fails_g = [r for r in g if r["success"] == 0]
        if not fails_g:
            print("    (本分支无失败样本)")
            continue
        c = Counter(bucket(r["dist_pred_to_gt"]) for r in fails_g)
        lt1 = c.get("<1m", 0)
        ge1 = len(fails_g) - lt1
        print(
            f"    失败{len(fails_g)}: dist<1m {lt1} ({100.0 * lt1 / len(fails_g):.0f}%), "
            f"dist>=1m {ge1} ({100.0 * ge1 / len(fails_g):.0f}%)  "
            f"细分 1-2/2-5/>=5: {c.get('1-2m',0)}/{c.get('2-5m',0)}/{c.get('>=5m',0)}"
        )


def summarize(rows: List[dict]) -> None:
    n = len(rows)
    if n == 0:
        print(
            "未找到 [episode_goal_summary] 行。请确认 yaml 中 log_episode_goal_summary: True，"
            "且使用 run.py --log_file 时同时检查同目录 <主名>_episode_goal_summary.log，"
            "或用 shell: python -u run.py ... 2>&1 | tee full.log 。"
        )
        return
    succ_n = sum(1 for r in rows if r["success"] == 1)
    fail_n = n - succ_n
    base_rate = 100.0 * succ_n / n if n else 0.0
    print("== episode_goal_summary ==")
    print("total_episodes:", n, "| success:", succ_n, "| fail:", fail_n, f"| 全体 Success={base_rate:.1f}%")
    _summarize_by_goal_source(rows, base_rate)
    fails = [r for r in rows if r["success"] == 0]
    if fails:
        print("\n-- 失败样本 dist_pred_to_gt 分桶 --")
        c = Counter(bucket(r["dist_pred_to_gt"]) for r in fails)
        for k in ("<1m", "1-2m", "2-5m", ">=5m"):
            if k in c:
                print(f"  {k}: {c[k]} ({100.0 * c[k] / len(fails):.1f}% of fails)")
        print("\n-- 失败样本 goal_source 计数 --")
        for gs, cnt in Counter(r["goal_source"] for r in fails).most_common():
            print(f"  {gs}: {cnt}")
        print("\n-- 失败样本 forced (无预测目标) --", sum(1 for r in fails if r["forced"] == 1))
    print("\n-- 全体 dist_pred_to_gt 分桶 --")
    c2 = Counter(bucket(r["dist_pred_to_gt"]) for r in rows)
    for k in ("<1m", "1-2m", "2-5m", ">=5m"):
        if k in c2:
            print(f"  {k}: {c2[k]}")


def _text_and_rows_for_path(p: Path) -> tuple[str, int, str]:
    """返回 (读入文本, 解析条数, 实际使用的说明：主文件或侧车）。"""
    text = p.read_text(encoding="utf-8", errors="replace")
    n0 = len(parse_rows(text))
    if n0 > 0 or "_episode_goal_summary" in p.name:
        return text, n0, str(p)
    side = p.parent / f"{p.stem}_episode_goal_summary.log"
    if side.is_file():
        t2 = side.read_text(encoding="utf-8", errors="replace")
        n2 = len(parse_rows(t2))
        if n2 > 0:
            return t2, n2, f"{p} -> {side}"
    return text, n0, str(p)


def main(argv: List[str]) -> None:
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        print("用法: python scripts/summarize_episode_goal_log.py <log1> [log2 ...]")
        sys.exit(1)
    all_rows: List[dict] = []
    for p in paths:
        if not p.is_file():
            print("跳过（非文件）:", p)
            continue
        t, n, src = _text_and_rows_for_path(p)
        print(f"\n### {p} 解析到 {n} 条" + (f"  (来源: {src})" if n else ""))
        all_rows.extend(parse_rows(t))
    summarize(all_rows)


if __name__ == "__main__":
    main(sys.argv)
