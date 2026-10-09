#!/usr/bin/env python3
"""汇总「标准 IIN 协议」train 消融：log-dir 下 {preset}_{ep}ep.log。

预设列表与 configs/ablation_fullsys_std_Agroup_g6_train 一致（含 wo_sg_fusion_g6）。
默认写入 --log-dir/summary_{ep}ep_table.md。
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_NAV = Path(__file__).resolve().parents[4]
_DEFAULT_LOG_DIR = (
    _NAV
    / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_train"
)

PAPER_ORDER = [
    "full",
    "wo_dyn",
    "wo_bench",
    "wo_ov",
    "wo_sg_semgeom",
    "wo_sg_fusion_g6",
    "wo_sg_geom",
]
PAPER_LABELS: dict[str, str] = {
    "full": "Full (G6, SG+OV+dyn., std IIN)",
    "wo_dyn": "w.o. dynamic semantic map",
    "wo_bench": "w.o. open-vocab benchmark (std 下近似退化)",
    "wo_ov": "w.o. OV retrieval (SG only)",
    "wo_sg_semgeom": "w.o. SG (OV sem.-only, α=1, β=0)",
    "wo_sg_fusion_g6": "w.o. SG (OV fusion G6, α=0.8, β=0.6)",
    "wo_sg_geom": "Baseline (OV geom.-only; α=0, β=1)",
}

PAT = re.compile(
    r"succ\s*/\s*spl\s*/\s*dist.*?:\s*"
    r"(?P<succ>[0-9.]+)/(?P<spl>[0-9.]+)/(?P<dist>[0-9.]+)/"
    r"(?P<noncov>[0-9.]+)/(?P<gtfb>[0-9.]+)\((?P<n>\d+)\)"
)


def last_n_metrics(path: Path, want_n: int) -> tuple[float, float, float, int] | None:
    last_want: tuple[float, float, float, int] | None = None
    with path.open("r", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            n = int(m.group("n"))
            if n != want_n:
                continue
            last_want = (
                float(m.group("succ")),
                float(m.group("spl")),
                float(m.group("dist")),
                n,
            )
    return last_want


def _collect_rows(
    ep: int,
    log_dir: Path,
) -> list[tuple[str, str, str, tuple[float, float, float, int] | None]]:
    """(preset_key, paper_label, log_path, metrics_or_none)"""
    out: list[tuple[str, str, str, tuple[float, float, float, int] | None]] = []
    for key in PAPER_ORDER:
        label = PAPER_LABELS[key]
        p = log_dir / f"{key}_{ep}ep.log"
        path_s = str(p)
        if not p.is_file():
            out.append((key, label, path_s, None))
            continue
        out.append((key, label, path_s, last_n_metrics(p, ep)))
    return out


def _paper_markdown(rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]]) -> str:
    lines = [
        "| Ablations | Success ↑ | SPL ↑ |",
        "| --- | ---: | ---: |",
    ]
    for _k, lab, _path, m in rows:
        if m is None:
            lines.append(f"| {lab} | — | — |")
            continue
        s, sp, _d, _n = m
        lines.append(f"| {lab} | {s:.3f} | {sp:.3f} |")
    return "\n".join(lines) + "\n"


def _paper_latex(
    rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]],
    model: str,
    ep: int,
) -> str:
    lines = [
        r"% " + model + r" std-IIN ablation (train); Success/SPL three decimals",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Standard IIN ablation of {model} (train, $n={ep}$).}}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Ablations} & \textbf{Success} $\uparrow$ & \textbf{SPL} $\uparrow$ \\",
        r"\midrule",
    ]
    for _k, lab, _path, m in rows:
        lab_tex = lab.replace("&", r"\&").replace("_", r"\_")
        if m is None:
            lines.append(rf"{lab_tex} & --- & --- \\")
            continue
        s, sp, _d, _n = m
        lines.append(rf"{lab_tex} & {s:.3f} & {sp:.3f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _markdown_detail(
    rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]],
) -> str:
    lines = [
        "| preset | succ | spl | dist | n | log |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, _lab, path, m in rows:
        if m is None:
            lines.append(f"| `{key}` | — | — | — | — | `{path}` |")
            continue
        s, sp, d, n = m
        lines.append(f"| `{key}` | {s:.5f} | {sp:.5f} | {d:.5f} | {n} | `{path}` |")
    return "\n".join(lines) + "\n"


def _tsv(rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    w.writerow(["preset", "paper_label", "succ", "spl", "dist", "n", "log_path"])
    for key, lab, path, m in rows:
        if m is None:
            w.writerow([key, lab, "", "", "", "", path])
            continue
        s, sp, d, n = m
        w.writerow([key, lab, f"{s:.5f}", f"{sp:.5f}", f"{d:.5f}", n, path])
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    ap.add_argument("--ep", type=int, default=1000)
    ap.add_argument(
        "--model-name",
        default="GaussNav++",
        help="写入 LaTeX caption / 表题用的模型名",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="写入汇总文件（默认: <log-dir>/summary_{ep}ep_table.md）",
    )
    ap.add_argument("--no-save", action="store_true", help="只打印终端，不写文件")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许缺少部分 log（表中为 —），不强制要求 full 已跑完",
    )
    args = ap.parse_args()

    rows = _collect_rows(args.ep, args.log_dir)
    full_row = next((m for k, _, _, m in rows if k == "full"), None)
    if full_row is None and not args.allow_partial:
        print("[err] full 行无法解析：请确认已跑 full 且存在", file=sys.stderr)
        print(f"      {args.log_dir / f'full_{args.ep}ep.log'}", file=sys.stderr)
        print("[hint] 尚未跑完时可加 --allow-partial", file=sys.stderr)
        sys.exit(1)

    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    caption = (
        f"**标准 IIN 协议消融**（**{args.model_name}** | train | $n={args.ep}$ | "
        "`use_open_vocab_benchmark: false`）："
        "锚点 `full` = G6 + SG + OV + 动态语义；"
        "`wo_*` 相对 full 关模块；"
        "主表 **Baseline** = `wo_sg_geom`（OV 纯几何，α=0, β=1）。"
    )

    paper_md = _paper_markdown(rows)
    # 修复 _paper_ascii 里错误的 em dash 拼接
    paper_ascii_lines = []
    w = 56
    paper_ascii_lines.append("Ablations".ljust(w) + "  Success ↑   SPL ↑")
    paper_ascii_lines.append("-" * (w + 22))
    for _k, lab, _path, m in rows:
        if m is None:
            paper_ascii_lines.append(
                (lab if len(lab) <= w else lab[: w - 3] + "...").ljust(w)
                + "        —        —"
            )
            continue
        s, sp, _d, _n = m
        short = lab if len(lab) <= w else lab[: w - 3] + "..."
        paper_ascii_lines.append(short.ljust(w) + f"  {s:9.3f}   {sp:7.3f}")
    paper_ascii = "\n".join(paper_ascii_lines) + "\n"

    latex_block = "```latex\n" + _paper_latex(rows, args.model_name, args.ep).rstrip("\n") + "\n```\n"
    detail_md = _markdown_detail(rows)
    tsv_block = "```tsv\n" + _tsv(rows).rstrip("\n") + "\n```\n"

    full_doc = (
        f"<!-- {args.model_name} std-IIN ablation | ep={args.ep} | {iso} -->\n"
        f"# {args.model_name} 标准 IIN 协议消融汇总\n\n"
        f"{caption}\n\n"
        f"## TABLE（Success / SPL）\n\n"
        f"{paper_md}\n"
        f"## LaTeX（booktabs，按需粘贴）\n\n"
        f"{latex_block}\n"
        f"## 终端宽表（与上表同序）\n\n"
        f"```\n{paper_ascii.rstrip()}\n```\n\n"
        f"## 原始 preset / dist / 路径\n\n"
        f"{detail_md}\n"
        f"## TSV\n\n"
        f"{tsv_block}"
    )

    print(paper_ascii, end="")

    if not args.no_save:
        out_path = args.output or (args.log_dir / f"summary_{args.ep}ep_table.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_doc, encoding="utf-8")
        print(f"\n[ok] 已写入: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
