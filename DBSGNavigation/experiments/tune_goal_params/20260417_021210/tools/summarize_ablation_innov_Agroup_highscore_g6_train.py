#!/usr/bin/env python3
"""汇总 train 上「以 Agroup_highscore_g6 为超参母版」的 innov 消融日志；full 行固定读 G6 官方 1000ep 日志；
full_dynamic_benchmark 行读 log-dir 下同名 *_ep.log（需单独跑 full_dynamic_benchmark.yaml）。

默认写入 --log-dir/summary_{ep}ep_table.md：含论文 Table II 风格的「GaussNav++ | Success↑ | SPL↑」表（三位小数）及 TSV / LaTeX 片段。
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
_DEFAULT_LOG_DIR = _NAV / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innov_Agroup_highscore_g6_train"
_DEFAULT_FULL_LOG = (
    _NAV
    / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/08_ext_gm40_k8_train_tuned_Agroup_highscore_g6_1000ep.log"
)

# 与 configs/ablation_innov_Agroup_highscore_g6_train/*.yaml 注释一致；论文式命名用 GaussNav++
PAPER_ORDER = [
    "full",
    "full_dynamic_benchmark",
    "baseline",
    "innov2",
    "innov3",
    "innov4",
    "innov5",
    "innov6",
]
PAPER_LABELS: dict[str, str] = {
    "full": "GaussNav++",
    "full_dynamic_benchmark": "GaussNav++ (dyn. sem. + OV benchmark on, rest G6)",
    "baseline": "GaussNav++ w.o. Semantic Gaussian, OV",
    "innov2": "GaussNav++ w.o. Open-Vocab Retrieval",
    "innov3": "GaussNav++ w.o. Semantic Gaussian (OV, geom.-only)",
    "innov4": "GaussNav++ w.o. Semantic Gaussian (OV, sem.-geom.)",
    "innov5": "GaussNav++ w. Open-vocab benchmark",
    "innov6": "GaussNav++ w. Dynamic semantic map",
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
    full_log: Path,
) -> list[tuple[str, str, str, tuple[float, float, float, int] | None]]:
    """(preset_key, paper_label, log_path, metrics_or_none)"""
    out: list[tuple[str, str, str, tuple[float, float, float, int] | None]] = []
    for key in PAPER_ORDER:
        label = PAPER_LABELS[key]
        if key == "full":
            p = full_log
        else:
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


def _paper_ascii(rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]]) -> str:
    w = 52
    out = [
        "Ablations".ljust(w) + "  Success ↑   SPL ↑",
        "-" * (w + 22),
    ]
    for _k, lab, _path, m in rows:
        if m is None:
            out.append(lab[: w - 1].ljust(w) + "  " + "—".rjust(9) + "   " + "—".rjust(7))
            continue
        s, sp, _d, _n = m
        short = lab if len(lab) <= w else lab[: w - 3] + "..."
        out.append(short.ljust(w) + f"  {s:9.3f}   {sp:7.3f}")
    return "\n".join(out) + "\n"


def _paper_latex(
    rows: list[tuple[str, str, str, tuple[float, float, float, int] | None]],
    model: str,
    ep: int,
) -> str:
    """需 \\usepackage{booktabs}。与论文 Table II 同列：Ablations / Success / SPL。"""
    lines = [
        r"% " + model + r" ablation (train); Success/SPL three decimals",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Ablation study of {model} (train, $n={ep}$).}}",
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
    ap.add_argument("--full-log", type=Path, default=_DEFAULT_FULL_LOG, help="full 行固定使用的 G6 日志")
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
    args = ap.parse_args()

    rows = _collect_rows(args.ep, args.log_dir, args.full_log)
    full_row = next((m for k, _, _, m in rows if k == "full"), None)
    if full_row is None:
        print("[err] full 行无法解析，请检查 --full-log 路径", file=sys.stderr)
        sys.exit(1)

    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    caption = (
        f"对齐论文 **Table II** 版式（**{args.model_name}** | Ablations | Success↑ | SPL↑，三位小数；"
        f"train split，每行 $n={args.ep}$）。"
        " 行名对应 `ablation_innov_Agroup_highscore_g6_train` 各 yaml 的消融语义。"
    )

    paper_md = _paper_markdown(rows)
    paper_ascii = _paper_ascii(rows)
    latex_block = "```latex\n" + _paper_latex(rows, args.model_name, args.ep).rstrip("\n") + "\n```\n"
    detail_md = _markdown_detail(rows)
    tsv_block = "```tsv\n" + _tsv(rows).rstrip("\n") + "\n```\n"

    full_doc = (
        f"<!-- {args.model_name} train innov ablation | ep={args.ep} | {iso} -->\n"
        f"# {args.model_name} 消融汇总（论文 Table II 风格）\n\n"
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
