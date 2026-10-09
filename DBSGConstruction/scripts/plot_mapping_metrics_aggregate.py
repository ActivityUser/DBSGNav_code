#!/usr/bin/env python3
"""Aggregate per-scene mapping eval curves and plot like eval/metrics.png.

Usage (CWD = DBSGConstruction):
  python scripts/plot_mapping_metrics_aggregate.py
  python scripts/plot_mapping_metrics_aggregate.py --map-root experiments/habitat_val_final \\
      --out docs/figures/habitat_val_final_metrics_aggregate
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.normpath(os.path.join(GC_ROOT, "..", "docs"))
REPO = os.path.normpath(os.path.join(GC_ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))
from paper_matplotlib import (  # noqa: E402
    BODY_PT,
    LATEX_TEXTWIDTH_CM,
    hide_top_right_spines,
    latex_matched_fontsize,
    setup_matplotlib,
)

# 单栏 figure* 通常 \\includegraphics[width=\\textwidth]
LATEX_FIG_WIDTH_CM = LATEX_TEXTWIDTH_CM


def output_stem(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    if ext:
        return root
    return out_path


def _load_series(path: str) -> np.ndarray:
    arr = np.loadtxt(path, ndmin=1)
    return np.atleast_1d(arr).astype(float)


def collect_series(map_root: str) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    scenes: List[str] = []
    psnr_series: List[np.ndarray] = []
    l1_series: List[np.ndarray] = []
    for run_dir in sorted(glob(os.path.join(map_root, "*_seed4_clip"))):
        eval_dir = os.path.join(run_dir, "eval")
        psnr_path = os.path.join(eval_dir, "psnr.txt")
        l1_path = os.path.join(eval_dir, "l1.txt")
        if not (os.path.isfile(psnr_path) and os.path.isfile(l1_path)):
            continue
        psnr = _load_series(psnr_path)
        l1 = _load_series(l1_path)
        if len(psnr) != len(l1):
            n = min(len(psnr), len(l1))
            psnr, l1 = psnr[:n], l1[:n]
        scenes.append(os.path.basename(run_dir).replace("_seed4_clip", ""))
        psnr_series.append(psnr)
        l1_series.append(l1)
    return scenes, psnr_series, l1_series


def aggregate_at_steps(
    series_list: List[np.ndarray], *, use_median: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(s) for s in series_list)
    center = np.full(max_len, np.nan)
    lower = np.full(max_len, np.nan)
    upper = np.full(max_len, np.nan)
    for step in range(max_len):
        vals = np.array([s[step] for s in series_list if step < len(s)])
        if len(vals) == 0:
            continue
        if use_median:
            center[step] = np.median(vals)
            lower[step] = np.percentile(vals, 25)
            upper[step] = np.percentile(vals, 75)
        else:
            center[step] = vals.mean()
            lower[step] = center[step] - vals.std(ddof=0)
            upper[step] = center[step] + vals.std(ddof=0)
    steps = np.arange(max_len)
    return steps, center, lower, upper


def plot_aggregate(
    scenes: List[str],
    psnr_series: List[np.ndarray],
    l1_series: List[np.ndarray],
    out_path: str,
    show_std: bool = True,
    *,
    body_pt: float = BODY_PT,
    latex_width_cm: float = LATEX_FIG_WIDTH_CM,
) -> None:
    all_psnr = np.concatenate(psnr_series)
    all_l1_cm = np.concatenate(l1_series) * 100.0
    avg_psnr = float(all_psnr.mean())
    avg_l1_cm = float(all_l1_cm.mean())
    med_psnr = float(np.median([s.mean() for s in psnr_series]))
    med_l1_cm = float(np.median([s.mean() for s in l1_series]) * 100.0)

    psnr_steps, psnr_mean, psnr_lo, psnr_hi = aggregate_at_steps(psnr_series, use_median=False)
    l1_steps, l1_med, l1_lo, l1_hi = aggregate_at_steps(l1_series, use_median=True)
    l1_med_cm = l1_med * 100.0
    l1_lo_cm = l1_lo * 100.0
    l1_hi_cm = l1_hi * 100.0

    fig_w, fig_h = 12.0, 4.0
    font_pt = latex_matched_fontsize(fig_w, body_pt=body_pt, latex_width_cm=latex_width_cm)

    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    C_MEAN   = "#FFA23E"
    C_MEDIAN = "#FED961"
    C_BAND   = "#A9D18E"

    axs[0].plot(psnr_steps, psnr_mean, color=C_MEAN, alpha=0.80, linewidth=2, label="mean")
    if show_std:
        axs[0].fill_between(psnr_steps, psnr_lo, psnr_hi, color=C_BAND, alpha=0.40, label="±1 std")
    axs[0].set_title("RGB PSNR", fontsize=font_pt)
    axs[0].set_xlabel("Eval Step", fontsize=font_pt)
    axs[0].set_ylabel("PSNR (dB)", fontsize=font_pt)
    axs[0].tick_params(axis="both", labelsize=font_pt)
    axs[0].grid(False)
    hide_top_right_spines(axs[0])
    axs[0].legend(loc="upper right", fontsize=font_pt, framealpha=0, facecolor="none", edgecolor="none")

    axs[1].plot(l1_steps, l1_med_cm, color=C_MEDIAN, alpha=0.80, linewidth=2, label="median")
    if show_std:
        axs[1].fill_between(l1_steps, l1_lo_cm, l1_hi_cm, color=C_BAND, alpha=0.40, label="IQR")
    axs[1].set_title("Depth L1", fontsize=font_pt)
    axs[1].set_xlabel("Eval Step", fontsize=font_pt)
    axs[1].set_ylabel("L1 (cm)", fontsize=font_pt)
    axs[1].tick_params(axis="both", labelsize=font_pt)
    axs[1].grid(False)
    hide_top_right_spines(axs[1])
    axs[1].legend(loc="upper right", fontsize=font_pt, framealpha=0, facecolor="none", edgecolor="none")
    l1_ylim = float(np.nanpercentile(l1_hi_cm, 95)) * 1.15
    if l1_ylim > 0:
        axs[1].set_ylim(0, max(l1_ylim, 20))

    fig.tight_layout()

    x_max = float(max(psnr_steps[-1], l1_steps[-1]))
    for ax in axs:
        ax.set_autoscalex_on(False)
        ax.set_xlim(0, x_max)
        ax.set_xmargin(0)
    axs[1].set_autoscaley_on(False)
    _, l1_yhi = axs[1].get_ylim()
    axs[1].set_ylim(0, l1_yhi)
    axs[1].set_ymargin(0)

    # Left panel: y-axis stops at 30; keep autoscale ymin (green band unchanged).
    y_cap = 30.0
    ylo0, _ = axs[0].get_ylim()
    axs[0].set_autoscaley_on(False)
    axs[0].set_ylim(ylo0, y_cap)
    axs[0].set_ymargin(0)
    ylo1, yhi1 = axs[1].get_ylim()
    axs[1].set_ylim(ylo1, min(yhi1, y_cap))
    axs[1].set_ymargin(0)

    stem = output_stem(out_path)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".pdf", bbox_inches="tight", format="pdf")
    fig.savefig(stem + ".eps", bbox_inches="tight", format="eps")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {stem}.pdf")
    print(f"Wrote {stem}.eps")
    print(f"Wrote {stem}.png")
    print(f"  scenes={len(scenes)}  mean_psnr={avg_psnr:.2f}  mean_l1_cm={avg_l1_cm:.2f}")
    print(f"  median_psnr_per_scene={med_psnr:.2f}  median_l1_cm_per_scene={med_l1_cm:.2f}")
    print(f"  font={font_pt:.1f}pt (→ ~{body_pt:.0f}pt @ {latex_width_cm:.1f}cm LaTeX width)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate mapping eval metrics.")
    parser.add_argument(
        "--map-root",
        default=os.path.join(GC_ROOT, "experiments", "habitat_val_final"),
        help="Root with *_seed4_clip runs",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(DOCS, "figures", "habitat_val_final_metrics_aggregate"),
        help="Output path stem or file path (.pdf/.png); always writes .pdf, .eps, .png",
    )
    parser.add_argument("--no-std", action="store_true", help="Hide ±1 std band")
    parser.add_argument("--body-pt", type=float, default=BODY_PT)
    parser.add_argument("--latex-width-cm", type=float, default=LATEX_FIG_WIDTH_CM)
    args = parser.parse_args()

    setup_matplotlib()

    map_root = os.path.normpath(
        args.map_root if os.path.isabs(args.map_root) else os.path.join(GC_ROOT, args.map_root)
    )
    out_path = os.path.normpath(
        args.out if os.path.isabs(args.out) else os.path.join(GC_ROOT, args.out)
    )

    scenes, psnr_series, l1_series = collect_series(map_root)
    if not scenes:
        raise SystemExit(f"No eval metrics found under {map_root}")
    plot_aggregate(
        scenes,
        psnr_series,
        l1_series,
        out_path,
        show_std=not args.no_std,
        body_pt=args.body_pt,
        latex_width_cm=args.latex_width_cm,
    )


if __name__ == "__main__":
    main()
