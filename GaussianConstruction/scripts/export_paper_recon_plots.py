#!/usr/bin/env python3
"""Re-export eval reconstruction grids for paper: vector titles + high-res panels.

Reads composite PNG from eval/plots/, splits the 2x3 grid, rebuilds with Times New Roman
subplot titles (no Time Step suptitle). Titles live in dedicated rows above images.

Usage (CWD = GaussianConstruction):
  python scripts/export_paper_recon_plots.py \\
    --scene 4ok3usBNeis_seed4_clip --plots 0000 0299 \\
    --out ../docs/paperfile/figs
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(GC_ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "docs", "paperfile"))
from paper_matplotlib import BODY_PT, latex_matched_fontsize, setup_matplotlib  # noqa: E402

SAVE_DPI = 150
# Fig.5 两场景并排：每个子图约 0.45\\textwidth
LATEX_FIG_WIDTH_CM = 0.45 * 17.5
TITLE_PAD_IN = 0.04
ROW_GAP_IN = 0.06

ROW1 = (98, 355)
ROW2 = (446, 705)
COLS = [(10, 392), (406, 788), (803, 1185)]


def crop_panels(src_png: str) -> list[np.ndarray]:
    im = np.array(Image.open(src_png).convert("RGB"))
    panels = []
    for row in (ROW1, ROW2):
        for col in COLS:
            x0, x1 = col
            y0, y1 = row
            panels.append(im[y0:y1, x0:x1])
    return panels


def load_metrics(run_dir: str, plot_name: str) -> tuple[float, float]:
    plots_dir = os.path.join(run_dir, "eval", "plots")
    names = sorted(n[:-4] for n in os.listdir(plots_dir) if n.endswith(".png"))
    idx = names.index(plot_name)
    psnr = float(np.loadtxt(os.path.join(run_dir, "eval", "psnr.txt"), ndmin=1)[idx])
    l1 = float(np.loadtxt(os.path.join(run_dir, "eval", "l1.txt"), ndmin=1)[idx])
    return psnr, l1


def target_title_fontsize(
    fig_w_px: int,
    panel_w_px: int,
    titles: list[str],
    *,
    body_pt: float = BODY_PT,
    latex_width_cm: float = LATEX_FIG_WIDTH_CM,
) -> float:
    """Max pt matching LaTeX body size, capped so longest title fits panel width."""
    fig_w_in = fig_w_px / SAVE_DPI
    target = latex_matched_fontsize(fig_w_in, body_pt=body_pt, latex_width_cm=latex_width_cm)
    panel_w_in = panel_w_px / SAVE_DPI
    longest = max(max(len(line) for line in t.split("\n")) for t in titles)
    width_cap = panel_w_in * 72.0 / (longest * 0.50)
    return float(min(target, width_cap))


def title_block_height_in(font_pt: float, n_lines: int) -> float:
    line_h = font_pt / 72.0 * 1.12
    return line_h * n_lines + TITLE_PAD_IN


def export_figure(
    panels: list[np.ndarray],
    psnr: float,
    l1: float,
    out_stem: str,
    *,
    body_pt: float = BODY_PT,
    latex_width_cm: float = LATEX_FIG_WIDTH_CM,
) -> None:
    titles = [
        "Ground Truth RGB",
        "Ground Truth Depth",
        "Rasterized Silhouette",
        f"Rasterized RGB\nPSNR: {psnr:.2f}",
        f"Rasterized Depth\nL1: {l1:.2f}",
        "Diff Depth L1",
    ]
    ncols = 3
    panel_h = max(p.shape[0] for p in panels)
    panel_w = max(p.shape[1] for p in panels)
    fig_w_px = panel_w * ncols

    sub_pt = target_title_fontsize(
        fig_w_px, panel_w, titles, body_pt=body_pt, latex_width_cm=latex_width_cm
    )

    row0_title_h = title_block_height_in(sub_pt, 1)
    row1_title_h = title_block_height_in(sub_pt, 2)
    panel_h_in = panel_h / SAVE_DPI
    panel_w_in = panel_w / SAVE_DPI
    fig_w_in = panel_w_in * ncols
    fig_h_in = (
        row0_title_h
        + panel_h_in
        + ROW_GAP_IN
        + row1_title_h
        + panel_h_in
        + TITLE_PAD_IN
    )

    title_row0_ratio = row0_title_h / panel_h_in
    title_row1_ratio = row1_title_h / panel_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=SAVE_DPI)
    gs = gridspec.GridSpec(
        4,
        ncols,
        figure=fig,
        height_ratios=[title_row0_ratio, 1, title_row1_ratio, 1],
        hspace=0.08,
        wspace=0.04,
        top=0.99,
        bottom=0.01,
        left=0.01,
        right=0.99,
    )

    for idx, (img, title) in enumerate(zip(panels, titles)):
        row_pair = 0 if idx < 3 else 1
        col = idx % 3
        title_row = row_pair * 2
        img_row = title_row + 1

        ax_t = fig.add_subplot(gs[title_row, col])
        ax_t.axis("off")
        ax_t.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=sub_pt,
            linespacing=1.05,
            transform=ax_t.transAxes,
        )

        h, w = img.shape[:2]
        ax_i = fig.add_subplot(gs[img_row, col])
        ax_i.imshow(img, aspect="equal", interpolation="nearest")
        ax_i.set_xlim(-0.5, w - 0.5)
        ax_i.set_ylim(h - 0.5, -0.5)
        ax_i.axis("off")

    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    save_kw = dict(bbox_inches="tight", pad_inches=0.02, dpi=SAVE_DPI)
    fig.savefig(out_stem + ".pdf", format="pdf", **save_kw)
    fig.savefig(out_stem + ".eps", format="eps", **save_kw)
    fig.savefig(out_stem + ".png", **save_kw)
    plt.close(fig)
    print(f"  saved {out_stem}.pdf/.eps/.png  (title={sub_pt:.1f}pt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map-root",
        default=os.path.join(GC_ROOT, "experiments", "habitat_val_final"),
    )
    ap.add_argument("--scene", default="4ok3usBNeis_seed4_clip")
    ap.add_argument("--plots", nargs="+", default=["0000", "0299"])
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "paperfile", "figs"))
    ap.add_argument("--body-pt", type=float, default=BODY_PT)
    ap.add_argument("--latex-width-cm", type=float, default=LATEX_FIG_WIDTH_CM)
    args = ap.parse_args()

    setup_matplotlib()
    plt.rcParams["image.composite_image"] = False

    run_dir = os.path.join(args.map_root, args.scene)
    for plot_name in args.plots:
        src = os.path.join(run_dir, "eval", "plots", f"{plot_name}.png")
        if not os.path.isfile(src):
            raise SystemExit(f"Missing {src}")
        panels = crop_panels(src)
        psnr, l1 = load_metrics(run_dir, plot_name)
        out_stem = os.path.join(args.out, plot_name)
        print(f"{args.scene} {plot_name}  PSNR={psnr:.2f} L1={l1:.2f}")
        export_figure(
            panels,
            psnr,
            l1,
            out_stem,
            body_pt=args.body_pt,
            latex_width_cm=args.latex_width_cm,
        )


if __name__ == "__main__":
    main()
