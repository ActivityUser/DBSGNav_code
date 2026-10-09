"""Export standardized Fig.4 near-goal symbolic subflow thumbnails."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "figs", "method_thumbs", "near_goal_subflows")

BG = "#f8f7f2"
PANEL = "#ffffff"
BOX = "#f7f3e8"
EDGE = "#4b5563"
TXT = "#111827"
MUTED = "#6b7280"
GOAL = "#ef4444"
AGENT = "#22c55e"
SEM = "#f59e0b"
PATH = "#2563eb"
DEC = "#14b8a6"
NEG = "#e5e7eb"


def ensure_out() -> None:
    os.makedirs(OUT, exist_ok=True)


def _setup_ax(title: str, sid: str):
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0.015, 0.02), 0.97, 0.96, facecolor=PANEL, edgecolor="#d1d5db", lw=1.0))
    ax.text(0.03, 0.945, sid, ha="left", va="top", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(0.50, 0.945, title, ha="center", va="top", fontsize=13, color=TXT, fontweight="bold")
    return fig, ax


def _box(ax, xy, wh, text, fs=8.3, fc=BOX, lw=1.0):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.010",
        linewidth=lw,
        edgecolor=EDGE,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TXT)
    return rect


def _arrow(ax, start, end, text=None, fs=7.2):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.25, shrinkA=2, shrinkB=2))
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.03, text, ha="center", va="center", fontsize=fs, color=MUTED)


def _mini_legend(ax, items):
    x = 0.03
    y = 0.07
    for color, label in items:
        ax.add_patch(patches.Circle((x, y), 0.010, facecolor=color, edgecolor=EDGE, lw=0.35))
        ax.text(x + 0.018, y, label, ha="left", va="center", fontsize=7.0, color=MUTED)
        x += 0.16


def _grid(ax, x, y, w, h, show_corr=False, expanded=False, robot=False):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#eef2f7", edgecolor=EDGE, lw=0.9))
    for i in range(1, 7):
        ax.plot([x + w * i / 7, x + w * i / 7], [y, y + h], color="#ffffff", lw=0.8)
        ax.plot([x, x + w], [y + h * i / 7, y + h * i / 7], color="#ffffff", lw=0.8)
    ax.add_patch(patches.Rectangle((x + 0.14 * w, y + 0.58 * h), 0.20 * w, 0.20 * h, facecolor="#334155", edgecolor="none"))
    ax.add_patch(patches.Rectangle((x + 0.60 * w, y + 0.18 * h), 0.22 * w, 0.20 * h, facecolor="#334155", edgecolor="none"))
    gx, gy = x + 0.70 * w, y + 0.66 * h
    if show_corr:
        for r, a in [(0.29, 0.20), (0.21, 0.32), (0.13, 0.52)]:
            ax.add_patch(patches.Circle((gx, gy), r * min(w, h), facecolor=SEM, alpha=a, edgecolor="none"))
    if expanded:
        ax.add_patch(patches.Ellipse((gx, gy), 0.44 * w, 0.30 * h, facecolor=SEM, alpha=0.45, edgecolor="none"))
    ax.add_patch(patches.Circle((gx, gy), 0.036 * min(w, h), facecolor=GOAL, edgecolor="white", lw=0.75))
    if robot:
        ax.add_patch(patches.Circle((x + 0.27 * w, y + 0.28 * h), 0.050 * min(w, h), facecolor=AGENT, edgecolor="white", lw=0.75))


def _save(fig, name: str) -> str:
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def step1():
    fig, ax = _setup_ax("Predicted goal position", "(1)")
    _box(ax, (0.07, 0.52), (0.22, 0.18), "SG/OV winner")
    _box(ax, (0.36, 0.52), (0.23, 0.18), "pred_goal_pos (3D)")
    _grid(ax, 0.68, 0.36, 0.24, 0.42)
    _arrow(ax, (0.29, 0.61), (0.36, 0.61))
    _arrow(ax, (0.59, 0.61), (0.68, 0.58), "project to map")
    ax.text(0.80, 0.30, "grid anchor p*", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(GOAL, "goal anchor p*"), (AGENT, "agent")])
    return _save(fig, "step_01_predicted_goal_position.png")


def step2():
    fig, ax = _setup_ax("Near-goal gate", "(2)")
    dia = patches.RegularPolygon((0.50, 0.58), 4, radius=0.16, orientation=np.pi / 4, facecolor=BOX, edgecolor=EDGE, lw=1.0)
    ax.add_patch(dia)
    ax.text(0.50, 0.58, "d_goal <= tau ?", ha="center", va="center", fontsize=8.6, color=TXT)
    _box(ax, (0.08, 0.48), (0.23, 0.17), "No: single-point FMM", fs=7.6, fc=NEG)
    _box(ax, (0.69, 0.48), (0.23, 0.17), "Yes: semantic optimize", fs=7.6, fc="#dcfce7")
    _arrow(ax, (0.34, 0.58), (0.34, 0.58))
    _arrow(ax, (0.66, 0.58), (0.69, 0.58), "Yes")
    ax.plot([0.12, 0.88], [0.31, 0.31], color="#9ca3af", lw=2.0)
    ax.scatter([0.51], [0.31], s=45, color=SEM, zorder=3)
    ax.text(0.50, 0.22, "Condition: semantic layer ready and timestep >= T_min", ha="center", fontsize=7.7, color=MUTED)
    _mini_legend(ax, [(DEC, "decision"), (SEM, "distance marker")])
    return _save(fig, "step_02_near_goal_gate.png")


def step3():
    fig, ax = _setup_ax("Current RGB", "(3)")
    _box(ax, (0.08, 0.52), (0.24, 0.17), "Agent execution")
    _box(ax, (0.39, 0.52), (0.20, 0.17), "Habitat obs")
    ax.add_patch(patches.Rectangle((0.67, 0.40), 0.24, 0.34, facecolor="#a16207", edgecolor=EDGE, lw=0.9))
    ax.add_patch(patches.Rectangle((0.70, 0.47), 0.16, 0.08, facecolor="#7f1d1d"))
    ax.add_patch(patches.Rectangle((0.70, 0.58), 0.16, 0.07, facecolor="#fca5a5"))
    _arrow(ax, (0.32, 0.60), (0.39, 0.60))
    _arrow(ax, (0.59, 0.60), (0.67, 0.57), "obs['rgb']")
    ax.text(0.79, 0.31, "current RGB frame", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [("#a16207", "RGB frame"), (EDGE, "data flow")])
    return _save(fig, "step_03_current_rgb.png")


def step4():
    fig, ax = _setup_ax("CLIP encode", "(4)")
    ax.add_patch(patches.Rectangle((0.08, 0.41), 0.20, 0.30, facecolor="#a16207", edgecolor=EDGE, lw=0.9))
    _box(ax, (0.38, 0.50), (0.20, 0.16), "CLIP ViT-B/32")
    vals = np.array([0.13, 0.21, 0.33, 0.52, 0.37, 0.68, 0.42, 0.76, 0.58, 0.83], dtype=np.float32)
    for i, v in enumerate(vals):
        ax.add_patch(patches.Rectangle((0.69 + i * 0.018, 0.42), 0.012, 0.28 * float(v), facecolor=PATH, alpha=0.80, edgecolor="none"))
    _arrow(ax, (0.28, 0.56), (0.38, 0.58), "RGB")
    _arrow(ax, (0.58, 0.58), (0.69, 0.56), "L2 norm -> v_t")
    ax.text(0.79, 0.33, "feature vector v_t (512-d)", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(PATH, "feature magnitude"), (EDGE, "encoding flow")])
    return _save(fig, "step_04_clip_encode.png")


def step5():
    fig, ax = _setup_ax("EMA semantic layer", "(5)")
    _grid(ax, 0.08, 0.33, 0.24, 0.42, show_corr=True)
    _box(ax, (0.38, 0.49), (0.29, 0.17), "S_t = (1-alpha) S + alpha v_t", fs=8.0)
    _grid(ax, 0.73, 0.33, 0.20, 0.42, show_corr=True)
    _arrow(ax, (0.32, 0.56), (0.38, 0.57), "old S")
    _arrow(ax, (0.67, 0.57), (0.73, 0.57), "write back")
    ax.text(0.50, 0.22, "2D occupancy-aligned semantic_clip_map", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(SEM, "semantic intensity"), (GOAL, "goal-related cell")])
    return _save(fig, "step_05_ema_semantic_layer.png")


def step6():
    fig, ax = _setup_ax("Goal correlation map", "(6)")
    _box(ax, (0.07, 0.54), (0.18, 0.15), "Goal image")
    _box(ax, (0.07, 0.33), (0.18, 0.15), "CLIP -> v_g")
    _grid(ax, 0.38, 0.31, 0.22, 0.41, show_corr=True)
    _grid(ax, 0.72, 0.31, 0.22, 0.41, show_corr=True)
    _arrow(ax, (0.16, 0.54), (0.16, 0.48))
    _arrow(ax, (0.25, 0.41), (0.38, 0.51), "cos(S, v_g)")
    _arrow(ax, (0.60, 0.51), (0.72, 0.51))
    ax.text(0.49, 0.23, "semantic map S", ha="center", fontsize=8.0, color=MUTED)
    ax.text(0.83, 0.23, "correlation sem_corr", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(SEM, "high correlation"), (GOAL, "anchor p*")])
    return _save(fig, "step_06_goal_correlation_map.png")


def step7():
    fig, ax = _setup_ax("Expand FMM goal", "(7)")
    _grid(ax, 0.07, 0.33, 0.24, 0.42, show_corr=True, expanded=False)
    _box(ax, (0.39, 0.49), (0.25, 0.17), "sem_corr >= theta\nwithin radius R", fs=7.8)
    _grid(ax, 0.72, 0.33, 0.22, 0.42, show_corr=True, expanded=True)
    _arrow(ax, (0.31, 0.56), (0.39, 0.57), "from p*")
    _arrow(ax, (0.64, 0.57), (0.72, 0.57), "expanded mask")
    ax.text(0.50, 0.22, "single anchor -> multi-cell FMM goal region", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(GOAL, "original p*"), (SEM, "added goal cells")])
    return _save(fig, "step_07_expand_fmm_goal.png")


def step8():
    fig, ax = _setup_ax("Refined stopping view", "(8)")
    _grid(ax, 0.08, 0.30, 0.32, 0.46, show_corr=True, expanded=True, robot=True)
    ax.plot([0.17, 0.22, 0.29], [0.40, 0.48, 0.56], color=PATH, lw=2.5)
    _box(ax, (0.50, 0.54), (0.18, 0.16), "FMM STG")
    _box(ax, (0.76, 0.54), (0.16, 0.16), "STOP < 25cm", fc="#fee2e2")
    _arrow(ax, (0.40, 0.53), (0.50, 0.62), "follow path")
    _arrow(ax, (0.68, 0.62), (0.76, 0.62))
    ax.text(0.58, 0.29, "effect: final pose is refined stopping viewpoint", ha="center", fontsize=8.0, color=MUTED)
    _mini_legend(ax, [(PATH, "planned path"), (AGENT, "agent pose"), (GOAL, "goal region")])
    return _save(fig, "step_08_refined_stopping_view.png")


def overview(paths):
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.1), dpi=170)
    fig.patch.set_facecolor(BG)
    for ax, path in zip(axes.ravel(), paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.02, hspace=0.06)
    out = os.path.join(OUT, "near_goal_subflows_overview.png")
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def main() -> None:
    ensure_out()
    paths = [step1(), step2(), step3(), step4(), step5(), step6(), step7(), step8()]
    paths.append(overview(paths))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
