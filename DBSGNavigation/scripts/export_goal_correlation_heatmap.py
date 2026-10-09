"""Export sem_corr heatmap assets for Fig.4 Goal correlation map panel."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.export_near_goal_real_subflows import make_state, overlay_map


def render_heatmap_only(sem_corr: np.ndarray, out_size: int = 480) -> np.ndarray:
    h = np.nan_to_num(sem_corr, nan=0.0).astype(np.float32)
    if h.max() > h.min() + 1e-8:
        h = (h - h.min()) / (h.max() - h.min())
    rgb = (plt.get_cmap("inferno")(np.clip(h, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    return cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def save_panel(state, out_dir: str, prefix: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []

    # 1) pure heatmap
    pure = render_heatmap_only(state.sem_corr)
    p1 = os.path.join(out_dir, f"{prefix}_sem_corr_heatmap.png")
    cv2.imwrite(p1, cv2.cvtColor(pure, cv2.COLOR_RGB2BGR))
    paths.append(p1)

    # 2) overlay on occupancy
    overlay = overlay_map(state, heat=np.nan_to_num(state.sem_corr, nan=0))
    p2 = os.path.join(out_dir, f"{prefix}_sem_corr_overlay_occ.png")
    cv2.imwrite(p2, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    paths.append(p2)

    # 3) publication-style figure with colorbar
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=180)
    fig.patch.set_facecolor("#f8f7f2")

    goal = cv2.resize(state.goal_rgb, (220, 165))
    axes[0].imshow(goal)
    axes[0].set_title("Goal image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    axes[0].text(0.5, -0.06, "CLIP -> v_g (512-d features)", ha="center", va="top", transform=axes[0].transAxes, fontsize=9)

    corr = np.nan_to_num(state.sem_corr, nan=0.0).astype(np.float32)
    im = axes[1].imshow(corr, cmap="inferno", vmin=0.0, vmax=max(float(corr.max()), 1e-6))
    axes[1].set_title("sem_corr heatmap", fontsize=11, fontweight="bold")
    gr, gc = state.goal_rc
    axes[1].scatter([gc], [gr], s=36, c="#ef4444", edgecolors="white", linewidths=0.8, zorder=3)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("cos(S, v_g)", fontsize=9)
    axes[1].text(0.5, -0.06, "240 x 240 local grid", ha="center", va="top", transform=axes[1].transAxes, fontsize=9)

    fig.suptitle("Goal correlation map", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    p3 = os.path.join(out_dir, f"{prefix}_goal_correlation_map_panel.png")
    fig.savefig(p3, bbox_inches="tight", facecolor="#f8f7f2")
    plt.close(fig)
    paths.append(p3)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="4ok3usBNeis")
    ap.add_argument("--data_root", default=os.path.join(ROOT, "..", "DBSGConstruction/configs/data/habitat_val"))
    ap.add_argument("--goal_info", default="examples/viz_0.8-0.6/0/3/goal_info.txt")
    ap.add_argument("--replay_steps", type=int, default=10)
    ap.add_argument("--current_frame", type=int, default=480)
    ap.add_argument("--corr_thr", type=float, default=0.52)
    ap.add_argument("--corr_radius", type=int, default=28)
    ap.add_argument("--peaks_only", action="store_true")
    ap.add_argument("--use_open_clip", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="../docs/figs/method_thumbs/goal_correlation")
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.join(ROOT, args.out_dir))
    state = make_state(args)
    paths = save_panel(state, out_dir, prefix="real")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
