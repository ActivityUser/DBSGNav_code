"""定性统计图 A1/A2/A3：从 episode_goal_summary 逐局日志生成 TPAMI 用图。

用法:
  python scripts/plot_qualitative.py \
    --log "experiments/.../1000(0.8-0.6_inst_v3_vlm)/2/full_scene_crop_1000ep_episode_goal_summary.log" \
    --out docs/figs
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = [
    "goal_source", "dist_pred_to_gt", "success", "spl", "dist_to_goal",
]


def parse_log(path):
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            if "[episode_goal_summary]" not in line:
                continue
            d = {}
            for k in FIELDS:
                m = re.search(k + r"=([^\s]+)", line)
                if m:
                    d[k] = m.group(1)
            rows.append(d)
    return rows


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def fig_a1_cdf(dpg, out):
    x = np.sort(dpg)
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(x, y, lw=2)
    for thr in [0.5, 1.0, 2.0, 3.0]:
        frac = float(np.mean(dpg <= thr))
        ax.axvline(thr, color="gray", ls=":", lw=0.8)
        ax.annotate(f"{frac*100:.0f}%", (thr, frac), fontsize=7,
                    xytext=(thr + 0.1, frac - 0.06))
    ax.set_xlim(0, 6)
    ax.set_xlabel("Goal localization error to GT (m)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Localization error CDF (n=%d)" % len(dpg), fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_a1_dpg_cdf.pdf"))
    fig.savefig(os.path.join(out, "fig_a1_dpg_cdf.png"), dpi=160)
    plt.close(fig)


def fig_a2_succ_vs_err(dpg, succ, out):
    edges = [0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
    labels = ["0-.5", ".5-1", "1-2", "2-3", "3-5", ">5"]
    rates, counts = [], []
    for i in range(len(edges) - 1):
        m = (dpg >= edges[i]) & (dpg < edges[i + 1])
        counts.append(int(m.sum()))
        rates.append(float(succ[m].mean()) if m.sum() else 0.0)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bars = ax.bar(labels, rates, color="#3b75af")
    for b, c in zip(bars, counts):
        ax.annotate(f"n={c}", (b.get_x() + b.get_width() / 2, b.get_height() + 0.02),
                    ha="center", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Localization error to GT (m)")
    ax.set_ylabel("Success rate")
    ax.set_title("Success vs. localization error", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_a2_succ_vs_dpg.pdf"))
    fig.savefig(os.path.join(out, "fig_a2_succ_vs_dpg.png"), dpi=160)
    plt.close(fig)


def fig_a3_goal_source(sources, out):
    from collections import Counter
    c = Counter(sources)
    labels = list(c.keys())
    sizes = [c[k] for k in labels]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%", startangle=90,
           textprops={"fontsize": 7})
    ax.set_title("Final goal source distribution", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_a3_goal_source.pdf"))
    fig.savefig(os.path.join(out, "fig_a3_goal_source.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="docs/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = parse_log(args.log)
    dpg = np.array([fnum(r.get("dist_pred_to_gt")) for r in rows])
    succ = np.array([fnum(r.get("success")) for r in rows])
    sources = [r.get("goal_source", "?") for r in rows]
    mask = ~np.isnan(dpg)
    dpg, succ = dpg[mask], succ[mask]
    print(f"parsed {len(rows)} episodes; valid dpg={len(dpg)}")
    print(f"succ overall={succ.mean():.4f}  dpg mean={dpg.mean():.3f} median={np.median(dpg):.3f}")
    fig_a1_cdf(dpg, args.out)
    fig_a2_succ_vs_err(dpg, succ, args.out)
    fig_a3_goal_source(sources, args.out)
    print("saved A1/A2/A3 to", args.out)


if __name__ == "__main__":
    main()
