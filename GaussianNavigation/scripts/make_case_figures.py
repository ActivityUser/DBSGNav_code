"""案例型定性图 B1/B2/B3：目标图 + 轨迹关键帧拼图。

从 examples/<...>/<ep>/ 读取 goal_query.png 与 *-Vis.mp4，抽若干关键帧拼成一行，
顶部标注 case 元信息（类别/来源/成功/SPL/定位误差）。
"""
import argparse
import glob
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_info(case_dir):
    d = {}
    gi = os.path.join(case_dir, "goal_info.txt")
    if os.path.isfile(gi):
        for line in open(gi):
            m = re.match(r"(\w+)=(.*)", line.strip())
            if m:
                d[m.group(1)] = m.group(2)
    return d


def grab_frames(mp4, n=4):
    cap = cv2.VideoCapture(mp4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # fallback: read sequentially
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[:, :, ::-1])
        cap.release()
        if not frames:
            return []
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]
    idx = np.linspace(0, total - 1, n).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(fr[:, :, ::-1])
    cap.release()
    return out


def make_case_row(case_dir, title, out_path, n_frames=4):
    info = read_info(case_dir)
    gq = os.path.join(case_dir, "goal_query.png")
    mp4 = glob.glob(os.path.join(case_dir, "*-Vis.mp4"))
    frames = grab_frames(mp4[0], n_frames) if mp4 else []
    panels = []
    labels = []
    if os.path.isfile(gq):
        panels.append(cv2.imread(gq)[:, :, ::-1])
        labels.append("Goal image")
    fl = ["start", "early", "mid", "end"]
    for k, fr in enumerate(frames):
        panels.append(fr)
        labels.append(fl[k] if k < len(fl) else f"t{k}")
    if not panels:
        print("no panels for", case_dir)
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.3))
    if len(panels) == 1:
        axes = [axes]
    for ax, img, lb in zip(axes, panels, labels):
        ax.imshow(img)
        ax.set_title(lb, fontsize=8)
        ax.axis("off")
    succ = info.get("success", "?")
    sub = "%s | cat=%s | src=%s | success=%s | SPL=%s | err2gt=%sm" % (
        title, info.get("object_category", "?"), info.get("goal_source", "?"),
        succ, info.get("spl", "?"), info.get("dist_pred_to_gt_m", info.get("dist_pred_to_gt", "?")),
    )
    fig.suptitle(sub, fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path, "(panels=%d)" % len(panels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples_root", default="examples")
    ap.add_argument("--out", default="../docs/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.examples_root
    cases = [
        (os.path.join(R, "viz_0.8-0.6/0/3"), "B1 Success (OV retrieval)", "fig_b1_success_case"),
        (os.path.join(R, "0/8"), "B2 Failure (semantic-gaussian mismatch)", "fig_b2_fail_sg_force"),
        (os.path.join(R, "0/18"), "B3 Failure (long-tail localization)", "fig_b3_fail_longtail"),
    ]
    for cd, title, name in cases:
        if os.path.isdir(cd):
            make_case_row(cd, title, os.path.join(args.out, name))
        else:
            print("missing case dir:", cd)


if __name__ == "__main__":
    main()
