"""Visualize Semantic–geometric grounding (Innovation 4) for OV Top-K candidates.

Example:
  python scripts/viz_semantic_geometric_grounding.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "DBSGConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"
ALPHA, BETA = 0.8, 0.6
COCO_IDS = [56, 57, 58, 59, 61, 62]


def resolve_rgb(scene_dir, frame_id):
    fid = str(frame_id)
    if not fid.startswith("frame_"):
        fid = f"frame_{int(fid):05d}"
    if not fid.endswith(".png"):
        fid += ".png"
    return os.path.join(scene_dir, "rgb", fid)


def crop_center(rgb, size=320):
    h, w = rgb.shape[:2]
    side = min(h, w, size)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return cv2.resize(rgb[y0 : y0 + side, x0 : x0 + side], (size, int(size * 0.75)))


def geo_disk_lightglue(goal_rgb, inst_rgb, device="cpu"):
    """Match count via DISK+LightGlue (same as navigation method=1)."""
    import torch
    from lightglue import DISK, LightGlue
    from lightglue.utils import match_pair, numpy_image_to_torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ext = DISK(max_num_keypoints=2048).eval().to(dev)
    matcher = LightGlue(features="disk").eval().to(dev)
    ob = numpy_image_to_torch(goal_rgb).to(dev)
    gi = numpy_image_to_torch(inst_rgb).to(dev)
    with torch.no_grad():
        _, _, matches01 = match_pair(ext, matcher, ob, gi)
        m = matches01["matches"]
        n = int(m.shape[0]) if m.shape[0] <= 2048 else 0
    return n


def geo_orb(goal_bgr, inst_bgr):
    g0 = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(inst_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:40]
    vis = cv2.drawMatches(
        goal_bgr, k0, inst_bgr, k1, matches, None,
        matchColor=(0, 200, 0), flags=2,
    )
    return len(matches), vis


def draw_disk_vis(goal_rgb, inst_rgb, device="cpu"):
    try:
        import torch
        from lightglue import DISK, LightGlue
        from lightglue.utils import match_pair, numpy_image_to_torch, rbd

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        ext = DISK(max_num_keypoints=2048).eval().to(dev)
        matcher = LightGlue(features="disk").eval().to(dev)
        ob = numpy_image_to_torch(goal_rgb).to(dev)
        gi = numpy_image_to_torch(inst_rgb).to(dev)
        with torch.no_grad():
            feats0, feats1, matches01 = match_pair(ext, matcher, ob, gi)
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            k0, k1 = feats0["keypoints"], feats1["keypoints"]
            m = matches01["matches"]
            if m.shape[0] > 2048:
                return None, 0
            h0, w0 = goal_rgb.shape[:2]
            h1, w1 = inst_rgb.shape[:2]
            canvas_h = max(h0, h1)
            canvas = np.zeros((canvas_h, w0 + w1 + 20, 3), dtype=np.uint8)
            canvas[:h0, :w0] = goal_rgb
            canvas[:h1, w0 + 20 : w0 + 20 + w1] = inst_rgb
            for idx in m[:80]:
                p0 = tuple(k0[idx[0]].astype(int))
                p1 = (int(k1[idx[1]][0]) + w0 + 20, int(k1[idx[1]][1]))
                cv2.line(canvas, p0, p1, (0, 220, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, p0, 2, (0, 0, 255), -1)
                cv2.circle(canvas, p1, 2, (255, 0, 0), -1)
            return canvas, int(m.shape[0])
    except Exception as e:
        print("DISK viz fallback:", e)
    return None, 0


def fuse_scores(sem, geo, alpha=ALPHA, beta=BETA):
    from map_planning_utils.semantic_geometric_fusion import fuse_scores as _fuse

    return _fuse(
        np.array(sem, dtype=np.float64),
        np.array(geo, dtype=np.float64),
        alpha,
        beta,
        normalize_geometric_mode="minmax",
        use_semantic=True,
        use_geometric=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--goal_image", default="examples/viz_0.8-0.6/0/3/goal_query.png")
    ap.add_argument("--ov_top_k", type=int, default=6)
    ap.add_argument("--out", default="../docs/figs/semantic_geometric_grounding_explainer.png")
    ap.add_argument("--export_dir", default="../docs/figs/semantic_geometric_exports/4ok3usBNeis_ep3")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    goal_path = os.path.abspath(os.path.join(ROOT, args.goal_image))
    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    export_dir = os.path.abspath(os.path.join(ROOT, args.export_dir))
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    g_bgr = cv2.imread(goal_path)
    g_rgb = g_bgr[:, :, ::-1]

    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    cv_name = "instance_clip_vectors_v2.json"
    if not os.path.isfile(os.path.join(scene_dir, cv_name)):
        cv_name = "instance_clip_vectors.json"
    r = OpenVocabRetriever(
        torch.device(args.device), clip_vectors_filename=cv_name
    )
    r.load_scene(scene_dir, instance_clip_vectors_path=os.path.join(scene_dir, cv_name))
    cands = r.retrieve(g_rgb, top_k=args.ov_top_k)

    sem_scores, geo_scores, frames = [], [], []
    for ent, pos, sem in cands:
        fid = ent[1]
        inst_bgr = cv2.imread(resolve_rgb(scene_dir, fid))
        if inst_bgr is None:
            sem_scores.append(sem)
            geo_scores.append(0.0)
            frames.append(fid)
            continue
        inst_rgb = inst_bgr[:, :, ::-1]
        try:
            geo = float(geo_disk_lightglue(g_rgb, inst_rgb, args.device))
            geo_method = "DISK+LightGlue (#matches)"
        except Exception as e:
            print("geo DISK failed, ORB fallback:", e)
            geo, _ = geo_orb(g_bgr, inst_bgr)
            geo_method = "ORB (#matches, preview fallback)"
        sem_scores.append(float(sem))
        geo_scores.append(geo)
        frames.append(fid)

    fused = fuse_scores(sem_scores, geo_scores)
    best_i = int(np.argmax(fused))

    # --- per-candidate match PNGs ---
    for i, (ent, sem, geo, fid) in enumerate(zip([c[0] for c in cands], sem_scores, geo_scores, frames)):
        inst_bgr = cv2.imread(resolve_rgb(scene_dir, fid))
        vis, n_disk = draw_disk_vis(g_rgb, inst_bgr[:, :, ::-1], args.device)
        if vis is None:
            _, vis = geo_orb(g_bgr, inst_bgr)
            tag = "orb"
        else:
            tag = "disk"
        name = f"cand{i+1:02d}_fused{fused[i]:.3f}_sem{sem:.3f}_geo{geo:.0f}_{tag}_frame{fid}"
        if vis is not None:
            cv2.imwrite(
                os.path.join(export_dir, name + ".png"), vis,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
            )
        write_meta = {
            "rank_clip": i + 1,
            "frame": fid,
            "semantic_clip": sem,
            "geometric_matches": geo,
            "fused_score": float(fused[i]),
            "selected": i == best_i,
        }
        with open(os.path.join(export_dir, name + ".json"), "w") as f:
            json.dump(write_meta, f, indent=2)

    # --- explainer figure ---
    fig = plt.figure(figsize=(14, 8), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(
        0, 1,
        "Semantic–geometric grounding (OV branch, per candidate)\n\n"
        "1) Semantic score = CLIP cosine(goal, instance crop)\n"
        "2) Geometric score = DISK+LightGlue feature matches (count)\n"
        f"3) Fused = {ALPHA}·semantic + {BETA}·geo_norm  (geo min–max within Top-K)\n"
        "4) Pick candidate with highest fused score → 3D goal position",
        fontsize=9, va="top", family="monospace",
    )

    ax_match = fig.add_subplot(gs[0, 1])
    best_bgr = cv2.imread(resolve_rgb(scene_dir, frames[best_i]))
    vis_best, n_best = draw_disk_vis(g_rgb, best_bgr[:, :, ::-1], args.device)
    if vis_best is None:
        _, vis_best = geo_orb(g_bgr, best_bgr)
    if vis_best is not None:
        vis_best = cv2.cvtColor(
            cv2.resize(vis_best, (640, 280)), cv2.COLOR_BGR2RGB
        )
        ax_match.imshow(vis_best)
    ax_match.set_title(
        f"Best after fusion: cand {best_i+1} | frame {frames[best_i]} | "
        f"{int(geo_scores[best_i])} geom. matches",
        fontsize=9,
    )
    ax_match.axis("off")

    ax_bar = fig.add_subplot(gs[1, :])
    x = np.arange(len(cands))
    w = 0.25
    from map_planning_utils.semantic_geometric_fusion import normalize_geometric

    geo_norm = normalize_geometric(np.array(geo_scores), mode="minmax")
    ax_bar.bar(x - w, sem_scores, w, label="semantic (CLIP)", color="#c05621")
    ax_bar.bar(x, geo_norm, w, label="geometric (norm.)", color="#2c5282")
    ax_bar.bar(x + w, fused, w, label="fused", color="#2ca02c")
    ax_bar.axvline(best_i, color="red", linestyle="--", alpha=0.5, label="selected")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"#{i+1}\n{frame}" for i, frame in enumerate(frames)], fontsize=7)
    ax_bar.set_ylabel("score")
    ax_bar.set_title("(b) Top-K candidates: semantic + geometric → fused", fontsize=10)
    ax_bar.legend(fontsize=8, ncol=4, loc="upper right")
    ax_bar.set_ylim(0, max(max(fused) * 1.1, 1.0))

    fig.suptitle(
        f"Semantic–geometric grounding — scene {args.scene}  |  same goal image, verify each OV candidate",
        fontsize=11, y=0.98,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if vis_best is not None:
        out_best = vis_best if len(vis_best.shape) == 3 and vis_best.shape[2] == 3 else cv2.cvtColor(vis_best, cv2.COLOR_RGB2BGR)
        if out_best.dtype != np.uint8:
            out_best = np.clip(out_best, 0, 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(export_dir, "00_best_match.png"),
            out_best if out_best.shape[2] == 3 else best_bgr,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
    print("saved", out_path)
    print("per-candidate matches in", export_dir)


if __name__ == "__main__":
    main()
