"""Export one PNG per step in Fig.2 Semantic Gaussian branch flow.

Example:
  python scripts/viz_semantic_gaussian_branch.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png \\
    --out_dir ../docs/figs/sg_branch_exports/4ok3usBNeis_ep3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.viz_semantic_gaussian_map import (
    DEFAULT_MAP_ROOT,
    DEFAULT_SCENE,
    encode_goal,
    load_params,
    resolve_params_path,
    subsample_idx,
    topdown_scatter,
)

DEFAULT_GOAL = "examples/viz_0.8-0.6/0/3/goal_query.png"
DEFAULT_GOAL_INFO = "examples/viz_0.8-0.6/0/3/goal_info.txt"
DATA_ROOT = os.path.join(ROOT, "..", "GaussianConstruction/configs/data/habitat_val")

# Match typical navigable-goal config (full_scene_crop / inst_v3_vlm)
NAV = dict(
    sim_floor=0.20,
    top_k_cluster=200,
    eps_m=0.6,
    min_cluster_size=6,
    snap_distance_m=0.8,
    floor_y_band=1.5,
    floor_y_offset=0.05,
    api_rerank_topk=3,
    api_rerank_sim_margin=0.02,
)


def read_goal_info(path):
    info = {}
    if not os.path.isfile(path):
        return info
    for line in open(path):
        line = line.strip()
        if "=" not in line or line.startswith("---"):
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def save_fig(path, dpi=200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def agent_from_scene_start(scene_dir):
    tf = os.path.join(scene_dir, "transforms.json")
    if not os.path.isfile(tf):
        return None
    with open(tf) as f:
        m = np.array(json.load(f)["frames"][0]["transform_matrix"], dtype=np.float64)
    return m[:3, 3].astype(np.float64)


def plot_clustering_topdown(ax, means, prep_result, max_pts=25000):
    """Color high-sim candidates by DBSCAN cluster label."""
    from map_planning_utils.semantic_gaussian_goal import SemanticGaussianGoal

    sg = SemanticGaussianGoal(device="cpu", gaussian_map_root=".")
    # reuse prep via temporary instance — call prep on loaded data
    _sims, _max_sim, n_total, floor_y, clusters = prep_result
    # Re-run prep inline is heavy; clusters already have gaussian_indices
    xyz_all = means
    ax.set_facecolor("0.97")
    # background subsample
    idx_bg = subsample_idx(xyz_all.shape[0], min(max_pts, 40000))
    ax.scatter(
        xyz_all[idx_bg, 0],
        xyz_all[idx_bg, 2],
        c="0.82",
        s=0.08,
        linewidths=0,
        marker=".",
        alpha=0.35,
        rasterized=True,
    )
    cmap = plt.colormaps["tab10"].resampled(max(10, len(clusters)))
    for i, cl in enumerate(clusters[:10]):
        gidx = cl.get("gaussian_indices")
        if gidx is None:
            continue
        gidx = np.asarray(gidx, dtype=np.int64).reshape(-1)
        if gidx.size > 8000:
            gidx = gidx[np.linspace(0, gidx.size - 1, 8000, dtype=int)]
        pts = xyz_all[gidx]
        ax.scatter(
            pts[:, 0],
            pts[:, 2],
            c=[cmap(i % 10)],
            s=1.2,
            linewidths=0,
            label=f"C{i+1} (n={cl['cluster_size']}, score={cl['score']:.1f})",
        )
        rc = cl["raw_center"]
        ax.scatter(
            rc[0],
            rc[2],
            s=80,
            marker="x",
            c=[cmap(i % 10)],
            linewidths=1.5,
            zorder=5,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser(description="Fig.2 SG branch — one PNG per flow step")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--gaussian_map_root", default=DEFAULT_MAP_ROOT)
    ap.add_argument("--goal_image", default=DEFAULT_GOAL)
    ap.add_argument("--goal_info", default=DEFAULT_GOAL_INFO)
    ap.add_argument(
        "--out_dir",
        default="../docs/figs/sg_branch_exports/4ok3usBNeis_ep3",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_points", type=int, default=80000)
    args = ap.parse_args()

    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever
    from map_planning_utils.semantic_gaussian_goal import SemanticGaussianGoal

    out_dir = os.path.abspath(os.path.join(ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    goal_path = os.path.abspath(os.path.join(ROOT, args.goal_image))
    gi = read_goal_info(os.path.abspath(os.path.join(ROOT, args.goal_info)))
    scene_dir = os.path.abspath(os.path.join(DATA_ROOT, args.scene))

    g_bgr = cv2.imread(goal_path)
    if g_bgr is None:
        raise FileNotFoundError(goal_path)
    g_rgb = g_bgr[:, :, ::-1]

    params_path = resolve_params_path(args.scene, args.gaussian_map_root)
    means, clip, rgb = load_params(params_path)
    agent_pos = agent_from_scene_start(scene_dir)

    device = torch.device(args.device)
    r = OpenVocabRetriever(device)
    v_g = r.encode_image(g_rgb).astype(np.float32).reshape(-1)

    sg = SemanticGaussianGoal(
        device,
        gaussian_map_root=os.path.abspath(args.gaussian_map_root),
        clip_encode_image=r.encode_image,
    )
    if not sg.load_scene(args.scene):
        raise RuntimeError(sg.get_last_error())

    prep = sg._navigable_cluster_prep(
        g_rgb,
        agent_pos,
        NAV["sim_floor"],
        NAV["top_k_cluster"],
        NAV["eps_m"],
        NAV["min_cluster_size"],
        NAV["floor_y_band"],
    )
    if prep is None:
        raise RuntimeError("SG cluster prep failed")
    sims, max_sim_global, n_total, floor_y, clusters = prep

    clusters_top = sg.get_topk_navigable_cluster_candidates(
        g_rgb,
        top_k=NAV["api_rerank_topk"],
        agent_pos=agent_pos,
        sim_floor=NAV["sim_floor"],
        top_k_cluster=NAV["top_k_cluster"],
        eps_m=NAV["eps_m"],
        min_cluster_size=NAV["min_cluster_size"],
        snap_distance_m=NAV["snap_distance_m"],
        floor_y_band=NAV["floor_y_band"],
        floor_y_offset=NAV["floor_y_offset"],
    )
    if not clusters_top:
        raise RuntimeError("no navigable clusters")

    mean_sims = [float(c["mean_sim"]) for c in clusters_top]
    margin = float(mean_sims[0] - mean_sims[1]) if len(mean_sims) >= 2 else float("inf")
    ambiguous = len(clusters_top) >= 2 and margin <= NAV["api_rerank_sim_margin"]
    pick_i = 0

    # --- 01 Goal image ---
    cv2.imwrite(
        os.path.join(out_dir, "01_goal_image.png"),
        g_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
    )

    # --- 02 CLIP encoding ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
    axes[0].imshow(g_rgb)
    axes[0].set_title("Goal image (Instance ImageGoal)", fontsize=10)
    axes[0].axis("off")
    dims = np.arange(64)
    axes[1].bar(dims, v_g[:64], color="#4a7ebB", width=0.85)
    axes[1].set_xlabel("CLIP dimension (first 64 of 512)")
    axes[1].set_ylabel("value")
    axes[1].set_title(
        f"CLIP encoding (ViT-B/32, L2-norm)  |  ||v||={np.linalg.norm(v_g):.3f}",
        fontsize=10,
    )
    fig.suptitle("Step: CLIP encoding", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "02_clip_encoding.png"))

    # --- 03 Similarity on Gaussian map ---
    idx = subsample_idx(means.shape[0], args.max_points)
    x, z = means[idx, 0], means[idx, 2]
    sim_plot = np.dot(clip[idx], v_g)
    sim_n = (sim_plot - sim_plot.min()) / (sim_plot.max() - sim_plot.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    sc = topdown_scatter(ax, x, z, sim_n, "CLIP similarity on semantic Gaussian map", cmap="jet")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="cosine sim (norm.)")
    fig.suptitle("Step: Similarity on Gaussian map", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "03_similarity_gaussian_map.png"))

    # --- 04 Spatial clustering ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    plot_clustering_topdown(ax, means, prep, max_pts=args.max_points)
    ax.set_title("DBSCAN clusters on high-sim Gaussians (top-down x–z)", fontsize=10)
    fig.suptitle("Step: Spatial clustering", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "04_spatial_clustering.png"))

    # --- 05 Top clusters ambiguous? ---
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    labels = [f"Cluster {i+1}" for i in range(len(clusters_top))]
    scores = [float(c["cluster_score"]) for c in clusters_top]
    colors = ["#2ca02c" if i == 0 else "#aec7e8" for i in range(len(clusters_top))]
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("cluster score (sum of CLIP sim in cluster)")
    verdict = "Yes → VLM reranking" if ambiguous else "No → select best cluster (CLIP score)"
    ax.set_title(
        f"Top clusters ambiguous?  margin={margin:.3f} (thr={NAV['api_rerank_sim_margin']})  →  {verdict}",
        fontsize=10,
    )
    fig.suptitle("Step: Ambiguity check (mean_sim margin)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "05_top_clusters_ambiguous.png"))

    # --- 06 VLM reranking (candidate views) ---
    n_c = len(clusters_top)
    fig, axes = plt.subplots(1, n_c + 1, figsize=(2.2 * (n_c + 1), 2.8), dpi=200)
    axes[0].imshow(g_rgb)
    axes[0].set_title("Goal", fontsize=9)
    axes[0].axis("off")
    for i, c in enumerate(clusters_top):
        thumb = c.get("thumbnail_rgb")
        if thumb is None:
            thumb = np.full((128, 128, 3), 180, dtype=np.uint8)
        axes[i + 1].imshow(thumb)
        axes[i + 1].set_title(
            f"Cand {i+1}\nmean_sim={c['mean_sim']:.3f}",
            fontsize=8,
        )
        axes[i + 1].axis("off")
    sub = "(optional) Qwen-VL picks cluster" if ambiguous else "(skipped — margin clear)"
    fig.suptitle(f"Step: VLM reranking {sub}", fontsize=11, y=1.05)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "06_vlm_rerank.png"))

    # --- 07 Select cluster ---
    cl0 = clusters[int(pick_i)]
    chosen_nav = clusters_top[int(pick_i)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    plot_clustering_topdown(ax, means, prep, max_pts=args.max_points)
    rc = cl0["raw_center"]
    gidx = np.asarray(cl0["gaussian_indices"], dtype=np.int64).reshape(-1)
    if gidx.size > 12000:
        gidx = gidx[np.linspace(0, gidx.size - 1, 12000, dtype=int)]
    pts = means[gidx]
    ax.scatter(pts[:, 0], pts[:, 2], c="#2ca02c", s=2.5, linewidths=0, label="selected cluster")
    ax.scatter(rc[0], rc[2], s=200, marker="*", c="#2ca02c", edgecolors="k", zorder=6)
    ax.legend(fontsize=8)
    ax.set_title(f"Selected cluster #{pick_i+1} (score={cl0['score']:.1f})", fontsize=10)
    fig.suptitle("Step: Select best cluster", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "07_select_cluster.png"))

    # --- 08 Snap to navigable point ---
    raw = np.array(cl0["raw_center"], dtype=np.float64)
    pos_snap = np.array(chosen_nav["position"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    ax.scatter(raw[0], raw[2], s=180, marker="o", c="#d62728", label="cluster centroid (raw)", zorder=5)
    ax.scatter(pos_snap[0], pos_snap[2], s=220, marker="*", c="#2ca02c", label="snapped goal", zorder=6)
    if agent_pos is not None:
        ax.scatter(agent_pos[0], agent_pos[2], s=120, marker="^", c="#1f77b4", label="agent", zorder=5)
        ax.annotate(
            "",
            xy=(pos_snap[0], pos_snap[2]),
            xytext=(raw[0], raw[2]),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Snap toward agent (≤{NAV['snap_distance_m']} m in x–z)  |  snapped={chosen_nav.get('snapped', False)}",
        fontsize=10,
    )
    fig.suptitle("Step: Snap to navigable point", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "08_snap_navigable_point.png"))

    # --- 09 SG candidate position ---
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    idx_m = subsample_idx(means.shape[0], min(30000, means.shape[0]))
    ax.scatter(means[idx_m, 0], means[idx_m, 2], c="0.85", s=0.1, linewidths=0)
    for i, c in enumerate(clusters_top):
        p = c["position"]
        ax.scatter(
            p[0], p[2], s=100, c="#aec7e8" if i != pick_i else "#2ca02c",
            edgecolors="k", linewidths=0.5, zorder=4,
        )
        ax.annotate(f"SG{i+1}", (p[0], p[2]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    p_final = clusters_top[pick_i]["position"]
    ax.scatter(
        p_final[0], p_final[2], s=280, marker="*", c="#2ca02c",
        edgecolors="k", linewidths=0.8, zorder=6, label="SG candidate",
    )
    if gi.get("pred_goal_pos"):
        pred = json.loads(gi["pred_goal_pos"].replace(" ", ""))
        ax.scatter(
            pred[0], pred[2], s=160, marker="D", c="#d94801",
            edgecolors="k", linewidths=0.6, zorder=5, label="final pred (merit)",
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title("SG candidate position (3D navigable goal)", fontsize=10)
    fig.suptitle("Step: SG candidate position", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(os.path.join(out_dir, "09_sg_candidate_position.png"))

    # overview grid
    steps = [
        "01_goal_image.png",
        "02_clip_encoding.png",
        "03_similarity_gaussian_map.png",
        "04_spatial_clustering.png",
        "05_top_clusters_ambiguous.png",
        "06_vlm_rerank.png",
        "07_select_cluster.png",
        "08_snap_navigable_point.png",
        "09_sg_candidate_position.png",
    ]
    tiles = []
    for fn in steps:
        im = cv2.imread(os.path.join(out_dir, fn))
        if im is not None:
            tiles.append(cv2.resize(im, (420, 315)))
    if len(tiles) == 9:
        rows = []
        for r in range(3):
            rows.append(np.hstack(tiles[r * 3 : r * 3 + 3]))
        grid = np.vstack(rows)
        cv2.imwrite(
            os.path.join(out_dir, "00_sg_branch_overview.png"),
            grid,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )

    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Fig.2 Semantic Gaussian branch — step visuals\n"
            "=============================================\n\n"
            f"scene={args.scene}\n"
            f"goal={args.goal_image}\n"
            f"ambiguous={ambiguous}  margin={margin:.4f}\n"
            f"SG pick cluster #{pick_i+1}  pos={list(p_final)}\n\n"
        )
        for fn in steps:
            f.write(f"  {fn}\n")
        f.write("  00_sg_branch_overview.png\n")

    print(f"scene={args.scene}  clusters={len(clusters_top)}  ambiguous={ambiguous}  margin={margin:.4f}")
    print("saved", out_dir)
    for fn in ["00_sg_branch_overview.png"] + steps:
        print(" ", fn)


if __name__ == "__main__":
    main()
