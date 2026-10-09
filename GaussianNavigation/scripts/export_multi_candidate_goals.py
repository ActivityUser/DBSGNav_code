"""Export multi-candidate goal images (SG clusters + OV Top-K) as individual PNGs.

Multi-candidate goal generation = multiple 3D goal hypotheses for the *same*
Instance ImageGoal (not multiple object categories). Two branches:
  - SG: CLIP on semantic Gaussian map → several spatial clusters
  - OV: instance database Top-K → each with a navigable viewpoint

Example:
  python scripts/export_multi_candidate_goals.py \\
    --scene 4ok3usBNeis \\
    --goal_image examples/viz_0.8-0.6/0/3/goal_query.png \\
    --out_dir ../docs/figs/multi_candidate_exports/4ok3usBNeis_ep3
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
    ROOT, "..", "GaussianConstruction/configs/data/habitat_val"
)
DEFAULT_MAP_ROOT = os.path.join(
    ROOT, "..", "GaussianConstruction/experiments/habitat_val_final"
)
DEFAULT_SCENE = "4ok3usBNeis"
COCO_IDS = [56, 57, 58, 59, 61, 62]


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


def resolve_rgb(scene_dir, frame_id):
    fid = str(frame_id)
    if not fid.startswith("frame_"):
        fid = f"frame_{int(fid):05d}"
    if not fid.endswith(".png"):
        fid += ".png"
    return os.path.join(scene_dir, "rgb", fid)


def resolve_seg_npy(scene_dir, frame_id):
    fid = str(frame_id)
    if fid.startswith("frame_"):
        fid = fid[6:]
    if fid.endswith(".png"):
        fid = fid[:-4]
    return os.path.join(scene_dir, "seg_semantic", f"frame_{int(fid):05d}.npy")


def crop_instance(rgb, seg_path, coco_id, pad=0.25):
    H, W = rgb.shape[:2]
    if not os.path.isfile(seg_path):
        return cv2.resize(rgb, (320, 240))
    masks = np.load(seg_path, allow_pickle=True)
    if isinstance(masks, np.ndarray) and masks.dtype == object:
        masks = masks.item()
    if not isinstance(masks, dict) or coco_id not in masks or not masks[coco_id]:
        return cv2.resize(rgb, (320, 240))
    mask = masks[coco_id][0]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return cv2.resize(rgb, (320, 240))
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y2 - y1) * pad), int((x2 - x1) * pad)
    y1, y2 = max(0, y1 - ph), min(H - 1, y2 + ph)
    x1, x2 = max(0, x1 - pw), min(W - 1, x2 + pw)
    return cv2.resize(rgb[y1 : y2 + 1, x1 : x2 + 1], (320, 240))


def find_coco_for_entry(ir, ent):
    for cid in COCO_IDS:
        if ent in ir.get(str(cid), []):
            return cid
    return 57


def save_png(path, img_bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def write_meta(path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_map(out_path, goal_info, sg_list, ov_list):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    for i, c in enumerate(sg_list):
        p = c["position"]
        ax.scatter(p[0], p[2], s=120, c="#2166ac", edgecolors="k", linewidths=0.6, zorder=4)
        ax.annotate(f"SG{i+1}", (p[0], p[2]), fontsize=8, color="#2166ac", xytext=(4, 4),
                    textcoords="offset points")
    for i, c in enumerate(ov_list):
        p = c["position"]
        ax.scatter(p[0], p[2], s=120, c="#d94801", edgecolors="k", linewidths=0.6, zorder=4)
        ax.annotate(f"OV{i+1}", (p[0], p[2]), fontsize=8, color="#d94801", xytext=(4, 4),
                    textcoords="offset points")
    if goal_info.get("pred_goal_pos"):
        pred = json.loads(goal_info["pred_goal_pos"].replace(" ", ""))
        ax.scatter(pred[0], pred[2], s=220, marker="*", c="#2ca02c", edgecolors="k",
                   linewidths=0.8, zorder=6, label="final pred")
    if goal_info.get("gt_viewpoint_goal_pos"):
        gt = json.loads(goal_info["gt_viewpoint_goal_pos"].replace(" ", ""))
        ax.scatter(gt[0], gt[2], s=160, marker="x", c="#9467bd", linewidths=2,
                   zorder=6, label="GT viewpoint")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Candidate goal positions (top-down x–z)")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--gaussian_map_root", default=DEFAULT_MAP_ROOT)
    ap.add_argument("--goal_image", default="examples/viz_0.8-0.6/0/3/goal_query.png")
    ap.add_argument("--goal_info", default="examples/viz_0.8-0.6/0/3/goal_info.txt")
    ap.add_argument(
        "--out_dir",
        default="../docs/figs/multi_candidate_exports/4ok3usBNeis_ep3",
    )
    ap.add_argument("--ov_top_k", type=int, default=6)
    ap.add_argument("--sg_top_k", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_dir = os.path.abspath(os.path.join(ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    goal_path = os.path.abspath(os.path.join(ROOT, args.goal_image))
    gi_path = os.path.abspath(os.path.join(ROOT, args.goal_info))
    goal_info = read_goal_info(gi_path)

    g = cv2.imread(goal_path)
    if g is None:
        raise FileNotFoundError(goal_path)
    g_rgb = g[:, :, ::-1]
    save_png(os.path.join(out_dir, "00_goal_image.png"), g)

    with open(os.path.join(scene_dir, "instance_retrieval.json")) as f:
        ir = json.load(f)

    # --- OV Top-K ---
    import torch
    from map_planning_utils.open_vocab_retriever import OpenVocabRetriever

    cv_name = "instance_clip_vectors_v2.json"
    if not os.path.isfile(os.path.join(scene_dir, cv_name)):
        cv_name = "instance_clip_vectors.json"
    r = OpenVocabRetriever(
        torch.device(args.device), clip_vectors_filename=cv_name
    )
    r.load_scene(scene_dir, instance_clip_vectors_path=os.path.join(scene_dir, cv_name))
    ov_raw = r.retrieve(g_rgb, top_k=args.ov_top_k)

    ov_export = []
    for rank, (ent, pos, sem_sim) in enumerate(ov_raw, start=1):
        frame_id = ent[1]
        cid = find_coco_for_entry(ir, ent)
        rgb_path = resolve_rgb(scene_dir, frame_id)
        rgb = cv2.imread(rgb_path)
        crop = None
        if rgb is not None:
            crop = crop_instance(
                cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                resolve_seg_npy(scene_dir, frame_id),
                cid,
            )
        name = f"ov_{rank:02d}_clip{sem_sim:.3f}_frame{frame_id}"
        meta = {
            "branch": "open_vocab",
            "rank": rank,
            "clip_semantic_sim": float(sem_sim),
            "frame_id": str(frame_id),
            "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            "coco_id": int(cid),
            "note": "One instance record in the database; same goal category, may be same object from another view.",
        }
        if crop is not None:
            save_png(os.path.join(out_dir, name + ".png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        write_meta(os.path.join(out_dir, name + ".json"), meta)
        ov_export.append({"position": pos, "sem_sim": sem_sim, "name": name})
        print("wrote", name)

    # --- SG Top-K clusters ---
    from map_planning_utils.semantic_gaussian_goal import SemanticGaussianGoal

    sg = SemanticGaussianGoal(
        torch.device(args.device),
        gaussian_map_root=os.path.abspath(args.gaussian_map_root),
        clip_encode_image=r.encode_image,
    )
    sg_loaded = sg.load_scene(args.scene)
    sg_export = []
    if sg_loaded:
        # approximate agent at episode start from dist in goal_info (optional)
        agent_pos = None
        clusters = sg.get_topk_navigable_cluster_candidates(
            g_rgb, top_k=args.sg_top_k, agent_pos=agent_pos
        )
        if clusters:
            for rank, cl in enumerate(clusters, start=1):
                pos = cl["position"]
                name = f"sg_{rank:02d}_score{cl['cluster_score']:.2f}_size{cl['cluster_size']}"
                meta = {
                    "branch": "semantic_gaussian",
                    "rank": rank,
                    "cluster_score": float(cl["cluster_score"]),
                    "mean_sim": float(cl["mean_sim"]),
                    "cluster_size": int(cl["cluster_size"]),
                    "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "note": "SG cluster thumbnail is mean RGB of Gaussians in cluster (not a camera image).",
                }
                thumb = cl.get("thumbnail_rgb")
                if thumb is not None:
                    save_png(
                        os.path.join(out_dir, name + ".png"),
                        cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                    )
                write_meta(os.path.join(out_dir, name + ".json"), meta)
                sg_export.append({"position": pos, "name": name, "cl": cl})
                print("wrote", name)
    else:
        print("SG load failed:", sg.get_last_error())

    plot_map(os.path.join(out_dir, "candidates_map.png"), goal_info, sg_export, ov_export)

    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Multi-candidate goal generation (same Instance ImageGoal, multiple 3D hypotheses)\n"
            "================================================================================\n\n"
            "NOT multiple object categories — one goal image, several candidate GOAL POSITIONS.\n\n"
            "Branches:\n"
            f"  - OV: Top-{args.ov_top_k} from instance database (CLIP on instance crops)\n"
            f"  - SG: Top-{args.sg_top_k} spatial clusters on semantic Gaussian map\n\n"
            "Files:\n"
            "  00_goal_image.png     — query (Instance ImageGoal)\n"
            "  ov_XX_*.png           — OV candidate instance crop + .json metadata\n"
            "  sg_XX_*.png           — SG cluster color patch + .json metadata\n"
            "  candidates_map.png    — all positions on scene x-z plane\n\n"
            "Final navigation picks ONE position via merit / fusion (see goal_info pred_goal_pos).\n"
        )
    print("saved", readme)
    print("\nAll exports in:", out_dir)


if __name__ == "__main__":
    main()
