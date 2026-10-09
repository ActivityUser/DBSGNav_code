#!/usr/bin/env python3
"""verify_inst_aligned_clip.py

Sanity checks for build_instance_aligned_gaussian_clip.py output.

Reports:
  1. clip_vectors shape / non-zero / unique-vector ratio.
  2. Number of Gaussians whose vector exactly equals some entry in
     instance_clip_vectors_v2.json (top-1 hits, no aggregation).
  3. Mean cosine similarity between each Gaussian's new vector and
     its NEAREST v2 entry vector across all coco_ids (random subsample).
  4. Coverage diff against the input (broadcast / dense_v2) clip vectors.
  5. cluster diameter sample (cos > 0.9999) over a random 50K-Gaussian
     subset (to compare with dense_v2's ~1.5m median diameter).

Usage:
  python scripts/verify_inst_aligned_clip.py \
      --in_root  experiments/habitat_val_dense_clip_v2 \
      --new_root experiments/habitat_val_inst_aligned_v3 \
      --scene yr17PDCnDDW \
      --basedir configs/data/habitat_val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))


def _l2_normalise(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _load_v2_matrix(scene_dir: Path):
    p = scene_dir / "instance_clip_vectors_v2.json"
    if not p.is_file():
        return None
    with open(p) as f:
        v2 = json.load(f)
    rows = []
    for c in [56, 57, 58, 59, 61, 62]:
        arr = np.array(v2.get(str(c), []), dtype=np.float32)
        if arr.size == 0 or arr.shape[1] != 512:
            continue
        rows.append(arr)
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def verify_scene(in_run_dir: Path, new_run_dir: Path, scene_dir: Path, sample: int = 50_000):
    in_npz = in_run_dir / "params.npz"
    new_npz = new_run_dir / "params.npz"
    if not in_npz.is_file() or not new_npz.is_file():
        print(f"[skip] missing npz: in={in_npz.is_file()} new={new_npz.is_file()}")
        return None

    in_data = dict(np.load(in_npz, allow_pickle=True))
    new_data = dict(np.load(new_npz, allow_pickle=True))

    means_in = in_data["means3D"].astype(np.float32)
    means_new = new_data["means3D"].astype(np.float32)
    print(f"  N_in={means_in.shape[0]:,}  N_new={means_new.shape[0]:,}  "
          f"means_match={np.allclose(means_in, means_new)}")

    cv_in = in_data.get("clip_vectors")
    cv_new = new_data.get("clip_vectors")
    if cv_in is None or cv_new is None:
        print("  [warn] missing clip_vectors in one of the npz")
        return None
    cv_in = cv_in.astype(np.float32)
    cv_new = cv_new.astype(np.float32)
    print(f"  cv_in.shape={cv_in.shape}  cv_new.shape={cv_new.shape}")
    print(f"  cv_in nonzero rows={int((np.linalg.norm(cv_in, axis=1) > 1e-6).sum())}")
    print(f"  cv_new nonzero rows={int((np.linalg.norm(cv_new, axis=1) > 1e-6).sum())}")

    # changed rows
    diff = np.linalg.norm(cv_new - cv_in, axis=1)
    changed = diff > 1e-6
    print(f"  changed_rows={int(changed.sum())} ({changed.mean() * 100:.2f}%)")
    changed_ratio = float(changed.mean())

    # uniqueness via 16-d truncation (cheap proxy of dense_v2 verification metric)
    n = min(sample, cv_new.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(cv_new.shape[0], size=n, replace=False)
    cv_sub = cv_new[idx]
    keys = np.round(cv_sub[:, :16], 4)
    unique_keys = len(set(map(tuple, keys.tolist())))
    print(f"  cv_new unique-vector ratio (probe @ 16d, n={n}) = {unique_keys}/{n}  ({unique_keys / max(1, n) * 100:.2f}%)")

    # cluster diameter sample (cos > 0.9999)
    cv_norm = _l2_normalise(cv_sub)
    sims = cv_norm @ cv_norm.T
    cluster_diam = []
    visited = np.zeros(n, dtype=bool)
    means_sub = means_new[idx]
    for i in range(n):
        if visited[i]:
            continue
        members = np.where(sims[i] > 0.9999)[0]
        if members.size <= 1:
            continue
        visited[members] = True
        if members.size > 200:
            members = members[:200]
        pts = means_sub[members]
        d = float(np.linalg.norm(pts - pts.mean(axis=0, keepdims=True), axis=1).max() * 2.0)
        cluster_diam.append(d)
    if cluster_diam:
        cd = np.array(cluster_diam)
        cluster_median = float(np.median(cd))
        cluster_max = float(cd.max())
        print(f"  cluster_diameter (cos>0.9999, n={len(cd)} clusters): "
              f"median={cluster_median:.2f}m  max={cluster_max:.2f}m")
    else:
        cluster_median = float("nan")
        cluster_max = float("nan")
        print("  cluster_diameter: no near-duplicate clusters in subsample")

    # nearest-v2 cosine similarity over a small subsample
    v2_mat = _load_v2_matrix(scene_dir)
    if v2_mat is not None and v2_mat.shape[0] > 0:
        v2n = _l2_normalise(v2_mat)
        cv_n = _l2_normalise(cv_sub)
        sims_v2 = cv_n @ v2n.T  # (n, E)
        max_sim = sims_v2.max(axis=1)
        nearest_mean = float(max_sim.mean())
        nearest_9999 = float((max_sim >= 0.9999).mean())
        nearest_95 = float((max_sim >= 0.95).mean())
        print(f"  nearest-v2 cosine over subsample: mean={max_sim.mean():.4f}  "
              f">=0.9999 ratio={float((max_sim >= 0.9999).mean()):.4f}  "
              f">=0.95 ratio={float((max_sim >= 0.95).mean()):.4f}")
    else:
        nearest_mean = float("nan")
        nearest_9999 = float("nan")
        nearest_95 = float("nan")
        print("  [warn] v2 matrix missing for nearest-v2 stat")

    return {
        "scene": scene_dir.name,
        "N": int(cv_new.shape[0]),
        "means_match": bool(np.allclose(means_in, means_new)),
        "changed_rows": int(changed.sum()),
        "changed_ratio": changed_ratio,
        "unique_probe": int(unique_keys),
        "unique_ratio": float(unique_keys / max(1, n)),
        "cluster_median": cluster_median,
        "cluster_max": cluster_max,
        "nearest_mean": nearest_mean,
        "nearest_9999": nearest_9999,
        "nearest_95": nearest_95,
    }


def discover_scenes(root: Path, seed_tag: str):
    scenes = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.endswith(seed_tag):
            continue
        if (p / "params.npz").is_file():
            scenes.append(p.name[: -len(seed_tag)])
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="dense_v2 (or final) group dir")
    ap.add_argument("--new_root", required=True, help="inst_aligned_v3 group dir")
    ap.add_argument("--scene", default=None, help="scene id (without _seed4_clip)")
    ap.add_argument("--all", action="store_true", help="verify all scenes found under new_root")
    ap.add_argument("--scenes_file", default=None, help="newline-delimited scene ids")
    ap.add_argument("--basedir", default=str(_PROJ_ROOT / "configs/data/habitat_val"))
    ap.add_argument("--seed_tag", default="_seed4_clip")
    ap.add_argument("--sample", type=int, default=50_000)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    new_root = Path(args.new_root)

    if args.all:
        scenes = discover_scenes(new_root, args.seed_tag)
    elif args.scenes_file:
        scenes = [s.strip() for s in open(args.scenes_file) if s.strip()]
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("需要指定 --scene、--scenes_file 或 --all")

    print(f"[verify] n_scenes={len(scenes)} sample={args.sample}")
    summaries = []
    for i, scene in enumerate(scenes, 1):
        scene_dir = Path(args.basedir) / scene
        in_run_dir = in_root / f"{scene}{args.seed_tag}"
        new_run_dir = new_root / f"{scene}{args.seed_tag}"
        print(f"\n[{i}/{len(scenes)}] scene={scene}")
        print(f"  in:    {in_run_dir / 'params.npz'}")
        print(f"  new:   {new_run_dir / 'params.npz'}")
        print(f"  scene_dir: {scene_dir}")
        s = verify_scene(in_run_dir, new_run_dir, scene_dir, sample=args.sample)
        if s is not None:
            summaries.append(s)

    if summaries:
        print("\n[verify summary]")
        print(f"  scenes_ok={len(summaries)}/{len(scenes)}")
        print(f"  means_match_all={all(s['means_match'] for s in summaries)}")
        print(f"  changed_ratio mean={np.mean([s['changed_ratio'] for s in summaries]) * 100:.2f}% "
              f"min={np.min([s['changed_ratio'] for s in summaries]) * 100:.2f}% "
              f"max={np.max([s['changed_ratio'] for s in summaries]) * 100:.2f}%")
        print(f"  unique_ratio mean={np.mean([s['unique_ratio'] for s in summaries]) * 100:.2f}%")
        print(f"  cluster_median mean={np.nanmean([s['cluster_median'] for s in summaries]):.2f}m "
              f"median={np.nanmedian([s['cluster_median'] for s in summaries]):.2f}m")
        print(f"  nearest-v2 mean={np.nanmean([s['nearest_mean'] for s in summaries]):.4f} "
              f">=0.95 mean={np.nanmean([s['nearest_95'] for s in summaries]):.4f}")


if __name__ == "__main__":
    main()
