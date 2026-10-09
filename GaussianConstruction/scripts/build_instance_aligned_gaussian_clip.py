#!/usr/bin/env python3
"""build_instance_aligned_gaussian_clip.py

OFFLINE post-processing: replace per-Gaussian CLIP vectors in params.npz with
the *instance-aligned* CLIP vectors from `instance_clip_vectors_v2.json`,
unifying SG (params.npz['clip_vectors']) and OV (instance_clip_vectors_v2.json)
to the same semantic source.

What it does
------------
For every scene:
  1. Load means3D and the existing clip_vectors (kept as fallback).
  2. Load `instance_retrieval.json` and `instance_clip_vectors_v2.json`.
  3. For every keyframe (HabitatDataset frame fi -> "rgb/frame_XXXXX.png"):
       a. Project all Gaussians into the frame using GT pose + intrinsics
          (with optional GT-depth z-test, identical to recompute_clip_vectors.py).
       b. Build an `inst_label_map[H, W]` where each pixel either is -1
          (background) or carries a global entry id encoded as
          (coco_idx * 100000 + entry_idx_in_v2). The mask->entry mapping is
          *bit-identical* to `_select_masks` from build_instance_clip_vectors_v2.py.
       c. Look up the entry id under each visible Gaussian's pixel.
       d. Accumulate a top-K voting [N, K] table on GPU.
  4. Aggregate: for each Gaussian, take its top-1 entry (or mean of top-K),
     write the corresponding v2 CLIP vector to new_clip_vectors[g].
  5. Gaussians that never landed inside any seg_semantic mask keep their
     original (dense_v2) vector. This guarantees a graceful fallback even if
     this script's coverage is partial.

Output
------
out_root/<scene>_seed4_clip/params.npz with the SAME schema as the input but
clip_vectors replaced. Geometry/RGB/opacity/scales/poses are identical.

Usage
-----
  # Sanity test on yr17:
  python scripts/build_instance_aligned_gaussian_clip.py \
      --in_root  experiments/habitat_val_dense_clip_v2 \
      --out_root experiments/habitat_val_inst_aligned_v3 \
      --scene yr17PDCnDDW --verbose

  # All scenes:
  python scripts/build_instance_aligned_gaussian_clip.py \
      --in_root  experiments/habitat_val_dense_clip_v2 \
      --out_root experiments/habitat_val_inst_aligned_v3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from datasets.gradslam_datasets.habitat import HabitatDataset  # noqa: E402

# Reuse the *exact* mask/entry alignment rule from the v2 builder so SG and OV
# are guaranteed to be sourced from identical (frame, instance) units.
_BUILD_V2_PATH = _PROJ_ROOT.parent / "GaussianNavigation/gaussian_utils/build_instance_clip_vectors_v2.py"
_spec = importlib.util.spec_from_file_location("build_v2", _BUILD_V2_PATH)
_build_v2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_build_v2)  # type: ignore[union-attr]
COCO_IDS: List[int] = list(_build_v2.COCO_IDS)
_select_masks = _build_v2._select_masks  # type: ignore[attr-defined]
resolve_seg_path = _build_v2.resolve_seg_path  # type: ignore[attr-defined]

CLIP_DIM = 512
IMG_H, IMG_W = 480, 640
ENTRY_ENCODE_BASE = 100_000  # encodes (coco_idx, entry_idx) into one int64

# ----------------------------- helpers -------------------------------------- #


def _parse_keyframe_id_from_filepath(file_path: str) -> Optional[str]:
    """rgb/frame_00683.png -> '00683'."""
    name = os.path.basename(file_path)
    if not name.startswith("frame_") or not name.endswith(".png"):
        return None
    return name[len("frame_") : -len(".png")]


def _parse_fid_for_v2_alignment(ent: Any) -> Any:
    """Replicate v2's frame-id extraction from one instance_retrieval entry."""
    if isinstance(ent, list) and len(ent) >= 2 and isinstance(ent[1], (int, str)):
        return ent[1]
    if isinstance(ent, list) and len(ent) >= 3 and isinstance(ent[2], (int, str)):
        return ent[2]
    if isinstance(ent, list):
        for item in ent:
            if isinstance(item, (int, str)):
                return item
    return None


def _build_v2_lookup(
    instance_retrieval: Dict[str, List[Any]],
) -> Dict[Tuple[int, str], List[int]]:
    """For each (coco_id, frame_id_str) -> ordered list of entry indices into v2.

    Frame ids are normalised to *string* (no leading-zero changes) since v2's
    `parse_fid` keeps the original type; instance_retrieval files in this repo
    already store them as zero-padded strings like "00683".
    """
    out: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for c in COCO_IDS:
        entries = instance_retrieval.get(str(c), []) or []
        for i, ent in enumerate(entries):
            fid = _parse_fid_for_v2_alignment(ent)
            if fid is None:
                continue
            out[(c, str(fid))].append(i)
    return out


def _encode_entry(coco_idx: int, entry_idx: int) -> int:
    return coco_idx * ENTRY_ENCODE_BASE + entry_idx + 1  # +1 keeps 0 free for "background"


def _decode_entry(code: int) -> Tuple[int, int]:
    code = int(code) - 1
    return code // ENTRY_ENCODE_BASE, code % ENTRY_ENCODE_BASE


def _build_inst_label_map(
    seg: Dict[Any, Any],
    keyframe_id: str,
    coco_fid_to_entries: Dict[Tuple[int, str], List[int]],
    H: int = IMG_H,
    W: int = IMG_W,
) -> Optional[np.ndarray]:
    """Per-pixel encoded entry id (background = 0). Returns int64 (H, W).

    Conflict policy: when multiple instance masks overlap on the same pixel,
    the first written wins (early-coco-id, then by entries' v2 order). This
    matches the typical case where only the largest-area mask is used and
    the few overlapping pixels are negligible for SG voting.
    """
    label = np.zeros((H, W), dtype=np.int64)
    any_label = False
    for coco_idx, c in enumerate(COCO_IDS):
        raw_masks = seg.get(c) or []
        masks: List[np.ndarray] = []
        for m in raw_masks:
            if (
                isinstance(m, np.ndarray)
                and m.dtype == bool
                and m.shape == (H, W)
            ):
                masks.append(m)
        if not masks:
            continue
        entry_idxs = coco_fid_to_entries.get((c, keyframe_id), [])
        if not entry_idxs:
            continue
        slot_to_mask = _select_masks(masks, len(entry_idxs))
        for slot, e_idx in enumerate(entry_idxs):
            m = slot_to_mask[slot]
            if m is None:
                continue
            code = _encode_entry(coco_idx, e_idx)
            empty = label == 0
            wrote = m & empty
            if wrote.any():
                label[wrote] = code
                any_label = True
    return label if any_label else None


# ------------------------------ main proc ----------------------------------- #


@torch.no_grad()
def process_scene(
    in_run_dir: Path,
    out_run_dir: Path,
    scene_dir: Path,
    scene: str,
    basedir: str,
    device: torch.device,
    use_topk: int = 5,
    aggregate_topk: int = 3,
    depth_tol: Optional[float] = 0.5,
    progress_every: int = 50,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Process a single scene end-to-end."""
    params_path = in_run_dir / "params.npz"
    if not params_path.is_file():
        print(f"  [skip] missing {params_path}")
        return None

    ir_path = scene_dir / "instance_retrieval.json"
    v2_path = scene_dir / "instance_clip_vectors_v2.json"
    if not ir_path.is_file() or not v2_path.is_file():
        print(f"  [skip] missing instance_retrieval/v2 in {scene_dir}")
        return None

    # ---- load v2 vectors and build (coco, frame_id) -> ordered entry indices.
    with open(ir_path, "r") as f:
        instance_retrieval = json.load(f)
    with open(v2_path, "r") as f:
        v2_vectors = json.load(f)

    v2_mat: Dict[int, np.ndarray] = {}
    for c in COCO_IDS:
        arr = np.array(v2_vectors.get(str(c), []), dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((0, CLIP_DIM), dtype=np.float32)
        elif arr.shape[1] != CLIP_DIM:
            print(f"  [warn] v2[{c}] dim={arr.shape[1]} != {CLIP_DIM}; treating as empty")
            arr = np.zeros((0, CLIP_DIM), dtype=np.float32)
        v2_mat[c] = arr

    coco_fid_to_entries = _build_v2_lookup(instance_retrieval)

    # ---- load Gaussian geometry and old clip vectors ------------------------
    data = dict(np.load(params_path, allow_pickle=True))
    means3D_np = np.asarray(data["means3D"], dtype=np.float32)
    N = means3D_np.shape[0]
    if "clip_vectors" in data:
        old_clip = np.asarray(data["clip_vectors"], dtype=np.float32)
        if old_clip.shape != (N, CLIP_DIM):
            print(f"  [warn] existing clip_vectors shape {old_clip.shape} != ({N},{CLIP_DIM}); treating as zeros")
            old_clip = np.zeros((N, CLIP_DIM), dtype=np.float32)
    else:
        old_clip = np.zeros((N, CLIP_DIM), dtype=np.float32)

    means3D = torch.from_numpy(means3D_np).to(device)              # [N, 3]
    means3D_h = torch.cat(
        [means3D, torch.ones((N, 1), device=device, dtype=torch.float32)], dim=1
    )                                                              # [N, 4]

    # ---- top-K voting buffers (per Gaussian) --------------------------------
    # Stores up to `use_topk` distinct entry codes seen, with their counts.
    top_codes = torch.zeros((N, use_topk), dtype=torch.int64, device=device)
    top_cnts = torch.zeros((N, use_topk), dtype=torch.int32, device=device)

    dataset = HabitatDataset(
        basedir=basedir,
        sequence=scene,
        start=0,
        end=-1,
        stride=1,
        desired_height=IMG_H,
        desired_width=IMG_W,
        device=device,
        relative_pose=True,
        ignore_bad=False,
        use_train_split=True,
    )
    F = len(dataset)
    if verbose:
        print(f"  scene={scene}  N={N:,}  frames={F}  depth_tol={depth_tol}  K={use_topk}")

    # ---- iterate frames -----------------------------------------------------
    n_frames_with_label = 0
    n_visible_total = 0
    n_hits_total = 0
    t0 = time.time()
    for fi in range(F):
        # 1) keyframe id
        try:
            file_path = dataset.frames_metadata[fi]["file_path"]
        except Exception:
            continue
        kf_id = _parse_keyframe_id_from_filepath(file_path)
        if kf_id is None:
            continue

        # 2) load seg_semantic
        seg_p = resolve_seg_path(scene_dir, kf_id)
        if seg_p is None or not seg_p.exists():
            if verbose and fi < 3:
                print(f"    [{fi+1:4d}/{F}] no seg for {kf_id}")
            continue
        try:
            seg = np.load(seg_p, allow_pickle=True).item()
        except Exception as e:
            print(f"    [warn] seg load failed {seg_p}: {e}")
            continue

        # 3) build per-pixel entry label map; skip frame if no relevant labels
        label_np = _build_inst_label_map(seg, kf_id, coco_fid_to_entries, H=IMG_H, W=IMG_W)
        if label_np is None:
            continue
        n_frames_with_label += 1
        label_t = torch.from_numpy(label_np).to(device)            # int64 (H, W)

        # 4) project Gaussians (same logic as recompute_clip_vectors.py)
        color, depth, _seg, intrinsics, pose = dataset[fi]
        if color.dim() == 3 and color.shape[-1] == 3:
            color_chw = color.permute(2, 0, 1).contiguous()
        elif color.dim() == 3 and color.shape[0] == 3:
            color_chw = color
        else:
            raise RuntimeError(f"unexpected color shape {tuple(color.shape)}")
        H, W = color_chw.shape[1], color_chw.shape[2]
        depth_hw = depth.squeeze(-1) if (depth.dim() == 3 and depth.shape[-1] == 1) else depth.squeeze(0)
        depth_hw = depth_hw.float()

        w2c = torch.linalg.inv(pose.float())
        cam_pts = (w2c @ means3D_h.T).T[:, :3]
        z = cam_pts[:, 2]
        in_front = z > 1e-3
        fx, fy = intrinsics[0, 0].float(), intrinsics[1, 1].float()
        cx, cy = intrinsics[0, 2].float(), intrinsics[1, 2].float()
        u = (cam_pts[:, 0] * fx) / torch.clamp(z, min=1e-6) + cx
        v = (cam_pts[:, 1] * fy) / torch.clamp(z, min=1e-6) + cy
        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front

        if depth_tol is not None and depth_tol > 0:
            ui = torch.clamp(u.long(), 0, W - 1)
            vi = torch.clamp(v.long(), 0, H - 1)
            d_gt = depth_hw[vi, ui]
            valid_d = (d_gt > 1e-3) & ((z - d_gt).abs() <= depth_tol)
            visible = in_image & valid_d
        else:
            visible = in_image

        n_vis = int(visible.sum().item())
        if n_vis == 0:
            if verbose and ((fi % progress_every == 0) or fi == F - 1):
                elapsed = time.time() - t0
                print(f"    [{fi+1:4d}/{F}] visible=0  elapsed={elapsed:.1f}s")
            continue

        vis_idx = torch.where(visible)[0]
        u_vis = torch.clamp(u[vis_idx].long(), 0, W - 1)
        v_vis = torch.clamp(v[vis_idx].long(), 0, H - 1)
        codes_vis = label_t[v_vis, u_vis]                          # int64 (n_vis,)
        hit_mask = codes_vis > 0
        if not hit_mask.any():
            if verbose and ((fi % progress_every == 0) or fi == F - 1):
                elapsed = time.time() - t0
                print(f"    [{fi+1:4d}/{F}] visible={n_vis}  hits=0  elapsed={elapsed:.1f}s")
            continue
        sel = vis_idx[hit_mask]                                    # (m,)
        codes_sel = codes_vis[hit_mask]                            # (m,)
        m = sel.shape[0]
        n_visible_total += n_vis
        n_hits_total += m

        # 5) update top-K voting on GPU --------------------------------------
        cur_top_codes = top_codes[sel]                             # (m, K)
        cur_top_cnts = top_cnts[sel]                               # (m, K)
        eq = cur_top_codes == codes_sel.unsqueeze(1)               # (m, K)
        already = eq.any(dim=1)                                    # (m,)
        slot_already = eq.float().argmax(dim=1)                    # (m,) valid where already
        # smallest-count slot (ties broken by lowest index)
        min_cnt, slot_min = cur_top_cnts.min(dim=1)                # (m,)
        # for not-already entries: replace if min_cnt == 0 (empty slot exists)
        replaceable = (~already) & (min_cnt == 0)
        # final slot to update: already -> slot_already; replaceable -> slot_min; else: -1
        do_update = already | replaceable
        slot_idx = torch.where(already, slot_already, slot_min)    # (m,)

        # gather current cnt at slot_idx
        cur_cnt_at_slot = cur_top_cnts.gather(1, slot_idx.unsqueeze(1)).squeeze(1)
        new_cnt_at_slot = torch.where(
            already, cur_cnt_at_slot + 1, torch.ones_like(cur_cnt_at_slot)
        )

        # apply only where do_update is True; else keep
        write_mask = do_update                                      # (m,)
        if write_mask.any():
            sel_w = sel[write_mask]
            slot_w = slot_idx[write_mask]
            codes_w = codes_sel[write_mask]
            cnt_w = new_cnt_at_slot[write_mask]
            # scatter back per (g, slot)
            top_codes_rows = top_codes[sel_w]                      # (mw, K)
            top_cnts_rows = top_cnts[sel_w]                        # (mw, K)
            top_codes_rows.scatter_(1, slot_w.unsqueeze(1), codes_w.unsqueeze(1))
            top_cnts_rows.scatter_(1, slot_w.unsqueeze(1), cnt_w.unsqueeze(1).to(torch.int32))
            top_codes[sel_w] = top_codes_rows
            top_cnts[sel_w] = top_cnts_rows

        if verbose and ((fi % progress_every == 0) or fi == F - 1):
            elapsed = time.time() - t0
            print(
                f"    [{fi+1:4d}/{F}] vis={n_vis:>7d}  hits={m:>7d}  "
                f"frames_with_label={n_frames_with_label}  elapsed={elapsed:.1f}s"
            )

    # ------------------------ aggregate to new clip --------------------------
    new_clip = old_clip.copy()  # fallback for never-hit Gaussians
    top_codes_cpu = top_codes.detach().cpu().numpy()
    top_cnts_cpu = top_cnts.detach().cpu().numpy()

    has_any = (top_cnts_cpu > 0).any(axis=1)
    n_hit_g = int(has_any.sum())

    if n_hit_g > 0:
        # Sort each row by count descending; take aggregate_topk codes
        # We do it row-wise vectorised on CPU to keep memory low.
        order = np.argsort(-top_cnts_cpu, axis=1, kind="stable")
        sorted_codes = np.take_along_axis(top_codes_cpu, order, axis=1)  # (N, K)
        sorted_cnts = np.take_along_axis(top_cnts_cpu, order, axis=1)    # (N, K)

        # For each Gaussian with hits, average up to `aggregate_topk` v2 vectors
        K_use = min(aggregate_topk, top_codes_cpu.shape[1])
        for g in np.where(has_any)[0]:
            vecs: List[np.ndarray] = []
            for k in range(K_use):
                cnt = int(sorted_cnts[g, k])
                if cnt <= 0:
                    break
                code = int(sorted_codes[g, k])
                if code <= 0:
                    continue
                coco_idx, e_idx = _decode_entry(code)
                if coco_idx < 0 or coco_idx >= len(COCO_IDS):
                    continue
                c = COCO_IDS[coco_idx]
                mat = v2_mat.get(c)
                if mat is None or e_idx < 0 or e_idx >= mat.shape[0]:
                    continue
                v = mat[e_idx]
                if np.linalg.norm(v) <= 1e-8:
                    continue
                vecs.append(v)
            if not vecs:
                continue
            mean_v = np.mean(vecs, axis=0).astype(np.float32)
            n = float(np.linalg.norm(mean_v))
            if n <= 1e-8:
                continue
            new_clip[g] = mean_v / n

    coverage = float(n_hit_g) / float(N) if N > 0 else 0.0
    fallback = N - n_hit_g

    # ----------------------------- save out ---------------------------------- #
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_run_dir / "params.npz"
    save_dict = {k: data[k] for k in data.keys() if k != "clip_vectors"}
    save_dict["clip_vectors"] = new_clip.astype(np.float32)
    np.savez(out_path, **save_dict)

    print(
        f"  -> {out_path}\n"
        f"     N={N}  hit_with_v2={n_hit_g} ({coverage * 100:.2f}%)  "
        f"fallback_kept_old={fallback}\n"
        f"     frames_with_label={n_frames_with_label}/{F}  "
        f"visible_total={n_visible_total}  hits_total={n_hits_total}"
    )
    return {
        "scene": scene,
        "N": N,
        "hit_with_v2": n_hit_g,
        "coverage": coverage,
        "frames_with_label": n_frames_with_label,
        "frames_total": F,
        "out": str(out_path),
    }


def discover_scenes(in_root: Path, scene_arg: Optional[str], scenes_file: Optional[str], seed_tag: str) -> List[str]:
    if scene_arg:
        return [scene_arg]
    if scenes_file:
        return [s.strip() for s in open(scenes_file) if s.strip()]
    out = []
    for p in sorted(in_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.endswith(seed_tag):
            continue
        if not (p / "params.npz").is_file():
            continue
        out.append(p.name[: -len(seed_tag)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True,
                    help="experiments/habitat_val_dense_clip_v2 (or habitat_val_final)")
    ap.add_argument("--out_root", required=True,
                    help="output group dir (e.g. experiments/habitat_val_inst_aligned_v3)")
    ap.add_argument("--basedir",
                    default=str(_PROJ_ROOT / "configs/data/habitat_val"),
                    help="HabitatDataset basedir (contains transforms.json per scene)")
    ap.add_argument("--scene", default=None, help="single scene id (without _seed4_clip)")
    ap.add_argument("--scenes_file", default=None, help="newline-delimited scene ids")
    ap.add_argument("--seed_tag", default="_seed4_clip")
    ap.add_argument("--depth_tol", type=float, default=0.5)
    ap.add_argument("--top_k_track", type=int, default=5,
                    help="capacity of the per-Gaussian top-K voting buffer")
    ap.add_argument("--aggregate_topk", type=int, default=3,
                    help="how many top-K entries' v2 vectors to average for the final per-Gaussian vec")
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inst_aligned] device={device}")
    print(f"[inst_aligned] in_root={in_root}")
    print(f"[inst_aligned] out_root={out_root}")
    print(f"[inst_aligned] basedir={args.basedir}")

    scenes = discover_scenes(in_root, args.scene, args.scenes_file, args.seed_tag)
    print(f"[inst_aligned] scenes={len(scenes)}")

    t_all = time.time()
    summaries = []
    for si, scene in enumerate(scenes, 1):
        in_run_dir = in_root / f"{scene}{args.seed_tag}"
        out_run_dir = out_root / f"{scene}{args.seed_tag}"
        out_npz = out_run_dir / "params.npz"
        if args.skip_existing and out_npz.is_file():
            print(f"[{si}/{len(scenes)}] {scene}: skip (exists)")
            continue
        scene_dir = Path(args.basedir) / scene
        if not scene_dir.is_dir():
            print(f"[{si}/{len(scenes)}] {scene}: skip (basedir/scene not found: {scene_dir})")
            continue
        print(f"[{si}/{len(scenes)}] {scene}: start")
        try:
            r = process_scene(
                in_run_dir=in_run_dir,
                out_run_dir=out_run_dir,
                scene_dir=scene_dir,
                scene=scene,
                basedir=args.basedir,
                device=device,
                use_topk=args.top_k_track,
                aggregate_topk=args.aggregate_topk,
                depth_tol=args.depth_tol if args.depth_tol > 0 else None,
                progress_every=args.progress_every,
                verbose=args.verbose,
            )
            if r is not None:
                summaries.append(r)
        except KeyboardInterrupt:
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[{si}/{len(scenes)}] {scene}: FAILED")
            continue

    total = time.time() - t_all
    print(f"\n[inst_aligned] DONE in {total:.1f}s for {len(scenes)} scene(s)")
    if summaries:
        cov = np.mean([s["coverage"] for s in summaries])
        print(f"[inst_aligned] avg instance-coverage = {cov * 100:.2f}%")


if __name__ == "__main__":
    main()
