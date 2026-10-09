#!/usr/bin/env python3
"""build_instance_clip_vectors_v2.py
Per-instance CLIP encoding using `seg_semantic/frame_XXXXX.npy` masks.

Why this exists
---------------
`build_instance_clip_vectors_fixed.py` (v1) re-encodes the *entire frame* for every
instance entry in `instance_retrieval.json`. As a consequence every instance that
appears in the same frame ends up with an identical CLIP vector, which collapses
the semantic component of the open-vocabulary retriever to "frame-level".

This v2 script crops each instance from its frame using the per-category masks
stored in `seg_semantic/frame_XXXXX.npy`, then encodes the crop with CLIP. The
output JSON has the same structure / key order as v1 so it is a drop-in for
`OpenVocabRetriever.load_scene` once the filename is switched.

Matching rules (entries -> seg masks) per (frame, coco_id):
  - 0 mask in seg, N entries -> all entries get the full-frame embedding (rare;
    typically means seg/instance_retrieval are out of sync).
  - M >= 1 masks, 1 entry  -> use the LARGEST mask (covers the most pixels).
  - M masks, N==M entries  -> match each entry to a mask by `j_local` order.
  - M masks, N >  M entries -> use the largest mask for the extras.
  - M masks, N <  M entries -> use the top-N masks by pixel count.
The chosen mask is converted to a 2D bbox, padded by `--pad` (default 30%), and
the corresponding RGB crop is encoded with CLIP.

Output: `instance_clip_vectors_v2.json` next to the existing `instance_clip_vectors.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

COCO_IDS = [56, 57, 58, 59, 61, 62]
CLIP_DIM = 512
DEFAULT_MODEL = "ViT-B-32"
IMG_H, IMG_W = 480, 640


class CLIPEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda",
                 clip_pretrained_path: Optional[str] = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.clip_pretrained_path = (clip_pretrained_path or "").strip() or None
        self.model = None
        self.preprocess = None
        self.n_ok = 0
        self.n_fail = 0

    def _ensure(self):
        if self.model is not None:
            return
        import open_clip
        pretrained = self.clip_pretrained_path if (
            self.clip_pretrained_path and os.path.isfile(self.clip_pretrained_path)
        ) else "openai"
        print(f"[v2] loading CLIP {self.model_name} (pretrained={pretrained}) on {self.device}")
        m, _, pp = open_clip.create_model_and_transforms(self.model_name, pretrained=pretrained)
        self.model = m.to(self.device).eval()
        self.preprocess = pp

    def encode_pil(self, pil_img: Image.Image) -> List[float]:
        self._ensure()
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x).cpu().numpy().astype(np.float32).squeeze(0)
        n = float(np.linalg.norm(f))
        if n > 1e-8:
            f = f / n
            self.n_ok += 1
            return f.tolist()
        self.n_fail += 1
        return [0.0] * CLIP_DIM


def resolve_rgb_path(scene_dir: Path, frame_id: Any) -> Path:
    s = str(frame_id)
    if s.startswith("frame_") and s.endswith(".png"):
        return scene_dir / "rgb" / s
    if s.startswith("frame_"):
        s = s[6:]
    if s.endswith(".png"):
        s = s[:-4]
    cands = [f"frame_{s}.png", f"frame_{s.zfill(5)}.png", f"frame_{s.zfill(6)}.png"]
    if s.isdigit():
        cands += [f"{s}.png", f"{s.zfill(5)}.png", f"{s.zfill(6)}.png"]
    for c in cands:
        p = scene_dir / "rgb" / c
        if p.exists():
            return p
    return scene_dir / "rgb" / f"frame_{s}.png"


def resolve_seg_path(scene_dir: Path, frame_id: Any) -> Optional[Path]:
    s = str(frame_id)
    if s.startswith("frame_"):
        s = s[6:]
    if s.endswith(".png"):
        s = s[:-4]
    cands = [f"frame_{s}.npy", f"frame_{s.zfill(5)}.npy", f"frame_{s.zfill(6)}.npy"]
    for c in cands:
        p = scene_dir / "seg_semantic" / c
        if p.exists():
            return p
    return None


def _bbox_of_mask(mask: np.ndarray, pad_ratio: float, hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = hw
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    py = int(h * pad_ratio)
    px = int(w * pad_ratio)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(W - 1, x2 + px),
        min(H - 1, y2 + py),
    )


def _select_masks(masks: List[np.ndarray], n_entries: int) -> List[Optional[np.ndarray]]:
    """Pick one mask per entry. Returns list length n_entries; None means no mask."""
    if not masks:
        return [None] * n_entries
    sizes = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: -sizes[i])
    largest = masks[order[0]]
    if n_entries == 1:
        return [largest]
    if len(masks) >= n_entries:
        return [masks[i] for i in order[:n_entries]]
    out: List[Optional[np.ndarray]] = list(masks)
    while len(out) < n_entries:
        out.append(largest)
    return out


def process_scene(scene_dir: Path, encoder: CLIPEncoder, pad_ratio: float, out_name: str,
                  verbose: bool = False) -> Optional[Dict[str, Any]]:
    ir_path = scene_dir / "instance_retrieval.json"
    rgb_dir = scene_dir / "rgb"
    if not ir_path.is_file() or not rgb_dir.is_dir():
        if verbose:
            print(f"  [skip] {scene_dir.name}: missing instance_retrieval.json or rgb/")
        return None
    try:
        with open(ir_path, "r") as f:
            instance_retrieval = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [skip] {scene_dir.name}: ir parse error: {e}")
        return None

    out: Dict[str, List[List[float]]] = {str(c): [] for c in COCO_IDS}
    seg_cache: Dict[str, Dict[int, Any]] = {}
    vec_cache: Dict[Tuple[str, int, int, int, int], List[float]] = {}
    full_cache: Dict[str, List[float]] = {}

    n_total = 0
    n_mask = 0
    n_full = 0

    for coco_id in COCO_IDS:
        key = str(coco_id)
        entries = instance_retrieval.get(key, [])
        if not entries:
            out[key] = []
            continue

        frame_to_idxs: Dict[Any, List[int]] = defaultdict(list)
        for i, ent in enumerate(entries):
            fid = None
            if len(ent) >= 2 and isinstance(ent[1], (int, str)):
                fid = ent[1]
            elif len(ent) >= 3 and isinstance(ent[2], (int, str)):
                fid = ent[2]
            else:
                for item in ent:
                    if isinstance(item, (int, str)):
                        fid = item
                        break
            frame_to_idxs[fid].append(i)

        vectors: List[Optional[List[float]]] = [None] * len(entries)

        for fr, idxs in frame_to_idxs.items():
            rgb_p = resolve_rgb_path(scene_dir, fr)
            seg_p = resolve_seg_path(scene_dir, fr)

            if not rgb_p.exists():
                for j in idxs:
                    vectors[j] = [0.0] * CLIP_DIM
                continue

            try:
                rgb_pil = Image.open(rgb_p).convert("RGB")
            except Exception:
                for j in idxs:
                    vectors[j] = [0.0] * CLIP_DIM
                continue

            seg_key = str(rgb_p)
            seg = seg_cache.get(seg_key)
            if seg is None and seg_p is not None and seg_p.exists():
                try:
                    seg = np.load(seg_p, allow_pickle=True).item()
                except Exception:
                    seg = {}
                seg_cache[seg_key] = seg
            seg = seg or {}

            raw_masks = seg.get(coco_id) or []
            masks: List[np.ndarray] = []
            for m in raw_masks:
                if m is None:
                    continue
                if isinstance(m, np.ndarray) and m.dtype == bool and m.shape == (IMG_H, IMG_W):
                    masks.append(m)
            assigned = _select_masks(masks, len(idxs))

            for slot, j in enumerate(idxs):
                m = assigned[slot]
                box = _bbox_of_mask(m, pad_ratio, (IMG_H, IMG_W)) if m is not None else None
                if box is None or (box[2] - box[0] < 4) or (box[3] - box[1] < 4):
                    v = full_cache.get(seg_key)
                    if v is None:
                        v = encoder.encode_pil(rgb_pil)
                        full_cache[seg_key] = v
                    vectors[j] = v
                    n_full += 1
                else:
                    ck = (seg_key, box[0], box[1], box[2], box[3])
                    v = vec_cache.get(ck)
                    if v is None:
                        crop = rgb_pil.crop((box[0], box[1], box[2] + 1, box[3] + 1))
                        v = encoder.encode_pil(crop)
                        vec_cache[ck] = v
                    vectors[j] = v
                    n_mask += 1
                n_total += 1

        out[key] = [v if v is not None else [0.0] * CLIP_DIM for v in vectors]

    out_path = scene_dir / out_name
    with open(out_path, "w") as f:
        json.dump(out, f)

    if verbose:
        ratio = (n_mask / max(1, n_total)) * 100.0
        print(f"  {scene_dir.name}: total={n_total} mask={n_mask} ({ratio:.1f}%) full={n_full} -> {out_path.name}")

    return {
        "scene": scene_dir.name,
        "total": n_total,
        "mask": n_mask,
        "full": n_full,
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root", type=str,
                    default="../GaussianConstruction/configs/data/habitat_val",
                    help="root directory containing per-scene subdirectories")
    ap.add_argument("--scene", type=str, default=None,
                    help="run on a single scene name (relative to scenes_root) or absolute path")
    ap.add_argument("--scenes", type=str, default=None,
                    help="comma-separated list of scene names to process (subset of scenes_root)")
    ap.add_argument("--clip_model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--clip_pretrained_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--pad", type=float, default=0.30,
                    help="bbox padding ratio added on each side; default 0.30")
    ap.add_argument("--out_name", type=str, default="instance_clip_vectors_v2.json",
                    help="output json filename inside each scene dir")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-encode even if the output file already exists")
    args = ap.parse_args()

    encoder = CLIPEncoder(args.clip_model, args.device, args.clip_pretrained_path)

    if args.scene:
        p = Path(args.scene)
        if not p.is_absolute():
            p = Path(args.scenes_root) / args.scene
        if not p.is_dir():
            print(f"scene not found: {p}")
            sys.exit(2)
        scenes = [p]
    elif args.scenes:
        names = [n.strip() for n in args.scenes.split(",") if n.strip()]
        scenes = [Path(args.scenes_root) / n for n in names]
    else:
        scenes = sorted(
            d for d in Path(args.scenes_root).iterdir()
            if d.is_dir() and (d / "instance_retrieval.json").exists()
        )

    print(f"[v2] scenes_root={args.scenes_root}  n_scenes={len(scenes)}  pad={args.pad}  out={args.out_name}")

    results = []
    for sd in tqdm(scenes, desc="scenes"):
        out_p = sd / args.out_name
        if out_p.exists() and not args.overwrite:
            if args.verbose:
                print(f"  [skip exists] {sd.name} -> {out_p.name}")
            continue
        try:
            r = process_scene(sd, encoder, args.pad, args.out_name, verbose=args.verbose)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  [ERR] {sd.name}: {e}")
            continue
        if r is not None:
            results.append(r)

    n_total = sum(r["total"] for r in results)
    n_mask = sum(r["mask"] for r in results)
    n_full = sum(r["full"] for r in results)
    print(
        f"\n[v2] done. scenes_done={len(results)}  total_entries={n_total}  "
        f"mask_crop={n_mask} ({(n_mask / max(1, n_total)) * 100:.1f}%)  full_fallback={n_full}"
    )
    print(f"[v2] CLIP ok={encoder.n_ok}  zero={encoder.n_fail}")


if __name__ == "__main__":
    main()
