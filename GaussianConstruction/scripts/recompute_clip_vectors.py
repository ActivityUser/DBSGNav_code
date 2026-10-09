#!/usr/bin/env python3
"""Recompute per-Gaussian CLIP vectors as a post-processing step.

Why this script exists:
  The original habitat_splatam pipeline writes one image-level CLIP vector
  per keyframe and broadcasts it to every newly-added Gaussian, so the
  per-Gaussian clip_vectors are effectively frame-level (~1000 unique
  vectors across ~1.1M Gaussians, with cos>0.9999 clusters spanning meters
  in 3D). This breaks per-instance semantic retrieval inside a frame.

What this script does:
  Without re-running SLAM, load the existing params.npz, walk through the
  scene's RGB frames again, run a MaskCLIP-style dense forward through the
  OpenCLIP ViT-B-32 visual transformer to get patch tokens [gh*gw, D]
  (post ln_post + visual.proj, i.e. text-aligned), project every Gaussian
  into the current frame using GT pose + intrinsics, look up the patch
  token whose footprint contains each Gaussian's pixel, and accumulate.
  After all frames the per-Gaussian CLIP vector is the mean of all visible
  patch contributions (with an optional GT-depth z-test to avoid occluded
  pollution). Gaussians never visible keep their original (broadcast) vector
  as a safe fallback. All other fields in params.npz are preserved bit-exact.

Output:
  --out_root/<scene>_seed4_clip/params.npz with the SAME schema as the input
  but clip_vectors replaced. Geometry, RGB, opacity, scales, poses are
  identical to the input.

Usage:
  python scripts/recompute_clip_vectors.py \
      --in_root  experiments/habitat_val_final \
      --out_root experiments/habitat_val_dense_clip_v2 \
      --scene yr17PDCnDDW

  # All scenes:
  python scripts/recompute_clip_vectors.py \
      --in_root experiments/habitat_val_final \
      --out_root experiments/habitat_val_dense_clip_v2

  # From a scene list, with depth-test tolerance and explicit GPU:
  CUDA_VISIBLE_DEVICES=7 python scripts/recompute_clip_vectors.py \
      --in_root experiments/habitat_val_final \
      --out_root experiments/habitat_val_dense_clip_v2 \
      --scenes_file /tmp/val_scenes.txt --depth_tol 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from datasets.gradslam_datasets.habitat import HabitatDataset  # noqa: E402
from utils.common_utils import params2cpu  # noqa: E402


def _dense_clip_patch_tokens(visual, inp: torch.Tensor):
    """MaskCLIP-style dense forward through OpenCLIP VisionTransformer.

    Args:
        visual: open_clip model.visual (VisionTransformer).
        inp:    [1, 3, H_in, W_in] preprocessed image tensor on `device`.

    Returns:
        patch:  [1, gh*gw, output_dim] tokens after ln_post + visual.proj.
        gh, gw: int, spatial grid (e.g. 7x7 for ViT-B/32 @ 224).
    """
    x = visual.conv1(inp)                                  # [1, embed_dim, gh, gw]
    gh, gw = x.shape[2], x.shape[3]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                                 # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                                 # LND -> NLD
    x = visual.ln_post(x)
    patch = x[:, 1:, :]                                    # drop CLS
    if getattr(visual, "proj", None) is not None:
        patch = patch @ visual.proj
    return patch, gh, gw


def _load_clip(device: torch.device, model_name: str = "ViT-B-32"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess


def _build_dataset(scene: str, basedir: str, device: torch.device) -> HabitatDataset:
    return HabitatDataset(
        basedir=basedir,
        sequence=scene,
        start=0,
        end=-1,
        stride=1,
        desired_height=480,
        desired_width=640,
        device=device,
        relative_pose=True,
        ignore_bad=False,
        use_train_split=True,
    )


def _patch_indices_for_pixels(
    u: torch.Tensor, v: torch.Tensor, W: int, H: int, gh: int, gw: int
) -> torch.Tensor:
    """Map pixel coordinates to flat patch-token index.

    The patch grid (gh, gw) tiles the (entire) input image uniformly, which
    is what bilinear-upsampling a [gh, gw, D] feature to (H, W) would do
    under nearest-neighbour sampling. Equivalent to MaskCLIP's per-pixel
    patch ownership.
    """
    px = torch.clamp((u.float() / float(W) * gw).long(), 0, gw - 1)
    py = torch.clamp((v.float() / float(H) * gh).long(), 0, gh - 1)
    return py * gw + px


@torch.no_grad()
def recompute_scene_clip_vectors(
    in_run_dir: Path,
    out_run_dir: Path,
    scene: str,
    basedir: str,
    device: torch.device,
    model,
    preprocess,
    clip_dim: int = 512,
    depth_tol: Optional[float] = 0.5,
    progress_every: int = 50,
):
    """Process a single scene end-to-end."""
    params_path = in_run_dir / "params.npz"
    if not params_path.is_file():
        print(f"  [skip] missing {params_path}")
        return None

    data = dict(np.load(params_path, allow_pickle=True))
    means3D_np = np.asarray(data["means3D"], dtype=np.float32)
    N = means3D_np.shape[0]
    if "clip_vectors" in data:
        old_clip = np.asarray(data["clip_vectors"], dtype=np.float32)
        if old_clip.shape != (N, clip_dim):
            print(f"  [warn] existing clip_vectors shape {old_clip.shape} != ({N},{clip_dim}); ignoring")
            old_clip = np.zeros((N, clip_dim), dtype=np.float32)
    else:
        old_clip = np.zeros((N, clip_dim), dtype=np.float32)

    means3D = torch.from_numpy(means3D_np).to(device)              # [N, 3]
    ones = torch.ones((N, 1), device=device, dtype=torch.float32)
    means3D_h = torch.cat([means3D, ones], dim=1)                  # [N, 4]

    accum = torch.zeros((N, clip_dim), device=device, dtype=torch.float32)
    counts = torch.zeros((N,), device=device, dtype=torch.float32)

    dataset = _build_dataset(scene, basedir, device)
    F = len(dataset)
    print(f"  scene={scene}  N={N:,}  frames={F}  depth_tol={depth_tol}")

    t0 = time.time()
    for fi in range(F):
        color, depth, _seg, intrinsics, pose = dataset[fi]
        # color: (H, W, 3) in [0,1] float | (already preprocessed -- channels_last)
        # depth: (H, W, 1)
        # intrinsics: (4, 4); pose: (4, 4) c2w (relative-to-frame0)
        if color.dim() == 3 and color.shape[-1] == 3:
            color_chw = color.permute(2, 0, 1).contiguous()        # (3, H, W)
        elif color.dim() == 3 and color.shape[0] == 3:
            color_chw = color
        else:
            raise RuntimeError(f"unexpected color shape {tuple(color.shape)}")
        H, W = color_chw.shape[1], color_chw.shape[2]
        depth_hw = depth.squeeze(-1) if (depth.dim() == 3 and depth.shape[-1] == 1) else depth.squeeze(0)
        depth_hw = depth_hw.float()                                # (H, W) in meters

        # Project all Gaussians into this frame
        w2c = torch.linalg.inv(pose.float())                       # (4, 4)
        cam_pts = (w2c @ means3D_h.T).T[:, :3]                     # (N, 3) in camera space
        z = cam_pts[:, 2]
        in_front = z > 1e-3
        fx, fy = intrinsics[0, 0].float(), intrinsics[1, 1].float()
        cx, cy = intrinsics[0, 2].float(), intrinsics[1, 2].float()
        u = (cam_pts[:, 0] * fx) / torch.clamp(z, min=1e-6) + cx   # (N,)
        v = (cam_pts[:, 1] * fy) / torch.clamp(z, min=1e-6) + cy   # (N,)
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
            if (fi % progress_every == 0) or fi == F - 1:
                elapsed = time.time() - t0
                print(f"    [{fi+1:4d}/{F}] visible=0   elapsed={elapsed:.1f}s")
            continue

        # CLIP dense forward (once per frame)
        from PIL import Image
        rgb_np = (color_chw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_np)
        inp = preprocess(pil_img).unsqueeze(0).to(device)
        patch, gh, gw = _dense_clip_patch_tokens(model.visual, inp)
        patch_tokens = patch[0]                                    # (gh*gw, D)

        # Map visible Gaussians to patch indices
        vis_idx = torch.where(visible)[0]
        u_vis = u[vis_idx]
        v_vis = v[vis_idx]
        tok_idx = _patch_indices_for_pixels(u_vis, v_vis, W, H, gh, gw)
        feats = patch_tokens.index_select(0, tok_idx)              # (n_vis, D)
        accum.index_add_(0, vis_idx, feats.float())
        counts.index_add_(0, vis_idx, torch.ones_like(vis_idx, dtype=torch.float32))

        if (fi % progress_every == 0) or fi == F - 1:
            elapsed = time.time() - t0
            cov = float((counts > 0).sum().item()) / N
            print(f"    [{fi+1:4d}/{F}] visible={n_vis:>7d}  coverage={cov*100:5.1f}%  elapsed={elapsed:.1f}s")

    # Aggregate: mean of contributions; fall back to old broadcast vector if a Gaussian was never visible
    nonzero = counts > 0
    n_nonzero = int(nonzero.sum().item())
    cov_final = n_nonzero / float(N) if N > 0 else 0.0
    new_clip = torch.from_numpy(old_clip).to(device).clone()
    if n_nonzero > 0:
        new_clip[nonzero] = accum[nonzero] / counts[nonzero].unsqueeze(1)
    new_clip_np = new_clip.detach().cpu().numpy().astype(np.float32)

    # Save: keep everything else identical to the input
    out_run_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_run_dir / "params.npz"
    save_dict = {k: data[k] for k in data.keys() if k != "clip_vectors"}
    save_dict["clip_vectors"] = new_clip_np
    np.savez(out_path, **save_dict)

    print(
        f"  -> wrote {out_path}\n"
        f"     coverage={cov_final*100:.2f}% ({n_nonzero}/{N}); "
        f"fallback (kept old vec) for {N-n_nonzero}"
    )
    return out_path


def discover_scenes(in_root: Path, scene_arg: Optional[str], scenes_file: Optional[str]) -> List[str]:
    """List of scene ids (without _seed4_clip suffix)."""
    if scene_arg:
        return [scene_arg]
    if scenes_file:
        return [s.strip() for s in open(scenes_file) if s.strip()]
    scenes = []
    for p in sorted(in_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.endswith("_seed4_clip"):
            continue
        if not (p / "params.npz").is_file():
            continue
        scenes.append(p.name[: -len("_seed4_clip")])
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="experiments/habitat_val_final 或类似目录")
    ap.add_argument("--out_root", required=True, help="输出 group 目录")
    ap.add_argument("--basedir", default=str(_PROJ_ROOT / "configs/data/habitat_val"),
                    help="HabitatDataset 输入根目录 (含每个 scene 的 transforms.json)")
    ap.add_argument("--scene", default=None, help="只跑单场景 (scene id, 不含 _seed4_clip)")
    ap.add_argument("--scenes_file", default=None, help="从文件读取场景列表 (每行一个 scene id)")
    ap.add_argument("--seed_tag", default="_seed4_clip",
                    help="输出/输入子目录名后缀 (默认 _seed4_clip)")
    ap.add_argument("--depth_tol", type=float, default=0.5,
                    help="GT depth z-test 容差 (m); 0 或负值表示不做 z-test")
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_dim", type=int, default=512)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[recompute_clip] device={device}  clip_model={args.clip_model}")
    model, preprocess = _load_clip(device, args.clip_model)
    print("[recompute_clip] CLIP ready")

    depth_tol = args.depth_tol if (args.depth_tol is not None and args.depth_tol > 0) else None

    scenes = discover_scenes(in_root, args.scene, args.scenes_file)
    print(f"[recompute_clip] in_root={in_root}\n[recompute_clip] out_root={out_root}\n"
          f"[recompute_clip] scenes={len(scenes)}")

    t_all = time.time()
    for si, scene in enumerate(scenes, 1):
        in_run_dir = in_root / f"{scene}{args.seed_tag}"
        out_run_dir = out_root / f"{scene}{args.seed_tag}"
        out_npz = out_run_dir / "params.npz"
        if args.skip_existing and out_npz.is_file():
            print(f"[{si}/{len(scenes)}] {scene}: skip (exists {out_npz})")
            continue
        print(f"[{si}/{len(scenes)}] {scene}: start")
        try:
            recompute_scene_clip_vectors(
                in_run_dir=in_run_dir,
                out_run_dir=out_run_dir,
                scene=scene,
                basedir=args.basedir,
                device=device,
                model=model,
                preprocess=preprocess,
                clip_dim=args.clip_dim,
                depth_tol=depth_tol,
                progress_every=args.progress_every,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{si}/{len(scenes)}] {scene}: FAILED ({e})")
            continue
    total = time.time() - t_all
    print(f"[recompute_clip] DONE in {total:.1f}s for {len(scenes)} scene(s)")


if __name__ == "__main__":
    main()
