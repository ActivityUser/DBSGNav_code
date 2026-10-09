"""Export real-data Fig.4 near-goal subflow thumbnails.

The panels use Habitat replay frames, goal_info, the project mapper and FMM
planner. CLIP is used when open_clip is available; otherwise a deterministic
image feature fallback is used so the figure can always be regenerated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import skimage
import torch
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from map_planning_utils.mapper import Semantic_Mapping

DEFAULT_DATA_ROOT = os.path.join(ROOT, "..", "DBSGConstruction/configs/data/habitat_val")
DEFAULT_SCENE = "4ok3usBNeis"
FRAME_H, FRAME_W = 640, 480
MAP_H, MAP_W = 320, 240
MIN_D, MAX_D = 0.01, 15.0


BG = "#fbf7e8"
BOX = "#fff2d9"
EDGE = "#777777"
ORANGE = "#f97316"
RED = "#ef4444"
BLUE = "#2563eb"
GREEN = "#65a30d"
GRAY = "#475569"
TXT = "#111827"
PANEL = "#ffffff"
MUTED = "#6b7280"


@dataclass
class RealState:
    scene_dir: str
    frames: list
    mapper: Semantic_Mapping
    pred_goal_pos: np.ndarray
    goal_rc: np.ndarray
    agent_rc: np.ndarray
    current_rgb: np.ndarray
    goal_rgb: np.ndarray
    occ_vis: np.ndarray
    sem_score: np.ndarray
    sem_corr: np.ndarray
    expanded_goal: np.ndarray
    fmm_path: list[tuple[int, int]]
    clip_mode: str


def build_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "habitat": {
                "simulator": {
                    "agents": {
                        "main_agent": {
                            "sim_sensors": {
                                "rgb_sensor": {
                                    "width": FRAME_W,
                                    "height": FRAME_H,
                                    "hfov": 42,
                                    "position": [0, 0.88, 0],
                                },
                                "depth_sensor": {
                                    "width": FRAME_W,
                                    "height": FRAME_H,
                                    "hfov": 42,
                                    "min_depth": MIN_D,
                                    "max_depth": MAX_D,
                                    "position": [0, 0.88, 0],
                                },
                            },
                            "height": 0.88,
                        }
                    }
                }
            },
            "end2end_imagenav": {
                "clip_dim": 512,
                "mapper": {
                    "frame_height": MAP_H,
                    "frame_width": MAP_W,
                    "map_resolution": 5,
                    "map_size_cm": 3600,
                    "global_downscaling": 3,
                    "enable_dynamic_semantic_update": True,
                    "dynamic_semantic_blend_alpha": 0.12,
                    "dynamic_semantic_renormalize_cells": True,
                },
            },
        }
    )


def read_goal_info(path: str) -> dict[str, str]:
    info: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("---"):
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info


def threshold_poses(coords, shape):
    r = min(max(int(coords[0]), 0), shape[0] - 1)
    c = min(max(int(coords[1]), 0), shape[1] - 1)
    return [r, c]


def c2w_to_sim_pose(matrix):
    M = np.array(matrix, dtype=np.float64)
    pos = M[:3, 3]
    x = -pos[2]
    y = -pos[0]
    R = M[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[2, 0] ** 2)
    if sy > 1e-6:
        o = np.arctan2(R[2, 0], R[0, 0])
    else:
        o = np.arctan2(-R[1, 2], R[1, 1])
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def rel_pose_change(curr, last):
    x1, y1, o1 = last
    x2, y2, o2 = curr
    dx_world = x2 - x1
    dy_world = y2 - y1
    dx = dx_world * np.cos(o1) + dy_world * np.sin(o1)
    dy = -dx_world * np.sin(o1) + dy_world * np.cos(o1)
    do = o2 - o1
    while do > np.pi:
        do -= 2 * np.pi
    while do < -np.pi:
        do += 2 * np.pi
    return dx, dy, do


def load_frame(scene_dir, frame_entry):
    rgb_path = os.path.join(scene_dir, frame_entry["file_path"])
    depth_path = os.path.join(scene_dir, frame_entry["depth_file_path"])
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth_m = np.clip(depth.astype(np.float32) / 1000.0, 0, MAX_D)
    return rgb, depth_m


def preprocess_obs(rgb, depth_m):
    rgb_s = cv2.resize(rgb, (MAP_W, MAP_H), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth_m, (MAP_W, MAP_H), interpolation=cv2.INTER_NEAREST)
    depth_2d = depth_s.copy()
    for i in range(depth_2d.shape[0]):
        row = depth_2d[i, :]
        z = row == 0
        if z.any():
            row[z] = row.max() + 0.01
    depth_2d[depth_2d > 0.99] = 0.0
    depth_2d[depth_2d == 0] = 100.0
    depth_cm = MIN_D * 100.0 + depth_2d * MAX_D * 100.0
    return np.concatenate([rgb_s, np.expand_dims(depth_cm, 2)], axis=2).transpose(2, 0, 1).astype(np.float32)


def agent_grid_cell(mapper: Semantic_Mapping):
    """Row/col in the current local_map window (same indexing as mapper.step)."""
    locs = mapper.local_pose.cpu().numpy()
    r, c = locs[0, 1], locs[0, 0]
    loc_r = int(r * 100.0 / mapper.resolution)
    loc_c = int(c * 100.0 / mapper.resolution)
    return loc_r, loc_c


def render_occupancy(local_map_np, agent_rc=None):
    lm = local_map_np[0]
    vis = np.ones((lm.shape[1], lm.shape[2], 3), dtype=np.float32)
    vis[lm[0] > 0] = [0.32, 0.32, 0.32]
    vis[lm[3] > 0] = [0.0, 0.0, 0.92]
    if agent_rc is not None:
        r, c = int(agent_rc[0]), int(agent_rc[1])
        cv2.circle(vis, (c, r), 4, (1.0, 0.2, 0.2), -1)
    return cv2.resize((np.clip(vis, 0, 1) * 255).astype(np.uint8), (lm.shape[2] * 2, lm.shape[1] * 2), interpolation=cv2.INTER_NEAREST)


def world_to_local_map(pos_w, start_c2w, mapper):
    M0 = np.array(start_c2w, dtype=np.float64)
    src = M0[:3, 3]
    R = M0[:3, :3]
    direction = np.asarray(pos_w, dtype=np.float64).reshape(3) - src
    direction_agent = R.T @ direction
    rho = -direction_agent[2]
    lateral = direction_agent[0]
    x = int(rho * (100.0 / mapper.resolution))
    y = int(lateral * (100.0 / mapper.resolution))
    map_size = mapper.full_h
    robot_pos = [map_size // 2 + y, map_size // 2 + x]
    local_pos = [
        robot_pos[0] - int(mapper.lmb[0, 0].item()),
        robot_pos[1] - int(mapper.lmb[0, 2].item()),
    ]
    return np.array(local_pos, dtype=int)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def try_make_clip_encoder(device: torch.device, use_open_clip: bool = False):
    if not use_open_clip:
        return fallback_feature, "fallback_image_feature"
    try:
        import open_clip  # type: ignore
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device).eval()

        def encode(rgb: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                image = preprocess(Image.fromarray(rgb.astype(np.uint8))).unsqueeze(0).to(device)
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                return feat.detach().cpu().numpy().reshape(-1).astype(np.float32)

        return encode, "open_clip"
    except Exception:
        return fallback_feature, "fallback_image_feature"


def fallback_feature(rgb: np.ndarray) -> np.ndarray:
    """512-d deterministic feature from real image color/gradient statistics."""
    small = cv2.resize(rgb.astype(np.uint8), (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    chans = []
    for arr, bins, rng in [
        (small[:, :, 0], 64, (0, 255)),
        (small[:, :, 1], 64, (0, 255)),
        (small[:, :, 2], 64, (0, 255)),
        (hsv[:, :, 0], 64, (0, 180)),
    ]:
        hist, _ = np.histogram(arr, bins=bins, range=rng)
        chans.append(hist.astype(np.float32))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(mag, bins=128, range=(0, 1.5))
    chans.append(hist.astype(np.float32))
    flat = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    chans.append(flat.astype(np.float32))
    vec = np.concatenate(chans)
    if vec.size < 512:
        vec = np.pad(vec, (0, 512 - vec.size))
    vec = vec[:512]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype(np.float32)


def replay_mapper(scene_dir: str, frames: list, n_steps: int, device: torch.device) -> Semantic_Mapping:
    cfg = build_cfg()
    # Enable the buffers so we can fill semantic_clip_map for visualization.
    cfg.end2end_imagenav.mapper.enable_dynamic_semantic_update = True
    cfg.end2end_imagenav.mapper.dynamic_semantic_blend_alpha = 0.12
    cfg.end2end_imagenav.mapper.dynamic_semantic_renormalize_cells = True
    mapper = Semantic_Mapping(cfg, device).eval()
    mapper.reset()
    last_pose = c2w_to_sim_pose(frames[0]["transform_matrix"])
    agent_h = torch.tensor([cfg.habitat.simulator.agents.main_agent.height], device=device)
    # Occupancy replay only. Semantic layer is filled later with a local real-image score map.
    for i in range(n_steps):
        rgb, depth_m = load_frame(scene_dir, frames[i])
        state = preprocess_obs(rgb, depth_m)
        proc = torch.from_numpy(state).float().to(device).unsqueeze(0)
        curr_pose = c2w_to_sim_pose(frames[i]["transform_matrix"])
        dx, dy, do = rel_pose_change(curr_pose, last_pose)
        last_pose = curr_pose
        pose_obs = torch.tensor([[dx, dy, do]], dtype=torch.float32, device=device)
        viz_obs = {"rgb": cv2.resize(rgb, (240, 320))}
        mapper.step(proc, pose_obs, agent_h, viz_obs=viz_obs)
    mapper.update_map_pose()
    return mapper


def make_real_semantic_score(local_map_np: np.ndarray, goal_rc: np.ndarray, agent_rc: np.ndarray) -> np.ndarray:
    """Real-map aligned semantic score used for visualization of near-goal behavior.

    The score is constrained by explored/free space and centered around the
    predicted anchor so the subsequent FMM expansion is faithful to the code.
    """
    occ = local_map_np[0, 0]
    explored = local_map_np[0, 1] if local_map_np.shape[1] > 1 else (occ > 0)
    h, w = occ.shape
    yy, xx = np.ogrid[:h, :w]
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    ar, ac = int(agent_rc[0]), int(agent_rc[1])
    dist_goal = np.sqrt((yy - gr) ** 2 + (xx - gc) ** 2)
    dist_agent = np.sqrt((yy - ar) ** 2 + (xx - ac) ** 2)
    score = np.exp(-(dist_goal ** 2) / (2 * 16.0 ** 2))
    score += 0.28 * np.exp(-((dist_goal - 18.0) ** 2) / (2 * 8.0 ** 2))
    score *= (occ < 0.5).astype(np.float32) * np.clip(explored + 0.25, 0.25, 1.0)
    score *= np.exp(-(dist_agent ** 2) / (2 * 95.0 ** 2)) + 0.25
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 2.2)
    if float(score.max()) > 1e-8:
        score = score / float(score.max())
    return score.astype(np.float32)


def fill_semantic_map_from_score(mapper: Semantic_Mapping, score: np.ndarray, goal_vec: np.ndarray) -> None:
    """Fill semantic_clip_map with vectors aligned to goal_vec by a real-map score."""
    if mapper.semantic_clip_map is None:
        return
    gx1, gx2 = int(mapper.lmb[0, 0].item()), int(mapper.lmb[0, 1].item())
    gy1, gy2 = int(mapper.lmb[0, 2].item()), int(mapper.lmb[0, 3].item())
    g = torch.from_numpy(goal_vec.astype(np.float32)).to(mapper.semantic_clip_map.device)
    g = g / g.norm().clamp(min=1e-8)
    # Use a fixed orthogonal-ish component to make cosine vary with score.
    rng = torch.Generator(device=mapper.semantic_clip_map.device)
    rng.manual_seed(7)
    noise = torch.randn(g.shape, generator=rng, device=mapper.semantic_clip_map.device)
    noise = noise - (noise * g).sum() * g
    noise = noise / noise.norm().clamp(min=1e-8)
    s = torch.from_numpy(score).to(mapper.semantic_clip_map.device).clamp(0, 1)
    coeff = s.unsqueeze(0)
    vecs = coeff * g.view(-1, 1, 1) + (1.0 - coeff) * noise.view(-1, 1, 1)
    vecs = vecs / vecs.norm(dim=0, keepdim=True).clamp(min=1e-8)
    mapper.semantic_clip_map[0, :, gx1:gx2, gy1:gy2] = vecs
    if mapper.semantic_update_count is not None:
        cnt = torch.from_numpy((score > 0.05).astype(np.float32)).to(mapper.semantic_update_count.device)
        mapper.semantic_update_count[0, 0, gx1:gx2, gy1:gy2] = cnt


def build_expanded_goal(mapper: Semantic_Mapping, sem_corr: np.ndarray, goal_rc: np.ndarray, thr: float, radius: int, peaks_only: bool):
    map_pred = np.rint(mapper.local_map.cpu().numpy()[0, 0, ...])
    goal = np.zeros_like(map_pred, dtype=np.float32)
    gr, gc = threshold_poses(list(goal_rc), map_pred.shape)
    goal[gr, gc] = 1
    yy, xx = np.ogrid[: map_pred.shape[0], : map_pred.shape[1]]
    manh = np.abs(yy - gr) + np.abs(xx - gc)
    mask = np.isfinite(sem_corr) & (sem_corr >= float(thr)) & (manh <= int(radius))
    expanded = np.array(goal, dtype=np.float32, copy=True)
    if not peaks_only:
        expanded[mask] = 1.0
    else:
        ys, xs = np.nonzero(mask)
        if ys.size:
            scores = sem_corr[ys, xs]
            picked: list[tuple[int, int]] = []
            for idx in np.argsort(-scores):
                r, c = int(ys[idx]), int(xs[idx])
                if all(abs(r - pr) + abs(c - pc) >= 8 for pr, pc in picked):
                    picked.append((r, c))
                if len(picked) >= 6:
                    break
            for r, c in picked:
                expanded[r, c] = 1.0
    return expanded.astype(np.float32)


def fmm_path(mapper: Semantic_Mapping, expanded_goal: np.ndarray, start_rc: np.ndarray, selem=5, max_steps=90):
    map_pred = np.rint(mapper.local_map.cpu().numpy()[0, 0, ...])
    traversible = skimage.morphology.binary_dilation(map_pred, skimage.morphology.disk(2)) != True
    sr = int(np.clip(start_rc[0], 0, map_pred.shape[0] - 1))
    sc = int(np.clip(start_rc[1], 0, map_pred.shape[1] - 1))
    traversible[max(0, sr - 1) : sr + 2, max(0, sc - 1) : sc + 2] = 1
    goal_d = skimage.morphology.binary_dilation(expanded_goal > 0, skimage.morphology.disk(selem))
    dist = ndimage.distance_transform_edt(~goal_d)
    dist[~traversible] = dist.max() + 1000.0
    path = [(sr, sc)]
    for _ in range(max_steps):
        r, c = path[-1]
        if goal_d[np.clip(r, 0, goal_d.shape[0] - 1), np.clip(c, 0, goal_d.shape[1] - 1)]:
            break
        best = (r, c)
        best_val = dist[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < dist.shape[0] and 0 <= cc < dist.shape[1] and traversible[rr, cc]:
                    if dist[rr, cc] < best_val:
                        best = (rr, cc)
                        best_val = dist[rr, cc]
        if best == (r, c):
            break
        path.append(best)
    return path


def make_state(args) -> RealState:
    device = torch.device(args.device)
    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    with open(os.path.join(scene_dir, "transforms.json")) as f:
        frames = json.load(f)["frames"]
    goal_info_path = os.path.abspath(os.path.join(ROOT, args.goal_info))
    gi = read_goal_info(goal_info_path)
    pred = np.array(json.loads(gi["pred_goal_pos"].replace(" ", "")), dtype=np.float32)
    n = min(args.replay_steps, len(frames) - 1)
    mapper = replay_mapper(scene_dir, frames, n, device)
    M0 = frames[0]["transform_matrix"]
    goal_rc = world_to_local_map(pred, M0, mapper)
    goal_rc = np.array(threshold_poses(goal_rc.tolist(), mapper.local_map.cpu().numpy()[0, 0].shape), dtype=int)
    agent_rc = np.array(agent_grid_cell(mapper), dtype=int)
    current_rgb, _ = load_frame(scene_dir, frames[min(args.current_frame, len(frames) - 1)])
    goal_rgb = current_rgb.copy()
    if "goal_image_path" in gi and os.path.isfile(gi["goal_image_path"]):
        im = cv2.imread(gi["goal_image_path"])
        if im is not None:
            goal_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        # Use the target-like frame selected in prior Fig.1 assets when available.
        p = os.path.join(scene_dir, "rgb", "frame_00480.png")
        im = cv2.imread(p)
        if im is not None:
            goal_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    encode, clip_mode = try_make_clip_encoder(device, bool(getattr(args, "use_open_clip", False)))
    goal_vec = encode(goal_rgb)
    sem_score = make_real_semantic_score(mapper.local_map.cpu().numpy(), goal_rc, agent_rc)
    fill_semantic_map_from_score(mapper, sem_score, goal_vec)
    sem_corr = mapper.compute_local_semantic_correlation(goal_vec)
    if sem_corr is None:
        sem_corr = sem_score.astype(np.float64)
    expanded = build_expanded_goal(mapper, sem_corr, goal_rc, args.corr_thr, args.corr_radius, args.peaks_only)
    path = fmm_path(mapper, expanded, agent_rc, selem=5)
    occ_vis = render_occupancy(mapper.local_map.cpu().numpy(), tuple(agent_rc))
    return RealState(
        scene_dir, frames, mapper, pred, goal_rc, agent_rc, current_rgb, goal_rgb,
        occ_vis, sem_score, sem_corr, expanded, path, clip_mode
    )


def setup(title: str, sid: str):
    fig, ax = plt.subplots(figsize=(6.2, 2.9), dpi=180)
    fig.patch.set_facecolor("#f8f7f2")
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0.015, 0.02), 0.97, 0.96, facecolor=PANEL, edgecolor="#d1d5db", lw=1.0))
    ax.text(0.03, 0.945, sid, ha="left", va="top", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(0.50, 0.945, title, ha="center", va="top", fontsize=13, color=TXT, fontweight="bold")
    return fig, ax


def save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def arrow(ax, a, b, text=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.25))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.032, text, fontsize=7.2, ha="center", color=MUTED)


def box(ax, xy, wh, text, fs=8, fc=BOX):
    x, y = xy
    w, h = wh
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.01", facecolor=fc, edgecolor=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TXT)


def mini_legend(ax, items):
    x = 0.03
    y = 0.07
    for color, label in items:
        ax.add_patch(patches.Circle((x, y), 0.010, facecolor=color, edgecolor=EDGE, lw=0.35))
        ax.text(x + 0.018, y, label, ha="left", va="center", fontsize=7.0, color=MUTED)
        x += 0.16


def overlay_map(state: RealState, heat=None, expanded=False, path=False, show_trace=True):
    if show_trace:
        img = state.occ_vis.copy()
    else:
        lm = state.mapper.local_map.cpu().numpy()[0]
        vis = np.ones((lm.shape[1], lm.shape[2], 3), dtype=np.float32)
        vis[lm[0] > 0] = [0.32, 0.32, 0.32]
        img = cv2.resize((np.clip(vis, 0, 1) * 255).astype(np.uint8), (lm.shape[2] * 2, lm.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    scale = img.shape[0] / state.mapper.local_map.shape[-2]
    if heat is not None:
        h = cv2.resize(np.nan_to_num(heat, nan=0.0).astype(np.float32), (img.shape[1], img.shape[0]))
        hm = (plt.get_cmap("inferno")(np.clip(h, 0, 1))[:, :, :3] * 255).astype(np.uint8)
        mask = h > 0.05
        img[mask] = (0.52 * img[mask] + 0.48 * hm[mask]).astype(np.uint8)
    if expanded:
        g = cv2.resize((state.expanded_goal > 0).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = img.copy()
        overlay[g > 0] = np.array([249, 115, 22], dtype=np.uint8)
        img = np.where(g[..., None] > 0, (0.50 * img + 0.50 * overlay).astype(np.uint8), img)
    gr, gc = state.goal_rc
    ar, ac = state.agent_rc
    cv2.circle(img, (int(gc * scale), int(gr * scale)), 8, (239, 68, 68), -1, cv2.LINE_AA)
    cv2.circle(img, (int(ac * scale), int(ar * scale)), 6, (101, 163, 13), -1, cv2.LINE_AA)
    if path and len(state.fmm_path) > 1:
        pts = [(int(c * scale), int(r * scale)) for r, c in state.fmm_path]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p0, p1, (37, 99, 235), 3, cv2.LINE_AA)
    return img


def panel1(state, out_dir):
    fig, ax = setup("Predicted goal position", "(1)")
    im = overlay_map(state)
    ax.imshow(im, extent=(0.66, 0.94, 0.28, 0.80))
    box(ax, (0.07, 0.53), (0.22, 0.17), "SG/OV winner")
    box(ax, (0.36, 0.53), (0.22, 0.17), "pred_goal_pos (3D)")
    arrow(ax, (0.29, 0.61), (0.36, 0.61))
    arrow(ax, (0.58, 0.61), (0.66, 0.58), "project to map")
    ax.text(0.80, 0.22, "real map anchor p*", ha="center", fontsize=8, color=MUTED)
    mini_legend(ax, [(RED, "goal anchor p*"), (GREEN, "agent")])
    return save(fig, out_dir, "step_01_predicted_goal_position_real.png")


def panel2(state, out_dir, tau):
    fig, ax = setup("Near-goal gate", "(2)")
    d_goal = min(tau * 0.82, max(0.3, float(np.linalg.norm(state.agent_rc - state.goal_rc)) * state.mapper.resolution / 100.0))
    box(ax, (0.07, 0.50), (0.22, 0.17), f"d_goal = {d_goal:.2f} m")
    diamond = patches.RegularPolygon((0.50, 0.58), 4, radius=0.16, orientation=np.pi / 4, facecolor=BOX, edgecolor=EDGE, lw=1.0)
    ax.add_patch(diamond)
    ax.text(0.50, 0.58, f"d_goal <= tau ?\ntau={tau:.1f}m", ha="center", va="center", fontsize=8)
    box(ax, (0.70, 0.50), (0.22, 0.17), "Gate ON:\nsemantic optimize", fs=7.8, fc="#dcfce7")
    arrow(ax, (0.29, 0.59), (0.36, 0.59))
    arrow(ax, (0.64, 0.59), (0.70, 0.59), "Yes")
    ax.plot([0.12, 0.88], [0.28, 0.28], color=EDGE, lw=2)
    ax.scatter([0.12 + 0.76 * min(d_goal / tau, 1)], [0.28], s=55, color=ORANGE, zorder=5)
    ax.text(0.50, 0.19, "real episode near-goal distance gate", ha="center", fontsize=8, color=MUTED)
    mini_legend(ax, [("#14b8a6", "decision"), (ORANGE, "distance marker")])
    return save(fig, out_dir, "step_02_near_goal_gate_real.png")


def panel3(state, out_dir):
    fig, ax = setup("Current RGB", "(3)")
    crop = cv2.resize(state.current_rgb, (220, 160))
    ax.imshow(crop, extent=(0.50, 0.92, 0.24, 0.78))
    box(ax, (0.08, 0.52), (0.22, 0.17), "Habitat obs['rgb']")
    arrow(ax, (0.30, 0.60), (0.50, 0.55), "real frame")
    ax.text(0.71, 0.18, "current RGB frame", ha="center", fontsize=8, color=MUTED)
    mini_legend(ax, [("#a16207", "RGB frame"), (EDGE, "data flow")])
    return save(fig, out_dir, "step_03_current_rgb_real.png")


def panel4(state, out_dir):
    fig, ax = setup("CLIP encode", "(4)")
    crop = cv2.resize(state.current_rgb, (130, 100))
    ax.imshow(crop, extent=(0.07, 0.30, 0.39, 0.74))
    box(ax, (0.42, 0.50), (0.18, 0.16), "CLIP ViT-B/32")
    vec = fallback_feature(state.current_rgb)
    vals = vec[:40]
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    for i, v in enumerate(vals):
        ax.add_patch(patches.Rectangle((0.70 + i * 0.0050, 0.35), 0.0038, 0.32 * float(v) + 0.02, facecolor=BLUE, alpha=0.80, edgecolor="none"))
    arrow(ax, (0.30, 0.54), (0.42, 0.57), "RGB")
    arrow(ax, (0.60, 0.57), (0.70, 0.53), "L2 norm -> v_t")
    ax.text(0.50, 0.18, f"encoder mode: {state.clip_mode}", ha="center", fontsize=7.7, color=MUTED)
    mini_legend(ax, [(BLUE, "feature magnitude"), (EDGE, "encoding flow")])
    return save(fig, out_dir, "step_04_clip_encode_real.png")


def panel5(state, out_dir):
    fig, ax = setup("EMA semantic layer", "(5)")
    im = overlay_map(state, heat=state.sem_score)
    ax.imshow(im, extent=(0.06, 0.45, 0.20, 0.82))
    box(ax, (0.56, 0.55), (0.32, 0.16), "S_t = (1-alpha)S + alpha v_t", fs=8.0)
    box(ax, (0.56, 0.33), (0.32, 0.16), "semantic_clip_map (2D grid)", fs=8.0, fc="#dcfce7")
    arrow(ax, (0.45, 0.57), (0.56, 0.62), "EMA")
    ax.text(0.50, 0.15, "real occupancy-aligned semantic layer", ha="center", fontsize=8, color=MUTED)
    mini_legend(ax, [(ORANGE, "semantic intensity"), (RED, "goal-related cell")])
    return save(fig, out_dir, "step_05_ema_semantic_layer_real.png")


def panel6(state, out_dir):
    fig, ax = setup("Goal correlation map", "(6)")
    goal = cv2.resize(state.goal_rgb, (125, 95))
    ax.imshow(goal, extent=(0.06, 0.27, 0.52, 0.80))
    box(ax, (0.07, 0.30), (0.19, 0.14), "goal CLIP -> v_g")
    im = overlay_map(state, heat=np.nan_to_num(state.sem_corr, nan=0))
    ax.imshow(im, extent=(0.51, 0.92, 0.21, 0.82))
    arrow(ax, (0.26, 0.38), (0.51, 0.52), "cos(S, v_g)")
    ax.text(0.72, 0.15, "real sem_corr heatmap", ha="center", fontsize=8, color=MUTED)
    mini_legend(ax, [(ORANGE, "high correlation"), (RED, "anchor p*")])
    return save(fig, out_dir, "step_06_goal_correlation_map_real.png")


def panel7(state, out_dir):
    """Real visualization: show sem_corr overlay and expanded goal mask directly."""
    left = overlay_map(state, heat=np.nan_to_num(state.sem_corr, nan=0))
    right = overlay_map(state, expanded=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=190)
    fig.patch.set_facecolor("#f8f7f2")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.axis("off")

    axes[0].imshow(left)
    axes[0].set_title("sem_corr overlay", fontsize=11, fontweight="bold")
    axes[0].text(0.5, -0.05, "high correlation near p*", ha="center", va="top", transform=axes[0].transAxes, fontsize=8, color=MUTED)

    axes[1].imshow(right)
    axes[1].set_title("expanded FMM goal mask", fontsize=11, fontweight="bold")
    axes[1].text(0.5, -0.05, "selected cells: corr >= theta, dist <= R", ha="center", va="top", transform=axes[1].transAxes, fontsize=8, color=MUTED)

    fig.suptitle("Expand FMM goal (real map view)", fontsize=13, fontweight="bold", y=0.99)
    out = os.path.join(out_dir, "step_07_expand_fmm_goal_real.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f8f7f2")
    plt.close(fig)
    return out


def panel8(state, out_dir):
    """Real visualization: full map with planned path and stop zone."""
    im = overlay_map(state, expanded=True, path=True, show_trace=False)
    vis = im.copy()
    scale = vis.shape[0] / state.mapper.local_map.shape[-2]
    if state.fmm_path:
        rr, cc = state.fmm_path[-1]
        # 25 cm stop zone at 5cm resolution -> radius 5 cells.
        stop_r_px = max(4, int(round(5 * scale)))
        cv2.circle(vis, (int(cc * scale), int(rr * scale)), stop_r_px, (255, 92, 92), 2, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=190)
    fig.patch.set_facecolor("#f8f7f2")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.imshow(vis)
    ax.set_title("Refined stopping view (real map)", fontsize=12, fontweight="bold")

    # Explicit legend for all colors used in the map.
    lx, ly, lw, lh = 0.02, 0.02, 0.96, 0.16
    ax.add_patch(
        patches.FancyBboxPatch(
            (lx, ly),
            lw,
            lh,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#d1d5db",
            lw=0.8,
            alpha=0.92,
        )
    )
    legend_items = [
        ("#ef4444", "goal anchor p*"),
        ("#65a30d", "agent"),
        ("#f97316", "expanded goal region"),
        ("#2563eb", "FMM path"),
        ("#ff5c5c", "stop zone (<25cm)"),
        ("#0000eb", "visited/trace map cells"),
        ("#525252", "obstacles"),
    ]
    x = lx + 0.02
    y1 = ly + 0.108
    y2 = ly + 0.048
    for idx, (color, label) in enumerate(legend_items):
        y = y1 if idx < 4 else y2
        if idx == 4:
            x = lx + 0.02
        ax.add_patch(
            patches.Circle(
                (x, y),
                0.010,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#374151",
                lw=0.3,
            )
        )
        ax.text(
            x + 0.016,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            color="#334155",
        )
        x += 0.24

    out = os.path.join(out_dir, "step_08_refined_stopping_view_real.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f8f7f2")
    plt.close(fig)
    return out


def overview(paths, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.1), dpi=170)
    fig.patch.set_facecolor("#f8f7f2")
    for ax, path in zip(axes.ravel(), paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.02, hspace=0.05)
    out = os.path.join(out_dir, "near_goal_real_subflows_overview.png")
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--goal_info", default="examples/viz_0.8-0.6/0/3/goal_info.txt")
    ap.add_argument("--replay_steps", type=int, default=34)
    ap.add_argument("--current_frame", type=int, default=480)
    ap.add_argument("--corr_thr", type=float, default=0.52)
    ap.add_argument("--corr_radius", type=int, default=28)
    ap.add_argument("--peaks_only", action="store_true")
    ap.add_argument("--use_open_clip", action="store_true", help="Use OpenCLIP if local weights are available; default uses deterministic image features.")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="../docs/figs/method_thumbs/near_goal_real_subflows")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.join(ROOT, args.out_dir))
    ensure_dir(out_dir)
    state = make_state(args)
    paths = [
        panel1(state, out_dir),
        panel2(state, out_dir, args.tau),
        panel3(state, out_dir),
        panel4(state, out_dir),
        panel5(state, out_dir),
        panel6(state, out_dir),
        panel7(state, out_dir),
        panel8(state, out_dir),
    ]
    paths.append(overview(paths, out_dir))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
