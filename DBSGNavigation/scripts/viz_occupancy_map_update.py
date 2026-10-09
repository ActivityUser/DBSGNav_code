"""Export occupancy map before / after one mapper update step.

Replays RGB-D from habitat_val transforms with Semantic_Mapping (same as online nav).

Example:
  python scripts/viz_occupancy_map_update.py --scene 4ok3usBNeis --warmup_steps 25 --update_step 26
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
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from map_planning_utils.mapper import Semantic_Mapping
from utils import pose as pu

DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "DBSGConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"
FRAME_H, FRAME_W = 640, 480
MAP_H, MAP_W = 320, 240
MIN_D, MAX_D = 0.01, 15.0


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
                "mapper": {
                    "frame_height": MAP_H,
                    "frame_width": MAP_W,
                    "map_resolution": 5,
                    "map_size_cm": 3600,
                    "global_downscaling": 3,
                    "enable_dynamic_semantic_update": False,
                }
            },
        }
    )


def c2w_to_sim_pose(matrix):
    """Match instance_imagenav_env.get_sim_location convention."""
    M = np.array(matrix, dtype=np.float64)
    pos = M[:3, 3]
    x = -pos[2]
    y = -pos[0]
    R = M[:3, :3]
    # yaw from rotation (same branch as env)
    sy = np.sqrt(R[0, 0] ** 2 + R[2, 0] ** 2)
    if sy > 1e-6:
        o = np.arctan2(R[2, 0], R[0, 0])
    else:
        o = np.arctan2(-R[1, 2], R[1, 1])
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def load_frame(scene_dir, frame_entry):
    rgb_path = os.path.join(scene_dir, frame_entry["file_path"])
    depth_path = os.path.join(scene_dir, frame_entry["depth_file_path"])
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(depth_path)
    if d.ndim == 3:
        d = d[:, :, 0]
    depth_m = d.astype(np.float32) / 1000.0
    depth_m = np.clip(depth_m, 0, MAX_D)
    return rgb, depth_m


def preprocess_obs(rgb, depth_m):
    """Same layout as NiceEnv._preprocess_obs (no seg): 320x240 (H x W)."""
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
    state = np.concatenate([rgb_s, np.expand_dims(depth_cm, 2)], axis=2).transpose(2, 0, 1)
    return state.astype(np.float32)


def render_occupancy(local_map_np, agent_rc=None):
    """Gray = explored, blue = obstacle (mapper.visualize style)."""
    lm = local_map_np[0]
    vis = np.ones((lm.shape[1], lm.shape[2], 3), dtype=np.float32)
    vis[lm[0] > 0] = [0.32, 0.32, 0.32]
    vis[lm[3] > 0] = [0.0, 0.0, 0.92]
    if agent_rc is not None:
        r, c = int(agent_rc[0]), int(agent_rc[1])
        cv2.circle(vis, (c, r), 4, (1.0, 0.2, 0.2), -1)
    vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(vis, (lm.shape[2] * 2, lm.shape[1] * 2), interpolation=cv2.INTER_NEAREST)


def agent_grid_cell(mapper):
    locs = mapper.local_pose.cpu().numpy()
    r, c = locs[0, 1], locs[0, 0]
    loc_r = int(r * 100.0 / mapper.resolution)
    loc_c = int(c * 100.0 / mapper.resolution)
    gx1 = int(mapper.lmb[0, 0].item())
    gy1 = int(mapper.lmb[0, 2].item())
    return loc_r - gx1, loc_c - gy1


def crop_occ_from_vis_mp4(mp4_path, frame_idx):
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    y1, y2 = 12, h - 12
    usable = w - 80
    pw = usable // 3
    x0 = 20
    occ = fr[y1:y2, x0 + pw + 20 : x0 + 2 * pw + 20]
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--warmup_steps", type=int, default=25,
                    help="replay this many steps before the shown update")
    ap.add_argument("--update_step", type=int, default=26,
                    help="frame index in transforms.json for the 'after' step")
    ap.add_argument("--out_dir", default="../docs/figs/occupancy_map_exports/4ok3usBNeis")
    ap.add_argument("--vis_mp4", default="examples/viz_0.8-0.6/0/3/4ok3usBNeis-4-Vis.mp4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    out_dir = os.path.abspath(os.path.join(ROOT, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(scene_dir, "transforms.json")) as f:
        frames = json.load(f)["frames"]
    n = len(frames)
    upd = min(args.update_step, n - 1)
    warm = min(args.warmup_steps, upd)

    cfg = build_cfg()
    device = torch.device(args.device)
    mapper = Semantic_Mapping(cfg, device).eval()
    mapper.reset()

    last_pose = c2w_to_sim_pose(frames[0]["transform_matrix"])
    agent_h = torch.tensor([cfg.habitat.simulator.agents.main_agent.height], device=device)

    def run_step(idx):
        nonlocal last_pose
        rgb, depth_m = load_frame(scene_dir, frames[idx])
        state = preprocess_obs(rgb, depth_m)
        proc = torch.from_numpy(state).float().to(device).unsqueeze(0)
        curr_pose = c2w_to_sim_pose(frames[idx]["transform_matrix"])
        dx, dy, do = pu.get_rel_pose_change(curr_pose, last_pose)
        last_pose = curr_pose
        pose_obs = torch.tensor([[dx, dy, do]], dtype=torch.float32, device=device)
        viz_obs = {"rgb": cv2.resize(rgb, (MAP_W, MAP_H))}
        mapper.step(proc, pose_obs, agent_h, viz_obs=viz_obs)

    for i in range(warm):
        run_step(i)

    before = render_occupancy(mapper.local_map.cpu().numpy(), agent_grid_cell(mapper))
    cv2.imwrite(os.path.join(out_dir, "occupancy_before_update.png"), before,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    run_step(upd)
    after = render_occupancy(mapper.local_map.cpu().numpy(), agent_grid_cell(mapper))
    cv2.imwrite(os.path.join(out_dir, "occupancy_after_update.png"), after,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # side-by-side explainer
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    axes[0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"(a) Before update (after {warm} steps)", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"(b) After update (step {upd})", fontsize=10)
    axes[1].axis("off")
    fig.suptitle(
        "Occupancy map update — gray: explored, blue: obstacle, red dot: agent",
        fontsize=11,
    )
    expl = os.path.join(out_dir, "occupancy_update_explainer.png")
    fig.tight_layout()
    fig.savefig(expl, bbox_inches="tight")
    plt.close(fig)

    # optional: real nav Vis.mp4 panels
    mp4 = os.path.abspath(os.path.join(ROOT, args.vis_mp4))
    if os.path.isfile(mp4):
        occ_b = crop_occ_from_vis_mp4(mp4, max(0, warm))
        occ_a = crop_occ_from_vis_mp4(mp4, min(upd, 38))
        if occ_b is not None:
            cv2.imwrite(os.path.join(out_dir, "occupancy_vismp4_before.png"), occ_b)
        if occ_a is not None:
            cv2.imwrite(os.path.join(out_dir, "occupancy_vismp4_after.png"), occ_a)

    # method thumbs
    thumb_dir = os.path.join(ROOT, "..", "docs/figs/method_thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    cv2.imwrite(os.path.join(thumb_dir, "occupancy_update.png"), after,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    print("saved", out_dir)
    print("  occupancy_before_update.png")
    print("  occupancy_after_update.png")
    print("  occupancy_update_explainer.png")


if __name__ == "__main__":
    main()
