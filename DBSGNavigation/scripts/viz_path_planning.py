"""Visualize FMM path planning on occupancy map (one figure).

Example:
  python scripts/viz_path_planning.py --scene 4ok3usBNeis \\
    --goal_info examples/viz_0.8-0.6/0/3/goal_info.txt \\
    --replay_steps 34
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from map_planning_utils.fmm_planner import FMMPlanner
from map_planning_utils.mapper import Semantic_Mapping
from map_planning_utils.planner import Planner
from utils import pose as pu

# reuse occupancy replay helpers
from viz_occupancy_map_update import (
    build_cfg,
    c2w_to_sim_pose,
    load_frame,
    preprocess_obs,
    render_occupancy,
)

DEFAULT_DATA_ROOT = os.path.join(
    ROOT, "..", "DBSGConstruction/configs/data/habitat_val"
)
DEFAULT_SCENE = "4ok3usBNeis"


def read_goal_info(path):
    info = {}
    for line in open(path):
        line = line.strip()
        if "=" not in line or line.startswith("---"):
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


def rot_from_c2w(M):
    import quaternion

    R = np.array(M, dtype=np.float64)[:3, :3]
    return quaternion.from_rotation_matrix(R)


def world_to_local_map(pos_w, start_pos_w, start_rot_q, mapper):
    """Mirror NiceEnv.from_env_to_map (episode start as origin)."""
    source_position = np.array(start_pos_w, dtype=np.float32)
    goal_position = np.array(pos_w, dtype=np.float32)
    direction_vector = goal_position - source_position
    direction_vector_agent = pu.quaternion_rotate_vector(
        start_rot_q.inverse(), direction_vector
    )
    rho, phi = pu.cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    dist, angle = rho, -phi
    x = int(dist * np.cos(angle) * (100.0 / mapper.resolution))
    y = int(dist * np.sin(angle) * (100.0 / mapper.resolution))
    map_size = mapper.full_h
    robot_pos = [map_size // 2 + y, map_size // 2 + x]
    local_pos = [
        robot_pos[0] - int(mapper.lmb[0, 0].item()),
        robot_pos[1] - int(mapper.lmb[0, 2].item()),
    ]
    return np.array(local_pos, dtype=int)


def agent_local_cell(mapper):
    """Same indexing as Planner.plan start cell."""
    start_x, start_y, _, planning_window = mapper.get_planner_pose_inputs()
    r, c = start_y, start_x
    start = [
        int(r * 100.0 / mapper.resolution - planning_window[0]),
        int(c * 100.0 / mapper.resolution - planning_window[2]),
    ]
    map_pred = np.rint(mapper.local_map.cpu().numpy()[0, 0, ...])
    return np.array(pu.threshold_poses(start, map_pred.shape), dtype=int)


def build_fmm_and_trace(mapper, planner_obj, start_rc, goal_rc, selem=5, max_steps=120):
    """Replicate Planner._get_stg setup and trace short-term goals."""
    map_pred = np.rint(mapper.local_map.cpu().numpy()[0, 0, ...])
    gx1, gx2, gy1, gy2 = [int(x) for x in mapper.get_planner_pose_inputs()[3]]
    x1, y1, x2, y2 = 0, 0, map_pred.shape[0], map_pred.shape[1]

    start = pu.threshold_poses(list(start_rc), map_pred.shape)
    goal = np.zeros_like(map_pred, dtype=np.float32)
    gr, gc = pu.threshold_poses(list(goal_rc), map_pred.shape)
    goal[gr, gc] = 1.0

    def add_boundary(mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2), dtype=np.float32) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    traversible = skimage.morphology.binary_dilation(
        map_pred[x1:x2, y1:y2], planner_obj.selem
    ) != True
    traversible[
        planner_obj.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0
    ] = 1
    traversible[
        planner_obj.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1
    ] = 0
    sr, sc = int(start[0]), int(start[1])
    traversible[max(0, sr - 1) : sr + 2, max(0, sc - 1) : sc + 2] = 1

    traversible_b = add_boundary(traversible)
    goal_b = add_boundary(goal, value=0)
    se = skimage.morphology.disk(selem)
    goal_d = skimage.morphology.binary_dilation(goal_b, se) != True
    goal_d = 1 - goal_d * 1.0

    fmm = FMMPlanner(traversible_b)
    fmm.set_multi_goal(goal_d)

    state = [start[0] - x1 + 1, start[1] - y1 + 1]
    path = [(sr, sc)]
    for _ in range(max_steps):
        stg_x, stg_y, replan, stop = fmm.get_short_term_goal(state)
        if replan:
            stg_x, stg_y, _, stop = fmm.get_short_term_goal(state, 2)
        gx = int(stg_x + x1 - 1)
        gy = int(stg_y + y1 - 1)
        if stop or (gx, gy) == path[-1]:
            break
        path.append((gx, gy))
        state = [stg_x, stg_y]
        if math.hypot(gx - gr, gy - gc) < 3:
            break
    return path, fmm, map_pred


def draw_path_figure(local_map_np, path, start_rc, goal_rc, out_path):
    canvas = render_occupancy(local_map_np, agent_rc=tuple(start_rc))
    h, w = local_map_np[0, 0].shape
    scale = 2
    pts = [(int(c * scale), int(r * scale)) for r, c in path]
    orange_bgr = (0, 165, 255)
    green_bgr = (60, 220, 60)
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], orange_bgr, 3, cv2.LINE_AA)
    cv2.circle(
        canvas,
        (int(goal_rc[1] * scale), int(goal_rc[0] * scale)),
        8,
        green_bgr,
        -1,
        lineType=cv2.LINE_AA,
    )

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=200)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("FMM path planning on occupancy map", fontsize=11)
    ax.axis("off")
    ax.text(
        0.02, 0.98,
        "gray: explored, blue: obstacle | orange: FMM path | red: agent | green: goal",
        transform=ax.transAxes, fontsize=8, va="top", color="#333",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    cv2.imwrite(out_path.replace(".png", "_raw.png"), canvas,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def replay_mapper(scene_dir, frames, n_steps, device):
    cfg = build_cfg()
    mapper = Semantic_Mapping(cfg, device).eval()
    mapper.reset()
    last_pose = c2w_to_sim_pose(frames[0]["transform_matrix"])
    agent_h = torch.tensor([cfg.habitat.simulator.agents.main_agent.height], device=device)

    for i in range(n_steps):
        rgb, depth_m = load_frame(scene_dir, frames[i])
        state = preprocess_obs(rgb, depth_m)
        proc = torch.from_numpy(state).float().to(device).unsqueeze(0)
        curr_pose = c2w_to_sim_pose(frames[i]["transform_matrix"])
        dx, dy, do = pu.get_rel_pose_change(curr_pose, last_pose)
        last_pose = curr_pose
        pose_obs = torch.tensor([[dx, dy, do]], dtype=torch.float32, device=device)
        viz_obs = {"rgb": cv2.resize(rgb, (240, 320))}
        mapper.step(proc, pose_obs, agent_h, viz_obs=viz_obs)
    return mapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--goal_info", default="examples/viz_0.8-0.6/0/3/goal_info.txt")
    ap.add_argument("--replay_steps", type=int, default=34,
                    help="mapper replay length; ~34 matches mid-navigation before goal")
    ap.add_argument("--out", default="../docs/figs/path_planning_explainer.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    scene_dir = os.path.abspath(os.path.join(args.data_root, args.scene))
    gi = read_goal_info(os.path.abspath(os.path.join(ROOT, args.goal_info)))
    pred = json.loads(gi["pred_goal_pos"].replace(" ", ""))

    with open(os.path.join(scene_dir, "transforms.json")) as f:
        frames = json.load(f)["frames"]
    n = min(args.replay_steps, len(frames) - 1)

    device = torch.device(args.device)
    mapper = replay_mapper(scene_dir, frames, n, device)

    M0 = frames[0]["transform_matrix"]
    start_pos_w = np.array(M0, dtype=np.float64)[:3, 3]
    start_rot_q = rot_from_c2w(M0)

    goal_rc = world_to_local_map(pred, start_pos_w, start_rot_q, mapper)
    start_rc = agent_local_cell(mapper)

    planner_obj = Planner()
    planner_obj.reset(mapper)

    path, _, map_pred = build_fmm_and_trace(
        mapper, planner_obj, start_rc, goal_rc, selem=5, max_steps=150
    )
    local_map_np = mapper.local_map.cpu().numpy()

    out_path = os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    draw_path_figure(local_map_np, path, start_rc, goal_rc, out_path)

    thumb = os.path.join(ROOT, "..", "docs/figs/method_thumbs/path_planning.png")
    os.makedirs(os.path.dirname(thumb), exist_ok=True)
    raw = out_path.replace(".png", "_raw.png")
    if os.path.isfile(raw):
        import shutil
        shutil.copy(raw, thumb)

    print("replay_steps", n, "path_len", len(path))
    print("start", start_rc.tolist(), "goal", goal_rc.tolist())
    print("saved", out_path)
    print("saved", thumb)


if __name__ == "__main__":
    main()
