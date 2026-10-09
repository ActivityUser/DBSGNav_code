import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import sys
import skimage
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from utils.model import get_grid, ChannelPool, Flatten, NNBase
import utils.depth_utils as du

import cv2



class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, cfg, device):
        super(Semantic_Mapping, self).__init__()

        self.device = device
        self.screen_h = cfg.end2end_imagenav.mapper.frame_height
        self.screen_w = cfg.end2end_imagenav.mapper.frame_width
        self.resolution = cfg.end2end_imagenav.mapper.map_resolution
        self.z_resolution = cfg.end2end_imagenav.mapper.map_resolution
        self.global_downscaling = cfg.end2end_imagenav.mapper.global_downscaling
        self.map_size_cm = cfg.end2end_imagenav.mapper.map_size_cm // self.global_downscaling
        self.global_map_size_cm = cfg.end2end_imagenav.mapper.map_size_cm
        self.max_depth = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        self.n_channels = 3
        self.vision_range = 100
        self.dropout = 0.5
        self.fov = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
        self.du_scale = 1
        self.cat_pred_threshold = 5.0
        self.exp_pred_threshold = 1.0
        self.map_pred_threshold = 1.0
        self.num_sem_categories = 0

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-80 / self.z_resolution)
        self.agent_height = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1] * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)
        self.vfov = np.arctan((self.screen_h/2.) / self.camera_matrix.f)
        self.min_vision = self.agent_height / np.tan(self.vfov) # cm

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            1, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            1, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        # mapper utils 
        map_size = self.map_size_cm // self.resolution * self.global_downscaling
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / cfg.end2end_imagenav.mapper.global_downscaling)
        self.local_h = int(self.full_h / cfg.end2end_imagenav.mapper.global_downscaling)

        self.full_map = torch.zeros(1, 4, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(1, 4, self.local_w,
                                self.local_h).float().to(self.device)
        
        # 创新点6：动态语义更新——可选增量 CLIP 语义图
        self.enable_dynamic_semantic_update = getattr(
            cfg.end2end_imagenav.mapper, "enable_dynamic_semantic_update", False
        )
        self.dynamic_semantic_blend_alpha = float(getattr(
            cfg.end2end_imagenav.mapper, "dynamic_semantic_blend_alpha", 0.3
        ))
        self.dynamic_semantic_update_interval = max(
            1, int(getattr(cfg.end2end_imagenav.mapper, "dynamic_semantic_update_interval", 1))
        )
        # 仅在与历史语义足够一致时更新（<=-1 关闭门控）。
        self.dynamic_semantic_update_min_consistency = float(
            getattr(cfg.end2end_imagenav.mapper, "dynamic_semantic_update_min_consistency", -1.0)
        )
        # 动态 α：false | "count" | "consistency"（与论文：越一致 α 越小）
        _ada = getattr(
            cfg.end2end_imagenav.mapper, "dynamic_semantic_adaptive_alpha", False
        )
        if isinstance(_ada, str):
            _ada = _ada.strip().lower()
            self.dynamic_semantic_adaptive_alpha = _ada if _ada in ("count", "consistency") else False
        else:
            self.dynamic_semantic_adaptive_alpha = False
        self.dynamic_semantic_alpha_min = float(getattr(
            cfg.end2end_imagenav.mapper, "dynamic_semantic_alpha_min", 0.05
        ))
        self.dynamic_semantic_alpha_decay = float(getattr(
            cfg.end2end_imagenav.mapper, "dynamic_semantic_alpha_decay", 0.5
        ))
        self.dynamic_semantic_renormalize_cells = bool(getattr(
            cfg.end2end_imagenav.mapper, "dynamic_semantic_renormalize_cells", False
        ))
        self._clip_dim = int(getattr(cfg.end2end_imagenav, "clip_dim", 512))
        self.semantic_clip_map = None  # (1, clip_dim, full_w, full_h)
        self.semantic_update_count = None  # (1, 1, full_w, full_h) 可选，用于平均
        if self.enable_dynamic_semantic_update:
            self.semantic_clip_map = torch.zeros(
                1, self._clip_dim, self.full_w, self.full_h, device=self.device, dtype=torch.float32
            )
            self.semantic_update_count = torch.zeros(
                1, 1, self.full_w, self.full_h, device=self.device, dtype=torch.float32
            )
        
        self.full_pose = torch.zeros(1, 3).float().to(self.device)
        self.local_pose = torch.zeros(1, 3).float().to(self.device)

        self.origins = torch.zeros((1, 3)).float().to(self.device)
        self.lmb = torch.zeros((1, 4)).long().to(self.device)

        self.local_steps = 10

        # visualization utils
        self.vis_map = np.ones((self.local_w, self.local_h, 3)) 

        # planner utils
        self.planner_pose = np.zeros((3, ))


    def reset(self):

        self.timestep = 0
        self.init_map_pose()


    def init_map_pose(self):

        def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if self.global_downscaling > 1:
                gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
                gx2, gy2 = gx1 + local_w, gy1 + local_h
                if gx1 < 0:
                    gx1, gx2 = 0, local_w
                if gx2 > full_w:
                    gx1, gx2 = full_w - local_w, full_w

                if gy1 < 0:
                    gy1, gy2 = 0, local_h
                if gy2 > full_h:
                    gy1, gy2 = full_h - local_h, full_h
            else:
                gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

            return [gx1, gx2, gy1, gy2]

        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.global_map_size_cm / 100.0 / 2.0
        # 创新点6：重置时清空动态语义图（可选保留以支持跨 episode 累积）
        if self.semantic_clip_map is not None:
            self.semantic_clip_map.fill_(0.)
        if self.semantic_update_count is not None:
            self.semantic_update_count.fill_(0.)

        locs = self.full_pose
        self.planner_pose[0], self.planner_pose[1], self.planner_pose[2] = \
            locs[0, 1].item(), locs[0, 1].item(), locs[0, 2].item()
        # planner_pose_inputs[:, :3] = locs

        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]

        self.full_map[0, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        temp_lmb = get_local_map_boundaries((loc_r, loc_c),
                                            (self.local_w, self.local_h),
                                            (self.full_w, self.full_h))
        self.lmb[0, 0], self.lmb[0, 1], self.lmb[0, 2], self.lmb[0, 3] = \
            int(temp_lmb[0]), int(temp_lmb[1]), int(temp_lmb[2]), int(temp_lmb[3])

        # planner_pose_inputs[e, 3:] = lmb[e]
        self.origins[0, 0], self.origins[0,1], self.origins[0,2] = \
            self.lmb[0, 2] * self.resolution / 100.0, self.lmb[0, 0] * self.resolution / 100.0, 0.


        self.local_map[0] = self.full_map[0, :,
                                self.lmb[0, 0]:self.lmb[0, 1],
                                self.lmb[0, 2]:self.lmb[0, 3]]
        self.local_pose[0] = self.full_pose[0] - \
            self.origins[0]


    def update_map_pose(self):

        def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
            loc_r, loc_c = agent_loc
            local_w, local_h = local_sizes
            full_w, full_h = full_sizes

            if self.global_downscaling > 1:
                gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
                gx2, gy2 = gx1 + local_w, gy1 + local_h
                if gx1 < 0:
                    gx1, gx2 = 0, local_w
                if gx2 > full_w:
                    gx1, gx2 = full_w - local_w, full_w

                if gy1 < 0:
                    gy1, gy2 = 0, local_h
                if gy2 > full_h:
                    gy1, gy2 = full_h - local_h, full_h
            else:
                gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

            return [gx1, gx2, gy1, gy2]


        self.full_map[0, :, self.lmb[0, 0]:self.lmb[0, 1], self.lmb[0, 2]:self.lmb[0, 3]] = \
            self.local_map[0]
        self.full_pose[0] = self.local_pose[0] + \
            self.origins[0]

        locs = self.full_pose[0]
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]

        temp_lmb = get_local_map_boundaries((loc_r, loc_c),
                                            (self.local_w, self.local_h),
                                            (self.full_w, self.full_h))
        self.lmb[0, 0], self.lmb[0, 1], self.lmb[0, 2], self.lmb[0, 3] = \
            int(temp_lmb[0]), int(temp_lmb[1]), int(temp_lmb[2]), int(temp_lmb[3])

        # planner_pose_inputs[e, 3:] = lmb[e]
        self.origins[0, 0], self.origins[0,1], self.origins[0,2] = \
            self.lmb[0, 2] * self.resolution / 100.0, self.lmb[0, 0] * self.resolution / 100.0, 0.

        self.local_map[0] = self.full_map[0, :,
                                self.lmb[0, 0]:self.lmb[0, 1],
                                self.lmb[0, 2]:self.lmb[0, 3]]
        self.local_pose[0] = self.full_pose[0] - \
            self.origins[0]



    def step(self, obs, pose_obs, agent_heights, viz_obs=None, viz_image_goal=None, clip_encoder=None):

        self.timestep += 1

        _, self.local_map, _, self.local_pose = \
              self.forward(obs, pose_obs, self.local_map, self.local_pose, agent_heights)
        
        locs = self.local_pose.cpu().numpy()
        origins = self.origins.cpu().numpy()

        # prepare for planner pose input
        self.planner_pose = locs[0] + origins[0]

        self.local_map[:, 2, :, :].fill_(0.)
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]
        self.local_map[0, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        if self.timestep % self.local_steps == 0:
            self.update_map_pose()

        # 创新点6：基于新观测增量更新语义（CLIP）图
        if (
            self.enable_dynamic_semantic_update
            and clip_encoder is not None
            and self.semantic_clip_map is not None
            and self.timestep % self.dynamic_semantic_update_interval == 0
        ):
            self._update_semantic_from_observation(obs, viz_obs, clip_encoder)

        self.visualize(obs, viz_obs, viz_image_goal)
        
        map_struct = {
            'map_pose': (loc_r, loc_c, locs[0, 2]),
            'local_map': self.local_map.cpu().numpy(),
            'vis_map': self.vis_map
        }

        return map_struct

    def get_planner_pose_inputs(self):
        gx1, gx2, gy1, gy2 = int(self.lmb[0, 0].item()), int(self.lmb[0, 1].item()), \
                             int(self.lmb[0, 2].item()), int(self.lmb[0, 3].item())
        planning_window = [gx1, gx2, gy1, gy2]
        return self.planner_pose[0], self.planner_pose[1], self.planner_pose[2], \
               planning_window

    def _update_semantic_from_observation(self, obs, viz_obs, clip_encoder, blend_alpha=None):
        """
        创新点6：基于当前帧观测增量更新语义 CLIP 图。
        可见窗口内 EMA：S_new = (1-α) S_old + α v_t；v_t 已 L2 归一化。
        α 由 mapper.dynamic_semantic_adaptive_alpha 决定：固定值 / count 衰减 / consistency（与 S_old 余弦一致则 α 小）。
        """
        if blend_alpha is None:
            blend_alpha = self.dynamic_semantic_blend_alpha
        # 获取 RGB：优先 viz_obs（与传感器一致），否则 obs 前 3 通道
        if viz_obs is not None and isinstance(viz_obs, dict) and "rgb" in viz_obs:
            rgb = np.asarray(viz_obs["rgb"], dtype=np.uint8)
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                pass
            else:
                rgb = (obs[0, :3].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            o = obs[0, :3].permute(1, 2, 0).cpu().numpy()
            rgb = (o * 255).clip(0, 255).astype(np.uint8) if o.max() <= 1.01 else o.astype(np.uint8)
        try:
            vec = clip_encoder(rgb)
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec).float().to(self.device)
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)
            # L2 归一化便于后续相似度等使用
            vec = vec / (vec.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            gx1, gx2 = int(self.lmb[0, 0].item()), int(self.lmb[0, 1].item())
            gy1, gy2 = int(self.lmb[0, 2].item()), int(self.lmb[0, 3].item())
            clip_dim = vec.shape[-1]
            if clip_dim != self._clip_dim or gx2 <= gx1 or gy2 <= gy1:
                return
            region = self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2]
            vt_b = vec.view(-1, 1, 1).expand(clip_dim, gx2 - gx1, gy2 - gy1)
            alpha_base = float(blend_alpha)
            raw_norm = region.norm(dim=0)
            seen_mask = raw_norm >= 1e-7
            trust_mask = None
            if self.dynamic_semantic_update_min_consistency > -1.0:
                safe_norm = raw_norm.clamp(min=1e-8)
                sim_gate = (region * vt_b).sum(dim=0) / safe_norm
                sim_gate = sim_gate.clamp(-1.0, 1.0)
                trust_mask = (~seen_mask) | (
                    sim_gate >= self.dynamic_semantic_update_min_consistency
                )
                if not bool(trust_mask.any()):
                    return
                trust_exp = trust_mask.unsqueeze(0).expand_as(region)
            if self.dynamic_semantic_adaptive_alpha == "consistency":
                # 每格：sim = cos(S_old, v_t)，sim_01=(sim+1)/2；α=α_min+(α_base-α_min)(1-sim_01)；无先验(全零)用 α_base
                unseen = raw_norm < 1e-7
                safe_norm = raw_norm.clamp(min=1e-8)
                sim = (region * vt_b).sum(dim=0) / safe_norm
                sim = sim.clamp(-1.0, 1.0)
                sim01 = (sim + 1.0) * 0.5
                a_min, a_max = self.dynamic_semantic_alpha_min, alpha_base
                alpha_map = a_min + (a_max - a_min) * (1.0 - sim01)
                alpha_map = alpha_map.clamp(min=a_min, max=a_max)
                if unseen.any():
                    alpha_map = alpha_map.clone()
                    alpha_map[unseen] = alpha_base
                alpha_exp = alpha_map.unsqueeze(0).expand_as(region)
                new_region = (
                    (1.0 - alpha_exp) * region + alpha_exp * vt_b
                )
                if trust_mask is not None:
                    new_region = torch.where(trust_exp, new_region, region)
                if self.dynamic_semantic_renormalize_cells:
                    new_region = Semantic_Mapping._l2_normalize_semantic_patch(new_region)
                self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2] = new_region
            elif self.dynamic_semantic_adaptive_alpha == "count" and self.semantic_update_count is not None:
                cnt = self.semantic_update_count[0, 0, gx1:gx2, gy1:gy2]
                k = self.dynamic_semantic_alpha_decay
                alpha_map = alpha_base / (1.0 + k * cnt)
                alpha_map = alpha_map.clamp(
                    min=self.dynamic_semantic_alpha_min, max=alpha_base
                )
                alpha_exp = alpha_map.unsqueeze(0).expand_as(region)
                new_region = (
                    (1.0 - alpha_exp) * region + alpha_exp * vt_b
                )
                if trust_mask is not None:
                    new_region = torch.where(trust_exp, new_region, region)
                if self.dynamic_semantic_renormalize_cells:
                    new_region = Semantic_Mapping._l2_normalize_semantic_patch(new_region)
                self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2] = new_region
            else:
                # 固定 α
                a = alpha_base
                new_region = (
                    (1.0 - a) * region + a * vt_b
                )
                if trust_mask is not None:
                    new_region = torch.where(trust_exp, new_region, region)
                if self.dynamic_semantic_renormalize_cells:
                    new_region = Semantic_Mapping._l2_normalize_semantic_patch(new_region)
                self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2] = new_region
            if self.semantic_update_count is not None:
                if trust_mask is None:
                    self.semantic_update_count[0, 0, gx1:gx2, gy1:gy2] += 1.0
                else:
                    self.semantic_update_count[0, 0, gx1:gx2, gy1:gy2] += (
                        trust_mask.float()
                    )
        except Exception:
            pass

    @staticmethod
    def _l2_normalize_semantic_patch(patch: torch.Tensor) -> torch.Tensor:
        """(D,H,W) 每格 CLIP 向量单位化；近零格不变，便于与 goal 余弦一致。"""
        norms = patch.norm(dim=0, keepdim=True).clamp(min=1e-8)
        n0 = patch.norm(dim=0)
        zero = n0 < 1e-7
        normed = patch / norms
        return torch.where(zero.unsqueeze(0), patch, normed)

    def get_semantic_clip_region(self, gx1, gx2, gy1, gy2):
        """
        创新点6：读取语义 CLIP 图中某一区域的向量（可用于目标匹配等）。
        返回 (clip_dim,) 或 (n, clip_dim) 的 L2 归一化向量；若未启用动态语义则返回 None。
        """
        if self.semantic_clip_map is None:
            return None
        gx1, gx2 = max(0, gx1), min(self.full_w, gx2)
        gy1, gy2 = max(0, gy1), min(self.full_h, gy2)
        if gx2 <= gx1 or gy2 <= gy1:
            return None
        region = self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2]
        region = region.reshape(self._clip_dim, -1).mean(dim=1)
        region = region / (region.norm().clamp(min=1e-8))
        return region.cpu().numpy()

    def compute_local_semantic_correlation(self, goal_clip: np.ndarray):
        """
        创新点 6 接入规划：当前局部窗口内，每个格子与目标图 CLIP 的余弦相似度。
        与 local_map[0,0] 同形状；从未更新过的格子为 nan。
        """
        if self.semantic_clip_map is None:
            return None
        gx1, gx2 = int(self.lmb[0, 0].item()), int(self.lmb[0, 1].item())
        gy1, gy2 = int(self.lmb[0, 2].item()), int(self.lmb[0, 3].item())
        if gx2 <= gx1 or gy2 <= gy1:
            return None
        sem = self.semantic_clip_map[0, :, gx1:gx2, gy1:gy2]
        g = torch.from_numpy(
            np.asarray(goal_clip, dtype=np.float32).reshape(-1)
        ).to(sem.device)
        g = g / (g.norm().clamp(min=1e-8))
        corr = (sem * g.view(-1, 1, 1)).sum(dim=0)
        norms = sem.norm(dim=0)
        corr_np = corr.cpu().numpy().astype(np.float64)
        norms_np = norms.cpu().numpy()
        out = np.full_like(corr_np, np.nan, dtype=np.float64)
        valid = norms_np > 1e-6
        # 目标向量已 L2 归一化；栅格向量经 EMA 后一般非单位范数，须除 ||sem|| 才是余弦相似度（与 docstring 一致）。
        cos_vals = corr_np[valid] / np.maximum(norms_np[valid], 1e-8)
        out[valid] = np.clip(cos_vals, -1.0, 1.0)
        return out

    def visualize(self, obs, viz_obs, viz_image_goal):

        local_map = self.local_map.cpu().numpy()

        vis_local_map = np.ones((self.local_h, self.local_h, 3))
        vis_local_map[local_map[0, 0, ...] > 0] = [0.32, 0.32, 0.32]
        vis_local_map[local_map[0, 3, ...] > 0] = [0., 0., 0.92]
        vis_local_map = (vis_local_map * 255).astype(np.uint8)
        vis_local_map = cv2.resize(vis_local_map, (self.local_h*2, self.local_w*2), \
                                   interpolation=cv2.INTER_NEAREST)

        resized_h = self.local_h * 2
        resized_w = int(self.local_h / self.screen_h * self.screen_w * 2)
        if viz_image_goal is None:
            self.vis_map = (np.ones((self.local_h*2 + 20*2, self.local_w*2 + 20*3 + \
                                    resized_w, 3)) * 255).astype(np.uint8)
        else:
            self.vis_map = (np.ones((self.local_h*2 + 20*2, self.local_w*2 + 20*4 + \
                                    resized_w + resized_w, 3)) * 255).astype(np.uint8)
            resized_image_goal = cv2.resize(viz_image_goal, \
                                    (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        if viz_obs is None:
            resized_rgb = cv2.resize(obs[0, :3, ...].cpu().numpy().transpose(1,2,0), \
                                    (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        else:
            rgb = viz_obs['rgb'].astype(np.uint8)
            resized_rgb = cv2.resize(rgb, \
                                    (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
            
        self.vis_map[20:self.local_h*2+20, 20:resized_w + 20, :] = \
            resized_rgb[:, :, ::-1]
        self.vis_map[20:self.local_h*2+20, 20 + resized_w + 20: \
                     20 + resized_w + 20 + self.local_w*2] = vis_local_map
        if viz_image_goal is not None:
            self.vis_map[20:self.local_h*2+20, 20 + resized_w + 20 + self.local_w*2 + 20: \
                        20 + resized_w + 20 + self.local_w*2 + 20 + resized_w, :] = \
            resized_image_goal[:, :, ::-1]
        

    def euclidean_distance_to_goal(self, goal_pos):
        '''
        the goal pos here is local pose
        '''

        locs = self.local_pose.cpu().numpy()
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [int(r * 100.0 / self.resolution),
                        int(c * 100.0 / self.resolution)]
        
        return np.sqrt(np.square((loc_r - goal_pos[0]) * self.resolution / 100.) + \
                       np.square(loc_c - goal_pos[1]) * self.resolution / 100.)

    def compute_goalmap_from_mask(self, obs, goal_id):
        '''
        mask should only be 0 or 1
        '''
        semantics = obs['semantic'][..., 0]
        mask = np.zeros_like(semantics)
        mask[semantics == goal_id] = 1

        rgb = obs['rgb']
        depth = obs['depth']

        depth = depth[:, :, 0] * self.max_depth * 100.

        start_x, start_y, start_o, planning_window = self.get_planner_pose_inputs()
        r, c = start_y, start_x
        start = int(r * 100.0 / self.resolution - planning_window[0]), \
                 int(c * 100.0 / self.resolution - planning_window[2])

        selem = skimage.morphology.disk(3)
        semantic_mask = skimage.morphology.erosion(mask, selem)

        if not np.any(semantic_mask > 0):
            return None
        else:

            depth_h, depth_w = np.where(semantic_mask > 0)
            goal_dis = depth[depth_h, depth_w] / self.resolution

            goal_angle = - self.fov / 2 * (depth_w - rgb.shape[1]/2) \
            / (rgb.shape[1]/2)
            goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
            goal_map = np.zeros((self.local_w, self.local_h))
            goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
            goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
            goal_map[goal[0], goal[1]] = 1
        
            return goal_map 


    def forward(self, obs, pose_obs, maps_last, poses_last, agent_heights):

        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]
        #feature the last row of obs in the last dimention
        # obs[:, -1, :, :] = 0.
        # obs[:, -1, -1, :] = 1.

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        # agent_view_t = du.transform_camera_view_t(
        #     point_cloud_t, self.agent_height, 0, self.device)
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, agent_heights * 100., 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        # self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
        #     obs[:, 4:, :, :]
        # ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(35 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)
        floor_z = int(-35 / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)


        min_vision_std = int(self.min_vision // z_resolution)
        around_floor_proj = voxels[..., :min_z].sum(4) 
        under_floor_proj = voxels[..., floor_z:min_z].sum(4) 
        under_floor_proj = (under_floor_proj == 0.0).float()# have floor = 0, no floor or not deteced = 1
        under_floor_proj = under_floor_proj * around_floor_proj # no floor and detected = 1

        # sever condition
        # index = (voxels[:, -1, :, :, :].sum((1, 2, 3))-voxels[:, -1, :, :, floor_z:].sum((1, 2, 3))) 
        # index = voxels[:, -1, :, :, :floor_z].sum((1, 2, 3))
        # index = torch.nonzero(index > 10)

        # strategy 2 : compare the last row of depth image's depth value
        # preprocess the depth map due to invalid value of depth
        # if the number is greater than a half then ...
        replace_element = torch.ones_like(depth[:, -1, :]) * self.min_vision
        re_depth = torch.where(depth[:, -1, :] < 3000, depth[:, -1, :], replace_element)
        count = ((re_depth - self.min_vision - 60) > 0).sum(dim=1)
        index = torch.nonzero(count > (re_depth.shape[1] / 4))
        # index = torch.nonzero(((re_depth - self.min_vision - 30) > 0).any(dim=1) )

        under_floor_proj[index, 0:1, min_vision_std:min_vision_std+1, \
                (self.vision_range-6)//2 : (self.vision_range+6)//2] \
                    = 1.

        # if torch.equal(torch.zeros_like(around_floor_proj[:, :, :2*min_vision_std, \
        #     (self.vision_range-min_vision_std)//2 : (self.vision_range+min_vision_std)//2]), \
        #         around_floor_proj[:, :, :min_vision_std, \
        #     (self.vision_range-min_vision_std)//2 : (self.vision_range+min_vision_std)//2]):
        #     # extreme condition
        #     under_floor_proj[:, 0:1, 1:min_vision_std//2, \
        #         (self.vision_range-min_vision_std)//2 : (self.vision_range+min_vision_std)//2] \
        #             = 1.

        fp_map_pred = agent_height_proj[:, 0:1, :, :] + under_floor_proj[:, 0:1, :, :]
        # fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses
