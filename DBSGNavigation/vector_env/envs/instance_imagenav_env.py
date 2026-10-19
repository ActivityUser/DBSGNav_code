import json
import base64
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
import cv2
import magnum as mn
import Quaternion
from tqdm import tqdm
# from base_env import base_env
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
from vector_env.envs.base_env import base_env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from torchvision import transforms
import utils.pose as pu
from map_planning_utils.mapper import Semantic_Mapping
from map_planning_utils.planner import Planner
from map_planning_utils.open_vocab_retriever import OpenVocabRetriever
from map_planning_utils.semantic_geometric_fusion import adaptive_weights
from map_planning_utils.semantic_gaussian_goal import SemanticGaussianGoal
from PIL import Image
import torch
import os
from glob import glob
from natsort import natsorted
import logging
from urllib import request as urllib_request
from urllib import error as urllib_error


class NiceEnv(base_env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env, dataset, rank):
        super().__init__(config_env, dataset, rank)
        self.goal_radius = 1.0
        self.follower = ShortestPathFollower(
            self.habitat_env.sim, self.goal_radius, False
        )
        self.base_dump_location = config_env.end2end_imagenav.dump_location
        self.base_dump_location = os.path.join(self.base_dump_location, str(rank))
        print(f"rank: {self.rank} env initialize suceessful !")

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((self.frame_height, self.frame_width),
                               interpolation=Image.NEAREST)])

        gpu_id = config_env.end2end_imagenav.gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.mapper = Semantic_Mapping(config_env, self.device)

        self.planner = Planner()
        _mp = config_env.end2end_imagenav.mapper
        self.dynamic_semantic_planner = bool(
            getattr(_mp, "dynamic_semantic_planner", True)
        )
        self.dynamic_semantic_planner_thr = float(
            getattr(_mp, "dynamic_semantic_planner_thr", 0.18)
        )
        self.dynamic_semantic_planner_radius = int(
            getattr(_mp, "dynamic_semantic_planner_radius", 22)
        )
        self.dynamic_semantic_planner_min_timestep = int(
            getattr(_mp, "dynamic_semantic_planner_min_timestep", 8)
        )
        self.dynamic_semantic_planner_peaks_only = bool(
            getattr(_mp, "dynamic_semantic_planner_peaks_only", False)
        )
        self.dynamic_semantic_planner_max_peaks = int(
            getattr(_mp, "dynamic_semantic_planner_max_peaks", 12)
        )
        self.dynamic_semantic_planner_peak_min_sep = int(
            getattr(_mp, "dynamic_semantic_planner_peak_min_sep", 3)
        )
        self.dynamic_semantic_planner_max_goal_dist = float(
            getattr(
                config_env.end2end_imagenav,
                "dynamic_semantic_planner_max_goal_dist",
                0.0,
            )
        )
        self.dynamic_semantic_planner_disable_when_ov_geo_floor_weak = bool(
            getattr(
                config_env.end2end_imagenav,
                "dynamic_semantic_planner_disable_when_ov_geo_floor_weak",
                False,
            )
        )

        # viz params
        self.viz = config_env.end2end_imagenav.viz_params.viz
        self.viz_failure_case = config_env.end2end_imagenav.viz_params.viz_failure_case
        self.viz_failure_max_saved = int(
            getattr(config_env.end2end_imagenav.viz_params, "viz_failure_max_saved", 0)
        )
        self._viz_failure_saved_count = 0
        # Quota mode: viz=false, save up to N failure episodes (video + goal assets) per process.
        self._viz_failure_quota_mode = bool(
            self.viz_failure_case and not self.viz and self.viz_failure_max_saved > 0
        )
        _viz_save_goal = getattr(
            config_env.end2end_imagenav.viz_params, "viz_save_goal_assets", None
        )
        if _viz_save_goal is None:
            self.viz_save_goal_assets = bool(self.viz or self._viz_failure_quota_mode)
        else:
            self.viz_save_goal_assets = bool(_viz_save_goal)
        if self.viz or self._viz_failure_quota_mode:
            self.viz_obs = []

        self.similarity_method = config_env.end2end_imagenav.particle_params.method
        self.remap_goal = config_env.end2end_imagenav.particle_params.remap_goal

        self.use_open_vocab_retrieval = getattr(
            config_env.end2end_imagenav, "use_open_vocab_retrieval", False
        )
        self.open_vocab_top_k = getattr(
            config_env.end2end_imagenav, "open_vocab_top_k", 10
        )
        self.open_vocab_semantic_weight = getattr(
            config_env.end2end_imagenav, "open_vocab_semantic_weight", 0.3
        )
        # 创新点4：语义-几何融合权重与归一化
        self.fusion_geometric_weight = getattr(
            config_env.end2end_imagenav, "fusion_geometric_weight", None
        )
        if self.fusion_geometric_weight is None:
            self.fusion_geometric_weight = 1.0 - self.open_vocab_semantic_weight
        self.fusion_normalize_geometric = getattr(
            config_env.end2end_imagenav, "fusion_normalize_geometric", "minmax"
        )
        self.fusion_adaptive_weights = getattr(
            config_env.end2end_imagenav, "fusion_adaptive_weights", False
        )
        # OV 融合：是否把语义/几何分计入加权求和（False 时该项为 0，α/β 仍保留在配置中）
        self.open_vocab_fusion_use_semantic_score = bool(
            getattr(
                config_env.end2end_imagenav,
                "open_vocab_fusion_use_semantic_score",
                True,
            )
        )
        self.open_vocab_fusion_use_geometric_score = bool(
            getattr(
                config_env.end2end_imagenav,
                "open_vocab_fusion_use_geometric_score",
                True,
            )
        )
        self.base_scene_path = getattr(
            config_env.end2end_imagenav, "base_scene_path", ""
        ).strip()
        clip_model_name = getattr(
            config_env.end2end_imagenav, "clip_model_name", "ViT-B-32"
        )
        clip_dim = getattr(config_env.end2end_imagenav, "clip_dim", 512)
        _nav_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        clip_pretrained_path = getattr(
            config_env.end2end_imagenav, "clip_pretrained_path", ""
        )
        if clip_pretrained_path and not os.path.isabs(clip_pretrained_path):
            clip_pretrained_path = os.path.normpath(os.path.join(_nav_root, clip_pretrained_path))
        self.use_semantic_gaussian_goal = getattr(
            config_env.end2end_imagenav, "use_semantic_gaussian_goal", False
        )
        self.gaussian_map_root = getattr(
            config_env.end2end_imagenav, "gaussian_map_root", ""
        ).strip()
        if self.gaussian_map_root and not os.path.isabs(self.gaussian_map_root):
            self.gaussian_map_root = os.path.normpath(os.path.join(_nav_root, self.gaussian_map_root))
        if self.base_scene_path and not os.path.isabs(self.base_scene_path):
            self.base_scene_path = os.path.normpath(os.path.join(_nav_root, self.base_scene_path))
        self.semantic_gaussian_top_k = getattr(
            config_env.end2end_imagenav, "semantic_gaussian_top_k", 1
        )
        self.semantic_gaussian_min_sim = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_min_sim", 0.85)
        )
        # SG confidence gate: if SG is not highly confident, prefer OV result (with geometric verification).
        self.semantic_gaussian_high_conf_sim = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_high_conf_sim", 0.90)
        )
        # SG-OV 目标分歧门：两者相距过远时优先 OV，除非 SG 置信度极高。
        self.semantic_gaussian_ov_disagree_dist = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_ov_disagree_dist", 3.0)
        )
        self.semantic_gaussian_ov_disagree_high_conf = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_ov_disagree_high_conf",
                0.997,
            )
        )
        # SG/OV 相距超过该值(米)时强制采用 OV（不受 sg_sim 上限绕过），修复「极高 sim 仍锁错 SG」的洞
        self.semantic_gaussian_ov_disagree_force_dist = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_ov_disagree_force_dist",
                0.0,
            )
        )
        # OV 候选融合分时对「离 agent 更远」施加惩罚，抑制 CLIP 长尾误检（0 关闭）
        self.open_vocab_agent_dist_penalty = float(
            getattr(config_env.end2end_imagenav, "open_vocab_agent_dist_penalty", 0.0)
        )
        # SG-agent 距离门：离当前 agent 过远的 SG 候选直接拒绝，防止大偏移点误导导航。
        self.semantic_gaussian_max_agent_goal_dist = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_max_agent_goal_dist", 10.0)
        )
        # OV 长尾诊断：离 agent 过远时打印告警；在 assist 模式下可作为「OV证据偏弱」信号。
        self.open_vocab_max_agent_goal_dist = float(
            getattr(config_env.end2end_imagenav, "open_vocab_max_agent_goal_dist", 8.0)
        )
        # DISK+LightGlue 匹配点数量下限：融合分最高者若低于此值则按融合分顺序尝试次优（0 关闭）
        self.open_vocab_min_geometric_matches = float(
            getattr(config_env.end2end_imagenav, "open_vocab_min_geometric_matches", 0.0)
        )
        # OV 熔断：当 OV 证据弱时，不允许其覆盖 SG（默认关闭，保持历史行为）
        self.open_vocab_circuit_breaker_geo_floor = bool(
            getattr(config_env.end2end_imagenav, "open_vocab_circuit_breaker_geo_floor", False)
        )
        self.open_vocab_circuit_breaker_long_tail = bool(
            getattr(config_env.end2end_imagenav, "open_vocab_circuit_breaker_long_tail", False)
        )
        # SG assist mode：OV 主导，SG 仅在 OV 证据偏弱且 SG 满足高置信条件时辅助接管。
        self.semantic_gaussian_assist_mode = bool(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_assist_mode", False)
        )
        self.semantic_gaussian_assist_min_sim = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_assist_min_sim", 0.96)
        )
        # <=0 关闭 gap 门控；>0 时要求 SG/OV 预测点间距不超过该阈值（米）。
        self.semantic_gaussian_assist_max_gap = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_assist_max_gap", 4.0)
        )
        self.semantic_gaussian_assist_only_when_ov_weak = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_assist_only_when_ov_weak",
                True,
            )
        )
        self.semantic_gaussian_assist_require_sg_closer_than_ov = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_assist_require_sg_closer_than_ov",
                True,
            )
        )
        # SG 与 OV 到 agent 距离比较时允许的松弛量（米）。
        self.semantic_gaussian_assist_sg_closer_margin = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_assist_sg_closer_margin",
                0.5,
            )
        )
        # SG-navigable goal：把"高 cosine Gaussian 中心"转成"可导航视点"
        # （top-K 聚类 + 地面投影 + 朝 agent snap），默认开启；可通过 yaml 关闭以保留旧行为。
        self.semantic_gaussian_use_navigable_goal = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_use_navigable_goal",
                True,
            )
        )
        self.semantic_gaussian_nav_sim_floor = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_sim_floor",
                0.20,
            )
        )
        self.semantic_gaussian_nav_top_k_cluster = int(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_top_k_cluster",
                200,
            )
        )
        self.semantic_gaussian_nav_eps_m = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_eps_m",
                0.6,
            )
        )
        self.semantic_gaussian_nav_min_cluster_size = int(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_min_cluster_size",
                6,
            )
        )
        self.semantic_gaussian_nav_snap_distance_m = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_snap_distance_m",
                0.8,
            )
        )
        self.semantic_gaussian_nav_floor_y_band = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_nav_floor_y_band",
                1.5,
            )
        )
        # assist 门新增：要求 SG cluster 大小不低于此值（与 sim 互补，更稳健）。
        self.semantic_gaussian_assist_min_cluster_size = int(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_assist_min_cluster_size",
                0,
            )
        )
        # 方案 A：把 SG 当作 OV 候选 reranking 的 3D spatial prior。
        # final_score = α·sem + β·geom + γ·sg_spatial_prior。weight=0 关闭。
        self.sg_spatial_prior_weight = float(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_weight", 0.0)
        )
        self.sg_spatial_prior_radius_m = float(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_radius_m", 1.5)
        )
        self.sg_spatial_prior_topk_local = int(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_topk_local", 20)
        )
        self.sg_spatial_prior_aggregator = str(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_aggregator", "topk_mean")
        )
        self.sg_spatial_prior_normalize = str(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_normalize", "minmax")
        )
        # 信号强度门控：若 raw_max-raw_min 小于该阈值（cos sim 量纲），认为是噪声，跳过 rerank。
        self.sg_spatial_prior_min_range = float(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_min_range", 0.02)
        )
        # === 路径 1a：颜色直方图 prior ===
        # SG 周围 Gaussian rgb_colors 算 HSV 直方图，与 target image 直方图比对。
        self.sg_color_prior_weight = float(
            getattr(config_env.end2end_imagenav, "sg_color_prior_weight", 0.0)
        )
        self.sg_color_prior_radius_m = float(
            getattr(config_env.end2end_imagenav, "sg_color_prior_radius_m", 1.5)
        )
        self.sg_color_prior_bins = int(
            getattr(config_env.end2end_imagenav, "sg_color_prior_bins", 8)
        )
        self.sg_color_prior_min_neighbors = int(
            getattr(config_env.end2end_imagenav, "sg_color_prior_min_neighbors", 30)
        )
        self.sg_color_prior_normalize = str(
            getattr(config_env.end2end_imagenav, "sg_color_prior_normalize", "minmax")
        )
        self.sg_color_prior_min_range = float(
            getattr(config_env.end2end_imagenav, "sg_color_prior_min_range", 0.02)
        )

        # === 路径 2a：3D 拓扑 prior（contextual reasoning over instance scene graph）===
        self.sg_topo_prior_weight = float(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_weight", 0.0)
        )
        self.sg_topo_prior_radius_m = float(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_radius_m", 2.5)
        )
        self.sg_topo_prior_topn_classes = int(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_topn_classes", 4)
        )
        self.sg_topo_prior_softmax_temp = float(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_softmax_temp", 0.1)
        )
        self.sg_topo_prior_normalize = str(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_normalize", "minmax")
        )
        self.sg_topo_prior_min_range = float(
            getattr(config_env.end2end_imagenav, "sg_topo_prior_min_range", 0.02)
        )

        # 是否在 SG load_scene 失败时仍允许 OV 单独工作（默认开）。
        self.sg_spatial_prior_skip_if_no_scene = bool(
            getattr(config_env.end2end_imagenav, "sg_spatial_prior_skip_if_no_scene", True)
        )
        # API-based candidate rerank (optional): ask a local OpenAI-compatible endpoint
        # to choose one candidate among top-k using target/candidate images.
        self.open_vocab_api_rerank_enable = bool(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_enable", False)
        )
        self.open_vocab_api_rerank_url = str(
            getattr(
                config_env.end2end_imagenav,
                "open_vocab_api_rerank_url",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
        )
        self.open_vocab_api_rerank_model = str(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_model", "")
        )
        self.open_vocab_api_rerank_topk = int(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_topk", 3)
        )
        self.open_vocab_api_rerank_accept_mode = str(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_accept_mode", "always")
        ).strip().lower()
        if self.open_vocab_api_rerank_accept_mode not in ("always", "score_margin"):
            print(
                "[ov_api_rerank] invalid accept_mode=%s, fallback to always"
                % self.open_vocab_api_rerank_accept_mode
            )
            self.open_vocab_api_rerank_accept_mode = "always"
        self.open_vocab_api_rerank_margin_threshold = float(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_margin_threshold", 0.05)
        )
        self.open_vocab_api_rerank_timeout_s = float(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_timeout_s", 12.0)
        )
        self.open_vocab_api_rerank_temperature = float(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_temperature", 0.0)
        )
        self.open_vocab_api_rerank_enable_thinking = bool(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_enable_thinking", False)
        )
        self.open_vocab_api_rerank_image_max_side = int(
            getattr(config_env.end2end_imagenav, "open_vocab_api_rerank_image_max_side", 448)
        )
        # SG-stage cluster-candidate path (for fair ablation, can be enabled
        # independently from API rerank).
        self.semantic_gaussian_cluster_disambig_enable = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_cluster_disambig_enable",
                False,
            )
        )
        # SG-stage VLM disambiguation (inst_v3_vlm ablation only when enabled in yaml).
        self.semantic_gaussian_api_rerank_enable = bool(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_api_rerank_enable", False)
        )
        self.semantic_gaussian_api_rerank_url = str(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_url",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
        )
        self.semantic_gaussian_api_rerank_model = str(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_api_rerank_model", "")
        )
        self.semantic_gaussian_api_rerank_topk = int(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_api_rerank_topk", 3)
        )
        self.semantic_gaussian_api_rerank_sim_margin = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_api_rerank_sim_margin", 0.03)
        )
        self.semantic_gaussian_api_rerank_accept_mode = str(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_accept_mode",
                "always",
            )
        ).strip().lower()
        if self.semantic_gaussian_api_rerank_accept_mode not in ("always", "score_margin"):
            self.semantic_gaussian_api_rerank_accept_mode = "always"
        self.semantic_gaussian_api_rerank_margin_threshold = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_margin_threshold",
                0.05,
            )
        )
        self.semantic_gaussian_api_rerank_timeout_s = float(
            getattr(config_env.end2end_imagenav, "semantic_gaussian_api_rerank_timeout_s", 12.0)
        )
        self.semantic_gaussian_api_rerank_temperature = float(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_temperature",
                0.0,
            )
        )
        self.semantic_gaussian_api_rerank_enable_thinking = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_enable_thinking",
                False,
            )
        )
        self.semantic_gaussian_api_rerank_image_max_side = int(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_image_max_side",
                448,
            )
        )
        # Optional SG-VLM diagnostics (default off; does not change decisions).
        self.semantic_gaussian_api_rerank_debug_log = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_debug_log",
                False,
            )
        )
        # SG-VLM prompt mode: "default" (legacy short prompt) | "smart" (instance-aware,
        # anti position-bias, focus on identity-discriminating cues). Default keeps
        # historical behavior so existing experiments are unaffected.
        self.semantic_gaussian_api_rerank_prompt_mode = str(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_prompt_mode",
                "default",
            )
        ).strip().lower()
        if self.semantic_gaussian_api_rerank_prompt_mode not in ("default", "smart"):
            self.semantic_gaussian_api_rerank_prompt_mode = "default"
        # SG-VLM candidate image source for the K candidates:
        #   "thumbnail"  : cluster mean-color patch (legacy default; almost no info)
        #   "scene_crop" : nearest OV instance RGB crop (real photo)
        #   "both"       : scene_crop preferred, fallback to thumbnail
        self.semantic_gaussian_api_rerank_candidate_image = str(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_api_rerank_candidate_image",
                "thumbnail",
            )
        ).strip().lower()
        if self.semantic_gaussian_api_rerank_candidate_image not in (
            "thumbnail", "scene_crop", "both"
        ):
            self.semantic_gaussian_api_rerank_candidate_image = "thumbnail"
        # Goal arbitration: ov_first (default assist), merit (inst_v3_vlm).
        self.goal_arbitration_mode = str(
            getattr(config_env.end2end_imagenav, "goal_arbitration_mode", "ov_first")
        ).strip().lower()
        if self.goal_arbitration_mode not in ("ov_first", "merit"):
            print(
                "[goal] invalid goal_arbitration_mode=%s, fallback ov_first"
                % self.goal_arbitration_mode
            )
            self.goal_arbitration_mode = "ov_first"
        # merit 仲裁时 OV 侧分数：fused（默认，与历史实验一致）| sem（与 sg_sim 同量纲）。
        self.goal_arbitration_merit_ov_metric = str(
            getattr(
                config_env.end2end_imagenav,
                "goal_arbitration_merit_ov_metric",
                "fused",
            )
        ).strip().lower()
        if self.goal_arbitration_merit_ov_metric not in ("fused", "sem"):
            self.goal_arbitration_merit_ov_metric = "fused"
        # Practical (not probabilistic) scale used to compare the OV merit
        # score with SG cosine similarity. 1.0 preserves all archived runs.
        self.goal_arbitration_merit_ov_scale = float(
            getattr(
                config_env.end2end_imagenav,
                "goal_arbitration_merit_ov_scale",
                1.0,
            )
        )
        if not np.isfinite(self.goal_arbitration_merit_ov_scale):
            self.goal_arbitration_merit_ov_scale = 1.0
        # merit 仲裁时若 sg_sim >= 该阈值则硬性优先 SG。<=0 表示关闭（默认）。
        self.goal_arbitration_merit_sg_high_conf_thr = float(
            getattr(
                config_env.end2end_imagenav,
                "goal_arbitration_merit_sg_high_conf_thr",
                0.0,
            )
        )
        # 动态语义更新门控（创新点6调参）：距离过远或 OV 几何证据过弱时跳过本帧更新。
        self.dynamic_semantic_update_max_goal_dist = float(
            getattr(config_env.end2end_imagenav, "dynamic_semantic_update_max_goal_dist", 0.0)
        )
        self.dynamic_semantic_update_disable_when_ov_geo_floor_weak = bool(
            getattr(
                config_env.end2end_imagenav,
                "dynamic_semantic_update_disable_when_ov_geo_floor_weak",
                False,
            )
        )
        # 每回合结束打印 [episode_goal_summary] 一行，便于统计失败与 dist_pred_to_gt
        self.log_episode_goal_summary = bool(
            getattr(config_env.end2end_imagenav, "log_episode_goal_summary", False)
        )
        self.episode_goal_summary_append_path = str(
            getattr(config_env.end2end_imagenav, "episode_goal_summary_append_path", "")
            or ""
        )
        # Habitat Success 要求 is_stop 且 geodesic 到 VIEW_POINTS < success_distance；Planner 可能已到附近却未发 STOP
        self.auto_stop_when_within_success_dist = bool(
            getattr(
                config_env.end2end_imagenav,
                "auto_stop_when_within_success_dist",
                True,
            )
        )
        # >0 时作为测地距离阈值（米）；=0 则用 habitat success_distance - 0.02
        self.auto_stop_geodesic_max = float(
            getattr(config_env.end2end_imagenav, "auto_stop_geodesic_max", 0.0)
        )
        self.prefer_open_vocab_when_semantic_gaussian_low_conf = bool(
            getattr(
                config_env.end2end_imagenav,
                "prefer_open_vocab_when_semantic_gaussian_low_conf",
                True,
            )
        )
        self.semantic_gaussian_enable_fallback_open_vocab = bool(
            getattr(
                config_env.end2end_imagenav,
                "semantic_gaussian_enable_fallback_open_vocab",
                True,
            )
        )
        # 非GT 模式：为 False 时不允许回退到 GT。
        # 当目标预测失败时，处理策略由 non_gt_fallback_handling 决定：
        #   - "count_as_failure": 不跳过，直接按失败计入（默认，更严谨）
        #   - "skip_episode": 跳过该 episode（旧行为，可能带来样本偏差）
        self.allow_gt_fallback = bool(
            getattr(config_env.end2end_imagenav, "allow_gt_fallback", True)
        )
        self.non_gt_fallback_handling = str(
            getattr(config_env.end2end_imagenav, "non_gt_fallback_handling", "count_as_failure")
        ).strip().lower()
        if self.non_gt_fallback_handling not in ("count_as_failure", "skip_episode"):
            print(
                "[goal] Warning: invalid non_gt_fallback_handling=%s, fallback to count_as_failure"
                % self.non_gt_fallback_handling
            )
            self.non_gt_fallback_handling = "count_as_failure"
        # 统计目标预测覆盖率（用于报告非GT覆盖率，避免只看 succ/spl）
        self.goal_episode_total = 0
        self.goal_non_gt_count = 0
        self.goal_gt_fallback_count = 0
        self.goal_skipped_count = 0
        self.ov_api_rerank_call_count = 0
        self.ov_api_rerank_parse_success_count = 0
        self.ov_api_rerank_changed_count = 0
        self.ov_api_rerank_fallback_count = 0
        # 当 non_gt_fallback_handling=count_as_failure 且预测失败时启用
        self._force_fail_episode = False
        self._force_fail_reason = ""
        self.open_vocab_retriever = None
        # Allow switching between v1 (frame-level) and v2 (per-instance crop)
        # CLIP vector files without touching dataset paths. Default keeps v1.
        self.instance_clip_vectors_filename = str(getattr(
            config_env.end2end_imagenav,
            "instance_clip_vectors_filename",
            "instance_clip_vectors.json",
        ) or "instance_clip_vectors.json")
        if self.use_open_vocab_retrieval or self.use_semantic_gaussian_goal:
            self.open_vocab_retriever = OpenVocabRetriever(
                self.device,
                clip_model_name=clip_model_name,
                clip_dim=clip_dim,
                clip_pretrained_path=clip_pretrained_path or None,
                clip_vectors_filename=self.instance_clip_vectors_filename,
            )
        self.semantic_gaussian_goal = None
        if self.use_semantic_gaussian_goal and self.gaussian_map_root:
            clip_encoder = (
                self.open_vocab_retriever.encode_image
                if self.open_vocab_retriever is not None
                else None
            )
            self.semantic_gaussian_goal = SemanticGaussianGoal(
                self.device,
                self.gaussian_map_root,
                clip_encode_image=clip_encoder,
                clip_model_name=clip_model_name,
                clip_dim=clip_dim,
            )

        # # use lightglue
        if self.similarity_method == 1:
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='disk').eval().to(self.device)
            # self.matcher.compile(mode='reduce-overhead')

    def _auto_stop_geodesic_threshold(self) -> float:
        if self.auto_stop_geodesic_max > 0.0:
            return float(self.auto_stop_geodesic_max)
        cfg = self.config
        for keys in (
            ("task", "measurements", "success", "success_distance"),
            ("habitat", "task", "measurements", "success", "success_distance"),
        ):
            try:
                node = cfg
                for k in keys:
                    node = node[k]
                sd = float(node)
                return max(0.1, sd - 0.02)
            except Exception:
                continue
        return 0.98

    def _current_geodesic_to_goal_viewpoints(self) -> float:
        """与 task distance_to_goal (VIEW_POINTS) 一致的测地距离。"""
        ep = self.habitat_env.current_episode
        agent_pos = self.habitat_env.sim.get_agent_state().position
        view_points = [
            view_point.agent_state.position
            for goal in ep.goals
            for view_point in goal.view_points
        ]
        return float(
            self.habitat_env.sim.geodesic_distance(agent_pos, view_points, ep)
        )

    def _update_goal_source_stats(self, goal_source, fallback_reasons):
        """统计 goal 来源与非GT覆盖率，写入 self.info 便于日志与评估汇报。"""
        self.goal_episode_total += 1
        if goal_source == "GT":
            self.goal_gt_fallback_count += 1
        else:
            self.goal_non_gt_count += 1
        total = max(1, self.goal_episode_total)
        self.info["goal_source"] = goal_source
        self.info["goal_predicted"] = int(goal_source != "GT")
        self.info["goal_non_gt_coverage"] = float(self.goal_non_gt_count) / float(total)
        self.info["goal_gt_fallback_rate"] = float(self.goal_gt_fallback_count) / float(total)
        self.info["goal_skipped_count"] = int(self.goal_skipped_count)
        self.info["goal_episode_total"] = int(self.goal_episode_total)
        self.info["non_gt_fallback_handling"] = self.non_gt_fallback_handling
        self.info["goal_fallback_reasons"] = "; ".join(fallback_reasons) if fallback_reasons else ""

    def _override_metrics_for_forced_failure(self):
        """
        对“无预测目标按失败计入”的 episode 做安全修正：
        若 stop 后环境原始 success/spl 非零，则强制按失败（0）计入。
        """
        try:
            raw_spl, raw_success, _ = self.get_metrics()
        except Exception:
            raw_spl, raw_success = 0.0, 0.0
        raw_spl = float(raw_spl)
        raw_success = float(raw_success)
        if raw_success > 0.0 or raw_spl > 0.0:
            self.succ_sum -= raw_success
            self.spl_sum -= raw_spl
            self.succ_sum = max(self.succ_sum, 0.0)
            self.spl_sum = max(self.spl_sum, 0.0)
            denom = max(1, int(self.episode_no))
            self.info["success"] = self.succ_sum / denom
            self.info["spl"] = self.spl_sum / denom
            msg = (
                "[goal] forced_failure override metrics | raw_success=%.3f raw_spl=%.3f | scene=%s episode=%s"
                % (raw_success, raw_spl, self.scene_name, self.episode_no)
            )
            logging.info(msg)
            print(msg)
        self.info["forced_no_prediction_failure"] = 1
        self.info["forced_failure_reason"] = self._force_fail_reason

    def _goal_image_to_hwc_uint8(self, goal_img: np.ndarray) -> np.ndarray:
        """instance_imagegoal -> HWC uint8 RGB for saving / CLIP."""
        img = np.asarray(goal_img)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1], img.shape[2]):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img[..., :3]

    def _save_episode_goal_viz_assets(self, out_dir=None, finalize: bool = False) -> None:
        """Save query goal image + metadata next to episode Vis.mp4 for manual inspection."""
        if not getattr(self, "viz_save_goal_assets", False):
            return
        out_dir = out_dir or getattr(self, "dump_location", None)
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return

        goal = getattr(self, "instance_imagegoal", None)
        if goal is not None and not finalize:
            img = self._goal_image_to_hwc_uint8(goal)
            cv2.imwrite(
                os.path.join(out_dir, "goal_query.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

        gs = getattr(self, "_episode_goal_source", "")
        dpg = getattr(self, "_episode_dist_pred_to_gt", float("nan"))
        pred = getattr(self, "pred_goal_pos", None)
        gt_vp = getattr(self, "min_viewpoint_goal_w_env", None)
        agent = None
        try:
            agent = np.array(self.habitat_env.sim.get_agent_state(0).position, dtype=np.float64)
        except Exception:
            pass
        lines = [
            "scene=%s" % self.scene_name,
            "episode_no=%d" % int(self.episode_no),
            "object_category=%s" % getattr(self, "gt_goal_category", ""),
            "goal_object_id=%s" % getattr(self, "gt_goal_object_id", ""),
            "goal_source=%s" % gs,
            "dist_pred_to_gt_m=%.4f" % float(dpg),
        ]
        if pred is not None:
            lines.append("pred_goal_pos=%s" % np.asarray(pred, dtype=np.float64).tolist())
        if gt_vp is not None:
            lines.append("gt_viewpoint_goal_pos=%s" % np.asarray(gt_vp, dtype=np.float64).tolist())
        if agent is not None and pred is not None:
            lines.append("dist_pred_to_agent_m=%.4f" % float(np.linalg.norm(np.asarray(pred) - agent)))
        if gt_vp is not None and agent is not None:
            lines.append("dist_gt_viewpoint_to_agent_m=%.4f" % float(np.linalg.norm(np.asarray(gt_vp) - agent)))

        if finalize:
            try:
                spl_ep, succ_ep, dist_ep = self.get_metrics()
            except Exception:
                spl_ep, succ_ep, dist_ep = 0.0, 0.0, 0.0
            lines.extend([
                "timesteps=%d" % int(self.timestep),
                "success=%d" % int(round(float(succ_ep))),
                "spl=%.5f" % float(spl_ep),
                "dist_to_goal=%.4f" % float(dist_ep),
                "forced_fail=%d" % int(getattr(self, "_episode_was_forced_fail", False)),
            ])

        info_path = os.path.join(out_dir, "goal_info.txt")
        mode = "a" if finalize and os.path.isfile(info_path) else "w"
        with open(info_path, mode, encoding="utf-8") as f:
            if finalize and mode == "a":
                f.write("\n--- episode result ---\n")
            f.write("\n".join(lines) + "\n")

    def _should_collect_viz_frames(self) -> bool:
        if self.viz:
            return True
        if self._viz_failure_quota_mode and self._viz_failure_saved_count < self.viz_failure_max_saved:
            return True
        return False

    def _maybe_save_failure_viz_package(self, force_failure: bool = False) -> bool:
        """Save video + goal_query.png + goal_info.txt for one failure episode (quota mode)."""
        if not self._viz_failure_quota_mode:
            return False
        if self._viz_failure_saved_count >= self.viz_failure_max_saved:
            return False
        if not force_failure:
            try:
                _spl, success, _dist = self.get_metrics()
            except Exception:
                success = 1
            if int(round(float(success))) != 0:
                return False
        out_dir = self.dump_location
        if self.viz_save_goal_assets:
            self._save_episode_goal_viz_assets(out_dir, finalize=False)
            self._save_episode_goal_viz_assets(out_dir, finalize=True)
        if len(getattr(self, "viz_obs", [])) > 0:
            self.visualize_video(self.viz_obs, out_dir)
        self._viz_failure_saved_count += 1
        print(
            "[viz] saved failure package %d/%d -> %s (Vis.mp4 + goal_query.png + goal_info.txt)"
            % (self._viz_failure_saved_count, self.viz_failure_max_saved, out_dir)
        )
        self.viz_obs = []
        return True

    def _emit_episode_goal_summary(self, forced: bool = False) -> None:
        """回合结束一行 TSV 风格日志，供 scripts/summarize_episode_goal_log.py 聚合。"""
        if not self.log_episode_goal_summary:
            return
        try:
            spl_ep, succ_ep, dist_ep = self.get_metrics()
        except Exception:
            spl_ep, succ_ep, dist_ep = 0.0, 0.0, 0.0
        if forced:
            succ_ep, spl_ep = 0.0, 0.0
        gs = getattr(self, "_episode_goal_source", "")
        dpg = getattr(self, "_episode_dist_pred_to_gt", float("nan"))
        _sp_st = getattr(self, "_sg_prior_status", "off")
        _sp_det = getattr(self, "_sg_prior_detail", "")
        _col_st = getattr(self, "_sg_color_status", "off")
        _col_det = getattr(self, "_sg_color_detail", "")
        _topo_st = getattr(self, "_sg_topo_status", "off")
        _topo_det = getattr(self, "_sg_topo_detail", "")
        _vlm_st = getattr(self, "_ov_api_rerank_status", "off")
        _vlm_rank = int(getattr(self, "_ov_api_rerank_picked_rank", -1))
        _vlm_changed = int(getattr(self, "_ov_api_rerank_changed", 0))
        _vlm_old = int(getattr(self, "_ov_api_rerank_old_best", -1))
        _vlm_new = int(getattr(self, "_ov_api_rerank_new_best", -1))
        _vlm_topk = int(getattr(self, "_ov_api_rerank_topk_used", 0))
        _vlm_margin = float(getattr(self, "_ov_api_rerank_score_margin", float("nan")))
        _vlm_accepted = int(getattr(self, "_ov_api_rerank_accepted", 0))
        line = (
            "[episode_goal_summary] scene=%s ep=%d goal_source=%s dist_pred_to_gt=%.4f "
            "success=%d spl=%.5f dist_to_goal=%.4f timesteps=%d forced=%d sg_prior=%s"
            % (
                self.scene_name,
                int(self.episode_no),
                gs,
                dpg,
                int(round(float(succ_ep))),
                float(spl_ep),
                float(dist_ep),
                int(self.timestep),
                int(1 if forced else 0),
                _sp_st,
            )
        )
        if _sp_det:
            line += " sg_prior_detail=%s" % _sp_det
        line += " sg_color=%s" % _col_st
        if _col_det:
            line += " sg_color_detail=%s" % _col_det
        line += " sg_topo=%s" % _topo_st
        if _topo_det:
            line += " sg_topo_detail=%s" % _topo_det
        line += (
            " vlm_status=%s vlm_picked_rank=%d vlm_changed=%d"
            " vlm_old_best=%d vlm_new_best=%d vlm_topk=%d"
            " vlm_margin=%.6f vlm_accepted=%d"
            % (
                _vlm_st,
                _vlm_rank,
                _vlm_changed,
                _vlm_old,
                _vlm_new,
                _vlm_topk,
                _vlm_margin,
                _vlm_accepted,
            )
        )
        if bool(getattr(self, "semantic_gaussian_api_rerank_enable", False)):
            line += (
                " sg_vlm_status=%s sg_vlm_rank=%d sg_vlm_changed=%d"
                " sg_vlm_margin=%.6f sg_vlm_accepted=%d goal_arbitration=%s"
                % (
                    getattr(self, "_sg_vlm_status", "off"),
                    int(getattr(self, "_sg_vlm_picked_rank", -1)),
                    int(getattr(self, "_sg_vlm_changed", 0)),
                    float(getattr(self, "_sg_vlm_score_margin", float("nan"))),
                    int(getattr(self, "_sg_vlm_accepted", 0)),
                    getattr(self, "goal_arbitration_mode", "ov_first"),
            )
        )
        print(line)
        logging.info(line)
        # VectorEnv 子进程（forkserver）的 print 不经过主进程 run.py 的 tee，故追加到配置路径供 summarize 使用
        _ap = self.episode_goal_summary_append_path
        if isinstance(_ap, str) and _ap.strip():
            try:
                with open(_ap.strip(), "a", encoding="utf-8", buffering=1) as f:
                    f.write(line + "\n")
            except OSError as e:
                logging.warning("episode_goal_summary 追加到 %s 失败: %s", _ap, e)

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # 仅 skip_episode 模式使用：限制连续跳过次数，避免异常配置下无限循环
        max_gt_skip = 20
        skip_count = 0
        while True:
            self._force_fail_episode = False
            self._force_fail_reason = ""
            self._episode_was_forced_fail = False
            obs, self.info = super().reset()
            self.obs = obs
            rgb = obs['rgb'].astype(np.uint8)
            self.instance_imagegoal = obs['instance_imagegoal']
            if self.last_scene_name != self.scene_name:
                self.follower = ShortestPathFollower(
                    self.habitat_env.sim, self.goal_radius, False
                )

            self.dump_location = os.path.join(self.base_dump_location, str(self.episode_no))

            self.mapper.reset()
            self.last_sim_location = self.get_sim_location()

            self.planner.reset(self.mapper)

            # Goal position candidates:
            # SG: semantic gaussian, OV: open-vocab (with geometric verification), GT fallback.
            self.pred_goal_pos = self.min_viewpoint_goal_w_env
            goal_source = "GT"
            fallback_reasons = []  # 用于打印为何没用上创新点
            sg_candidate = None
            sg_sim = None
            sg_dist_agent = None
            sg_cluster_size = 0
            sg_cluster_compactness = float("inf")
            ov_candidate = None
            ov_dist_agent = None
            ov_geo_floor_weak = False
            self._ov_api_rerank_status = "off"
            self._ov_api_rerank_picked_rank = -1
            self._ov_api_rerank_changed = 0
            self._ov_api_rerank_old_best = -1
            self._ov_api_rerank_new_best = -1
            self._ov_api_rerank_topk_used = 0
            self._ov_api_rerank_score_margin = float("nan")
            self._ov_api_rerank_accepted = 0
            self._last_ov_api_rerank_status = "off"
            self._last_ov_api_rerank_raw = ""
            self._sg_vlm_status = "off"
            self._sg_vlm_picked_rank = -1
            self._sg_vlm_changed = 0
            self._sg_vlm_score_margin = float("nan")
            self._sg_vlm_accepted = 0
            self._last_sg_vlm_raw = ""
            self._last_ov_merit_fused_score = float("nan")
            self._last_ov_merit_sem_score = float("nan")

            if self.use_semantic_gaussian_goal and self.semantic_gaussian_goal is not None:
                sg_loaded = self.semantic_gaussian_goal.load_scene(self.scene_name)
                if sg_loaded:
                    # Pre-fetch agent_pos so navigable_goal can floor-project + snap.
                    _sg_agent_state = self.habitat_env.sim.get_agent_state(0)
                    _sg_agent_pos_np = np.array(_sg_agent_state.position, dtype=np.float32)
                    sg_info = None
                    position, max_sim, sg_info = self._resolve_semantic_gaussian_candidate(
                        _sg_agent_pos_np
                    )
                    if position is not None:
                        if max_sim >= self.semantic_gaussian_min_sim:
                            sg_candidate = np.array(position, dtype=np.float32)
                            sg_sim = float(max_sim)
                            if isinstance(sg_info, dict):
                                sg_cluster_size = int(sg_info.get("cluster_size", 0))
                                sg_cluster_compactness = float(
                                    sg_info.get("cluster_compactness", float("inf"))
                                )
                            agent_pos = _sg_agent_pos_np
                            dist_agent = float(np.linalg.norm(sg_candidate - agent_pos))
                            if (
                                self.semantic_gaussian_max_agent_goal_dist > 0.0
                                and dist_agent > self.semantic_gaussian_max_agent_goal_dist
                            ):
                                reason = (
                                    "semantic_gaussian: dist_to_agent %.3f > max_agent_goal_dist %.3f"
                                    % (dist_agent, self.semantic_gaussian_max_agent_goal_dist)
                                )
                                fallback_reasons.append(reason)
                                print("[goal] reject semantic_gaussian | " + reason)
                                sg_candidate = None
                                sg_sim = None
                                sg_cluster_size = 0
                                sg_cluster_compactness = float("inf")
                            else:
                                sg_dist_agent = dist_agent
                                dist_gt = float(np.linalg.norm(sg_candidate - self.min_viewpoint_goal_w_env))
                                logging.info(
                                    "[goal] semantic_gaussian | scene=%s | pred_goal_pos=%s | sim=%.4f | dist_to_GT=%.3f | dist_to_agent=%.3f | cluster_size=%d | compact=%.3f",
                                    self.scene_name, sg_candidate.tolist(), max_sim, dist_gt, dist_agent,
                                    sg_cluster_size, sg_cluster_compactness,
                                )
                                print(
                                    "[goal] semantic_gaussian | scene=%s | dist_to_GT=%.3fm | dist_to_agent=%.3fm | sim=%.4f | cluster=%d/%.2fm"
                                    % (
                                        self.scene_name, dist_gt, dist_agent, max_sim,
                                        sg_cluster_size, sg_cluster_compactness,
                                    )
                                )
                        else:
                            reason = (
                                "semantic_gaussian: sim %.4f < min_sim %.4f"
                                % (max_sim, self.semantic_gaussian_min_sim)
                            )
                            fallback_reasons.append(reason)
                            print("[goal] reject semantic_gaussian | " + reason)
                    else:
                        fallback_reasons.append("semantic_gaussian: get_goal_position=None")
                else:
                    err = self.semantic_gaussian_goal.get_last_error() or "load_scene=False"
                    fallback_reasons.append("semantic_gaussian: %s" % err)

            should_try_open_vocab = (
                self.use_open_vocab_retrieval
                and self.open_vocab_retriever is not None
                and (
                    not self.use_semantic_gaussian_goal
                    or self.semantic_gaussian_enable_fallback_open_vocab
                )
            )
            if should_try_open_vocab:
                scene_path = self._resolve_scene_path()
                if scene_path is None:
                    fallback_reasons.append("open_vocab: scene_path=None (base_scene_path下无匹配场景目录)")
                elif not self.open_vocab_retriever.load_scene(scene_path):
                    fallback_reasons.append("open_vocab: load_scene失败 (缺instance_retrieval.json或instance_clip_vectors.json)")
                else:
                    candidates = self.open_vocab_retriever.retrieve(
                        self.instance_imagegoal, top_k=self.open_vocab_top_k
                    )
                    if candidates:
                        _ov_agent = np.array(
                            self.habitat_env.sim.get_agent_state(0).position,
                            dtype=np.float32,
                        )
                        ov_candidate = self._open_vocab_goal_with_geometric_verification(
                            candidates, scene_path, agent_pos=_ov_agent
                        )
                        ov_geo_floor_weak = bool(getattr(self, "_ov_geo_floor_weak", False))
                        pred_np = np.array(ov_candidate, dtype=np.float32)
                        gt_np = np.array(self.min_viewpoint_goal_w_env, dtype=np.float32)
                        dist_gt = float(np.linalg.norm(pred_np - gt_np))
                        ov_agent_state = self.habitat_env.sim.get_agent_state(0)
                        ov_agent_pos = np.array(ov_agent_state.position, dtype=np.float32)
                        ov_dist_agent = float(np.linalg.norm(pred_np - ov_agent_pos))
                        logging.info(
                            "[goal] open_vocab | scene_path=%s | pred_goal_pos=%s | dist_to_GT=%.3f | dist_to_agent=%.3f",
                            scene_path, pred_np.tolist(), dist_gt, ov_dist_agent,
                        )
                        print(
                            "[goal] open_vocab | scene_path=%s | dist_to_GT=%.3fm | dist_to_agent=%.3fm"
                            % (scene_path, dist_gt, ov_dist_agent)
                        )
                        if (
                            self.open_vocab_max_agent_goal_dist > 0.0
                            and ov_dist_agent > self.open_vocab_max_agent_goal_dist
                        ):
                            print(
                                "[goal] warn open_vocab_long_tail | ov_dist_agent=%.3f > max=%.3f (mark OV weak)"
                                % (ov_dist_agent, self.open_vocab_max_agent_goal_dist)
                            )
                    else:
                        fallback_reasons.append("open_vocab: candidates=[]")

            # Final source selection (ov_first assist vs merit arbitration for inst_v3_vlm).
            if (
                sg_candidate is not None
                and ov_candidate is not None
                and self.goal_arbitration_mode == "merit"
            ):
                goal_source = self._arbitrate_sg_ov_merit(
                    sg_candidate,
                    sg_sim,
                    sg_dist_agent,
                    ov_candidate,
                    ov_dist_agent,
                    ov_geo_floor_weak,
                )
                if goal_source.startswith("semantic_gaussian"):
                    self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                else:
                    self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                print(
                    "[goal] merit_arbitration | pick=%s sg_sim=%.4f ov_fused=%.4f "
                    "ov_sem=%.4f ov_scale=%.3f"
                    % (
                        goal_source,
                        (sg_sim if sg_sim is not None else -1.0),
                        getattr(self, "_last_ov_merit_fused_score", float("nan")),
                        getattr(self, "_last_ov_merit_sem_score", float("nan")),
                        self.goal_arbitration_merit_ov_scale,
                    )
                )
            elif sg_candidate is not None and ov_candidate is not None:
                sg_ov_gap = float(np.linalg.norm(np.array(sg_candidate) - np.array(ov_candidate)))
                ov_long_tail_weak = (
                    ov_dist_agent is not None
                    and self.open_vocab_max_agent_goal_dist > 0.0
                    and ov_dist_agent > self.open_vocab_max_agent_goal_dist
                )
                ov_is_weak = (
                    (self.open_vocab_circuit_breaker_geo_floor and ov_geo_floor_weak)
                    or (self.open_vocab_circuit_breaker_long_tail and ov_long_tail_weak)
                )
                if self.semantic_gaussian_assist_mode:
                    # Assist mode: still keep hard safety for huge SG/OV divergence, but
                    # avoid over-relying on legacy OV-refine/high-conf thresholds that can
                    # completely suppress SG (e.g., high_conf_sim=0.999).
                    if (
                        self.semantic_gaussian_ov_disagree_force_dist > 0.0
                        and sg_ov_gap >= self.semantic_gaussian_ov_disagree_force_dist
                    ):
                        self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                        goal_source = "open_vocab_disagree_force"
                        print(
                            "[goal] select open_vocab_disagree_force | sg_ov_gap=%.3f >= force_dist=%.3f (ignore sg_sim)"
                            % (sg_ov_gap, self.semantic_gaussian_ov_disagree_force_dist)
                        )
                    elif (
                        self.semantic_gaussian_ov_disagree_dist > 0.0
                        and sg_ov_gap > self.semantic_gaussian_ov_disagree_dist
                    ):
                        # In assist mode, let SG rescue moderate-disagreement cases when
                        # SG quality is strong enough and SG is not farther than OV.
                        disagree_assist_ok = True
                        if sg_sim is None or sg_sim < self.semantic_gaussian_assist_min_sim:
                            disagree_assist_ok = False
                        if (
                            disagree_assist_ok
                            and self.semantic_gaussian_assist_require_sg_closer_than_ov
                            and sg_dist_agent is not None
                            and ov_dist_agent is not None
                            and sg_dist_agent
                            > (ov_dist_agent + self.semantic_gaussian_assist_sg_closer_margin)
                        ):
                            disagree_assist_ok = False
                        if disagree_assist_ok:
                            self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                            goal_source = "semantic_gaussian_assist_disagree"
                            print(
                                "[goal] assist_disagree | use semantic_gaussian (sg_ov_gap=%.3f, sg_sim=%.4f, sg_da=%.3f, ov_da=%.3f)"
                                % (
                                    sg_ov_gap,
                                    (sg_sim if sg_sim is not None else -1.0),
                                    (sg_dist_agent if sg_dist_agent is not None else -1.0),
                                    (ov_dist_agent if ov_dist_agent is not None else -1.0),
                                )
                            )
                        else:
                            self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                            goal_source = "open_vocab_disagree"
                            print(
                                "[goal] select open_vocab_disagree | sg_ov_gap=%.3f > disagree_dist=%.3f (assist veto, sg_sim=%.4f, sg_da=%.3f, ov_da=%.3f)"
                                % (
                                    sg_ov_gap,
                                    self.semantic_gaussian_ov_disagree_dist,
                                    (sg_sim if sg_sim is not None else -1.0),
                                    (sg_dist_agent if sg_dist_agent is not None else -1.0),
                                    (ov_dist_agent if ov_dist_agent is not None else -1.0),
                                )
                            )
                    else:
                        assist_gate_ok = True
                        if (
                            self.semantic_gaussian_assist_only_when_ov_weak
                            and not ov_is_weak
                        ):
                            assist_gate_ok = False
                        if (
                            assist_gate_ok
                            and (sg_sim is None or sg_sim < self.semantic_gaussian_assist_min_sim)
                        ):
                            assist_gate_ok = False
                        if (
                            assist_gate_ok
                            and self.semantic_gaussian_assist_min_cluster_size > 0
                            and sg_cluster_size < self.semantic_gaussian_assist_min_cluster_size
                        ):
                            assist_gate_ok = False
                        if (
                            assist_gate_ok
                            and self.semantic_gaussian_assist_max_gap > 0.0
                            and sg_ov_gap > self.semantic_gaussian_assist_max_gap
                        ):
                            assist_gate_ok = False
                        if (
                            assist_gate_ok
                            and self.semantic_gaussian_assist_require_sg_closer_than_ov
                            and sg_dist_agent is not None
                            and ov_dist_agent is not None
                            and sg_dist_agent
                            > (ov_dist_agent + self.semantic_gaussian_assist_sg_closer_margin)
                        ):
                            assist_gate_ok = False
                        if assist_gate_ok:
                            self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                            goal_source = "semantic_gaussian_assist"
                            print(
                                "[goal] assist | use semantic_gaussian (ov_weak=%s, sg_sim=%.4f, sg_ov_gap=%.3f, sg_da=%.3f, ov_da=%.3f)"
                                % (
                                    str(ov_is_weak),
                                    (sg_sim if sg_sim is not None else -1.0),
                                    sg_ov_gap,
                                    (sg_dist_agent if sg_dist_agent is not None else -1.0),
                                    (ov_dist_agent if ov_dist_agent is not None else -1.0),
                                )
                            )
                        else:
                            self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                            goal_source = "open_vocab"
                            print(
                                "[goal] assist | keep open_vocab (ov_weak=%s, sg_sim=%.4f, sg_ov_gap=%.3f)"
                                % (
                                    str(ov_is_weak),
                                    (sg_sim if sg_sim is not None else -1.0),
                                    sg_ov_gap,
                                )
                            )
                else:
                    if ov_is_weak:
                        self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                        goal_source = "semantic_gaussian"
                        print(
                            "[goal] circuit_breaker | keep semantic_gaussian (geo_floor_weak=%s, long_tail_weak=%s)"
                            % (str(ov_geo_floor_weak), str(ov_long_tail_weak))
                        )
                    elif (
                        self.semantic_gaussian_ov_disagree_force_dist > 0.0
                        and sg_ov_gap >= self.semantic_gaussian_ov_disagree_force_dist
                    ):
                        self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                        goal_source = "open_vocab_disagree_force"
                        print(
                            "[goal] select open_vocab_disagree_force | sg_ov_gap=%.3f >= force_dist=%.3f (ignore sg_sim)"
                            % (sg_ov_gap, self.semantic_gaussian_ov_disagree_force_dist)
                        )
                    elif (
                        self.semantic_gaussian_ov_disagree_dist > 0.0
                        and sg_ov_gap > self.semantic_gaussian_ov_disagree_dist
                        and (
                            sg_sim is None
                            or sg_sim < self.semantic_gaussian_ov_disagree_high_conf
                        )
                    ):
                        self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                        goal_source = "open_vocab_disagree"
                        print(
                            "[goal] select open_vocab_disagree | sg_ov_gap=%.3f > disagree_dist=%.3f and sg_sim=%.4f < disagree_high_conf=%.4f"
                            % (
                                sg_ov_gap,
                                self.semantic_gaussian_ov_disagree_dist,
                                (sg_sim if sg_sim is not None else -1.0),
                                self.semantic_gaussian_ov_disagree_high_conf,
                            )
                        )
                    elif (
                        self.prefer_open_vocab_when_semantic_gaussian_low_conf
                        and sg_sim is not None
                        and sg_sim < self.semantic_gaussian_high_conf_sim
                    ):
                        self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                        goal_source = "open_vocab_refine"
                        print(
                            "[goal] select open_vocab_refine | sg_sim=%.4f < high_conf=%.4f"
                            % (sg_sim, self.semantic_gaussian_high_conf_sim)
                        )
                    else:
                        self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                        goal_source = "semantic_gaussian"
            elif sg_candidate is not None:
                self.pred_goal_pos = np.array(sg_candidate, dtype=np.float32)
                goal_source = "semantic_gaussian"
            elif ov_candidate is not None:
                self.pred_goal_pos = np.array(ov_candidate, dtype=np.float32)
                goal_source = "open_vocab"

            if goal_source == "GT" and (self.use_semantic_gaussian_goal or self.use_open_vocab_retrieval):
                msg = "[goal] fallback GT: " + "; ".join(fallback_reasons) if fallback_reasons else "[goal] fallback GT"
                logging.info(msg)
                print(msg)
                if fallback_reasons and not getattr(self, "_goal_path_debug_done", False):
                    self._goal_path_debug_done = True
                    _g = self.gaussian_map_root
                    _b = self.base_scene_path
                    _sg_path = self.semantic_gaussian_goal._resolve_params_path(self.scene_name) if (self.semantic_gaussian_goal and self.gaussian_map_root) else None
                    _ov_path = self._resolve_scene_path()
                    print(
                        "[goal] 路径诊断(首次) | scene_name=%s | gaussian_map_root=%s exists=%s | base_scene_path=%s exists=%s | params.npz=>%s | open_vocab_scene=>%s"
                        % (self.scene_name, _g, os.path.isdir(_g) if _g else False, _b, os.path.isdir(_b) if _b else False, _sg_path, _ov_path)
                    )

            # 记录目标来源统计（含 coverage/fallback rate）
            self._update_goal_source_stats(goal_source, fallback_reasons)

            # 回合级诊断：预测目标与 GT viewpoint 的直线距离（米）
            self._episode_goal_source = goal_source
            self._episode_dist_pred_to_gt = float(
                np.linalg.norm(
                    np.array(self.pred_goal_pos, dtype=np.float64)
                    - np.array(self.min_viewpoint_goal_w_env, dtype=np.float64)
                )
            )
            if self.viz_save_goal_assets and self.viz:
                self._save_episode_goal_viz_assets(self.dump_location, finalize=False)

            # 非GT 模式下不允许使用 GT：根据策略决定“跳过”或“按失败计入”
            if goal_source == "GT" and not self.allow_gt_fallback:
                if self.non_gt_fallback_handling == "skip_episode":
                    self.goal_skipped_count += 1
                    skip_count += 1
                    self.info["goal_skipped_count"] = int(self.goal_skipped_count)
                    if skip_count <= max_gt_skip:
                        print("[goal] 非GT模式：跳过 GT fallback episode，重新 reset (%d/%d)" % (skip_count, max_gt_skip))
                    else:
                        logging.warning(
                            "非GT模式：连续 %d 次均为 GT fallback，不使用 GT，继续尝试获取非 GT episode", skip_count
                        )
                        if skip_count == max_gt_skip + 1 or skip_count % 20 == 0:
                            print("[goal] 非GT模式：已连续 %d 次 GT fallback，不使用 GT，继续 reset" % skip_count)
                    # 安全上限，防止无限循环（若数据集中所有 episode 均 fallback 到 GT）
                    if skip_count >= max_gt_skip * 50:
                        raise RuntimeError(
                            "非GT模式：连续 %d 次均为 GT fallback，无法获取非 GT episode，终止运行"
                            % skip_count
                        )
                    continue
                # count_as_failure：不跳过 episode，不使用 GT 导航，直接在 step 首帧 stop 并按失败计入
                self._force_fail_episode = True
                self._episode_was_forced_fail = True
                self._force_fail_reason = "; ".join(fallback_reasons) if fallback_reasons else "goal_source=GT"
                msg = (
                    "[goal] 非GT模式：目标预测失败，按失败计入该 episode（non_gt_fallback_handling=count_as_failure） | reason=%s"
                    % self._force_fail_reason
                )
                logging.info(msg)
                print(msg)
            break

        # # use preprocessed semantic gaussian for goal map prediction
        # self.target_obs = {'rgb':self.obs['instance_imagegoal']}
        # # normal
        # self.pred_goal_pos = self.compare_bbox(self.instance_imagegoal, self.gt_goal_coco_id)
        # # random select instance
        # self.pred_goal_pos = self.select_instance(self.instance_imagegoal, self.gt_goal_coco_id)


        if self.viz:
            if len(self.viz_obs) > 0:
                if not self.viz_failure_case:
                    temp_dump_location = os.path.join(self.base_dump_location, str(self.episode_no - 1))
                    self.visualize_video(self.viz_obs, temp_dump_location)
                    self.viz_obs = []
        elif self._should_collect_viz_frames():
            self.viz_obs = []

        return rgb.transpose(2,0,1), self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        if action["finished"]:
        # if isinstance(action, dict) and action.get("action", -1) == 0:
            return np.zeros((3, self.env_frame_height, self.env_frame_width)), 0., False, self.info

        # 非GT模式且目标预测失败时：首步直接 stop，按失败计入，避免使用 GT 目标或跳过样本
        if self._force_fail_episode:
            obs, rew, done, _ = super().step({"action": 0})
            self.obs = obs
            rgb = obs['rgb'].astype(np.uint8)
            if done:
                if not self._maybe_save_failure_viz_package(force_failure=True):
                    if self.viz_save_goal_assets and self.viz:
                        self._save_episode_goal_viz_assets(self.dump_location, finalize=True)
                elif self._viz_failure_quota_mode:
                    self.viz_obs = []
                self._emit_episode_goal_summary(forced=True)
                self._override_metrics_for_forced_failure()
                self._force_fail_episode = False
            return rgb.transpose(2, 0, 1), rew, done, self.info


        # here, we directly get the goal position from environment
        # we will update the code to get the goal position from Semantic Gaussian
        use_stop = False
        if self.auto_stop_when_within_success_dist:
            try:
                gd = self._current_geodesic_to_goal_viewpoints()
                thr = self._auto_stop_geodesic_threshold()
                if np.isfinite(gd) and gd < thr:
                    use_stop = False
            except Exception:
                use_stop = False
        if use_stop:
            action = {"action": 0}
        else:
            sem_corr = None
            if (
                self.dynamic_semantic_planner
                and getattr(self.mapper, "enable_dynamic_semantic_update", False)
                and self.open_vocab_retriever is not None
                and self.mapper.semantic_clip_map is not None
                and self.mapper.timestep >= self.dynamic_semantic_planner_min_timestep
            ):
                allow_planner_sem = True
                if self.dynamic_semantic_planner_max_goal_dist > 0.0:
                    pgd = float(self.info.get("distance_to_goal", np.inf))
                    if np.isfinite(pgd) and pgd > self.dynamic_semantic_planner_max_goal_dist:
                        allow_planner_sem = False
                if (
                    allow_planner_sem
                    and self.dynamic_semantic_planner_disable_when_ov_geo_floor_weak
                    and bool(getattr(self, "_ov_geo_floor_weak", False))
                ):
                    allow_planner_sem = False
                if allow_planner_sem:
                    try:
                        vg = self.open_vocab_retriever.encode_image(
                            self.instance_imagegoal
                        )
                        sem_corr = self.mapper.compute_local_semantic_correlation(vg)
                    except Exception:
                        sem_corr = None
            action = self.planner.plan(
                self.from_env_to_map(self.pred_goal_pos).tolist(),
                self.mapper,
                selem=5,
                semantic_goal_corr=sem_corr,
                semantic_goal_corr_thr=self.dynamic_semantic_planner_thr,
                semantic_goal_corr_radius=self.dynamic_semantic_planner_radius,
                semantic_goal_peaks_only=self.dynamic_semantic_planner_peaks_only,
                semantic_goal_max_peaks=self.dynamic_semantic_planner_max_peaks,
                semantic_goal_peak_min_sep=self.dynamic_semantic_planner_peak_min_sep,
            )
        obs, rew, done, _ = super().step(action)
        self.obs = obs
        rgb = obs['rgb'].astype(np.uint8)
        # self.visualize_image(rgb, self.dump_location, 'rgb')

        processed_obs = self._preprocess_obs(obs)
        processed_obs = torch.from_numpy(processed_obs).float().to(self.device).unsqueeze(0)

        dx, dy, do = self.get_pose_change()
        pose_obs = torch.Tensor([dx, dy, do]).float().to(self.device).unsqueeze(0)

        agent_heights = torch.Tensor([self.agent_height]).float().to(self.device).unsqueeze(0)

        # map_struct = self.mapper.step(processed_obs, pose_obs, agent_heights, viz_obs=obs, viz_image_goal=self.target_obs['rgb'])
        clip_encoder = None
        if (
            getattr(self.mapper, "enable_dynamic_semantic_update", False)
            and self.open_vocab_retriever is not None
        ):
            allow_dynamic_update = True
            if self.dynamic_semantic_update_max_goal_dist > 0.0:
                curr_dist = float(self.info.get("distance_to_goal", np.inf))
                if np.isfinite(curr_dist) and curr_dist > self.dynamic_semantic_update_max_goal_dist:
                    allow_dynamic_update = False
            if (
                allow_dynamic_update
                and self.dynamic_semantic_update_disable_when_ov_geo_floor_weak
                and bool(getattr(self, "_ov_geo_floor_weak", False))
            ):
                allow_dynamic_update = False
            if allow_dynamic_update:
                clip_encoder = self.open_vocab_retriever.encode_image
        map_struct = self.mapper.step(
            processed_obs, pose_obs, agent_heights, viz_obs=obs, clip_encoder=clip_encoder
        )

        # visualize goal position
        if self._should_collect_viz_frames():
            vis_map = map_struct['vis_map']
            goal_pos_w_map = self.from_env_to_map(self.pred_goal_pos)
            vis_map[20:500, 40+360:400+480][goal_pos_w_map[0]*2-5:goal_pos_w_map[0]*2+5, \
                                            goal_pos_w_map[1]*2-5:goal_pos_w_map[1]*2+5] = [255, 0, 0]
            self.viz_obs.append(vis_map)
            
            if done and self.viz and self.viz_failure_case:
                spl, success, dist = self.get_metrics()
                if success == 0:
                    self.visualize_video(self.viz_obs, self.dump_location)
                self.viz_obs = []

        if done:
            saved_failure_pkg = self._maybe_save_failure_viz_package()
            if self._viz_failure_quota_mode and not saved_failure_pkg:
                self.viz_obs = []
            elif self.viz_save_goal_assets and self.viz and not saved_failure_pkg:
                self._save_episode_goal_viz_assets(self.dump_location, finalize=True)
            self._emit_episode_goal_summary(forced=False)

        return rgb.transpose(2,0,1), rew, done, self.info


    def select_instance(self, target_obs, goal_coco_id):
        base_scene_path = self.base_scene_path
        posssible_scenes = glob(os.path.join(base_scene_path, f'{self.scene_name}*'))
        for i in range(len(posssible_scenes)):
            if abs(float((posssible_scenes[i].split('floor')[-1])[1:]) - self.scene_height[-1]) < 1.0:
                scene_path = posssible_scenes[i]
                transforms_json_path = os.path.join(scene_path, 'transforms.json')
                instance_json_path = os.path.join(scene_path, 'instance_retrieval.json')
                break
        try:
            instance_retrieval = json.load(open(instance_json_path, "r"))
        except:
            import sys
            print(posssible_scenes)
            print(self.scene_height)
            print(self.scene_name)
            sys.exit()
        curr_instance_list = instance_retrieval[str(goal_coco_id)]
        
        if len(curr_instance_list) > 0:
            max_index = np.random.randint(len(curr_instance_list))
            sele_bbox = curr_instance_list[max_index][0][0]
            sele_goal_pos = [sele_bbox[0], sele_bbox[2], sele_bbox[4]]
        else:
            sele_goal_pos = [0, 0, 0]
        return sele_goal_pos

    def _log_sg_vlm_debug(self, msg: str) -> None:
        """Print SG-VLM diagnostics when semantic_gaussian_api_rerank_debug_log is enabled."""
        if not bool(getattr(self, "semantic_gaussian_api_rerank_debug_log", False)):
            return
        line = "[sg_vlm_debug] scene=%s ep=%d %s" % (
            getattr(self, "scene_name", ""),
            int(getattr(self, "episode_no", -1)),
            msg,
        )
        print(line)
        logging.info(line)

    def _resolve_semantic_gaussian_candidate(self, agent_pos_np):
        """SG goal position; optional cluster-level VLM disambiguation (inst_v3_vlm)."""
        sg = self.semantic_gaussian_goal
        sg_info = None
        position, max_sim = None, None
        use_sg_cluster_disambig = bool(
            self.semantic_gaussian_use_navigable_goal
            and (
                self.semantic_gaussian_cluster_disambig_enable
                or self.semantic_gaussian_api_rerank_enable
            )
        )
        if use_sg_cluster_disambig:
            clusters = sg.get_topk_navigable_cluster_candidates(
                self.instance_imagegoal,
                top_k=max(2, int(self.semantic_gaussian_api_rerank_topk)),
                agent_pos=agent_pos_np,
                sim_floor=self.semantic_gaussian_nav_sim_floor,
                top_k_cluster=self.semantic_gaussian_nav_top_k_cluster,
                eps_m=self.semantic_gaussian_nav_eps_m,
                min_cluster_size=self.semantic_gaussian_nav_min_cluster_size,
                snap_distance_m=self.semantic_gaussian_nav_snap_distance_m,
                floor_y_band=self.semantic_gaussian_nav_floor_y_band,
            )
            if clusters:
                pick_i = 0
                mean_sims = [float(c.get("mean_sim", 0.0)) for c in clusters]
                if len(mean_sims) >= 2:
                    margin = float(mean_sims[0] - mean_sims[1])
                else:
                    margin = float("inf")
                self._sg_vlm_score_margin = margin
                trigger_vlm = bool(
                    self.semantic_gaussian_api_rerank_enable
                    and len(clusters) >= 2
                    and margin <= float(self.semantic_gaussian_api_rerank_sim_margin)
                )
                if trigger_vlm:
                    scene_path = self._resolve_scene_path()
                    cand_img_mode = getattr(
                        self,
                        "semantic_gaussian_api_rerank_candidate_image",
                        "thumbnail",
                    )
                    cand_imgs = []
                    cand_src = []
                    for c in clusters:
                        thumb = c.get("thumbnail_rgb")
                        crop = None
                        if cand_img_mode in ("scene_crop", "both") and scene_path is not None:
                            crop = self._sg_vlm_scene_crop_for_position(
                                c["position"], scene_path
                            )
                        if cand_img_mode == "thumbnail":
                            img = thumb
                            src = "thumbnail"
                            if img is None and scene_path is not None:
                                img = self._sg_vlm_scene_crop_for_position(
                                    c["position"], scene_path
                                )
                                src = "scene_crop_fallback"
                        elif cand_img_mode == "scene_crop":
                            img = crop
                            src = "scene_crop"
                            if img is None:
                                img = thumb
                                src = "thumbnail_fallback"
                        else:  # both
                            img = crop if crop is not None else thumb
                            src = "scene_crop" if crop is not None else "thumbnail"
                        if img is not None:
                            cand_imgs.append(img)
                            cand_src.append(src)
                    self._sg_vlm_candidate_image_src = ",".join(cand_src) if cand_src else "none"
                    if len(cand_imgs) >= 2:
                        picked = self._call_sg_api_rerank(self.instance_imagegoal, cand_imgs)
                        self._sg_vlm_status = getattr(self, "_sg_vlm_status", "off")
                        if picked is not None:
                            self._sg_vlm_picked_rank = int(picked)
                            old_i = 0
                            new_i = int(picked)
                            accept = (
                                self.semantic_gaussian_api_rerank_accept_mode == "always"
                                or new_i == old_i
                                or margin <= self.semantic_gaussian_api_rerank_margin_threshold
                            )
                            if not accept:
                                self._sg_vlm_status = "ok_rejected_margin"
                            pick_i = new_i if accept else old_i
                            self._sg_vlm_accepted = int(accept)
                            self._sg_vlm_changed = int(accept and new_i != old_i)
                        else:
                            self._sg_vlm_picked_rank = -1
                    else:
                        self._sg_vlm_status = "image_missing"
                else:
                    if self.semantic_gaussian_api_rerank_enable:
                        self._sg_vlm_status = "skip_clear_margin"
                chosen = clusters[int(pick_i)]
                position = np.array(chosen["position"], dtype=np.float64)
                max_sim = float(chosen.get("max_sim", chosen.get("mean_sim", 0.0)))
                sg_info = {
                    "cluster_size": int(chosen.get("cluster_size", 0)),
                    "cluster_compactness": float(
                        chosen.get("cluster_compactness", float("inf"))
                    ),
                    "cluster_mean_sim": float(chosen.get("mean_sim", 0.0)),
                    "snapped": bool(chosen.get("snapped", False)),
                    "sg_vlm_triggered": int(trigger_vlm),
                }
                if bool(getattr(self, "semantic_gaussian_api_rerank_debug_log", False)):
                    _sims_dbg = ",".join("%.4f" % s for s in mean_sims[:6])
                    _raw_dbg = str(getattr(self, "_sg_vlm_raw", ""))[:120]
                    _accept_dbg = int(getattr(self, "_sg_vlm_accepted", 0))
                    _changed_dbg = int(getattr(self, "_sg_vlm_changed", 0))
                    _status_dbg = str(getattr(self, "_sg_vlm_status", "off"))
                    _picked_dbg = int(getattr(self, "_sg_vlm_picked_rank", -1))
                    _src_dbg = str(getattr(self, "_sg_vlm_candidate_image_src", "?"))
                    self._log_sg_vlm_debug(
                        "triggered=%d n_clusters=%d margin=%.6f mean_sims=[%s] "
                        "api_enable=%d picked=%d pick_i=%d accept=%d changed=%d "
                        "status=%s cand_src=%s raw=%s"
                        % (
                            int(trigger_vlm),
                            len(clusters),
                            float(margin),
                            _sims_dbg,
                            int(self.semantic_gaussian_api_rerank_enable),
                            _picked_dbg,
                            int(pick_i),
                            _accept_dbg,
                            _changed_dbg,
                            _status_dbg,
                            _src_dbg,
                            _raw_dbg,
                        )
                    )
                return position, max_sim, sg_info
        if self.semantic_gaussian_use_navigable_goal:
            out = sg.get_navigable_goal_position(
                self.instance_imagegoal,
                agent_pos=agent_pos_np,
                sim_floor=self.semantic_gaussian_nav_sim_floor,
                top_k_cluster=self.semantic_gaussian_nav_top_k_cluster,
                eps_m=self.semantic_gaussian_nav_eps_m,
                min_cluster_size=self.semantic_gaussian_nav_min_cluster_size,
                snap_distance_m=self.semantic_gaussian_nav_snap_distance_m,
                floor_y_band=self.semantic_gaussian_nav_floor_y_band,
            )
            if out is not None:
                position, max_sim, sg_info = out
        else:
            out = sg.get_goal_position(
                self.instance_imagegoal, top_k=self.semantic_gaussian_top_k
            )
            if out is not None:
                position, max_sim = out
        return position, max_sim, sg_info

    def _sg_vlm_scene_crop_for_position(self, position, scene_path):
        """Nearest OV instance RGB crop for a 3D position (SG-VLM candidate view)."""
        if (
            not self.use_open_vocab_retrieval
            or self.open_vocab_retriever is None
            or scene_path is None
        ):
            return None
        try:
            if not self.open_vocab_retriever.load_scene(scene_path):
                return None
            infos = getattr(self.open_vocab_retriever, "_instance_infos", None)
            if not infos:
                return None
            pos = np.asarray(position, dtype=np.float64).reshape(3)
            best_j = None
            best_d = float("inf")
            for j, entry in enumerate(infos):
                p = np.asarray(entry[1], dtype=np.float64).reshape(3)
                d = float(np.linalg.norm(p - pos))
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is None:
                return None
            frame_id = infos[best_j][0][1]
            img_name = "rgb/frame_" + frame_id + ".png"
            img = cv2.imread(os.path.join(scene_path, img_name))
            if img is None:
                return None
            return img[:, :, ::-1]
        except Exception:
            return None

    def _arbitrate_sg_ov_merit(
        self,
        sg_candidate,
        sg_sim,
        sg_dist_agent,
        ov_candidate,
        ov_dist_agent,
        ov_geo_floor_weak,
    ):
        sg_ov_gap = float(
            np.linalg.norm(np.array(sg_candidate) - np.array(ov_candidate))
        )
        if (
            self.semantic_gaussian_ov_disagree_force_dist > 0.0
            and sg_ov_gap >= self.semantic_gaussian_ov_disagree_force_dist
        ):
            return "open_vocab_merit_force"
        sg_score = float(sg_sim if sg_sim is not None else 0.0)
        # SG high-confidence shortcut: if sg_sim crosses threshold (>0), force SG.
        _sg_thr = float(getattr(self, "goal_arbitration_merit_sg_high_conf_thr", 0.0))
        if _sg_thr > 0.0 and sg_score >= _sg_thr:
            return "semantic_gaussian_merit_high_conf"
        if getattr(self, "goal_arbitration_merit_ov_metric", "fused") == "sem":
            ov_score = float(getattr(self, "_last_ov_merit_sem_score", 0.0))
        else:
            ov_score = float(getattr(self, "_last_ov_merit_fused_score", 0.0))
            if not np.isfinite(ov_score):
                ov_score = float(getattr(self, "_last_ov_merit_sem_score", 0.0))
        if not np.isfinite(ov_score):
            ov_score = 0.0
        ov_score *= self.goal_arbitration_merit_ov_scale
        if sg_score > ov_score + 1e-4:
            return "semantic_gaussian_merit"
        if ov_score > sg_score + 1e-4:
            return "open_vocab_merit"
        if sg_dist_agent is not None and ov_dist_agent is not None:
            if sg_dist_agent <= ov_dist_agent:
                return "semantic_gaussian_merit_tie_dist"
            return "open_vocab_merit_tie_dist"
        if ov_geo_floor_weak:
            return "semantic_gaussian_merit_ov_weak"
        return "open_vocab_merit_tie"

    @staticmethod
    def _vision_api_headers():
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("GAUSSNAV_VLM_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        return headers

    def _vision_api_rerank_cfg(self, prefix):
        """prefix: 'open_vocab_api' or 'semantic_gaussian_api'."""
        return {
            "enable": bool(getattr(self, prefix + "_rerank_enable", False)),
            "url": str(getattr(self, prefix + "_rerank_url", "")),
            "model": str(getattr(self, prefix + "_rerank_model", "")),
            "timeout_s": float(getattr(self, prefix + "_rerank_timeout_s", 12.0)),
            "temperature": float(getattr(self, prefix + "_rerank_temperature", 0.0)),
            "enable_thinking": bool(
                getattr(self, prefix + "_rerank_enable_thinking", False)
            ),
            "image_max_side": int(
                getattr(self, prefix + "_rerank_image_max_side", 448)
            ),
        }

    def _sg_vlm_smart_prompt_text(self) -> str:
        """Instance-discriminating prompt tailored to SG cluster-level ambiguity.

        针对当前问题（VLM 永远返回 rank 0、negative-margin 也不换）的几点：
        - 明确告知这是 instance-image navigation：同类、同色但不同实例。
        - 引导关注 identity-discriminating 线索（纹理/上下文/形状/比例），CLIP 已经吃透色块/类别。
        - 反 position bias：候选顺序无信息；不要默认 rank 0。
        - 严格 JSON 输出。
        """
        return (
            "Task: instance-image navigation candidate selection.\n"
            "You see ONE target image showing a SPECIFIC physical instance "
            "(e.g. one particular chair / sofa / bed / lamp). The K candidate "
            "images are different 3D regions in the SAME scene that all "
            "resemble the target at category / color level.\n\n"
            "Goal: identify which candidate is the SAME physical instance as "
            "the target — not just the same category.\n\n"
            "What to focus on (in order):\n"
            "1. Distinctive surface details: texture, fabric pattern, wood "
            "grain, scratches, stains, decals.\n"
            "2. Surrounding context visible behind the object: adjacent walls, "
            "floor, furniture, doors, windows, lighting tone.\n"
            "3. Exact shape & proportions: legs, armrests, cushions, headboard, "
            "handles, joints.\n"
            "4. Subtle color tone under similar lighting (avoid being fooled "
            "by category color).\n\n"
            "Important rules:\n"
            "- The candidate order is ARBITRARY. Do NOT default to rank 0.\n"
            "- Multiple candidates may look very similar; pick the one whose "
            "DETAILS and CONTEXT best match the target.\n"
            "- If unsure, prefer the candidate with the most uniquely matching "
            "detail (e.g. same stain, same painting on wall behind), not the "
            "first one you see.\n\n"
            "Output JSON ONLY, no prose, no explanation:\n"
            "{\"best_rank\": <int 0..K-1>}"
        )

    def _call_sg_api_rerank(self, target_rgb, cand_rgbs):
        cfg = self._vision_api_rerank_cfg("semantic_gaussian_api")
        prompt_mode = getattr(
            self, "semantic_gaussian_api_rerank_prompt_mode", "default"
        )
        prompt_text = (
            self._sg_vlm_smart_prompt_text()
            if prompt_mode == "smart"
            else None
        )
        return self._call_vision_api_rerank(
            target_rgb,
            cand_rgbs,
            cfg,
            status_attr_prefix="_sg_vlm",
            log_tag="sg_api_rerank",
            prompt_text=prompt_text,
        )

    def _call_vision_api_rerank(
        self,
        target_rgb,
        cand_rgbs,
        cfg,
        status_attr_prefix="_last_sg_vlm",
        log_tag="vision_api_rerank",
        prompt_text=None,
    ):
        setattr(self, status_attr_prefix + "_status", "off")
        setattr(self, status_attr_prefix + "_raw", "")
        if not cfg.get("enable"):
            setattr(self, status_attr_prefix + "_status", "disabled")
            return None
        if not cfg.get("model"):
            setattr(self, status_attr_prefix + "_status", "no_model")
            return None
        if not cand_rgbs:
            setattr(self, status_attr_prefix + "_status", "no_candidates")
            return None
        setattr(self, status_attr_prefix + "_status", "called")
        try:
            target_url = self._to_data_url_jpeg(
                target_rgb, max_side=cfg.get("image_max_side")
            )
            cand_urls = [
                self._to_data_url_jpeg(x, max_side=cfg.get("image_max_side"))
                for x in cand_rgbs
            ]
        except Exception as _e:
            setattr(self, status_attr_prefix + "_status", "encode_error")
            msg = "[%s] status=encode_error error=%s" % (log_tag, str(_e)[:120])
            print(msg)
            logging.info(msg)
            return None
        default_prompt = (
            "Given ONE target image and K candidate images of possible same instance, "
            "choose the best candidate index. Return JSON ONLY: {\"best_rank\": <int>} "
            "where best_rank is 0-based within candidate order."
        )
        prompt_used = (
            prompt_text
            if isinstance(prompt_text, str) and prompt_text
            else default_prompt
        )
        content = [
            {"type": "text", "text": prompt_used},
            {"type": "text", "text": "Target image:"},
            {"type": "image_url", "image_url": {"url": target_url}},
        ]
        for i, u in enumerate(cand_urls):
            content.append({"type": "text", "text": f"Candidate rank {i}:"})
            content.append({"type": "image_url", "image_url": {"url": u}})
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": float(cfg["temperature"]),
            "max_tokens": 32,
            "chat_template_kwargs": {
                "enable_thinking": bool(cfg["enable_thinking"])
            },
        }
        req = urllib_request.Request(
            cfg["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers=self._vision_api_headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                req, timeout=float(cfg["timeout_s"])
            ) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib_error.URLError as _e:
            setattr(self, status_attr_prefix + "_status", "request_failed")
            msg = "[%s] status=request_failed error=%s" % (log_tag, str(_e)[:120])
            print(msg)
            logging.info(msg)
            return None
        except Exception as _e:
            setattr(self, status_attr_prefix + "_status", "request_error")
            msg = "[%s] status=request_error error=%s" % (log_tag, str(_e)[:120])
            print(msg)
            logging.info(msg)
            return None
        try:
            obj = json.loads(body)
            txt = obj["choices"][0]["message"]["content"]
            setattr(self, status_attr_prefix + "_raw", str(txt).replace("\n", " ")[:240])
        except Exception:
            setattr(self, status_attr_prefix + "_status", "bad_response")
            return None
        try:
            j0 = txt.find("{")
            j1 = txt.rfind("}")
            if j0 >= 0 and j1 > j0:
                j = json.loads(txt[j0 : j1 + 1])
                r = int(j.get("best_rank", -1))
            else:
                r = int(str(txt).strip())
            if 0 <= r < len(cand_rgbs):
                setattr(self, status_attr_prefix + "_status", "ok")
                return r
            setattr(self, status_attr_prefix + "_status", "out_of_range")
            return None
        except Exception:
            setattr(self, status_attr_prefix + "_status", "parse_failed")
            return None

    def _to_data_url_jpeg(self, img_rgb: np.ndarray, max_side=None) -> str:
        """Encode RGB image as JPEG data URL for OpenAI-compatible vision APIs."""
        if img_rgb is None:
            raise ValueError("img_rgb is None")
        arr = np.asarray(img_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("invalid image shape")
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        max_side = max(int(arr.shape[0]), int(arr.shape[1]))
        lim = max(
            64,
            int(
                max_side
                if max_side is not None
                else self.open_vocab_api_rerank_image_max_side
            ),
        )
        if max_side > lim:
            scale = float(lim) / float(max_side)
            nw = max(32, int(arr.shape[1] * scale))
            nh = max(32, int(arr.shape[0] * scale))
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        # API expects true RGB bytes.
        ok, enc = cv2.imencode(".jpg", arr[:, :, ::-1][:, :, ::-1])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b64 = base64.b64encode(enc.tobytes()).decode("ascii")
        return "data:image/jpeg;base64," + b64

    def _call_open_vocab_api_rerank(self, target_rgb, cand_rgbs):
        """Return candidate rank (0..k-1) chosen by OpenAI-compatible API, or None."""
        self._last_ov_api_rerank_status = "off"
        self._last_ov_api_rerank_raw = ""
        if not self.open_vocab_api_rerank_enable:
            self._last_ov_api_rerank_status = "disabled"
            return None
        if not self.open_vocab_api_rerank_model:
            self._last_ov_api_rerank_status = "no_model"
            return None
        if not cand_rgbs:
            self._last_ov_api_rerank_status = "no_candidates"
            return None
        self.ov_api_rerank_call_count += 1
        self._last_ov_api_rerank_status = "called"
        try:
            target_url = self._to_data_url_jpeg(target_rgb)
            cand_urls = [self._to_data_url_jpeg(x) for x in cand_rgbs]
        except Exception as _e:
            self._last_ov_api_rerank_status = "encode_error"
            msg = "[ov_api_rerank] status=encode_error error=%s" % str(_e)[:120]
            print(msg)
            logging.info(msg)
            return None
        content = [
            {
                "type": "text",
                "text": (
                    "Given ONE target image and K candidate images of possible same instance, "
                    "choose the best candidate index. Return JSON ONLY: {\"best_rank\": <int>} "
                    "where best_rank is 0-based within candidate order."
                ),
            },
            {"type": "text", "text": "Target image:"},
            {"type": "image_url", "image_url": {"url": target_url}},
        ]
        for i, u in enumerate(cand_urls):
            content.append({"type": "text", "text": f"Candidate rank {i}:"})
            content.append({"type": "image_url", "image_url": {"url": u}})
        payload = {
            "model": self.open_vocab_api_rerank_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(self.open_vocab_api_rerank_temperature),
            "max_tokens": 32,
            "chat_template_kwargs": {
                "enable_thinking": bool(self.open_vocab_api_rerank_enable_thinking)
            },
        }
        req = urllib_request.Request(
            self.open_vocab_api_rerank_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._vision_api_headers(),
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=float(self.open_vocab_api_rerank_timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib_error.URLError as _e:
            self._last_ov_api_rerank_status = "request_failed"
            self.ov_api_rerank_fallback_count += 1
            msg = "[ov_api_rerank] status=request_failed error=%s" % str(_e)[:120]
            print(msg)
            logging.info(msg)
            return None
        except Exception as _e:
            self._last_ov_api_rerank_status = "request_error"
            self.ov_api_rerank_fallback_count += 1
            msg = "[ov_api_rerank] status=request_error error=%s" % str(_e)[:120]
            print(msg)
            logging.info(msg)
            return None
        try:
            obj = json.loads(body)
            txt = obj["choices"][0]["message"]["content"]
            self._last_ov_api_rerank_raw = str(txt).replace("\n", " ")[:240]
        except Exception:
            self._last_ov_api_rerank_status = "bad_response"
            self.ov_api_rerank_fallback_count += 1
            return None
        try:
            # prefer strict JSON object
            j0 = txt.find("{")
            j1 = txt.rfind("}")
            if j0 >= 0 and j1 > j0:
                j = json.loads(txt[j0 : j1 + 1])
                r = int(j.get("best_rank", -1))
            else:
                r = int(str(txt).strip())
            if 0 <= r < len(cand_rgbs):
                self._last_ov_api_rerank_status = "ok"
                self.ov_api_rerank_parse_success_count += 1
                return r
            self._last_ov_api_rerank_status = "out_of_range"
            self.ov_api_rerank_fallback_count += 1
        except Exception:
            self._last_ov_api_rerank_status = "parse_failed"
            self.ov_api_rerank_fallback_count += 1
            return None
        return None

    def _resolve_scene_path(self):
        """Resolve scene directory from base_scene_path, scene_name, and optional floor height."""
        if not self.base_scene_path or not os.path.isdir(self.base_scene_path):
            return None
        possible = glob(os.path.join(self.base_scene_path, f"{self.scene_name}*"))
        possible = [p for p in possible if os.path.isdir(p)]
        for p in possible:
            dirname = os.path.basename(p)
            if "floor" not in dirname.lower():
                # 目录名无 floor（如 CrMo8WxCyVb），直接采用第一个匹配
                return p
            try:
                if abs(float((p.split("floor")[-1])[1:]) - self.scene_height[-1]) < 1.0:
                    return p
            except (ValueError, IndexError):
                continue
        # 若所有候选都含 floor 但高度都不匹配，返回第一个候选以免完全退化为 GT
        return possible[0] if possible else None

    def _open_vocab_goal_with_geometric_verification(self, candidates, scene_path, agent_pos=None):
        """Fuse semantic similarity and geometric (DISK+LightGlue) score; return best position (创新点4)."""
        self._ov_geo_floor_weak = False
        alpha = self.open_vocab_semantic_weight
        beta = self.fusion_geometric_weight
        use_sem = self.open_vocab_fusion_use_semantic_score
        use_geo = self.open_vocab_fusion_use_geometric_score
        geo_scores = []
        for instance_info, position, sem_sim in candidates:
            if not use_geo:
                geo_scores.append(0.0)
                continue
            frame_id = instance_info[1]
            img_name = "rgb/frame_" + frame_id + ".png"
            instance_img = cv2.imread(os.path.join(scene_path, img_name))
            if instance_img is None:
                geo_scores.append(0.0)
                continue
            instance_img = instance_img[:, :, ::-1]
            gs = self.compute_image_pair_similarity(self.instance_imagegoal, instance_img)
            geo_scores.append(float(gs))
        sem_scores = [c[2] for c in candidates]
        if self.fusion_adaptive_weights and use_sem and use_geo:
            alpha, beta = adaptive_weights(
                np.array(sem_scores),
                np.array(geo_scores),
                base_alpha=alpha,
                base_beta=beta,
            )
        best_i, fused_arr = self.planner.fuse_semantic_geometric(
            sem_scores,
            geo_scores,
            alpha,
            beta,
            normalize_geometric_mode=self.fusion_normalize_geometric,
            use_semantic=use_sem,
            use_geometric=use_geo,
        )
        fused = np.asarray(fused_arr, dtype=np.float64)
        # === 方案 A：SG spatial prior reranking ===
        _sp_w = getattr(self, "sg_spatial_prior_weight", 0.0)
        _sp_sg = self.semantic_gaussian_goal if hasattr(self, "semantic_gaussian_goal") else None
        self._sg_prior_status = "off"
        if _sp_w > 0.0 and _sp_sg is not None and fused.size > 0:
            sg_ready = (
                getattr(_sp_sg, "_clip_vectors", None) is not None
                and getattr(_sp_sg, "_means3D", None) is not None
            )
            if not sg_ready:
                try:
                    _sp_sg.load_scene(self.scene_name)
                    sg_ready = (
                        getattr(_sp_sg, "_clip_vectors", None) is not None
                        and getattr(_sp_sg, "_means3D", None) is not None
                    )
                except Exception:
                    sg_ready = False
            if not sg_ready:
                self._sg_prior_status = "skip_not_ready"
            else:
                positions = np.array(
                    [list(c[1]) for c in candidates], dtype=np.float64
                )
                try:
                    sg_prior = _sp_sg.score_positions(
                        self.instance_imagegoal,
                        positions,
                        radius_m=self.sg_spatial_prior_radius_m,
                        topk_local=self.sg_spatial_prior_topk_local,
                        aggregator=self.sg_spatial_prior_aggregator,
                        use_xz_only=True,
                    )
                except Exception as _e:
                    self._sg_prior_status = "error:%s" % str(_e)[:60]
                    sg_prior = None
                if sg_prior is None:
                    if self._sg_prior_status == "off":
                        self._sg_prior_status = "skip_none"
                elif sg_prior.size != fused.size:
                    self._sg_prior_status = "skip_size(%d!=%d)" % (sg_prior.size, fused.size)
                else:
                    s_raw = np.asarray(sg_prior, dtype=np.float64)
                    raw_range = float(s_raw.max() - s_raw.min())
                    if raw_range < self.sg_spatial_prior_min_range:
                        self._sg_prior_status = "skip_low_signal(range=%.4f<%.4f)" % (
                            raw_range, self.sg_spatial_prior_min_range,
                        )
                        self._sg_prior_detail = "n=%d raw=%s" % (
                            len(candidates), np.round(sg_prior, 4).tolist(),
                        )
                    else:
                        if self.sg_spatial_prior_normalize == "minmax":
                            sg_norm = (s_raw - float(s_raw.min())) / (raw_range + 1e-12)
                        elif self.sg_spatial_prior_normalize == "raw":
                            sg_norm = s_raw
                        else:  # zscore
                            std = float(s_raw.std()) + 1e-9
                            sg_norm = (s_raw - float(s_raw.mean())) / std
                        prior_arr = self.sg_spatial_prior_weight * sg_norm
                        new_fused = fused + prior_arr
                        new_best = int(np.argmax(new_fused))
                        changed = (new_best != best_i)
                        self._sg_prior_status = "rerank(%d->%d)" % (best_i, new_best) if changed else "applied_no_change"
                        self._sg_prior_detail = "n=%d range=%.4f raw=%s norm=%s" % (
                            len(candidates), raw_range,
                            np.round(sg_prior, 4).tolist(),
                            np.round(sg_norm, 3).tolist(),
                        )
                        fused = new_fused
                        best_i = new_best

        # === 路径 1a + 2a：color + topology prior reranking ===
        # 不复用 SG CLIP；color 用 Gaussian rgb_colors，topo 用 OV instance scene graph。
        self._sg_color_status = "off"
        self._sg_topo_status = "off"
        # 修 detail 残留 bug：每回合开始清空，避免 skip_low_signal 时输出上一回合的 raw/norm。
        self._sg_color_detail = ""
        self._sg_topo_detail = ""
        _color_w = float(getattr(self, "sg_color_prior_weight", 0.0))
        _topo_w = float(getattr(self, "sg_topo_prior_weight", 0.0))
        if (_color_w > 0.0 or _topo_w > 0.0) and fused.size > 0:
            positions_arr = np.array(
                [list(c[1]) for c in candidates], dtype=np.float64
            )

            # ---- 1a 颜色直方图 ----
            if _color_w > 0.0:
                _sp_sg = self.semantic_gaussian_goal if hasattr(self, "semantic_gaussian_goal") else None
                sg_color_raw = None
                if _sp_sg is not None:
                    sg_ready = (
                        getattr(_sp_sg, "_means3D", None) is not None
                        and getattr(_sp_sg, "_rgb_colors", None) is not None
                    )
                    if not sg_ready:
                        try:
                            _sp_sg.load_scene(self.scene_name)
                            sg_ready = (
                                getattr(_sp_sg, "_means3D", None) is not None
                                and getattr(_sp_sg, "_rgb_colors", None) is not None
                            )
                        except Exception:
                            sg_ready = False
                    if not sg_ready:
                        self._sg_color_status = "skip_no_rgb"
                    else:
                        try:
                            sg_color_raw = _sp_sg.score_positions_color(
                                self.instance_imagegoal,
                                positions_arr,
                                radius_m=self.sg_color_prior_radius_m,
                                bins_per_channel=self.sg_color_prior_bins,
                                min_neighbors=self.sg_color_prior_min_neighbors,
                                use_xz_only=True,
                            )
                        except Exception as _e:
                            self._sg_color_status = "error:%s" % str(_e)[:60]
                            sg_color_raw = None
                else:
                    self._sg_color_status = "skip_no_sg"
                if sg_color_raw is not None and sg_color_raw.size == fused.size:
                    s = np.asarray(sg_color_raw, dtype=np.float64)
                    rng = float(s.max() - s.min())
                    if rng < self.sg_color_prior_min_range:
                        self._sg_color_status = "skip_low_signal(range=%.4f<%.4f)" % (
                            rng, self.sg_color_prior_min_range,
                        )
                    else:
                        if self.sg_color_prior_normalize == "minmax":
                            s_norm = (s - float(s.min())) / (rng + 1e-12)
                        elif self.sg_color_prior_normalize == "raw":
                            s_norm = s
                        else:  # zscore
                            std = float(s.std()) + 1e-9
                            s_norm = (s - float(s.mean())) / std
                        prev_best = int(best_i)
                        fused = fused + _color_w * s_norm
                        new_best = int(np.argmax(fused))
                        self._sg_color_status = (
                            "rerank(%d->%d)" % (prev_best, new_best) if new_best != prev_best else "applied_no_change"
                        )
                        self._sg_color_detail = "n=%d range=%.4f raw=%s norm=%s" % (
                            len(candidates), rng,
                            np.round(s, 4).tolist(),
                            np.round(s_norm, 3).tolist(),
                        )
                        best_i = new_best

            # ---- 2a 拓扑 prior ----
            if _topo_w > 0.0:
                try:
                    from map_planning_utils.instance_topology_prior import InstanceTopologyPrior
                except Exception:
                    InstanceTopologyPrior = None
                if InstanceTopologyPrior is None:
                    self._sg_topo_status = "skip_import_failed"
                else:
                    topo = getattr(self, "_topo_prior", None)
                    if (
                        topo is None
                        or getattr(topo, "_scene_path", None) != getattr(self.open_vocab_retriever, "_scene_path", None)
                    ):
                        topo = InstanceTopologyPrior(
                            coco_ids=self.open_vocab_retriever.COCO_IDS,
                            radius_m=self.sg_topo_prior_radius_m,
                            softmax_temperature=self.sg_topo_prior_softmax_temp,
                            topn_classes=self.sg_topo_prior_topn_classes,
                            use_xz_only=True,
                        )
                        ok = topo.load_from_retriever(self.open_vocab_retriever)
                        topo._scene_path = getattr(self.open_vocab_retriever, "_scene_path", None)
                        self._topo_prior = topo if ok else None
                        topo = self._topo_prior
                    if topo is None or not topo.is_ready():
                        self._sg_topo_status = "skip_not_ready"
                    else:
                        try:
                            v_target = self.open_vocab_retriever.encode_image(self.instance_imagegoal)
                            cand_cids = self.open_vocab_retriever.get_last_retrieve_coco_ids()
                            sg_topo_raw = topo.score_positions(
                                v_target, positions_arr, candidate_coco_ids=cand_cids,
                            )
                        except Exception as _e:
                            self._sg_topo_status = "error:%s" % str(_e)[:60]
                            sg_topo_raw = None
                        if sg_topo_raw is not None and sg_topo_raw.size == fused.size:
                            s = np.asarray(sg_topo_raw, dtype=np.float64)
                            rng = float(s.max() - s.min())
                            if rng < self.sg_topo_prior_min_range:
                                self._sg_topo_status = "skip_low_signal(range=%.4f<%.4f)" % (
                                    rng, self.sg_topo_prior_min_range,
                                )
                            else:
                                if self.sg_topo_prior_normalize == "minmax":
                                    s_norm = (s - float(s.min())) / (rng + 1e-12)
                                elif self.sg_topo_prior_normalize == "raw":
                                    s_norm = s
                                else:
                                    std = float(s.std()) + 1e-9
                                    s_norm = (s - float(s.mean())) / std
                                prev_best = int(best_i)
                                fused = fused + _topo_w * s_norm
                                new_best = int(np.argmax(fused))
                                self._sg_topo_status = (
                                    "rerank(%d->%d)" % (prev_best, new_best) if new_best != prev_best else "applied_no_change"
                                )
                                self._sg_topo_detail = "n=%d range=%.4f raw=%s norm=%s" % (
                                    len(candidates), rng,
                                    np.round(s, 4).tolist(),
                                    np.round(s_norm, 3).tolist(),
                                )
                                best_i = new_best

        if (
            agent_pos is not None
            and self.open_vocab_agent_dist_penalty > 0.0
            and fused.size > 0
        ):
            ap = np.asarray(agent_pos, dtype=np.float64).reshape(-1)[:3]
            dists = []
            for _inst, pos, _s in candidates:
                p = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
                dists.append(float(np.linalg.norm(p - ap)))
            dists = np.asarray(dists, dtype=np.float64)
            dmax = float(dists.max()) + 1e-6
            fused = fused - self.open_vocab_agent_dist_penalty * (dists / dmax)
            best_i = int(np.argmax(fused))
        if (
            use_geo
            and self.open_vocab_min_geometric_matches > 0.0
            and fused.size > 0
        ):
            geo_arr = np.asarray(geo_scores, dtype=np.float64)
            order = np.argsort(-fused)
            picked = None
            for j in order:
                ji = int(j)
                if geo_arr[ji] >= self.open_vocab_min_geometric_matches:
                    picked = ji
                    break
            if picked is not None and picked != best_i:
                print(
                    "[goal] open_vocab_geo_floor | use rank fused_idx=%d geo=%.0f over argmax_i=%d geo=%.0f (min=%.0f)"
                    % (
                        picked,
                        float(geo_arr[picked]),
                        best_i,
                        float(geo_arr[best_i]),
                        self.open_vocab_min_geometric_matches,
                    )
                )
                best_i = picked
            elif picked is None:
                self._ov_geo_floor_weak = True
                print(
                    "[goal] warn open_vocab_geo_floor | no candidate geo>=%.0f (keep argmax_i=%d geo=%.0f)"
                    % (
                        self.open_vocab_min_geometric_matches,
                        best_i,
                        float(geo_arr[best_i]),
                    )
                )
        # Optional API-based final rerank on top-k fused candidates (vision model expected).
        if self.open_vocab_api_rerank_enable and fused.size > 1:
            try:
                k = max(1, min(int(self.open_vocab_api_rerank_topk), int(fused.size)))
                top_idx = np.argsort(-fused)[:k].tolist()
                old_best = int(best_i)
                if len(top_idx) >= 2:
                    score_margin = float(fused[int(top_idx[0])] - fused[int(top_idx[1])])
                else:
                    score_margin = float("inf")
                self._ov_api_rerank_topk_used = int(k)
                self._ov_api_rerank_old_best = old_best
                self._ov_api_rerank_new_best = old_best
                self._ov_api_rerank_score_margin = score_margin
                self._ov_api_rerank_accepted = 0
                self._ov_api_rerank_status = "prepare"
                cand_imgs = []
                for ci in top_idx:
                    try:
                        frame_id = candidates[int(ci)][0][1]
                        img_name = "rgb/frame_" + frame_id + ".png"
                        img = cv2.imread(os.path.join(scene_path, img_name))
                        if img is None:
                            cand_imgs = []
                            self._ov_api_rerank_status = "image_missing"
                            break
                        cand_imgs.append(img[:, :, ::-1])
                    except Exception:
                        cand_imgs = []
                        self._ov_api_rerank_status = "image_error"
                        break
                if cand_imgs:
                    picked_rank = self._call_open_vocab_api_rerank(
                        self.instance_imagegoal, cand_imgs
                    )
                    status = getattr(self, "_last_ov_api_rerank_status", "unknown")
                    self._ov_api_rerank_status = status
                    if picked_rank is not None:
                        self._ov_api_rerank_picked_rank = int(picked_rank)
                        new_best = int(top_idx[int(picked_rank)])
                        accept_vlm = (
                            self.open_vocab_api_rerank_accept_mode == "always"
                            or new_best == old_best
                            or score_margin <= self.open_vocab_api_rerank_margin_threshold
                        )
                        if not accept_vlm:
                            status = "ok_rejected_margin"
                        accepted_best = new_best if accept_vlm else old_best
                        changed = int(accept_vlm and new_best != old_best)
                        self._ov_api_rerank_changed = changed
                        self._ov_api_rerank_new_best = accepted_best
                        self._ov_api_rerank_status = status
                        self._ov_api_rerank_accepted = int(accept_vlm)
                        if changed:
                            self.ov_api_rerank_changed_count += 1
                        msg = (
                            "[goal] ov_api_rerank | status=%s picked_rank=%d "
                            "old_best=%d vlm_best=%d final_best=%d changed=%d "
                            "accepted=%d accept_mode=%s margin=%.6f threshold=%.6f "
                            "topk=%d top_idx=%s raw=%s"
                            % (
                                status,
                                int(picked_rank),
                                old_best,
                                new_best,
                                accepted_best,
                                changed,
                                int(accept_vlm),
                                self.open_vocab_api_rerank_accept_mode,
                                score_margin,
                                self.open_vocab_api_rerank_margin_threshold,
                                k,
                                top_idx,
                                getattr(self, "_last_ov_api_rerank_raw", ""),
                            )
                        )
                        print(msg)
                        logging.info(msg)
                        best_i = accepted_best
                    else:
                        self._ov_api_rerank_picked_rank = -1
                        self._ov_api_rerank_changed = 0
                        self._ov_api_rerank_new_best = old_best
                        self._ov_api_rerank_accepted = 0
                        msg = (
                            "[goal] ov_api_rerank | status=%s picked_rank=-1 "
                            "old_best=%d new_best=%d changed=0 accepted=0 "
                            "accept_mode=%s margin=%.6f threshold=%.6f topk=%d top_idx=%s raw=%s"
                            % (
                                status,
                                old_best,
                                old_best,
                                self.open_vocab_api_rerank_accept_mode,
                                score_margin,
                                self.open_vocab_api_rerank_margin_threshold,
                                k,
                                top_idx,
                                getattr(self, "_last_ov_api_rerank_raw", ""),
                            )
                        )
                        print(msg)
                        logging.info(msg)
                else:
                    self.ov_api_rerank_fallback_count += 1
                    msg = (
                        "[goal] ov_api_rerank | status=%s picked_rank=-1 "
                        "old_best=%d new_best=%d changed=0 accepted=0 "
                        "accept_mode=%s margin=%.6f threshold=%.6f topk=%d top_idx=%s"
                        % (
                            self._ov_api_rerank_status,
                            old_best,
                            old_best,
                            self.open_vocab_api_rerank_accept_mode,
                            score_margin,
                            self.open_vocab_api_rerank_margin_threshold,
                            k,
                            top_idx,
                        )
                    )
                    print(msg)
                    logging.info(msg)
            except Exception as _e:
                self._ov_api_rerank_status = "error"
                self.ov_api_rerank_fallback_count += 1
                msg = "[goal] ov_api_rerank | status=error error=%s" % str(_e)[:120]
                print(msg)
                logging.info(msg)
        try:
            self._last_ov_merit_fused_score = float(fused[int(best_i)])
            self._last_ov_merit_sem_score = float(sem_scores[int(best_i)])
        except Exception:
            self._last_ov_merit_fused_score = float("nan")
            self._last_ov_merit_sem_score = float("nan")
        return candidates[best_i][1]

    def compare_bbox(self, target_obs, goal_coco_id):
        base_scene_path = self.base_scene_path
        posssible_scenes = glob(os.path.join(base_scene_path, f'{self.scene_name}*'))
        for i in range(len(posssible_scenes)):
            if abs(float((posssible_scenes[i].split('floor')[-1])[1:]) - self.scene_height[-1]) < 1.0:
                scene_path = posssible_scenes[i]
                transforms_json_path = os.path.join(scene_path, 'transforms.json')
                instance_json_path = os.path.join(scene_path, 'instance_retrieval.json')
                break
        try:
            instance_retrieval = json.load(open(instance_json_path, "r"))
        except:
            import sys
            print(posssible_scenes)
            print(self.scene_height)
            print(self.scene_name)
            sys.exit()
        curr_instance_list = instance_retrieval[str(goal_coco_id)]
        if len(curr_instance_list) > 0:
            matched_keypoints_list = []
            for bi_pairs in tqdm(curr_instance_list):
                img_name = 'rgb/frame_' + bi_pairs[1] + '.png'
                instance_img = cv2.imread(os.path.join(scene_path, img_name))[:, :, ::-1]
                matched_keypoints_list.append(self.compute_image_pair_similarity(target_obs, instance_img))
            matched_keypoints_array = np.array(matched_keypoints_list)
            max_index = np.argmax(matched_keypoints_array, axis=0)
            print(self.rank, self.episode_no, curr_instance_list[max_index][1])
            sele_bbox = curr_instance_list[max_index][0][0]
            sele_goal_pos = [sele_bbox[0], sele_bbox[2], sele_bbox[4]]
        else:
            sele_goal_pos = [0, 0, 0]
        return sele_goal_pos

    def compute_image_pair_similarity(self, obs1, obs2, resize_factors=4):

        method = self.similarity_method
        # method = 3

        if method == 0:
            # implement the histogram measurement
            rgb1 = obs1["rgb"] 
            depth1 = obs1["depth"][:, :, 0]
            rgb2 = obs2["rgb"] 
            depth2 = obs2["depth"][:, :, 0]

            rgb1 = cv2.resize(rgb1, (rgb1.shape[1] // resize_factors, rgb1.shape[0] // resize_factors))
            depth1 = cv2.resize(depth1, (depth1.shape[1] // resize_factors, depth1.shape[0] // resize_factors))
            rgb2 = cv2.resize(rgb2, (rgb2.shape[1] // resize_factors, rgb2.shape[0] // resize_factors))
            depth2 = cv2.resize(depth2, (depth2.shape[1] // resize_factors, depth2.shape[0] // resize_factors))

            hist_img1 = cv2.calcHist([rgb1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_img2 = cv2.calcHist([rgb2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            rgb_similarity_score = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

            depth_hist1, _ = np.histogram(depth1/(self.max_depth*1000), bins=256, range=(0, 1))
            depth_hist2, _ = np.histogram(depth2/(self.max_depth*1000), bins=256, range=(0, 1))
            depth_similarity_score = np.corrcoef(depth_hist1, depth_hist2)[0, 1]
            similarity_score = 0.5 * rgb_similarity_score + 0.5 * depth_similarity_score

        elif method == 1:
            with torch.set_grad_enabled(False):
                if isinstance(obs1, dict):
                    ob = numpy_image_to_torch(obs1['rgb']).to(self.device)
                    gi = numpy_image_to_torch(obs2['rgb']).to(self.device)
                else:
                    ob = numpy_image_to_torch(obs1).to(self.device)
                    gi = numpy_image_to_torch(obs2).to(self.device)
                try:
                    feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                        )
                    # indices with shape (K, 2)
                    matches = matches01['matches']
                    # in case that the matches collapse make a check
                    if matches.shape[0] > 2048:
                        similarity_score = 0
                    else:
                        similarity_score = matches.shape[0]
                except:
                    similarity_score = 0

        elif method == 2:
            if isinstance(obs1, dict):
                gray1 = cv2.cvtColor(obs1['rgb'], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(obs2['rgb'], cv2.COLOR_BGR2GRAY)
            else:
                gray1 = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(obs2, cv2.COLOR_BGR2GRAY)
            try:
                # Initialize SIFT detector
                sift = cv2.SIFT_create()

                # Detect keypoints and compute descriptors
                keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
                keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

                # Define FLANN matcher parameters
                index_params = dict(algorithm=1, trees=5)  # Using FLANN_INDEX_KDTREE algorithm
                search_params = dict(checks=50)  # Number of times the tree(s) in the index should be recursively traversed

                # Initialize FLANN matcher
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # Match descriptors
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)

                # Apply ratio test to find good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                similarity_score = len(good_matches)
            except:
                similarity_score = 0

        elif method == 3:
            with torch.set_grad_enabled(False):                
                if isinstance(obs1, dict):
                    ob = cv2.cvtColor(obs1['rgb'], cv2.COLOR_BGR2GRAY)
                    gi = cv2.cvtColor(obs2['rgb'], cv2.COLOR_BGR2GRAY)
                else:
                    ob = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
                    gi = cv2.cvtColor(obs2, cv2.COLOR_BGR2GRAY)
                torch_gray0 = numpy_image_to_torch(ob).to(self.device)[None]
                torch_gray1 = numpy_image_to_torch(gi).to(self.device)[None]
                try:
                    x = {'image0': torch_gray0, 'image1': torch_gray1}
                    pred = self.pipeline_gluestick(x)
                    pred = batch_to_np(pred)
                    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                    m0 = pred["matches0"]

                    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
                    line_matches = pred["line_matches0"]

                    valid_matches = m0 != -1
                    match_indices = m0[valid_matches]

                    similarity_score = match_indices.shape[0]
                    # matched_kps0 = kp0[valid_matches]
                    # matched_kps1 = kp1[match_indices]

                    # valid_matches = line_matches != -1
                    # match_indices = line_matches[valid_matches]
                    # matched_lines0 = line_seg0[valid_matches]
                    # matched_lines1 = line_seg1[match_indices]
                except:
                    similarity_score = 0
        
        return similarity_score


    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o
    

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
    
    def from_env_to_map(self, pos, rot=None):
        """Converts x, y coordinates from Habitat simulator to map coordinates."""
        if rot is None:
            source_position = np.array(self.start_pos_w_env, dtype=np.float32)
            source_rotation = pu.quaternion_from_coeff([self.start_rot_w_env.x, self.start_rot_w_env.y, \
                                                        self.start_rot_w_env.z, self.start_rot_w_env.w])
            goal_position = np.array(pos)
            direction_vector = goal_position - source_position
            direction_vector_agent = pu.quaternion_rotate_vector(
                source_rotation.inverse(), direction_vector
            )
            rho, phi = pu.cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            dist, angle = rho, -phi
            x = int(dist*np.cos(angle)*(100. / self.mapper.resolution))
            y = int(dist*np.sin(angle)*(100. / self.mapper.resolution))
            map_size = self.mapper.full_h
            robot_pos = [map_size // 2 + y,
                        map_size // 2 + x]
            global_pos = [robot_pos[0] , robot_pos[1]]
            local_pos = [robot_pos[0] - self.mapper.lmb[0, 0].item() , robot_pos[1] - self.mapper.lmb[0, 2].item()]
            local_pos = np.array(local_pos).astype(int)

            return local_pos

    def _preprocess_obs(self, obs, use_seg=True):

        rgb = obs['rgb']
        depth = obs['depth']

        depth = self._preprocess_depth(depth, self.min_depth, self.max_depth)

        ds = self.env_frame_width // self.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        # lj dataset
        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
