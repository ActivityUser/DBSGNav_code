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
