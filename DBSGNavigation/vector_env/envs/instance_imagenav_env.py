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
