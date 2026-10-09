import os
from collections import OrderedDict
from typing import Optional

import numpy as np
from omegaconf import OmegaConf
import habitat
import habitat_sim
from habitat.config.default import get_config
from habitat import make_dataset
from vector_env.envs.base_env import base_env
from utils.time import print_run_time
from vector_env.utils.vector_env import VectorEnv

def _scene_name_from_scene_id(scene_id: str) -> str:
    # scene_id 常见格式: ".../CrMo8WxCyVb.basis.glb"
    # 这里与 base_env 保持一致，取第一个 '.' 前的场景名
    return os.path.basename(scene_id).split(".")[0]


def _filter_dataset_by_scene_names(dataset, scene_names):
    if not scene_names:
        return dataset
    scene_names = set(scene_names)
    dataset.episodes = [
        ep for ep in dataset.episodes
        if _scene_name_from_scene_id(ep.scene_id) in scene_names
    ]
    return dataset


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            # scene = filename[: -len(scene_dataset_ext) + 4]
            scene = filename[: -len(scene_dataset_ext) ]
            scenes.append(scene)
    scenes.sort()
    return scenes


def _is_train_split(split_name: str) -> bool:
    return str(split_name).strip().lower() == "train"


def _reorder_episodes_by_scene(
    dataset,
    max_consecutive_scene_eps: int,
    interleave: bool,
    rng: Optional[np.random.RandomState] = None,
):
    """
    Reorder episodes to avoid very long single-scene streaks in train.
    This only changes episode order, not membership.
    """
    episodes = list(getattr(dataset, "episodes", []))
    if not episodes:
        return dataset

    if (not interleave) and max_consecutive_scene_eps <= 0:
        return dataset

    chunk_size = max(1, int(max_consecutive_scene_eps)) if max_consecutive_scene_eps > 0 else 1

    scene_to_eps = OrderedDict()
    for ep in episodes:
        scene_name = _scene_name_from_scene_id(ep.scene_id)
        scene_to_eps.setdefault(scene_name, []).append(ep)

    scene_order = list(scene_to_eps.keys())
    if rng is not None:
        rng.shuffle(scene_order)
        if interleave:
            for scene_name in scene_order:
                rng.shuffle(scene_to_eps[scene_name])

    reordered = []
    remaining = True
    while remaining:
        remaining = False
        for scene_name in scene_order:
            eps = scene_to_eps[scene_name]
            if not eps:
                continue
            remaining = True
            take_n = min(chunk_size, len(eps))
            reordered.extend(eps[:take_n])
            del eps[:take_n]

    dataset.episodes = reordered
    return dataset

# @print_run_time
def _load_data(cfg_path: str, episode_goal_summary_append_path: Optional[str] = None):
    configs = []
    datasets = []
    basic_config = get_config(cfg_path)
    NUM_ENVS = basic_config.end2end_imagenav.num_envs
    NUM_NODES = basic_config.end2end_imagenav.num_nodes
    NUM_GPUS = basic_config.end2end_imagenav.num_gpus
    num_processes = NUM_ENVS * NUM_NODES * NUM_GPUS

    with habitat.config.read_write(basic_config):
        split = basic_config.end2end_imagenav.split
        basic_config.habitat.dataset.split = split
    split_is_train = _is_train_split(split)
    global_train_scene_shuffle = bool(
        getattr(basic_config.end2end_imagenav, "train_scene_shuffle", False)
    )
    dataset = make_dataset(
        basic_config.habitat.dataset.type, config=basic_config.habitat.dataset
    )
    target_scenes = list(getattr(basic_config.end2end_imagenav, "target_scenes", []))
    if target_scenes:
        dataset = _filter_dataset_by_scene_names(dataset, target_scenes)
        scenes = sorted({_scene_name_from_scene_id(sid) for sid in dataset.scene_ids})
        if len(scenes) == 0:
            raise ValueError(
                f"target_scenes={target_scenes} 在 split='{split}' 中无可用 episodes。"
                "请检查 split（例如 CrMo8WxCyVb 应使用 val）或场景名拼写。"
            )
    else:
        scenes = basic_config.habitat.dataset.content_scenes
        if "*" in basic_config.habitat.dataset.content_scenes:
            scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)

    if len(scenes) > 0:
        scenes = list(scenes)
        if split_is_train and global_train_scene_shuffle:
            np.random.RandomState(int(basic_config.habitat.seed)).shuffle(scenes)
        if len(scenes) < num_processes:
            # 场景数少于进程数时重复场景列表，使每个进程至少分到一个场景（如 content_scenes: ["CrMo8WxCyVb"]）
            scenes = (scenes * (num_processes // len(scenes) + 1))[:num_processes]
        scene_split_sizes = [int(np.floor(len(scenes) / num_processes))
                             for _ in range(num_processes)]
        for i in range(len(scenes) % num_processes):
            scene_split_sizes[i] += 1
    print("Scenes per thread:")
    for i in range(num_processes):
        config_env = get_config(cfg_path)
        if episode_goal_summary_append_path:
            with habitat.config.read_write(config_env):
                # 合并到实验级 yaml 时可能没有该 struct 键，用 update 可补全
                OmegaConf.update(
                    config_env,
                    "end2end_imagenav.episode_goal_summary_append_path",
                    episode_goal_summary_append_path,
                )
        scene_targets = []
        with habitat.config.read_write(config_env):

            if len(scenes) > 0:
                scene_targets = scenes[
                    sum(scene_split_sizes[:i]):
                    sum(scene_split_sizes[:i + 1])
                ]
                # 注意：base_env 当前未使用传入 dataset 参数，实际 episode 来源由 config 决定。
                # 因此即使设置了 target_scenes，也必须把 content_scenes 显式设为 scene_targets，
                # 否则环境会回到 "*" 并加载全量场景。
                config_env.habitat.dataset.content_scenes = scene_targets
                print("Thread {}: {}".format(i, scene_targets))

            gpu_id = (i % (NUM_ENVS * NUM_GPUS)) // NUM_ENVS
            node_id = i // (NUM_ENVS * NUM_GPUS)
            config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
            config_env.end2end_imagenav.gpu_id = gpu_id
            config_env.end2end_imagenav.node_id = node_id
            split_name = str(getattr(config_env.end2end_imagenav, "split", split))
            split_is_train_env = _is_train_split(split_name)
            train_scene_shuffle = bool(
                getattr(config_env.end2end_imagenav, "train_scene_shuffle", False)
            )
            config_env.habitat.environment.iterator_options.shuffle = (
                split_is_train_env and train_scene_shuffle
            )

            # 创新点5：若指定预生成零样本数据集路径，则用其替换默认 data_path
            open_vocab_path = getattr(
                config_env.end2end_imagenav, "open_vocab_data_path", ""
            )
            if getattr(config_env.end2end_imagenav, "use_open_vocab_benchmark", False) and open_vocab_path:
                open_vocab_path = open_vocab_path.strip()
                if open_vocab_path and os.path.isfile(open_vocab_path):
                    config_env.habitat.dataset.data_path = open_vocab_path
                    config_env.habitat.dataset.content_scenes = []
                    print("Thread {}: using open_vocab_data_path={}".format(i, open_vocab_path))

        ds = habitat.make_dataset(
            config_env.habitat.dataset.type, config=config_env.habitat.dataset
        )
        # 创新点5：开放词汇评估基准——仅保留 COCO 类别外的 episode（未使用 open_vocab_data_path 时做内存过滤）
        if getattr(config_env.end2end_imagenav, "use_open_vocab_benchmark", False):
            open_vocab_path_used = getattr(config_env.end2end_imagenav, "open_vocab_data_path", "").strip()
            if not open_vocab_path_used or not os.path.isfile(open_vocab_path_used):
                try:
                    from datasets.open_vocab_iin import apply_open_vocab_filter_to_dataset
                    n_kept = apply_open_vocab_filter_to_dataset(ds)
                    if n_kept == 0:
                        print(
                            "[open_vocab_benchmark] Warning: 0 episodes after filtering. "
                            "Standard HM3D IIN only has 6 categories. Use a pre-built open-vocab dataset "
                            "(python -m datasets.open_vocab_iin build --source ... --out ...) or set open_vocab_data_path."
                        )
                    else:
                        print(f"[open_vocab_benchmark] Using {n_kept} open-vocab episodes.")
                except Exception as e:
                    print(f"[open_vocab_benchmark] Failed to filter dataset: {e}")
        if target_scenes and scene_targets:
            ds = _filter_dataset_by_scene_names(ds, scene_targets)
        if _is_train_split(getattr(config_env.end2end_imagenav, "split", split)):
            train_scene_interleave = bool(
                getattr(config_env.end2end_imagenav, "train_scene_interleave", False)
            )
            train_max_consecutive_scene_eps = int(
                getattr(config_env.end2end_imagenav, "train_max_consecutive_scene_eps", 0)
            )
            if train_scene_interleave or train_max_consecutive_scene_eps > 0:
                train_scene_shuffle = bool(
                    getattr(config_env.end2end_imagenav, "train_scene_shuffle", False)
                )
                rng = (
                    np.random.RandomState(int(config_env.habitat.seed) + i)
                    if train_scene_shuffle
                    else None
                )
                before_n = len(ds.episodes)
                ds = _reorder_episodes_by_scene(
                    ds,
                    max_consecutive_scene_eps=train_max_consecutive_scene_eps,
                    interleave=train_scene_interleave,
                    rng=rng,
                )
                print(
                    "Thread {} train episode reorder: interleave={}, max_consecutive_scene_eps={}, episodes={}".format(
                        i,
                        train_scene_interleave,
                        train_max_consecutive_scene_eps,
                        before_n,
                    )
                )
        # Deterministic evaluation slicing supports restartable long runs. It
        # is intentionally restricted to one process so start/count refer to
        # the global dataset order rather than a per-process scene subset.
        eval_episode_start = int(
            getattr(config_env.end2end_imagenav, "eval_episode_start", 0)
        )
        eval_episode_count = int(
            getattr(config_env.end2end_imagenav, "eval_episode_count", 0)
        )
        if eval_episode_start < 0 or eval_episode_count < 0:
            raise ValueError("eval_episode_start/count must be non-negative")
        if eval_episode_start > 0 or eval_episode_count > 0:
            if num_processes != 1:
                raise ValueError(
                    "Restartable episode slicing currently requires num_processes=1"
                )
            stop = (
                eval_episode_start + eval_episode_count
                if eval_episode_count > 0
                else len(ds.episodes)
            )
            total_before_slice = len(ds.episodes)
            ds.episodes = list(ds.episodes[eval_episode_start:stop])
            if not ds.episodes:
                raise ValueError(
                    "Empty evaluation slice start={} count={} total={}".format(
                        eval_episode_start,
                        eval_episode_count,
                        total_before_slice,
                    )
                )
            print(
                "Thread {} evaluation episode slice: [{}:{}) -> {} / {}".format(
                    i,
                    eval_episode_start,
                    stop,
                    len(ds.episodes),
                    total_before_slice,
                )
            )
        loaded_scene_names = sorted({_scene_name_from_scene_id(sid) for sid in ds.scene_ids})
        print("Thread {} loaded scenes: {}".format(i, loaded_scene_names))
        datasets.append(ds)
        # datasets.append(None)

        configs.append(config_env)

    return configs, datasets

def _make_env_func(config, dataset=None, rank=0):
    r"""Constructor for dummy habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    with habitat.config.read_write(config):
        config.habitat.simulator.scene = dataset.episodes[0].scene_id
    env_name = config.end2end_imagenav.env_name
    if env_name == 'instance_imagenav':
        from vector_env.envs.instance_imagenav_env import NiceEnv
    env = NiceEnv(config_env=config, dataset=dataset, rank=rank)
    if dataset is not None and env.number_of_episodes != len(dataset.episodes):
        raise RuntimeError(
            "Constructed environment discarded prepared dataset: "
            "expected {} episodes, got {}".format(
                len(dataset.episodes), env.number_of_episodes
            )
        )
    env.seed(config.habitat.seed + rank)
    return env

@print_run_time
def _vec_env_fn(configs, datasets, multiprocessing_start_method='forkserver'):

    num_envs = len(configs)
    envs = VectorEnv(
        make_env_fn=_make_env_func,
                env_fn_args=tuple(
            tuple(
                zip(configs, datasets, range(num_envs))
            )
        ),
    )
    print("habitat environments created successfully !!!")
    return envs


def construct_envs(
    cfg_path: str, episode_goal_summary_append_path: Optional[str] = None
):
    configs, datasets = _load_data(
        cfg_path, episode_goal_summary_append_path=episode_goal_summary_append_path
    )
    return _vec_env_fn(configs, datasets)
    