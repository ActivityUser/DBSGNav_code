import sys
sys.path.append("..")
from vector_env.threaded_env import construct_envs
from habitat.config.default import get_config
import time
from pprint import pprint
import numpy as np
from collections import deque
import logging
from typing import Optional


class Runner():


    def __init__(
        self,
        cfg_pth="configs/instance_imagenav.yaml",
        max_episodes: Optional[int] = None,
        episode_goal_summary_append_path: Optional[str] = None,
    ):

        self._agent = None
        self.obs_transforms = []
        self._envs = construct_envs(
            cfg_pth, episode_goal_summary_append_path=episode_goal_summary_append_path
        )
        self._config = get_config(cfg_pth)
        self.num_envs = self._config.end2end_imagenav.num_envs
        self.num_nodes = self._config.end2end_imagenav.num_nodes
        self.num_gpus = self._config.end2end_imagenav.num_gpus
        self.total_envs = self.num_envs * self.num_gpus * self.num_nodes

        # time utils
        self.start_time = time.time()

        # episodic parameters setting（可由 run.py --max_episodes 覆盖，无需改 yaml）
        self._max_episodes = self._config.end2end_imagenav.max_episodes
        if max_episodes is not None:
            self._max_episodes = int(max_episodes)
        self._timestep = 0
        split_name = str(getattr(self._config.end2end_imagenav, "split", "")).strip().lower()
        self._rolling_window = int(
            getattr(self._config.end2end_imagenav, "train_report_rolling_window", 0)
        )
        self._enable_train_rolling_report = (
            split_name == "train" and self._rolling_window > 0
        )
        self._rolling_success = deque(maxlen=self._rolling_window) if self._enable_train_rolling_report else None
        self._rolling_spl = deque(maxlen=self._rolling_window) if self._enable_train_rolling_report else None
        self._rolling_dist = deque(maxlen=self._rolling_window) if self._enable_train_rolling_report else None

        # 同时写 eval_metrics.log 与当前 sys.stdout。子进程内 print 不经过主进程 tee，见 run.py 侧车文件
        _fmt = "%(asctime)s - %(message)s"
        _handlers = [
            logging.FileHandler("eval_metrics.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
        try:
            logging.basicConfig(
                level=logging.INFO,
                format=_fmt,
                handlers=_handlers,
                force=True,  # py3.8+：重配 root，避免仅 FileHandler 时终端与 tee 都看不到 logging
            )
        except TypeError:
            logging.basicConfig(
                filename="eval_metrics.log",
                level=logging.INFO,
                format=_fmt,
            )
            if not any(
                isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
                for h in logging.getLogger().handlers
            ):
                _sh = logging.StreamHandler(sys.stdout)
                _sh.setFormatter(logging.Formatter(_fmt))
                logging.getLogger().addHandler(_sh)


    def _create_agent(self, resume_state, **kwargs):
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        # self._create_obs_transforms()
        # self._agent = end2end_imagenav_agent(
        #     resume_state,
        #     **kwargs
        # )
        pass

    def train(self):
        self._envs.reset()
        episodes_count = [0 for i in range(self.total_envs)]
        finished = [False for i in range(self.total_envs)]
        self.start_time = time.time()
        while True:
            # report info every 20 steps
            self._timestep += 1
            if self._timestep % 20 == 0:
                print("FPS: {:.3f}, current timestep: {}".format((self._timestep * self.total_envs)/(time.time()-self.start_time), \
                                                                  self._timestep * self.total_envs ))
            

            if np.array(episodes_count).sum() == self._max_episodes * self.total_envs:
                pprint("all episode over")
                break
            finished = [(episodes_count[i]==self._max_episodes) for i in range(self.total_envs)]            
            actions = [{'action': 0, "finished": finished[e]} for e in range(self.total_envs)]
            _, _, done, info = self._envs.step(actions)
            episodes_count = [episodes_count[i]+1 if done[i] else episodes_count[i] for i in range(self.total_envs) ]

            # Running metrics (print regularly even if no new episode finished this step)
            succ = np.array([info[i]['success'] for i in range(self.total_envs)])
            spl = np.array([info[i]['spl'] for i in range(self.total_envs)])
            dist = np.array([info[i]['distance_to_goal'] for i in range(self.total_envs)])
            count = np.array([info[i]['episode_count'] for i in range(self.total_envs)])
            goal_cov = np.array(
                [info[i].get('goal_non_gt_coverage', np.nan) for i in range(self.total_envs)],
                dtype=np.float32,
            )
            gt_fb_rate = np.array(
                [info[i].get('goal_gt_fallback_rate', np.nan) for i in range(self.total_envs)],
                dtype=np.float32,
            )
            total_count = np.sum(count)
            if self._timestep % 20 == 0:
                if total_count > 0:
                    mean_success = np.sum(succ * count) / total_count
                    mean_spl = np.sum(spl * count) / total_count
                    mean_dist = np.sum(dist * count) / total_count
                    mean_goal_cov = np.nanmean(goal_cov) if np.any(~np.isnan(goal_cov)) else np.nan
                    mean_gt_fb_rate = np.nanmean(gt_fb_rate) if np.any(~np.isnan(gt_fb_rate)) else np.nan
                    print(
                        " running succ / spl / dist / nonGT_cov / GT_fb_rate : "
                        "{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}({:.0f})".format(
                            np.mean(mean_success),
                            np.mean(mean_spl),
                            np.mean(mean_dist),
                            float(mean_goal_cov) if not np.isnan(mean_goal_cov) else float("nan"),
                            float(mean_gt_fb_rate) if not np.isnan(mean_gt_fb_rate) else float("nan"),
                            total_count,
                        )
                    )
                else:
                    print(" running succ / spl / dist : N/A (no finished episodes yet)")

            # eval parameters  
            done_rank = np.array([done[i] for i in range(self.total_envs)]).astype(int)
            if np.any(done_rank > 0):
                if self._enable_train_rolling_report:
                    appended = 0
                    for env_idx in range(self.total_envs):
                        if done_rank[env_idx] <= 0:
                            continue
                        last_success = info[env_idx].get("last_episode_success")
                        last_spl = info[env_idx].get("last_episode_spl")
                        last_dist = info[env_idx].get("last_episode_distance_to_goal")
                        if (
                            last_success is None
                            or last_spl is None
                            or last_dist is None
                        ):
                            continue
                        self._rolling_success.append(float(last_success))
                        self._rolling_spl.append(float(last_spl))
                        self._rolling_dist.append(float(last_dist))
                        appended += 1
                    if appended > 0 and len(self._rolling_success) > 0:
                        print(
                            " rolling{} succ / spl / dist : {:.5f}/{:.5f}/{:.5f}({})".format(
                                self._rolling_window,
                                float(np.mean(self._rolling_success)),
                                float(np.mean(self._rolling_spl)),
                                float(np.mean(self._rolling_dist)),
                                len(self._rolling_success),
                            )
                        )

                mean_success = np.sum(succ * count) / total_count
                mean_spl = np.sum(spl * count) / total_count
                mean_dist = np.sum(dist * count) / total_count
                mean_goal_cov = np.nanmean(goal_cov) if np.any(~np.isnan(goal_cov)) else np.nan
                mean_gt_fb_rate = np.nanmean(gt_fb_rate) if np.any(~np.isnan(gt_fb_rate)) else np.nan

                print( " Instance ImageGoal Navigation Metrics \n"
                       " succ / spl / dist / nonGT_cov / GT_fb_rate : "
                       "{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
                    np.mean(mean_success),
                    np.mean(mean_spl),
                    np.mean(mean_dist),
                    float(mean_goal_cov) if not np.isnan(mean_goal_cov) else float("nan"),
                    float(mean_gt_fb_rate) if not np.isnan(mean_gt_fb_rate) else float("nan"),
                    total_count) )
                
                log_info = " Instance ImageGoal Navigation Metrics \n" \
                           " succ / spl / dist / nonGT_cov / GT_fb_rate : {:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
                    np.mean(mean_success),
                    np.mean(mean_spl),
                    np.mean(mean_dist),
                    float(mean_goal_cov) if not np.isnan(mean_goal_cov) else float("nan"),
                    float(mean_gt_fb_rate) if not np.isnan(mean_gt_fb_rate) else float("nan"),
                    total_count)
                logging.info(log_info)



