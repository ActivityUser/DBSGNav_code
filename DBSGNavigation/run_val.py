"""GaussNav **验证集（val）消融** 评测入口：与 ``run.py`` 共用同一套 ``Runner.train()`` 代码路径。

仅在工程组织上与 ``run.py`` 区分，便于 val 消融统一用本文件启动；行为与 ``run.py`` 一致，
实际 split / 数据路径由 ``--config`` 指向的 yaml 决定（例如 ``ablation_fullsys_std_Agroup_g6_val``）。

``--log_file`` 与侧车 ``*_episode_goal_summary.log`` 的说明同 ``run.py``。
"""
from runners.instance_imagenav_runner import Runner
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

_DEFAULT_CFG = "configs/instance_imagenav.yaml"
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_max_episodes(argv):
    if "--max_episodes" in argv:
        idx = argv.index("--max_episodes")
        if idx + 1 < len(argv):
            return int(argv[idx + 1])
    return None


def _strip_log_file(argv):
    out = list(argv)
    if "--log_file" in out:
        i = out.index("--log_file")
        if i + 1 < len(out):
            path = out[i + 1]
            del out[i : i + 2]
            return path, out
    return None, out


def main():
    log_file, argv = _strip_log_file(sys.argv)
    sys.argv = argv

    cfg = _DEFAULT_CFG
    if log_file:
        from utils.tee_log import tee_session_log

        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        sidecar = log_path.parent / f"{log_path.stem}_episode_goal_summary.log"
        with tee_session_log(log_path):
            print(f"[run_val.py] 终端日志同时写入: {log_path.resolve()}", flush=True)
            print(
                f"[run_val.py] [episode_goal_summary] 子进程追加到: {sidecar.resolve()}",
                flush=True,
            )
            _train(episode_goal_summary_append_path=str(sidecar.resolve()))
    else:
        _train()


if __name__ == "__main__":
    main()
