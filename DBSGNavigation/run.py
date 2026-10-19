"""GaussNav 导航评测入口。支持消融实验配置。

**--log_file 与「终端里看到的」为何常不一致**

- ``--log_file`` 只在 **主进程** 内把 ``sys.stdout`` / ``sys.stderr`` 接到文件（Tee）。
- **Habitat 的 VectorEnv 在子进程里跑环境**：子进程里 ``print``/``logging`` 的 stdout **不是** 主进程 Tee，
  故 ``[episode_goal_summary]`` 在终端可见，但 **不会** 写入 ``--log_file``。解决：``run.py`` 会**自动**
  把每集摘要追加到与 ``--log_file`` 同目录的侧车文件 ``<stem>_episode_goal_summary.log``。
- 另：终端里还有 **C 库** 直写 fd1/2 的日志，不经过 Python，同样不会进上述 tee 文件；需要「整段终端重放」时仍用
  下文 shell ``tee``。
- 若需要 log 与 **整段终端输出** 尽量一致，请用 **shell 的 tee**（不要与 ``--log_file`` 混用，二选一即可）::

    PYTHONUNBUFFERED=1 python -u run.py --config ... --max_episodes 300 2>&1 | tee experiments/full.log

  或 ``./scripts/run_with_shell_tee.sh experiments/full.log -- --config ...``.

Runner 内 ``logging`` 会写 ``eval_metrics.log`` 并镜像到 ``sys.stdout``（含主进程 tee 时也会进主日志）。

汇总 ``[episode_goal_summary]`` 时：用 ``--log_file`` 请对侧车
``experiments/my_run_episode_goal_summary.log`` 运行
``python scripts/summarize_episode_goal_log.py``，该脚本在解析 0 条时会自动尝试该侧车路径。

示例::

  python run.py --config configs/instance_imagenav_val_nogt.yaml \\
    --max_episodes 200 \\
    --log_file experiments/my_run.log
"""
from runners.instance_imagenav_runner import Runner
import os
import sys
from pathlib import Path

# 与 run_full_pipeline 一致：open_clip/timm 可能经 huggingface_hub 拉权重，镜像可缓解直连 huggingface.co 失败
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

_DEFAULT_CFG = "configs/instance_imagenav.yaml"
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _strip_log_file(argv):
    """解析并移除 --log_file <path>，避免传入未识别的参数。"""
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
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            cfg = sys.argv[idx + 1]
    max_ep = _parse_max_episodes(sys.argv)

    def _train(*, episode_goal_summary_append_path=None):
        runner = Runner(
            cfg_pth=cfg,
            max_episodes=max_ep,
            episode_goal_summary_append_path=episode_goal_summary_append_path,
        )
        runner.train()

    if log_file:
        from utils.tee_log import tee_session_log

        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        sidecar = log_path.parent / f"{log_path.stem}_episode_goal_summary.log"
        with tee_session_log(log_path):
            print(f"[run.py] 终端日志同时写入: {log_path.resolve()}", flush=True)
            print(
                f"[run.py] [episode_goal_summary] 子进程追加到: {sidecar.resolve()}",
                flush=True,
            )
            _train(episode_goal_summary_append_path=str(sidecar.resolve()))
    else:
        _train()


if __name__ == "__main__":
    main()
