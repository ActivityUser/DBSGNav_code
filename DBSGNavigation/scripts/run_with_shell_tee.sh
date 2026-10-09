#!/usr/bin/env bash
# 与「你在终端里看到的」最一致地保存 log：在 shell 用 tee 接 python 的 stdout+stderr
#
# 说明：--log_file 时主进程 tee 也收不到子进程里 print 的 [episode_goal_summary]（run.py 会另写侧车
# *_episode_goal_summary.log）。若还缺 C 库直打终端的字，用本脚本的 tee 可一并收录；可仍加 --log_file
# 以同时得到侧车（子进程追加）+ tee 的完整流。
#
# 用法（在 DBSGNavigation 根目录，chmod +x 后）:
#   ./scripts/run_with_shell_tee.sh experiments/goal_full.log -- \
#     --config experiments/tune_goal_params/20260417_021210/configs/08_ext_gm40_k8.yaml \
#     --max_episodes 300
set -euo pipefail
GN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GN_ROOT"
if [[ $# -lt 2 ]]; then
  echo "用法: $0 <输出log路径> -- [传给 run.py 的参数，不要加 --log_file ]" >&2
  exit 1
fi
OUT="$1"
shift
if [[ "${1:-}" == "--" ]]; then shift; fi
mkdir -p "$(dirname "$OUT")"
export PYTHONUNBUFFERED=1
python -u run.py "$@" 2>&1 | tee "$OUT"
