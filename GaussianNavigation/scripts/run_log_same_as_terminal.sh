#!/usr/bin/env bash
# 尽量让「保存的 log」接近独占终端里看到的内容：用 script(1) 分配伪终端，子进程里 print 也会进 typescript。
# 并行多卡时请由调用方 setsid + 重定向 script 自身 stdout，避免 6 路输出打在同一终端。
#
# 用法（在 GaussianNavigation 根目录）:
#   ./scripts/run_log_same_as_terminal.sh path/to/out.log -- --config ... --max_episodes 1000
#
# 环境变量:
#   RUN_LOG_SCRIPT_QUIET=1  若设置：不向当前终端回显（适合后台 batch；仅写 out.log）
set -euo pipefail
GN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$GN_ROOT"
if [[ $# -lt 2 ]]; then
  echo "用法: $0 <输出log路径> -- [传给 run.py 的参数]" >&2
  exit 1
fi
OUT="$1"
shift
if [[ "${1:-}" == "--" ]]; then shift; fi
mkdir -p "$(dirname "$OUT")"
export PYTHONUNBUFFERED=1
# 默认 run.py；val 消融可设 GAUSSNAV_RUN_ENTRY=run_val.py
RUN_ENTRY="${GAUSSNAV_RUN_ENTRY:-run.py}"
PY=(stdbuf -oL -eL python -u "$RUN_ENTRY" "$@")

if ! command -v script >/dev/null 2>&1; then
  echo "[warn] 未找到 script(1)，回退为 stdbuf+tee（子进程部分行仍可能只在终端）" >&2
  exec "${PY[@]}" 2>&1 | tee "$OUT"
  exit 0
fi

cmd=$(printf '%q ' "${PY[@]}")
cmd="cd $(printf '%q' "$GN_ROOT") && export PYTHONUNBUFFERED=1 && exec $cmd"

if [[ -n "${RUN_LOG_SCRIPT_QUIET:-}" ]]; then
  # -qf：安静 + 尽快刷盘；并行多卡请用 QUIET，避免多路 script 抢同一终端。
  exec setsid script -qf -c "$cmd" "$OUT" </dev/null >/dev/null 2>&1
else
  exec script -qf -c "$cmd" "$OUT"
fi