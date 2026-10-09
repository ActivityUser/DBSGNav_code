#!/usr/bin/env bash
# 日志说明：并行模式用 scripts/run_log_same_as_terminal.sh（script+伪终端）写各 stem 的 *_ep.log，
# 避免 python|tee 时子进程输出不进文件。串行 all 仍用 tee：若仍嫌不全，可单条执行：
#   ./scripts/run_log_same_as_terminal.sh logs/.../baseline_1000ep.log -- --config ... --max_episodes 1000
# 用法：
#   串行（单卡依次跑）： bash run_ablation_innov_Agroup_highscore_g6_train.sh all [gpu] [ep]
#       例： bash ... all 0 1000
#   多卡并行（推荐：每个 stem 独占一张卡，日志完整且不穿插）：
#       bash run_ablation_innov_Agroup_highscore_g6_train.sh parallel [ep] <gpu0> <gpu1> ... <gpu5>
#       例： bash ... parallel 1000 0 1 2 3 4 5
#       需要至少 6 个 GPU 编号，与 baseline innov2 innov3 innov4 innov5 innov6 一一对应。
#   单跑某一预设： bash ... baseline 0 1000
#   可选「G6 + 动态语义 + OV benchmark」全开： bash ... full_dynamic_benchmark 0 1000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_innov_Agroup_highscore_g6_train"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innov_Agroup_highscore_g6_train"
G6_LOG="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/08_ext_gm40_k8_train_tuned_Agroup_highscore_g6_1000ep.log"

PRESET="${1:-all}"
SESSION_LOG=""
GPU=""
EP=""

STEMS_ALL=(baseline innov2 innov3 innov4 innov5 innov6)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  if [[ -n "${SESSION_LOG}" ]]; then
    (
      echo ""
      echo "========== $(date '+%Y-%m-%d %H:%M:%S') RUN ${stem} gpu=${GPU} ep=${EP} =========="
      echo "cfg=${cfg}"
      echo "per-stem log=${log}"
      echo "--------------------------------------------------------------"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run.py \
        --config "${cfg}" \
        --max_episodes "${EP}"
    ) 2>&1 | tee "${log}" | tee -a "${SESSION_LOG}"
  else
    (
      echo ""
      echo "========== $(date '+%Y-%m-%d %H:%M:%S') RUN ${stem} gpu=${GPU} ep=${EP} =========="
      echo "cfg=${cfg}"
      echo "per-stem log=${log}"
      echo "--------------------------------------------------------------"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run.py \
        --config "${cfg}" \
        --max_episodes "${EP}"
    ) 2>&1 | tee "${log}"
  fi
}

# 并行：用 script(1)+伪终端写各自 log，尽量收录子进程 print / C 库行（避免「终端在刷、tee 文件不涨」）。
run_one_parallel() {
  local stem="$1"
  local gpu="$2"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export RUN_LOG_SCRIPT_QUIET=1
    cd "${ROOT}"
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" -- \
      --config "${cfg}" \
      --max_episodes "${EP}"
  ) &
  echo "[parallel] started stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ -f "${G6_LOG}" ]]; then
  ln -sf "../$(basename "${G6_LOG}")" "${LOG_DIR}/full_1000ep_Agroup_highscore_g6_official.log"
  echo "[link] full 行 -> ${G6_LOG} （见 full_1000ep_Agroup_highscore_g6_official.log）"
else
  echo "[warn] 未找到 G6 官方日志，跳过 symlink: ${G6_LOG}"
fi

if [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  n_stems=${#STEMS_ALL[@]}
  if [[ ${#local_gpus[@]} -lt $n_stems ]]; then
    echo "用法: $0 parallel <ep> <${n_stems} 个 GPU 编号…>  例: $0 parallel 1000 0 1 2 3 4 5" >&2
    exit 1
  fi
  echo "[INFO] 多卡并行：各 stem 用 script 写 *_${EP}ep.log（尽量与独占终端一致）；此处仅打印启动行。"
  pids=()
  for ((i = 0; i < n_stems; i++)); do
    run_one_parallel "${STEMS_ALL[$i]}" "${local_gpus[$i]}"
    pids+=($!)
  done
  ec=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      ec=1
    fi
  done
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] 有子任务非 0 退出，请检查各 log 末尾" >&2
    exit "${ec}"
  fi
elif [[ "${PRESET}" == "all" ]]; then
  GPU="${2:-0}"
  EP="${3:-1000}"
  SESSION_LOG="${LOG_DIR}/session_all_${EP}ep_$(date +%Y%m%d_%H%M%S).log"
  ln -sf "$(basename "${SESSION_LOG}")" "${LOG_DIR}/session_all_${EP}ep.latest.log"
  {
    echo "[INFO] 串行 all：终端与「会话总日志」逐行一致 → ${SESSION_LOG}"
    echo "[INFO] baseline/innov2/… 各自 log 仅在轮到该任务时增长；看全程请 tail -f 本会话 session 文件。"
  } | tee -a "${SESSION_LOG}"
  for stem in "${STEMS_ALL[@]}"; do
    run_one "${stem}"
  done
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] ablation_innov_Agroup_highscore_g6_train preset=${PRESET} ep=${EP}"
echo "[NOTE] full 不重跑；指标见 FULL_row_official_g6_1000ep.txt 或 summarize 脚本。"
