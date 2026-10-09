#!/usr/bin/env bash
# 并行模式用 scripts/run_log_same_as_terminal.sh 写 log（见 innov 脚本头注释）。串行仍可用 tee。
# 用法：
#   串行： bash run_ablation_train_g6.sh all [gpu] [ep]
#   多卡并行： bash run_ablation_train_g6.sh parallel [ep] <gpu0> <gpu1> ... <gpu4>
#       例： bash ... parallel 1000 0 1 2 3 4
#       需要至少 5 个 GPU 编号。
#   单跑： bash ... 00_g6_full 0 1000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_train_g6"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_train_g6"

PRESET="${1:-all}"
SESSION_LOG=""
GPU=""
EP=""

STEMS_ALL=(00_g6_full 01_no_circuit_breaker 02_Agroup_disagree_routing 03_sg_only_no_ov_fallback 04_ov_only_no_sg)

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

if [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  n_stems=${#STEMS_ALL[@]}
  if [[ ${#local_gpus[@]} -lt $n_stems ]]; then
    echo "用法: $0 parallel <ep> <${n_stems} 个 GPU 编号…>  例: $0 parallel 1000 0 1 2 3 4" >&2
    exit 1
  fi
  mkdir -p "${LOG_DIR}"
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
  mkdir -p "${LOG_DIR}"
  SESSION_LOG="${LOG_DIR}/session_all_${EP}ep_$(date +%Y%m%d_%H%M%S).log"
  ln -sf "$(basename "${SESSION_LOG}")" "${LOG_DIR}/session_all_${EP}ep.latest.log"
  {
    echo "[INFO] 串行 all：终端与「会话总日志」逐行一致 → ${SESSION_LOG}"
    echo "[INFO] 各 stem 独立 log 仅在轮到该任务时增长；看全程请 tail -f 本会话 session 文件。"
  } | tee -a "${SESSION_LOG}"
  for stem in "${STEMS_ALL[@]}"; do
    run_one "${stem}"
  done
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] ablation_train_g6 preset=${PRESET} ep=${EP}"
