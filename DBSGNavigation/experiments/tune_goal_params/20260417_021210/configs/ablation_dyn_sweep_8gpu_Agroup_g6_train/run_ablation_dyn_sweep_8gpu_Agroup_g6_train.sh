#!/usr/bin/env bash
# 8 卡并行：动态语义/规划器小范围调参 sweep（母版 full_dyn_tuned，仅单项改动）。
# 用法：
#   parallel： bash run_ablation_dyn_sweep_8gpu_Agroup_g6_train.sh parallel [ep] <8 个 GPU>
#   单跑：   bash run_ablation_dyn_sweep_8gpu_Agroup_g6_train.sh s01_alpha005 0 1000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_dyn_sweep_8gpu_Agroup_g6_train"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_dyn_sweep_8gpu_Agroup_g6_train"

PRESET="${1:-parallel}"
GPU=""
EP=""

STEMS_ALL=(
  s01_alpha005
  s02_alpha015
  s03_int1
  s04_int3
  s05_cons025
  s06_cons045
  s07_plthr024
  s08_plmin4
)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  (
    echo ""
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') RUN ${stem} gpu=${GPU} ep=${EP} =========="
    echo "cfg=${cfg}"
    echo "log=${log}"
    echo "--------------------------------------------------------------"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run.py \
      --config "${cfg}" \
      --max_episodes "${EP}"
  ) 2>&1 | tee "${log}"
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
  echo "[parallel] stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  n_stems=${#STEMS_ALL[@]}
  if [[ ${#local_gpus[@]} -lt $n_stems ]]; then
    echo "用法: $0 parallel <ep> <${n_stems} 个 GPU 编号…>  例: $0 parallel 1000 0 1 2 3 4 5 6 7" >&2
    exit 1
  fi
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
    echo "[WARN] 有子任务非 0 退出" >&2
    exit "${ec}"
  fi
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] dyn_sweep_8gpu preset=${PRESET} ep=${EP}"
echo "[NOTE] 汇总: python experiments/tune_goal_params/20260417_021210/tools/summarize_dyn_sweep_8gpu.py --ep ${EP}"
