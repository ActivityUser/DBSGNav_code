#!/usr/bin/env bash
# 创新点5（use_open_vocab_benchmark）价值实验：仅比较 bench_off vs bench_on。
# 约束：两者除 use_open_vocab_benchmark 外完全一致（SG/OV/动态语义等均保持开启）。
# 用法：
#   单跑：bash run_ablation_innov5_benchmark_value.sh bench_on 0 1000
#   串行：bash run_ablation_innov5_benchmark_value.sh all 0 1000
#   双卡并行：bash run_ablation_innov5_benchmark_value.sh parallel 1000 0 1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_innov5_benchmark_value"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innov5_benchmark_value"

PRESET="${1:-all}"
GPU=""
EP=""
STEMS=(bench_off bench_on)

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
  echo "[parallel] started stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  if [[ ${#local_gpus[@]} -lt 2 ]]; then
    echo "用法: $0 parallel <ep> <gpu_for_bench_off> <gpu_for_bench_on>" >&2
    exit 1
  fi
  run_one_parallel bench_off "${local_gpus[0]}"
  pid1=$!
  run_one_parallel bench_on "${local_gpus[1]}"
  pid2=$!
  wait "${pid1}" && wait "${pid2}"
elif [[ "${PRESET}" == "all" ]]; then
  GPU="${2:-0}"
  EP="${3:-1000}"
  for stem in "${STEMS[@]}"; do
    run_one "${stem}"
  done
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] innov5 benchmark value experiment preset=${PRESET} ep=${EP}"
echo "[NOTE] 汇总: python experiments/tune_goal_params/20260417_021210/tools/summarize_innov5_benchmark_value.py --ep ${EP}"
