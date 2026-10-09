#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="${SCRIPT_DIR}"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_dense_v2_nosg_mainline_vlm)"

PRESET="${1:-parallel9}"
GPU=""
EP=""

STEMS_ALL=(mainline_nosg wo_ov_nosg wo_vlm_nosg wo_dyn_nosg wo_bench_nosg wo_geom_nosg wo_semantic_nosg w_sg replace_ov_with_sg)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py     --config "${cfg}"     --max_episodes "${EP}"     --log_file "${log}"
}

run_one_parallel() {
  local stem="$1"
  local gpu="$2"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export GAUSSNAV_RUN_ENTRY=run_val.py
    export RUN_LOG_SCRIPT_QUIET=1
    cd "${ROOT}"
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" --       --config "${cfg}"       --max_episodes "${EP}"
  ) &
  echo "[parallel] stem=${stem} cuda=${gpu} pid=$! log=${log}"
}

wait_pids_fail_any() {
  local ec=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      ec=1
    fi
  done
  return "${ec}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel9" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  G3="${6:-3}"
  G4="${7:-4}"
  G5="${8:-5}"
  G6="${9:-6}"
  G7="${10:-7}"
  G8="${11:-8}"
  pids=()
  run_one_parallel "mainline_nosg"    "${G0}"; pids+=($!)
  run_one_parallel "wo_ov_nosg"       "${G1}"; pids+=($!)
  run_one_parallel "wo_vlm_nosg"      "${G2}"; pids+=($!)
  run_one_parallel "wo_dyn_nosg"      "${G3}"; pids+=($!)
  run_one_parallel "wo_bench_nosg"    "${G4}"; pids+=($!)
  run_one_parallel "wo_geom_nosg"     "${G5}"; pids+=($!)
  run_one_parallel "wo_semantic_nosg" "${G6}"; pids+=($!)
  run_one_parallel "w_sg"             "${G7}"; pids+=($!)
  run_one_parallel "replace_ov_with_sg" "${G8}"; pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then ec=1; fi
  exit "${ec}"
fi

if [[ "${PRESET}" == "all" ]]; then
  GPU="${2:-0}"
  EP="${3:-1000}"
  for s in "${STEMS_ALL[@]}"; do
    run_one "${s}"
  done
  exit 0
fi

GPU="${2:-0}"
EP="${3:-1000}"
case "${PRESET}" in
  mainline_nosg|wo_ov_nosg|wo_vlm_nosg|wo_dyn_nosg|wo_bench_nosg|wo_geom_nosg|wo_semantic_nosg|w_sg|replace_ov_with_sg)
    run_one "${PRESET}"
    ;;
  *)
    echo "unknown preset: ${PRESET}" >&2
    echo "available: parallel9 | all | mainline_nosg | wo_ov_nosg | wo_vlm_nosg | wo_dyn_nosg | wo_bench_nosg | wo_geom_nosg | wo_semantic_nosg | w_sg | replace_ov_with_sg" >&2
    exit 1
    ;;
esac
