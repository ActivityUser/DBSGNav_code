#!/usr/bin/env bash
# inst_v3_vlm: SG-stage Qwen3-VL disambiguation + merit SG/OV arbitration.
# Logs: logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_vlm)/
#
# Usage:
#   bash run_ablation_inst_v3_vlm.sh parallel8           # 8 GPUs, 1000 ep
#   bash run_ablation_inst_v3_vlm.sh parallel8 100       # 100 ep smoke
#   bash run_ablation_inst_v3_vlm.sh full 0 1000         # single preset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="${SCRIPT_DIR}"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_vlm)"

PRESET="${1:-parallel8}"
GPU=""
EP=""

STEMS_ALL=(full wo_dyn wo_bench wo_ov wo_sg_semgeom wo_sg_fusion_g6 wo_sg_geom wo_vlm)

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
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" -- \
      --config "${cfg}" \
      --max_episodes "${EP}"
  ) &
  echo "[parallel] inst_v3_vlm stem=${stem} cuda=${gpu} pid=$! log=${log}"
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

if [[ "${PRESET}" == "parallel8" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  G3="${6:-3}"
  G4="${7:-4}"
  G5="${8:-5}"
  G6="${9:-6}"
  G7="${10:-7}"
  echo "[INFO] inst_v3_vlm parallel8 ep=${EP}"
  pids=()
  run_one_parallel "full"             "${G0}"; pids+=($!)
  run_one_parallel "wo_dyn"           "${G1}"; pids+=($!)
  run_one_parallel "wo_bench"         "${G2}"; pids+=($!)
  run_one_parallel "wo_ov"            "${G3}"; pids+=($!)
  run_one_parallel "wo_sg_semgeom"    "${G4}"; pids+=($!)
  run_one_parallel "wo_sg_fusion_g6"  "${G5}"; pids+=($!)
  run_one_parallel "wo_sg_geom"       "${G6}"; pids+=($!)
  run_one_parallel "wo_vlm"           "${G7}"; pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then ec=1; fi
  exit "${ec}"
fi

if [[ "${PRESET}" == "all" ]]; then
  EP="${2:-1000}"
  GPU="${3:-0}"
  for s in "${STEMS_ALL[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
      --config "${CFG_DIR}/${s}.yaml" \
      --max_episodes "${EP}" \
      --log_file "${LOG_DIR}/${s}_${EP}ep.log"
  done
  exit 0
fi

GPU="${2:-0}"
EP="${3:-1000}"
case "${PRESET}" in
  full|full_sg_vlm_debug|full_scene_crop|full_scene_crop_high_conf|probe_vlm_always|probe_vlm_smart|probe_vlm_scene_crop|probe_merit_sem|wo_dyn|wo_bench|wo_ov|wo_sg_semgeom|wo_sg_fusion_g6|wo_sg_geom|wo_vlm)
    mkdir -p "${LOG_DIR}"
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
      --config "${CFG_DIR}/${PRESET}.yaml" \
      --max_episodes "${EP}" \
      --log_file "${LOG_DIR}/${PRESET}_${EP}ep.log"
    ;;
  *)
    echo "unknown preset: ${PRESET}" >&2
    echo "available: parallel8 | all | full | full_sg_vlm_debug | full_scene_crop | full_scene_crop_high_conf | probe_vlm_always | probe_vlm_smart | probe_vlm_scene_crop | probe_merit_sem | wo_dyn | wo_bench | wo_ov | wo_sg_semgeom | wo_sg_fusion_g6 | wo_sg_geom | wo_vlm" >&2
    exit 1
    ;;
esac
