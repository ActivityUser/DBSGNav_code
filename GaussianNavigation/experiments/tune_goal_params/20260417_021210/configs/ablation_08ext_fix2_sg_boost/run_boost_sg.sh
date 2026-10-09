#!/usr/bin/env bash
# SG-boost 探针：让 SG 在更多 episode 中接管，撬动 full > wo_sg_geom 的边际。
#
# 3 个 boost preset（与 v2 母版 ablation_08ext_gm40_k8_val/full.yaml 对比仅改 SG 触发参数）：
#   boost1_compete          — assist_only_when_ov_weak=false; min_sim 0.95; max_gap 3.0
#   boost2_compete_strict   — only_when_ov_weak=false; min_sim 0.97; max_gap 2.0; closer_margin 0.3
#   boost3_disagree_more    — disagree_dist 10→5; min_sim 0.93（OV-weak 主门保留）
#
# 用法：
#   三卡并行（默认 ep=100 快速看趋势，GPU 0 1 3）：
#     bash run_boost_sg.sh parallel3 [ep] [g0] [g1] [g2]
#
#   单跑某个 preset：
#     bash run_boost_sg.sh <preset> [gpu] [ep]
#
#   汇总：
#     python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
#       --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_fix2_sg_boost \
#       --ep <ep> --model-name "08ext val sg-boost"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_fix2_sg_boost"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_fix2_sg_boost"

PRESET="${1:-parallel3}"
GPU=""
EP=""

STEMS_ALL=(boost1_compete boost2_compete_strict boost3_disagree_more boost4_realistic boost5_aggressive boost6_disagree_lowsim)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  echo ""
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') sg-boost ${stem} gpu=${GPU} ep=${EP} =========="
  echo "cfg=${cfg}"; echo "log=${log}"
  echo "--------------------------------------------------------------"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
    --config "${cfg}" --max_episodes "${EP}" --log_file "${log}"
}

run_one_parallel() {
  local stem="$1"; local gpu="$2"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export GAUSSNAV_RUN_ENTRY=run_val.py
    export RUN_LOG_SCRIPT_QUIET=1
    cd "${ROOT}"
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" -- \
      --config "${cfg}" --max_episodes "${EP}"
  ) &
  echo "[parallel] sg-boost stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

wait_pids_fail_any() {
  local ec=0
  for pid in "$@"; do wait "${pid}" || ec=1; done
  return "${ec}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel3" ]]; then
  EP="${2:-100}"
  G0="${3:-0}"; G1="${4:-1}"; G2="${5:-3}"
  echo "[INFO] sg-boost parallel3 ep=${EP} ${G0}=boost1 ${G1}=boost2 ${G2}=boost3"
  pids=()
  run_one_parallel "boost1_compete"        "${G0}"; pids+=($!)
  run_one_parallel "boost2_compete_strict" "${G1}"; pids+=($!)
  run_one_parallel "boost3_disagree_more"  "${G2}"; pids+=($!)
  ec=0; wait_pids_fail_any "${pids[@]}" || ec=1
  exit "${ec}"
fi

# 第二轮：贴合 val SG sim 分布（峰值 0.96，中位 0.906）的新 3 个 preset
if [[ "${PRESET}" == "parallel3_r2" ]]; then
  EP="${2:-100}"
  G0="${3:-0}"; G1="${4:-1}"; G2="${5:-3}"
  echo "[INFO] sg-boost parallel3_r2 ep=${EP} ${G0}=boost4 ${G1}=boost5 ${G2}=boost6"
  pids=()
  run_one_parallel "boost4_realistic"       "${G0}"; pids+=($!)
  run_one_parallel "boost5_aggressive"      "${G1}"; pids+=($!)
  run_one_parallel "boost6_disagree_lowsim" "${G2}"; pids+=($!)
  ec=0; wait_pids_fail_any "${pids[@]}" || ec=1
  exit "${ec}"
fi

if [[ "${PRESET}" == "all" ]]; then
  GPU="${2:-0}"; EP="${3:-100}"
  for s in "${STEMS_ALL[@]}"; do run_one "${s}"; done
  exit 0
fi

GPU="${2:-0}"; EP="${3:-100}"
case "${PRESET}" in
  boost1_compete|boost2_compete_strict|boost3_disagree_more|boost4_realistic|boost5_aggressive|boost6_disagree_lowsim)
    run_one "${PRESET}" ;;
  *)
    echo "未知 preset: ${PRESET}" >&2
    echo "可用: parallel3 | parallel3_r2 | all | boost1_compete | boost2_compete_strict | boost3_disagree_more | boost4_realistic | boost5_aggressive | boost6_disagree_lowsim" >&2
    exit 1 ;;
esac
