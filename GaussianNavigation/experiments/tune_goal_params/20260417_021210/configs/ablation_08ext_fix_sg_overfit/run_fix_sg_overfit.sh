#!/usr/bin/env bash
# SG 过拟合修复实验（4 fix 配置，val，08ext 底座）。
#
# 4 个 fix preset：
#   fix1_assist          — 开启 assist 模式（OV 主导，SG 辅助）
#   fix2_assist_ov_geom  — assist 模式 + OV 纯几何
#   fix3_sg_high_gate    — legacy 模式但大幅抬高 SG 门槛
#   fix4_assist_strict_ov_geom — 严格 assist + OV 纯几何（最保守）
#
# 用法：
#   四卡并行（默认 GPU 0,1,2,3）：
#     bash run_fix_sg_overfit.sh parallel4 [ep] [g0] [g1] [g2] [g3]
#
#   两卡两阶段（默认 GPU 0,1）：
#     bash run_fix_sg_overfit.sh parallel22 [ep] [g0] [g1]
#     阶段1：GPU0=fix1_assist  GPU1=fix2_assist_ov_geom
#     阶段2：GPU0=fix3_sg_high_gate  GPU1=fix4_assist_strict_ov_geom
#
#   单卡串行全部：
#     bash run_fix_sg_overfit.sh all [gpu] [ep]
#
#   单跑某一个：
#     bash run_fix_sg_overfit.sh fix1_assist [gpu] [ep]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_fix_sg_overfit"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_fix_sg_overfit"

PRESET="${1:-parallel4}"
GPU=""
EP=""

STEMS_ALL=(fix1_assist fix2_assist_ov_geom fix3_sg_high_gate fix4_assist_strict_ov_geom)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  echo ""
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') fix-sg ${stem} gpu=${GPU} ep=${EP} =========="
  echo "cfg=${cfg}"
  echo "log=${log}"
  echo "--------------------------------------------------------------"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
    --config "${cfg}" \
    --max_episodes "${EP}" \
    --log_file "${log}"
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
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" -- \
      --config "${cfg}" \
      --max_episodes "${EP}"
  ) &
  echo "[parallel] fix-sg stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
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

# ---- parallel4: 四卡各一路 ----
if [[ "${PRESET}" == "parallel4" ]]; then
  EP="${2:-100}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  G3="${6:-3}"
  echo "[INFO] fix-sg parallel4 ep=${EP} ${G0}=fix1 ${G1}=fix2 ${G2}=fix3 ${G3}=fix4"
  pids=()
  run_one_parallel "fix1_assist" "${G0}"; pids+=($!)
  run_one_parallel "fix2_assist_ov_geom" "${G1}"; pids+=($!)
  run_one_parallel "fix3_sg_high_gate" "${G2}"; pids+=($!)
  run_one_parallel "fix4_assist_strict_ov_geom" "${G3}"; pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then ec=1; fi
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] parallel4 有任务非 0 退出，请检查 log" >&2
  fi
  exit "${ec}"
fi

# ---- parallel22: 两卡两阶段 ----
if [[ "${PRESET}" == "parallel22" ]]; then
  EP="${2:-100}"
  G0="${3:-0}"
  G1="${4:-1}"
  echo "[INFO] fix-sg parallel22 ep=${EP} 阶段1: ${G0}=fix1 ${G1}=fix2 | 阶段2: ${G0}=fix3 ${G1}=fix4"
  pids1=()
  run_one_parallel "fix1_assist" "${G0}"; pids1+=($!)
  run_one_parallel "fix2_assist_ov_geom" "${G1}"; pids1+=($!)
  if ! wait_pids_fail_any "${pids1[@]}"; then
    echo "[ERROR] 阶段1 失败" >&2; exit 1
  fi
  echo "[INFO] 阶段1 结束，启动阶段2"
  pids2=()
  run_one_parallel "fix3_sg_high_gate" "${G0}"; pids2+=($!)
  run_one_parallel "fix4_assist_strict_ov_geom" "${G1}"; pids2+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids2[@]}"; then ec=1; fi
  exit "${ec}"
fi

# ---- all: 单卡串行 ----
if [[ "${PRESET}" == "all" ]]; then
  GPU="${2:-0}"
  EP="${3:-100}"
  for s in "${STEMS_ALL[@]}"; do
    run_one "${s}"
  done
  exit 0
fi

# ---- 单跑某一个 ----
GPU="${2:-0}"
EP="${3:-100}"
case "${PRESET}" in
  fix1_assist|fix2_assist_ov_geom|fix3_sg_high_gate|fix4_assist_strict_ov_geom)
    run_one "${PRESET}"
    ;;
  *)
    echo "未知 preset: ${PRESET}" >&2
    echo "可用: parallel4 | parallel22 | all | fix1_assist | fix2_assist_ov_geom | fix3_sg_high_gate | fix4_assist_strict_ov_geom" >&2
    exit 1
    ;;
esac
