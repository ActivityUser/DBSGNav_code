#!/usr/bin/env bash
# ablation_08ext_clip_v2_sg_fix: 在 clip_v2_probe/full（succ=0.799）基础上调 SG assist 行为，
# 让 SG 真正"辅助" OV 而非干扰。数据驱动诊断（来自 clip_v2_probe/1000/full_1000ep.log）：
#   - SG 在 disagree 通道 5 次介入全部失败（0/5，dpg 全部 10-18m）。
#   - OV 失败的 196 例反事实分析：SG 救场=0，SG 比 OV 更差 123 例（87.2%）。
# 三个变体：
#   fix_no_disagree         关闭 assist_disagree 通道（最稳，最确定收益）
#   fix_no_disagree_widen   再去掉 only_when_ov_weak 限制（看放宽 ov_weak 能否净正）
#   fix_no_disagree_sim95   再把 assist_min_sim 从 0.96 降到 0.95（与 widen 互斥）
#
# 用法：
#   3 卡并跑（默认 1000ep，GPU 0 1 2）：
#     bash run_sg_fix.sh
#   自定义 ep / GPU：
#     bash run_sg_fix.sh parallel3 [ep] [g0] [g1] [g2]
#   单跑一个：
#     bash run_sg_fix.sh <preset> [gpu] [ep]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_sg_fix"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_sg_fix"

PRESET="${1:-parallel3}"
GPU=""
EP=""

STEMS_ALL=(fix_no_disagree fix_no_disagree_widen fix_no_disagree_sim95)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  echo ""
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') sg_fix ${stem} gpu=${GPU} ep=${EP} =========="
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
  echo "[parallel] sg_fix stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

wait_pids_fail_any() {
  local ec=0
  for pid in "$@"; do
    if ! wait "${pid}"; then ec=1; fi
  done
  return "${ec}"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel3" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  echo "[INFO] sg_fix parallel3 ep=${EP}"
  echo "[INFO]   ${G0}=fix_no_disagree  ${G1}=fix_no_disagree_widen  ${G2}=fix_no_disagree_sim95"
  pids=()
  run_one_parallel "fix_no_disagree"        "${G0}"; pids+=($!)
  run_one_parallel "fix_no_disagree_widen"  "${G1}"; pids+=($!)
  run_one_parallel "fix_no_disagree_sim95"  "${G2}"; pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then ec=1; fi
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] parallel3 有任务非 0 退出，请检查对应 log" >&2
  fi
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
  fix_no_disagree|fix_no_disagree_widen|fix_no_disagree_sim95)
    run_one "${PRESET}"
    ;;
  *)
    echo "未知 preset: ${PRESET}" >&2
    echo "可用: parallel3 | all | fix_no_disagree | fix_no_disagree_widen | fix_no_disagree_sim95" >&2
    exit 1
    ;;
esac
