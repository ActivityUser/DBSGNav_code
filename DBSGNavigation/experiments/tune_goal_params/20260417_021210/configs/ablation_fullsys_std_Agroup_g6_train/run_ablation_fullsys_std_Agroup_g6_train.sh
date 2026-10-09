#!/usr/bin/env bash
# 「标准协议」train 消融：固定 use_open_vocab_benchmark=false，比较 full/wo_*（不含无 SG/OV 下界）。
# 用法：
#   串行： bash run_ablation_fullsys_std_Agroup_g6_train.sh all [gpu] [ep]
#   并行（7 preset 各一卡）： bash run_ablation_fullsys_std_Agroup_g6_train.sh parallel [ep] <7 个 GPU>
#       例： bash run_ablation_fullsys_std_Agroup_g6_train.sh parallel 1000 0 1 2 3 4 5 6
#   单跑： bash run_ablation_fullsys_std_Agroup_g6_train.sh wo_dyn 0 1000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_fullsys_std_Agroup_g6_train"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_train"

PRESET="${1:-all}"
SESSION_LOG=""
GPU=""
EP=""

# 串行顺序：主表 Baseline (wo_sg_geom) 先跑，其余随后，full 最后（按需可改顺序）。
STEMS_ALL=(wo_sg_geom wo_sg_semgeom wo_sg_fusion_g6 wo_ov wo_bench wo_dyn full)

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
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  n_stems=${#STEMS_ALL[@]}
  if [[ ${#local_gpus[@]} -lt $n_stems ]]; then
    echo "用法: $0 parallel <ep> <${n_stems} 个 GPU 编号…>  例: $0 parallel 1000 0 1 2 3 4 5 6" >&2
    exit 1
  fi
  echo "[INFO] 多卡并行：各 stem 写 *_${EP}ep.log；需 ${n_stems} 张卡。"
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
    echo "[INFO] 串行 all：会话总日志 → ${SESSION_LOG}"
  } | tee -a "${SESSION_LOG}"
  for stem in "${STEMS_ALL[@]}"; do
    run_one "${stem}"
  done
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] ablation_fullsys_std_Agroup_g6_train preset=${PRESET} ep=${EP}"
echo "[NOTE] 汇总: python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py --log-dir ${LOG_DIR} --ep ${EP}"
