#!/usr/bin/env bash
# 「标准协议」val 消融：与 train 套件字段对齐；评测入口统一用 run_val.py（内部仍调用 Runner.train，与 run.py 相同）。
# 用法：
#   串行： bash run_ablation_fullsys_std_Agroup_g6_val.sh all [gpu] [ep]
#   并行（7 preset 各一卡）： bash run_ablation_fullsys_std_Agroup_g6_val.sh parallel [ep] <7 个 GPU>
#   三卡 3+2+2（默认 GPU 0 1 7）： bash run_ablation_fullsys_std_Agroup_g6_val.sh parallel322 [ep] [g0] [g1] [g2]
#       例： bash run_ablation_fullsys_std_Agroup_g6_val.sh parallel322 1000 0 1 7
#   单跑： bash run_ablation_fullsys_std_Agroup_g6_val.sh wo_dyn 0 1000
#
# 生成 val yaml：
#   python experiments/tune_goal_params/20260417_021210/tools/make_ablation_fullsys_std_Agroup_g6_val_configs.py
#
# 汇总：
#   python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
#     --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_val \
#     --ep 1000 --model-name "GaussNav++ (val)"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_fullsys_std_Agroup_g6_val"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_fullsys_std_Agroup_g6_val"

PRESET="${1:-all}"
SESSION_LOG=""
GPU=""
EP=""

# 与 train 脚本相同顺序（parallel 7 卡时按数组下标绑 GPU）
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
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
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
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
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
    export GAUSSNAV_RUN_ENTRY=run_val.py
    export RUN_LOG_SCRIPT_QUIET=1
    cd "${ROOT}"
    exec "${ROOT}/scripts/run_log_same_as_terminal.sh" "${log}" -- \
      --config "${cfg}" \
      --max_episodes "${EP}"
  ) &
  echo "[parallel] started stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

# 单卡上串行跑若干 stem（用于 3+2+2）；每任务带 --log_file，与 run.py tee 行为一致
launch_worker_322() {
  local gpu="$1"
  shift
  local stems=( "$@" )
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    cd "${ROOT}"
    for stem in "${stems[@]}"; do
      cfg="${CFG_DIR}/${stem}.yaml"
      log="${LOG_DIR}/${stem}_${EP}ep.log"
      mkdir -p "${LOG_DIR}"
      echo ""
      echo "[parallel322] $(date '+%Y-%m-%d %H:%M:%S') gpu=${gpu} stem=${stem} ep=${EP}"
      echo "cfg=${cfg} log=${log}"
      PYTHONUNBUFFERED=1 python -u run_val.py \
        --config "${cfg}" \
        --max_episodes "${EP}" \
        --log_file "${log}"
    done
  ) &
  echo "[parallel322] worker gpu=${gpu} stems=${stems[*]} pid=$!"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ "${PRESET}" == "parallel322" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-7}"
  echo "[INFO] parallel322：GPU ${G0} 上 3 个、${G1} 上 2 个、${G2} 上 2 个；入口 run_val.py"
  pids=()
  launch_worker_322 "${G0}" wo_sg_geom wo_sg_semgeom wo_sg_fusion_g6
  pids+=($!)
  launch_worker_322 "${G1}" wo_ov wo_bench
  pids+=($!)
  launch_worker_322 "${G2}" wo_dyn full
  pids+=($!)
  ec=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      ec=1
    fi
  done
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] parallel322 有 worker 非 0 退出，请检查对应 log" >&2
    exit "${ec}"
  fi
elif [[ "${PRESET}" == "parallel" ]]; then
  EP="${2:-1000}"
  shift 2 || true
  local_gpus=( "$@" )
  n_stems=${#STEMS_ALL[@]}
  if [[ ${#local_gpus[@]} -lt $n_stems ]]; then
    echo "用法: $0 parallel <ep> <${n_stems} 个 GPU 编号…>  例: $0 parallel 1000 0 1 2 3 4 5 6" >&2
    exit 1
  fi
  echo "[INFO] 多卡并行（val，run_val.py）：各 stem 写 *_${EP}ep.log；需 ${n_stems} 张卡。"
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
    echo "[INFO] 串行 all（val，run_val.py）：会话总日志 → ${SESSION_LOG}"
  } | tee -a "${SESSION_LOG}"
  for stem in "${STEMS_ALL[@]}"; do
    run_one "${stem}"
  done
else
  GPU="${2:-0}"
  EP="${3:-1000}"
  run_one "${PRESET}"
fi

echo "[DONE] ablation_fullsys_std_Agroup_g6_val preset=${PRESET} ep=${EP}"
echo "[NOTE] 汇总: python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py --log-dir ${LOG_DIR} --ep ${EP} --model-name 'GaussNav++ (val)'"