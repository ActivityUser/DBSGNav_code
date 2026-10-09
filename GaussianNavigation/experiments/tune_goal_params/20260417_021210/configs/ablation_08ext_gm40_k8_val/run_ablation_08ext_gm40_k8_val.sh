#!/usr/bin/env bash
# Val 消融（7 preset，08_ext gm40/k8 底座）：入口 run_val.py。
#
# Preset 与 fullsys std 语义对齐（标量母版：full.yaml）：
#   full          — Full（SG+OV+dyn.，OV G6；本目录各 yaml 除消融键外与 full 逐键对齐）
#   wo_dyn        — w.o. dynamic semantic map（相对 full 仅 mapper）
#   wo_bench      — benchmark 开关对照（preset 名称沿用 wo_bench）
#   wo_ov         — w.o. OV retrieval（SG only）
#   wo_sg_fusion_g6 — w.o. SG（OV 与 full 同 G6；相对 full 仅 use_semantic_gaussian_goal: false）
#   wo_sg_geom    — w.o. SG（OV geom.-only；相对 full 关 SG + 仅改融合）
#   wo_sg_semgeom — w.o. SG（OV sem.-only；相对 full 关 SG + 仅改融合）
#
# 用法：
#   两阶段 parallel322（默认 GPU 0、1、7）— **先** 并行跑 3 路，**全部结束后自动** 再跑余下 4 路（2+2 在 GPU0/1，GPU7 第二阶段空闲）：
#     bash run_ablation_08ext_gm40_k8_val.sh parallel322 [ep] [g0] [g1] [g2]
#     阶段1（并行）：GPU0=full，GPU1=wo_dyn，GPU7=wo_sg_geom
#     阶段2（自动）：GPU0 串行 wo_sg_semgeom → wo_sg_fusion_g6；GPU1 串行 wo_ov → wo_bench
#
#   仅阶段 1（三卡各 1 路，不自动跑阶段 2）：
#     bash run_ablation_08ext_gm40_k8_val.sh parallel3 [ep] [g0] [g1] [g2]
#
#   七卡并行（与 STEMS_ALL 顺序一一对应，默认 GPU 0–6；可改 7 个 GPU 号）：
#     bash run_ablation_08ext_gm40_k8_val.sh parallel7 [ep] [g0] [g1] [g2] [g3] [g4] [g5] [g6]
#     g0=full, g1=wo_dyn, g2=wo_bench, g3=wo_ov, g4=wo_sg_semgeom, g5=wo_sg_fusion_g6, g6=wo_sg_geom
#     例：指定 0,1,2,3,4,5,7 → bash ... parallel7 1000 0 1 2 3 4 5 7
#
#   单卡串行 7 个：
#     bash run_ablation_08ext_gm40_k8_val.sh all [gpu] [ep]
#
#   单跑某一 preset：
#     bash run_ablation_08ext_gm40_k8_val.sh full 0 1000
#
# 汇总（与 train 汇总脚本共用，改 --log-dir）：
#   python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
#     --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_gm40_k8_val \
#     --ep 1000 --model-name "08ext val"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_gm40_k8_val"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_gm40_k8_val"

PRESET="${1:-parallel322}"
GPU=""
EP=""
SESSION_LOG=""

# 阶段 1：三卡各一路（优先跑满 full / wo_dyn / wo_sg_geom）
STEM_PHASE1=(full wo_dyn wo_sg_geom)

# 单卡串行「全 7」时的顺序（与 fullsys 表头一致）
STEMS_ALL=(full wo_dyn wo_bench wo_ov wo_sg_semgeom wo_sg_fusion_g6 wo_sg_geom)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  if [[ -n "${SESSION_LOG}" ]]; then
    (
      echo ""
      echo "========== $(date '+%Y-%m-%d %H:%M:%S') 08ext-val ${stem} gpu=${GPU} ep=${EP} =========="
      echo "cfg=${cfg}"
      echo "log=${log}"
      echo "--------------------------------------------------------------"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u run_val.py \
        --config "${cfg}" \
        --max_episodes "${EP}"
    ) 2>&1 | tee "${log}" | tee -a "${SESSION_LOG}"
  else
    (
      echo ""
      echo "========== $(date '+%Y-%m-%d %H:%M:%S') 08ext-val ${stem} gpu=${GPU} ep=${EP} =========="
      echo "cfg=${cfg}"
      echo "log=${log}"
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
  echo "[parallel] 08ext-val stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
}

# 单卡串行多 stem（阶段 2）；与 fullsys val 一致用 --log_file
launch_worker_serial() {
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
      echo "[08ext-val serial] $(date '+%Y-%m-%d %H:%M:%S') gpu=${gpu} stem=${stem} ep=${EP}"
      echo "cfg=${cfg} log=${log}"
      PYTHONUNBUFFERED=1 python -u run_val.py \
        --config "${cfg}" \
        --max_episodes "${EP}" \
        --log_file "${log}"
    done
  ) &
  echo "[08ext-val serial] worker gpu=${gpu} stems=${stems[*]} pid=$!"
}

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

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

if [[ "${PRESET}" == "parallel322" || "${PRESET}" == "parallel332" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-7}"
  echo "[INFO] 08ext val parallel322 两阶段 ep=${EP} G0=${G0} G1=${G1} G2=${G2}"
  echo "[INFO] 阶段1 并行: ${G0}=full ${G1}=wo_dyn ${G2}=wo_sg_geom"
  pids1=()
  run_one_parallel "${STEM_PHASE1[0]}" "${G0}"
  pids1+=($!)
  run_one_parallel "${STEM_PHASE1[1]}" "${G1}"
  pids1+=($!)
  run_one_parallel "${STEM_PHASE1[2]}" "${G2}"
  pids1+=($!)
  if ! wait_pids_fail_any "${pids1[@]}"; then
    echo "[ERROR] 阶段1 有任务非 0 退出，已中止（不启动阶段2）。检查上述 log。" >&2
    exit 1
  fi
  echo "[INFO] 阶段1 已全部结束。阶段2 自动开始: ${G0}=(wo_sg_semgeom wo_sg_fusion_g6) ${G1}=(wo_ov wo_bench)（${G2} 空闲）"
  pids2=()
  launch_worker_serial "${G0}" wo_sg_semgeom wo_sg_fusion_g6
  pids2+=($!)
  launch_worker_serial "${G1}" wo_ov wo_bench
  pids2+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids2[@]}"; then
    ec=1
  fi
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] 阶段2 有 worker 非 0 退出，请检查对应 log" >&2
  fi
  exit "${ec}"
fi

if [[ "${PRESET}" == "parallel3" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-7}"
  echo "[INFO] 08ext val parallel3：仅阶段1 三卡并行 ep=${EP} ${G0}=full ${G1}=wo_dyn ${G2}=wo_sg_geom"
  pids=()
  run_one_parallel "${STEM_PHASE1[0]}" "${G0}"
  pids+=($!)
  run_one_parallel "${STEM_PHASE1[1]}" "${G1}"
  pids+=($!)
  run_one_parallel "${STEM_PHASE1[2]}" "${G2}"
  pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then
    ec=1
  fi
  exit "${ec}"
fi

if [[ "${PRESET}" == "parallel7" ]]; then
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  G3="${6:-3}"
  G4="${7:-4}"
  G5="${8:-5}"
  G6="${9:-6}"
  echo "[INFO] 08ext val parallel7 七卡并行 ep=${EP}"
  echo "[INFO]   ${G0}=full ${G1}=wo_dyn ${G2}=wo_bench ${G3}=wo_ov ${G4}=wo_sg_semgeom ${G5}=wo_sg_fusion_g6 ${G6}=wo_sg_geom"
  pids=()
  run_one_parallel "full" "${G0}"
  pids+=($!)
  run_one_parallel "wo_dyn" "${G1}"
  pids+=($!)
  run_one_parallel "wo_bench" "${G2}"
  pids+=($!)
  run_one_parallel "wo_ov" "${G3}"
  pids+=($!)
  run_one_parallel "wo_sg_semgeom" "${G4}"
  pids+=($!)
  run_one_parallel "wo_sg_fusion_g6" "${G5}"
  pids+=($!)
  run_one_parallel "wo_sg_geom" "${G6}"
  pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then
    ec=1
  fi
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] parallel7 有任务非 0 退出，请检查对应 log" >&2
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

# 单 preset
GPU="${2:-0}"
EP="${3:-1000}"
case "${PRESET}" in
  full|wo_dyn|wo_bench|wo_ov|wo_sg_semgeom|wo_sg_fusion_g6|wo_sg_geom)
    run_one "${PRESET}"
    ;;
  full_strong)
    echo "[WARN] preset full_strong 已弃用，改用 full（full.yaml）" >&2
    run_one "full"
    ;;
  *)
    echo "未知 preset: ${PRESET}" >&2
    echo "可用: parallel322 | parallel3 | parallel7 | all | full | wo_dyn | wo_bench | wo_ov | wo_sg_semgeom | wo_sg_fusion_g6 | wo_sg_geom（兼容：full_strong→full）" >&2
    exit 1
    ;;
esac