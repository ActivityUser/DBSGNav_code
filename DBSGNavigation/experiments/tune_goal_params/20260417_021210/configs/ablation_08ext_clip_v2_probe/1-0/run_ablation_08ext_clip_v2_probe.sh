#!/usr/bin/env bash
# ablation_08ext_clip_v2_probe: 在 v2 assist 母版基础上，仅切换 OV 读取 instance_clip_vectors_v2.json
# （per-instance 掩码裁剪后再编码 CLIP），其余键完全一致。本实验用来回答：
#   修好 v1 同帧 cross-cat CLIP 退化（100% cos>0.999）后，OV 的语义判别力提升，
#   能否把 full 的 succ/SPL 真正抬到 wo_sg_geom 之上？
#
# 7 stems 顺序：full wo_dyn wo_bench wo_ov wo_sg_semgeom wo_sg_fusion_g6 wo_sg_geom
#
# 用法：
#   100ep 探测七卡并行（默认 GPU 0 1 2 3 4 5 6）：
#     bash run_ablation_08ext_clip_v2_probe.sh
#   1000ep：
#     bash run_ablation_08ext_clip_v2_probe.sh parallel7 1000
#   单 preset：
#     bash run_ablation_08ext_clip_v2_probe.sh full 0 100
#
# 汇总：
#   python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
#     --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe \
#     --ep 100 --model-name "08ext val clip_v2 probe"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_08ext_clip_v2_probe"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe"

PRESET="${1:-parallel7}"
GPU=""
EP=""

STEMS_ALL=(full wo_dyn wo_bench wo_ov wo_sg_semgeom wo_sg_fusion_g6 wo_sg_geom)

run_one() {
  local stem="$1"
  local cfg="${CFG_DIR}/${stem}.yaml"
  local log="${LOG_DIR}/${stem}_${EP}ep.log"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  echo ""
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') 08ext-val-clip_v2 ${stem} gpu=${GPU} ep=${EP} =========="
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
  echo "[parallel] 08ext-val-clip_v2 stem=${stem} CUDA_VISIBLE_DEVICES=${gpu} pid=$! log=${log}"
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

# ---- parallel7: 七卡并行 ----
if [[ "${PRESET}" == "parallel7" ]]; then
  EP="${2:-100}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-2}"
  G3="${6:-3}"
  G4="${7:-4}"
  G5="${8:-5}"
  G6="${9:-6}"
  echo "[INFO] 08ext-val-clip_v2 parallel7 ep=${EP}"
  echo "[INFO]   ${G0}=full ${G1}=wo_dyn ${G2}=wo_bench ${G3}=wo_ov ${G4}=wo_sg_semgeom ${G5}=wo_sg_fusion_g6 ${G6}=wo_sg_geom"
  pids=()
  run_one_parallel "full"             "${G0}"; pids+=($!)
  run_one_parallel "wo_dyn"           "${G1}"; pids+=($!)
  run_one_parallel "wo_bench"         "${G2}"; pids+=($!)
  run_one_parallel "wo_ov"            "${G3}"; pids+=($!)
  run_one_parallel "wo_sg_semgeom"    "${G4}"; pids+=($!)
  run_one_parallel "wo_sg_fusion_g6"  "${G5}"; pids+=($!)
  run_one_parallel "wo_sg_geom"       "${G6}"; pids+=($!)
  ec=0
  if ! wait_pids_fail_any "${pids[@]}"; then ec=1; fi
  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] parallel7 有任务非 0 退出，请检查对应 log" >&2
  fi
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

# ---- 单跑某一个 preset ----
GPU="${2:-0}"
EP="${3:-100}"
case "${PRESET}" in
  full|wo_dyn|wo_bench|wo_ov|wo_sg_semgeom|wo_sg_fusion_g6|wo_sg_geom)
    run_one "${PRESET}"
    ;;
  *)
    echo "未知 preset: ${PRESET}" >&2
    echo "可用: parallel7 | all | full | wo_dyn | wo_bench | wo_ov | wo_sg_semgeom | wo_sg_fusion_g6 | wo_sg_geom" >&2
    exit 1
    ;;
esac
