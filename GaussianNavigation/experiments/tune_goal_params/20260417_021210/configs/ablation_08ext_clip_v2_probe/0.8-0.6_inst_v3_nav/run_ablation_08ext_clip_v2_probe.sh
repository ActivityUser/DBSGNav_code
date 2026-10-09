#!/usr/bin/env bash
# ablation_08ext_clip_v2_probe / 0.8-0.6_inst_v3_nav：在 0.8-0.6_inst_v3 基础上开启 SG navigable goal（P0）。
# 日志目录：logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_nav)/
#
# 用法（本脚本在 0.8-0.6/ 子目录，只读同目录 yaml；日志默认写入 1000(0.8-0.6_inst_v3)/）：
#   七卡并行（默认 ep=1000，GPU 0 1 3 4 5 6 7，跳过 GPU2）：
#     bash run_ablation_08ext_clip_v2_probe.sh
#   100ep 探测：
#     bash run_ablation_08ext_clip_v2_probe.sh parallel7 100
#   自定义 ep / GPU：
#     bash run_ablation_08ext_clip_v2_probe.sh parallel7 1000 0 1 3 4 5 6 7
#   单 preset：
#     bash run_ablation_08ext_clip_v2_probe.sh full 0 1000
#
# 汇总：
#   python experiments/tune_goal_params/20260417_021210/tools/summarize_ablation_fullsys_std_Agroup_g6_train.py \
#     --log-dir experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3) \
#     --ep 1000 --model-name "08ext val clip_v2 0.8-0.6"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="${SCRIPT_DIR}"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_nav)"

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
  EP="${2:-1000}"
  G0="${3:-0}"
  G1="${4:-1}"
  G2="${5:-3}"
  G3="${6:-4}"
  G4="${7:-5}"
  G5="${8:-6}"
  G6="${9:-7}"
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
  EP="${3:-1000}"
  for s in "${STEMS_ALL[@]}"; do
    run_one "${s}"
  done
  exit 0
fi

# ---- 单跑某一个 preset ----
GPU="${2:-0}"
EP="${3:-1000}"
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
