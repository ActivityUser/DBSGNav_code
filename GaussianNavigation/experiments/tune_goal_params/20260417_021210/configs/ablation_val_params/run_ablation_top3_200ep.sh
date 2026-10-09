#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_val_params"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun"
EP=200
GPU=${1:-0}

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"

for cfg in   A0_baseline_top3   A1_force_dist_7   A2_disagree_dist_4p5   A3_min_geom_30   A4_sg_max_agent_12   A5_prefer_ov_lowconf_false; do
  echo "[RUN] $cfg"
  PYTHONUNBUFFERED=1 python -u run.py     --config "experiments/tune_goal_params/20260417_021210/configs/ablation_val_params/${cfg}.yaml"     --max_episodes "$EP"     --log_file "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/${cfg}_${EP}ep.log"
done
