#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
BASE_CFG="experiments/tune_goal_params/20260417_021210/configs/08_ext_gm40_k8.yaml"
LOG_DIR="experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun"
EP=100
GPU="${1:-0}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[A0] baseline_full36"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A0_baseline_${EP}ep.log"

echo "[A1] force_dist_7"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A1_force_dist_7_${EP}ep.log" \
  end2end_imagenav.semantic_gaussian_ov_disagree_force_dist=7.0

echo "[A2] disagree_dist_4p5"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A2_disagree_dist_4p5_${EP}ep.log" \
  end2end_imagenav.semantic_gaussian_ov_disagree_dist=4.5

echo "[A3] min_geom_30"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A3_min_geom_30_${EP}ep.log" \
  end2end_imagenav.open_vocab_min_geometric_matches=30

echo "[A4] sg_max_agent_12"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A4_sg_max_agent_12_${EP}ep.log" \
  end2end_imagenav.semantic_gaussian_max_agent_goal_dist=12.0

echo "[A5] prefer_ov_lowconf_false"
PYTHONUNBUFFERED=1 python -u run.py \
  --config "$BASE_CFG" \
  --max_episodes "$EP" \
  --log_file "$LOG_DIR/ablation_full36_A5_prefer_ov_lowconf_false_${EP}ep.log" \
  end2end_imagenav.prefer_open_vocab_when_semantic_gaussian_low_conf=false

echo "[DONE] full36 ablation finished."
