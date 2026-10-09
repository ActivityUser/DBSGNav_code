#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/run_val.py" && "${ROOT}" != "/" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
CFG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/configs/ablation_innovations_08ext"
LOG_DIR="$ROOT/experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innovations_08ext"

PRESET="${1:-all}"        # all | baseline | innov2 | innov3 | innov4 | innov5 | innov6 | full
GPU="${2:-0}"
EP="${3:-100}"

run_one() {
  local name="$1"
  echo "[RUN] $name (gpu=${GPU}, ep=${EP})"
  PYTHONUNBUFFERED=1 python -u run.py \
    --config "experiments/tune_goal_params/20260417_021210/configs/ablation_innovations_08ext/${name}.yaml" \
    --max_episodes "${EP}" \
    --log_file "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun/ablation_innovations_08ext/${name}_${EP}ep.log"
}

cd "$ROOT"
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$PRESET" == "all" ]]; then
  for name in baseline innov2 innov3 innov4 innov5 innov6 full; do
    run_one "$name"
  done
else
  run_one "$PRESET"
fi

echo "[DONE] 08_ext 创新点消融完成: $PRESET"
