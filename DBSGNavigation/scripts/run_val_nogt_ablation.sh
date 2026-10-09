#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS=(
  baseline
  innov2
  innov3
  innov4
  innov5
  innov6
  full
)

echo "=========================================="
echo "开始运行 val_nogt 消融套件"
echo "工作目录: $(pwd)"
echo "时间: $(date '+%F %T %z')"
echo "=========================================="

for p in "${PRESETS[@]}"; do
  echo ""
  echo "------------------------------------------"
  echo "[Ablation] preset=${p}"
  echo "------------------------------------------"
  python run_ablation.py --preset "${p}"
done

echo ""
echo "=========================================="
echo "全部消融运行完成"
echo "时间: $(date '+%F %T %z')"
echo "=========================================="
