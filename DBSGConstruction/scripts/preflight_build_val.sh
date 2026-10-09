#!/usr/bin/env bash
set -euo pipefail

# 运行 run_build_val_semantic_gaussians.py 之前的快速体检
# 用法:
#   bash scripts/preflight_build_val.sh
#   bash scripts/preflight_build_val.sh configs/data/habitat_val habitat_val_final

DATA_DIR="${1:-configs/data/habitat_val}"
GROUP_NAME="${2:-habitat_val_final}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${GC_ROOT}/scripts/run_build_val_semantic_gaussians.py"
SPLATAM_SCRIPT="${GC_ROOT}/scripts/habitat_splatam.py"
SPLATAM_CFG="${GC_ROOT}/configs/habitat/habitat_splatam.py"

echo "== Preflight =="
echo "pwd: $(pwd)"
echo "GC_ROOT: ${GC_ROOT}"
echo "python: $(command -v python || true)"
python -V || true

echo
echo "[1] 关键文件检查"
[[ -f "${RUN_SCRIPT}" ]] || { echo "ERROR: 缺少 ${RUN_SCRIPT}"; exit 1; }
[[ -f "${SPLATAM_SCRIPT}" ]] || { echo "ERROR: 缺少 ${SPLATAM_SCRIPT}"; exit 1; }
[[ -f "${SPLATAM_CFG}" ]] || { echo "ERROR: 缺少 ${SPLATAM_CFG}"; exit 1; }
echo "OK: scripts/config 都存在"

echo
echo "[2] 数据目录检查"
if [[ "${DATA_DIR}" = /* ]]; then
  DATA_ABS="${DATA_DIR}"
else
  DATA_ABS="${GC_ROOT}/${DATA_DIR}"
fi
[[ -d "${DATA_ABS}" ]] || { echo "ERROR: 数据目录不存在 ${DATA_ABS}"; exit 1; }
SCENE_COUNT="$(ls -1 "${DATA_ABS}" | wc -l || true)"
echo "OK: DATA_DIR=${DATA_ABS} (entries=${SCENE_COUNT})"

echo
echo "[3] 空间与显卡快照"
df -h "${GC_ROOT}" | sed -n '1,2p'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo "WARN: nvidia-smi 不可用"
fi

echo
echo "[4] 推荐运行命令"
cat <<EOF
cd "${GC_ROOT}"
python scripts/run_build_val_semantic_gaussians.py \\
  --data_dir "${DATA_DIR}" \\
  --group_name "${GROUP_NAME}" \\
  --max_frames -1 \\
  --stride 1
EOF
