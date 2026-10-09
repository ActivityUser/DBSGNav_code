#!/usr/bin/env bash
# 创新1 分割质量评测 — 5 场景 × 全帧（与 rgb/seg_semantic 对齐，stride=1）
# 在 DBSGNavigation 目录执行：
#   bash ../DBSGConstruction/scripts/run_seg_quality_full_frames.sh
set -euo pipefail

cd "$(dirname "$0")/../../DBSGNavigation"

SCENES=(4ok3usBNeis 5cdEh9F2hJL 6s7QHgap2fW a8BtkwhxdRV CrMo8WxCyVb)
PY_HAB="${PY_HAB:-python}"
# Mask-RCNN 也可用系统 python3（需 torchvision）；与 GT 同环境可统一用 PY_HAB
PY_MRCNN="${PY_MRCNN:-$PY_HAB}"
HSIM_GPU="${HSIM_GPU:-0}"
CUDA_MRCNN="${CUDA_MRCNN:-0}"

export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2

DINO_SUBDIR="${DINO_SUBDIR:-seg_semantic}"
OUT_MD="${OUT_MD:-../docs/seg_quality.md}"
STRIDE="${STRIDE:-1}"

echo "[1/3] GT 语义 (stride=${STRIDE}) ..."
for S in "${SCENES[@]}"; do
  echo "  -> $S"
  HSIM_GPU="$HSIM_GPU" "$PY_HAB" ../DBSGConstruction/scripts/export_gt_semantic.py \
    --scene "$S" --stride "$STRIDE"
done

echo "[2/3] Mask-RCNN (stride=${STRIDE}) ..."
for S in "${SCENES[@]}"; do
  echo "  -> $S"
  CUDA_VISIBLE_DEVICES="$CUDA_MRCNN" "$PY_MRCNN" ../DBSGConstruction/scripts/run_maskrcnn_seg.py \
    --scene "$S" --stride "$STRIDE" --score 0.5
done

echo "[3/3] 评测 (DINO=${DINO_SUBDIR}) ..."
"$PY_HAB" ../DBSGConstruction/scripts/eval_seg_quality.py \
  --scenes "${SCENES[@]}" --dino_subdir "$DINO_SUBDIR" --out "$OUT_MD"

echo "done: $OUT_MD"
cat "$OUT_MD"
