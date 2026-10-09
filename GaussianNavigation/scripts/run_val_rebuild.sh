#!/bin/bash
# Val 完整重建：清理 → 导出 → 分割 → 实例检索 → CLIP 向量
# 确保 export 已改为 5 位（zfill(5)），整个 pipeline 统一 5 位命名
set -e

cd "$(dirname "$0")/.."
VAL_ROOT="../GaussianConstruction/configs/data/habitat_val"

echo "=========================================="
echo "Step 0: 清理 habitat_val 派生数据"
echo "=========================================="
for d in "$VAL_ROOT"/*/; do
  [ -d "$d" ] || continue
  rm -rf "${d}rgb" "${d}depth" "${d}seg_semantic"
  rm -f "${d}transforms.json" "${d}instance_retrieval.json" "${d}instance_clip_vectors.json"
done
echo "清理完成"

echo ""
echo "=========================================="
echo "Step 1: 导出 rgb / depth / transforms.json (5 位)"
echo "=========================================="
python scripts/export_habitat_val_data.py \
  --output_dir "$VAL_ROOT" \
  --poses start_only \
  --skip_white_frames \
  --max_frames_per_scene 500 \
  --sampling_method fps

echo ""
echo "=========================================="
echo "Step 2: 开放词汇分割 → seg_semantic/*.npy"
echo "=========================================="
python gaussian_utils/seg_train_imgs.py \
  --config configs/seg_open_vocab.yaml \
  --scenes_root "$VAL_ROOT"

echo ""
echo "=========================================="
echo "Step 3: 生成 instance_retrieval.json"
echo "=========================================="
python gaussian_utils/transform_pointcloud.py \
  --scenes_root "$VAL_ROOT"

echo ""
echo "=========================================="
echo "Step 4: 生成 instance_clip_vectors.json"
echo "=========================================="
export HF_ENDPOINT=https://hf-mirror.com
python gaussian_utils/build_instance_clip_vectors.py \
  --scenes_root "$VAL_ROOT"

echo ""
echo "=========================================="
echo "Val 预处理完成！"
echo "接下来需在 GaussianConstruction 中建图："
echo "  cd ../GaussianConstruction"
echo "  python scripts/run_build_val_semantic_gaussians.py --data_dir configs/data/habitat_val --group_name habitat_val_final"
echo ""
echo "建图完成后运行消融："
echo "  cd ../GaussianNavigation"
echo "  bash scripts/run_val_nogt_ablation.sh"
echo "=========================================="
