"""
开放词汇分割模块的测试脚本。
用法：
  1) 仅检查输出格式约定（不需模型）：
     python test_open_vocab_segmenter.py --check-format

  2) 单张图测试（需已安装 Grounding DINO + SAM 并指定权重）：
     python test_open_vocab_segmenter.py --image /path/to/rgb.png \\
       --grounding_config /path/to/GroundingDINO_SwinT_OGC.py \\
       --grounding_checkpoint /path/to/groundingdino_swint_ogc.pth \\
       --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \\
       [--out_dir /path/to/save] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import os
import sys

# 保证可导入 open_vocab_segmenter
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from open_vocab_segmenter import (
    DEFAULT_COCO_KEYS,
    init_segmenter,
    segment_frame,
)


def check_format():
    """不加载模型，仅验证输出格式约定（key 集合、value 类型）。"""
    expected_keys = {56, 57, 58, 59, 61, 62}
    assert set(DEFAULT_COCO_KEYS) == expected_keys, f"DEFAULT_COCO_KEYS 应为 {expected_keys}"
    # 模拟 save_mask：全 None 的合法结构
    save_mask = {k: None for k in DEFAULT_COCO_KEYS}
    for k in save_mask:
        assert k in expected_keys
        assert save_mask[k] is None or (isinstance(save_mask[k], list))
    print("[check_format] 通过：save_mask 的 key 为 {56,57,58,59,61,62}，value 为 list 或 None。")


def test_single_image(
    image_path: str,
    grounding_config: str,
    grounding_checkpoint: str,
    sam_checkpoint: str,
    out_dir: str | None = None,
    device: str = "cuda:0",
):
    """对单张图跑 segment_frame，检查返回格式并可选保存 .npy / .png。"""
    import cv2
    import numpy as np

    rgb = cv2.imread(image_path)
    if rgb is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    rgb = rgb[:, :, ::-1]  # BGR -> RGB

    segmenter = init_segmenter(
        device=device,
        sam_checkpoint_path=sam_checkpoint,
        grounding_config_path=grounding_config,
        grounding_checkpoint_path=grounding_checkpoint,
    )
    save_mask, vis_image = segment_frame(segmenter, rgb)

    # 格式检查（与 transform_pointcloud / basedataset 兼容）
    assert set(save_mask.keys()) == set(DEFAULT_COCO_KEYS), (
        f"save_mask keys 应为 {set(DEFAULT_COCO_KEYS)}，得到 {set(save_mask.keys())}"
    )
    h, w = rgb.shape[:2]
    for k, v in save_mask.items():
        assert v is None or isinstance(v, list), f"key {k} 的值应为 None 或 list"
        if v is not None:
            for m in v:
                assert isinstance(m, np.ndarray) and m.ndim == 2 and m.shape == (h, w), (
                    f"key {k} 的 mask 应为 (H,W)=({h},{w}) 的 2D 数组"
                )

    print("[test_single_image] 通过：save_mask 格式符合约定。")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        npy_path = os.path.join(out_dir, base + ".npy")
        png_path = os.path.join(out_dir, base + "_vis.png")
        np.save(npy_path, save_mask)
        cv2.imwrite(png_path, vis_image)
        print(f"已保存: {npy_path}, {png_path}")

    return save_mask, vis_image


def main():
    parser = argparse.ArgumentParser(description="测试 open_vocab_segmenter")
    parser.add_argument("--check-format", action="store_true", help="仅检查输出格式约定，不加载模型")
    parser.add_argument("--image", type=str, help="单张 RGB 图像路径（用于完整推理测试）")
    parser.add_argument("--grounding_config", type=str, help="Grounding DINO 配置文件路径")
    parser.add_argument("--grounding_checkpoint", type=str, help="Grounding DINO 权重路径")
    parser.add_argument("--sam_checkpoint", type=str, help="SAM 权重路径")
    parser.add_argument("--out_dir", type=str, default=None, help="保存 .npy 与可视化 .png 的目录")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.check_format:
        check_format()
        return

    if args.image and args.grounding_config and args.grounding_checkpoint and args.sam_checkpoint:
        test_single_image(
            args.image,
            args.grounding_config,
            args.grounding_checkpoint,
            args.sam_checkpoint,
            out_dir=args.out_dir,
            device=args.device,
        )
        return

    print(__doc__)
    print("示例:")
    print("  python test_open_vocab_segmenter.py --check-format")
    print("  python test_open_vocab_segmenter.py --image ./rgb/0.png \\")
    print("    --grounding_config /path/to/GroundingDINO_SwinT_OGC.py \\")
    print("    --grounding_checkpoint /path/to/groundingdino_swint_ogc.pth \\")
    print("    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth --out_dir ./out")


if __name__ == "__main__":
    main()
