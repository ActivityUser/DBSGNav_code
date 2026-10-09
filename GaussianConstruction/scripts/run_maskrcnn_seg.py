"""torchvision Mask R-CNN (COCO 预训练) 对 rgb/ 推理，输出 COCO 6 类类别图。

代表"原版 Mask-RCNN 分割"基线，用于与 DINO+SAM (seg_semantic) 对比。
输出：<scene>/seg_maskrcnn/frame_XXXXX.npy —— int16 类别图，0=背景，否则 COCO id。

用法（任意带 torchvision 的 python）：
  python run_maskrcnn_seg.py --scene 4ok3usBNeis [--stride 20] [--score 0.5]
"""
import argparse
import glob
import os
import re

import numpy as np

FRAMES_ROOT = "../GaussianConstruction/configs/data/habitat_val"
# torchvision COCO 名称 -> 我们的 COCO 6 类 id
NAME_TO_COCO6 = {
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "toilet": 61, "tv": 62,
}


def main():
    import torch
    import torchvision
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
    )
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--mask_thr", type=float, default=0.5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    cats = weights.meta["categories"]  # index -> name
    label2coco6 = {i: NAME_TO_COCO6[n] for i, n in enumerate(cats) if n in NAME_TO_COCO6}
    model = maskrcnn_resnet50_fpn(weights=weights).eval().to(dev)

    scene_dir = os.path.join(FRAMES_ROOT, args.scene)
    rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "frame_*.png")))
    if args.stride > 1:
        rgb_files = rgb_files[:: args.stride]
    out_dir = os.path.join(scene_dir, "seg_maskrcnn")
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for fp in rgb_files:
        bgr = cv2.imread(fp)
        rgb = bgr[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(dev)
        with torch.no_grad():
            out = model([t])[0]
        cat = np.zeros((h, w), dtype=np.int16)
        best_score = np.zeros((h, w), dtype=np.float32)
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()  # (N,1,H,W)
        for i in range(len(labels)):
            if scores[i] < args.score:
                continue
            cid = label2coco6.get(int(labels[i]))
            if cid is None:
                continue
            m = masks[i, 0] >= args.mask_thr
            upd = m & (scores[i] > best_score)
            cat[upd] = cid
            best_score[upd] = scores[i]
        fid = re.sub(r".*frame_(\d+).*", r"\1", os.path.basename(fp))
        np.save(os.path.join(out_dir, f"frame_{fid}.npy"), cat)
        n += 1
    print(f"[{args.scene}] Mask-RCNN saved {n} maps -> {out_dir}")


if __name__ == "__main__":
    main()
