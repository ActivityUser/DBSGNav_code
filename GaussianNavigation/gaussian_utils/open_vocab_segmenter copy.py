"""
开放词汇分割模块：Grounding DINO（检测）+ SAM（分割）。
输出格式与 seg_train_imgs.get_sem_pred_internimg 兼容，供 transform_pointcloud 与 GaussianConstruction 使用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch

# 默认 6 类与 COCO id 对应（与 transform_pointcloud / basedataset 一致）
DEFAULT_COCO_KEYS = (56, 57, 58, 59, 61, 62)
DEFAULT_TEXT_PROMPT = "chair . couch . potted plant . bed . toilet . television ."
# phrase (lower) -> COCO id；支持多表述映射到同一 id
DEFAULT_PHRASE_TO_COCO: Dict[str, int] = {
    "chair": 56,
    "couch": 57,
    "sofa": 57,
    "potted plant": 58,
    "plant": 58,
    "bed": 59,
    "toilet": 61,
    "television": 62,
    "tv": 62,
}


def _load_grounding_dino(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str | torch.device,
) -> Any:
    # 需安装 Grounding DINO：clone https://github.com/IDEA-Research/GroundingDINO 后 pip install -e .
    from groundingdino.models import build_model  # type: ignore[import-untyped]
    from groundingdino.util.misc import clean_state_dict  # type: ignore[import-untyped]
    from groundingdino.util.slconfig import SLConfig  # type: ignore[import-untyped]

    device = str(device) if isinstance(device, torch.device) else device
    cfg = SLConfig.fromfile(model_config_path)
    cfg.device = device
    model = build_model(cfg)
    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model


def _load_sam_predictor(checkpoint_path: str, device: str | torch.device) -> Any:
    from segment_anything import sam_model_registry  # type: ignore[import-untyped]
    from segment_anything.predictor import SamPredictor  # type: ignore[import-untyped]

    device = torch.device(device) if isinstance(device, str) else device
    # 默认使用 vit_h；若需 vit_b/vit_l 可通过配置文件扩展
    model_type = "vit_h"
    if "vit_b" in checkpoint_path.lower():
        model_type = "vit_b"
    elif "vit_l" in checkpoint_path.lower():
        model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


def _preprocess_image_grounding_dino(rgb: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """将 RGB numpy (H,W,3) 转为 Grounding DINO 输入张量。"""
    from PIL import Image
    import groundingdino.datasets.transforms as T  # type: ignore[import-untyped]

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(rgb)
    out, _ = transform(pil, None)
    device = torch.device(device) if isinstance(device, str) else device
    return out.to(device)


def _run_grounding_dino(
    model: Any,
    rgb: np.ndarray,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """返回 (boxes_cxcywh_norm, logits, phrases)。"""
    from groundingdino.util.utils import get_phrases_from_posmap  # type: ignore[import-untyped]

    image_tensor = _preprocess_image_grounding_dino(rgb, device)
    caption_proc = caption.lower().strip()
    if not caption_proc.endswith("."):
        caption_proc = caption_proc + "."
    model = model.to(device)
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption_proc])
    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    pred_boxes = outputs["pred_boxes"].cpu()[0]
    mask = pred_logits.max(dim=1)[0] > box_threshold
    logits = pred_logits[mask].max(dim=1)[0]
    boxes = pred_boxes[mask]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption_proc)
    phrases = []
    for i in range(mask.sum().item()):
        ph = get_phrases_from_posmap(
            (pred_logits[mask][i] > text_threshold).cpu(),
            tokenized,
            tokenizer,
        ).replace(".", "").strip()
        phrases.append(ph)
    return boxes, logits, phrases


def _boxes_cxcywh_norm_to_xyxy(boxes: torch.Tensor, h: int, w: int) -> np.ndarray:
    """boxes: (N,4) 归一化 cxcywh -> (N,4) xyxy 像素坐标。"""
    from torchvision.ops import box_convert

    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    boxes_px = boxes * scale
    xyxy = box_convert(boxes_px, in_fmt="cxcywh", out_fmt="xyxy")
    return xyxy.cpu().numpy()


def _phrase_to_coco_id(phrase: str, phrase_to_coco: Dict[str, int]) -> Optional[int]:
    """将 Grounding DINO 的 phrase 映射到 COCO id，只接受 DEFAULT_COCO_KEYS。"""
    key = phrase.lower().strip()
    if key in phrase_to_coco:
        coco_id = phrase_to_coco[key]
        if coco_id in DEFAULT_COCO_KEYS:
            return coco_id
    for name, cid in phrase_to_coco.items():
        if name in key or key in name:
            if cid in DEFAULT_COCO_KEYS:
                return cid
    return None


def init_segmenter(
    device: str | torch.device,
    sam_checkpoint_path: str,
    grounding_config_path: str,
    grounding_checkpoint_path: str,
):
    """
    加载 Grounding DINO 与 SAM，返回一个 segmenter 对象，用于后续 segment_frame 调用。
    """
    device = torch.device(device) if isinstance(device, str) else device
    gd_model = _load_grounding_dino(
        grounding_config_path,
        grounding_checkpoint_path,
        device,
    )
    sam_predictor = _load_sam_predictor(sam_checkpoint_path, device)
    return {
        "gd_model": gd_model,
        "sam_predictor": sam_predictor,
        "device": device,
    }


def segment_frame(
    segmenter: Dict[str, Any],
    rgb_image: np.ndarray,
    text_prompt: Optional[str] = None,
    category_mapping: Optional[Dict[str, int]] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    sam_multimask_output: bool = False,
) -> Tuple[Dict[int, Optional[List[np.ndarray]]], np.ndarray]:
    """
    对一帧 RGB 做开放词汇检测+分割，输出与 get_sem_pred_internimg 兼容的 save_mask 与可视化图。

    Args:
        segmenter: init_segmenter 返回的对象
        rgb_image: (H, W, 3) RGB，uint8
        text_prompt: 检测文本，None 则用 DEFAULT_TEXT_PROMPT
        category_mapping: phrase -> COCO id，None 则用 DEFAULT_PHRASE_TO_COCO（仅输出 56,57,58,59,61,62）
        box_threshold: Grounding DINO 框阈值
        text_threshold: Grounding DINO 文本阈值
        sam_multimask_output: SAM 是否多 mask，False 则每框一个 mask

    Returns:
        save_mask: dict[int, list[np.ndarray]|None]，key 为 56,57,58,59,61,62
        vis_image: (H,W,3) BGR，用于保存 .png
    """
    gd_model = segmenter["gd_model"]
    sam_predictor = segmenter["sam_predictor"]
    device = segmenter["device"]

    if text_prompt is None:
        text_prompt = DEFAULT_TEXT_PROMPT
    if category_mapping is None:
        category_mapping = dict(DEFAULT_PHRASE_TO_COCO)

    h, w = rgb_image.shape[:2]
    # 1) Grounding DINO
    boxes_norm, logits, phrases = _run_grounding_dino(
        gd_model,
        rgb_image,
        text_prompt,
        box_threshold,
        text_threshold,
        device,
    )
    xyxy = _boxes_cxcywh_norm_to_xyxy(boxes_norm, h, w)

    # 2) 按 phrase 映射到 COCO id，只保留 56,57,58,59,61,62
    coco_ids: List[Optional[int]] = []
    for ph in phrases:
        coco_ids.append(_phrase_to_coco_id(ph, category_mapping))

    # 3) SAM 按框预测 mask，按 category 聚合
    save_mask: Dict[int, Optional[List[np.ndarray]]] = {k: None for k in DEFAULT_COCO_KEYS}
    lists: Dict[int, List[np.ndarray]] = {k: [] for k in DEFAULT_COCO_KEYS}

    if xyxy.shape[0] > 0:
        sam_predictor.set_image(rgb_image, image_format="RGB")
        for i in range(xyxy.shape[0]):
            cid = coco_ids[i] if i < len(coco_ids) else None
            if cid is None:
                continue
            box = xyxy[i].astype(np.float32)
            masks, iou_pred, _ = sam_predictor.predict(
                box=box,
                multimask_output=sam_multimask_output,
                return_logits=False,
            )
            idx = 0
            if masks.shape[0] > 1 and iou_pred is not None:
                idx = int(np.argmax(iou_pred))
            mask = masks[idx]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            lists[cid].append(mask)

    for k in DEFAULT_COCO_KEYS:
        save_mask[k] = lists[k] if lists[k] else None

    # 4) 可视化
    vis_image = _draw_vis(rgb_image, xyxy, phrases, logits, save_mask, coco_ids)
    return save_mask, vis_image


_VIS_COLORS = {
    56: [0.94, 0.78, 0.66],
    57: [0.89, 0.94, 0.66],
    58: [0.66, 0.94, 0.85],
    59: [0.71, 0.66, 0.94],
    61: [0.92, 0.66, 0.94],
    62: [0.94, 0.66, 0.75],
}


def _draw_vis(
    rgb: np.ndarray,
    xyxy: np.ndarray,
    phrases: List[str],
    logits: torch.Tensor,
    save_mask: Dict[int, Optional[List[np.ndarray]]],
    coco_ids: List[Optional[int]],
) -> np.ndarray:
    """在 RGB 上画框和 mask，返回 BGR 图。"""
    out = rgb.copy()
    if logits is not None and logits.numel() > 0:
        logits_np = logits.cpu().numpy()
    else:
        logits_np = np.zeros(0)
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        cid = coco_ids[i] if i < len(coco_ids) else None
        label = phrases[i] if i < len(phrases) else ""
        conf = float(logits_np[i]) if i < len(logits_np) else 0.0
        color = _VIS_COLORS.get(cid, [0.5, 0.5, 0.5])
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            out,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    for cid, masks in save_mask.items():
        if cid is None or masks is None:
            continue
        color = _VIS_COLORS.get(cid, [0.5, 0.5, 0.5])
        for m in masks:
            if m.shape[:2] == (out.shape[0], out.shape[1]):
                overlay = out.copy()
                overlay[m] = (
                    overlay[m] * 0.5
                    + np.array([color[0] * 255, color[1] * 255, color[2] * 255]) * 0.5
                ).astype(np.uint8)
                out = overlay
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
