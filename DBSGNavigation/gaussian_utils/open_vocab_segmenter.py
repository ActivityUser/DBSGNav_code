"""
开放词汇分割模块：Grounding DINO（检测）+ SAM（分割）。
输出格式与 seg_train_imgs.get_sem_pred_internimg 兼容，供 transform_pointcloud 与 DBSGConstruction 使用。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import traceback

import cv2
import numpy as np
import torch

# 默认 6 类与 COCO id 对应（与 transform_pointcloud / basedataset 一致）
DEFAULT_COCO_KEYS = (56, 57, 58, 59, 61, 62)
# 扩展默认文本提示，包含更多室内物体
DEFAULT_TEXT_PROMPT = "chair . couch . potted plant . bed . toilet . television . table . sofa . cabinet . shelf . lamp . pillow . curtain . rug . window . door . wall . floor . ceiling ."
# 仅 6 类 IIN：避免 wall/floor/lamp 等误映射到 plant/tv（分割评测与建图推荐）
STRICT_COCO6_TEXT_PROMPT = "chair . couch . potted plant . bed . toilet . television ."
STRICT_PHRASE_TO_COCO: Dict[str, int] = {
    "chair": 56,
    "couch": 57,
    "sofa": 57,
    "potted plant": 58,
    "bed": 59,
    "toilet": 61,
    "television": 62,
    "tv": 62,
}
# 按 COCO id 的 (box_threshold, text_threshold)；用于抑制 plant/tv 过检
DEFAULT_CATEGORY_THRESHOLDS: Dict[int, Tuple[float, float]] = {
    58: (0.50, 0.40),  # potted plant
    62: (0.50, 0.40),  # tv / television
}
# phrase (lower) -> COCO id；支持多表述映射到同一 id（宽松模式，易过检）
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
    "table": 56,
    "cabinet": 58,
    "shelf": 58,
    "lamp": 62,
    "pillow": 57,
    "curtain": 58,
    "rug": 58,
    "window": 58,
    "door": 58,
    "wall": 58,
    "floor": 58,
    "ceiling": 58,
}


def _load_grounding_dino(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str | torch.device,
) -> Any:
    """加载 Grounding DINO 模型"""
    try:
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
    except Exception as e:
        print(f"加载 Grounding DINO 失败: {e}")
        raise


def _load_sam_predictor(checkpoint_path: str, device: str | torch.device) -> Any:
    """加载 SAM 模型"""
    try:
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
    except Exception as e:
        print(f"加载 SAM 失败: {e}")
        raise


def _preprocess_image_grounding_dino(rgb: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """将 RGB numpy (H,W,3) 转为 Grounding DINO 输入张量。"""
    try:
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
    except Exception as e:
        print(f"图像预处理失败: {e}")
        raise


def _run_grounding_dino(
    model: Any,
    rgb: np.ndarray,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """返回 (boxes_cxcywh_norm, logits, phrases)。"""
    try:
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
    except Exception as e:
        print(f"Grounding DINO 运行失败: {e}")
        return torch.zeros((0, 4)), torch.zeros(0), []


def _boxes_cxcywh_norm_to_xyxy(boxes: torch.Tensor, h: int, w: int) -> np.ndarray:
    """boxes: (N,4) 归一化 cxcywh -> (N,4) xyxy 像素坐标。"""
    try:
        if boxes is None or boxes.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        from torchvision.ops import box_convert

        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes_px = boxes * scale
        xyxy = box_convert(boxes_px, in_fmt="cxcywh", out_fmt="xyxy")
        
        # 确保坐标在图像范围内
        xyxy = xyxy.cpu().numpy()
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w-1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h-1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], xyxy[:, 0]+1, w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], xyxy[:, 1]+1, h)
        
        return xyxy.astype(np.float32)
    except Exception as e:
        print(f"坐标转换失败: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def _nms_filter_xyxy(
    xyxy: np.ndarray,
    scores: torch.Tensor,
    phrases: List[str],
    nms_iou: float,
    max_boxes: Optional[int],
) -> Tuple[np.ndarray, torch.Tensor, List[str]]:
    """对检测框做 NMS，并按分数截断到 max_boxes。nms_iou 为 torchvision.ops.nms 的 IoU 阈值。"""
    if xyxy.shape[0] == 0:
        return xyxy, scores, phrases
    try:
        from torchvision.ops import nms as tv_nms

        boxes_t = torch.from_numpy(xyxy).float()
        sc = scores.detach().float().cpu()
        if sc.numel() != boxes_t.shape[0]:
            return xyxy, scores, phrases
        keep = tv_nms(boxes_t, sc, float(nms_iou))
        if max_boxes is not None and int(max_boxes) > 0 and keep.numel() > int(max_boxes):
            sub = sc[keep]
            _, ord_idx = torch.sort(sub, descending=True)
            keep = keep[ord_idx[: int(max_boxes)]]
        keep_idx = keep.cpu().numpy().astype(np.int64)
        xyxy_out = xyxy[keep_idx]
        keep_t = keep.to(device=scores.device, dtype=torch.long)
        scores_out = scores[keep_t] if scores.dim() > 0 else scores
        phrases_out = [phrases[int(i)] for i in keep_idx.tolist()]
        return xyxy_out, scores_out, phrases_out
    except Exception as e:
        print(f"NMS 失败，跳过: {e}")
        return xyxy, scores, phrases


def _phrase_to_coco_id(
    phrase: str,
    phrase_to_coco: Dict[str, int],
    strict: bool = False,
) -> Optional[int]:
    """将 Grounding DINO 的 phrase 映射到 COCO id，只接受 DEFAULT_COCO_KEYS。"""
    if not phrase:
        return None

    key = phrase.lower().strip()

    if key in phrase_to_coco:
        coco_id = phrase_to_coco[key]
        if coco_id in DEFAULT_COCO_KEYS:
            return coco_id

    if strict:
        return None

    for name, cid in phrase_to_coco.items():
        if name in key or key in name:
            if cid in DEFAULT_COCO_KEYS:
                return cid

    return None


def _box_passes_category_threshold(
    cid: Optional[int],
    score: float,
    box_threshold: float,
    category_thresholds: Optional[Dict[int, Tuple[float, float]]],
) -> bool:
    """检测框置信度是否达到该类阈值（未配置则用全局 box_threshold）。"""
    if cid is None:
        return False
    thr = box_threshold
    if category_thresholds and cid in category_thresholds:
        thr = float(category_thresholds[cid][0])
    return float(score) >= thr


def init_segmenter(
    device: str | torch.device,
    sam_checkpoint_path: str,
    grounding_config_path: str,
    grounding_checkpoint_path: str,
):
    """
    加载 Grounding DINO 与 SAM，返回一个 segmenter 对象，用于后续 segment_frame 调用。
    """
    try:
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
    except Exception as e:
        print(f"初始化分割器失败: {e}")
        traceback.print_exc()
        raise


def segment_frame(
    segmenter: Dict[str, Any],
    rgb_image: np.ndarray,
    text_prompt: Optional[str] = None,
    category_mapping: Optional[Dict[str, int]] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    sam_multimask_output: bool = False,
    nms_threshold: Optional[float] = None,
    max_boxes: Optional[int] = None,
    strict_coco6: bool = False,
    category_thresholds: Optional[Dict[int, Tuple[float, float]]] = None,
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
        nms_threshold: 检测后 NMS 的 IoU 阈值（与 torchvision.ops.nms 一致）；None 表示不做 NMS
        max_boxes: NMS 后最多保留框数；None 表示不限制
        strict_coco6: True 时仅用 STRICT 文本/短语表，避免 wall→plant、lamp→tv 误映射
        category_thresholds: 按 COCO id 提高 box 阈值，如 {58:(0.5,0.4), 62:(0.5,0.4)}

    Returns:
        save_mask: dict[int, list[np.ndarray]|None]，key 为 56,57,58,59,61,62
        vis_image: (H,W,3) BGR，用于保存 .png
    """
    # 初始化返回结构
    save_mask: Dict[int, Optional[List[np.ndarray]]] = {k: None for k in DEFAULT_COCO_KEYS}
    
    # 检查输入
    if rgb_image is None or rgb_image.size == 0:
        print("警告: 输入图像为空")
        h, w = 480, 640
        if rgb_image is not None and len(rgb_image.shape) >= 2:
            h, w = rgb_image.shape[:2]
        empty_vis = np.zeros((h, w, 3), dtype=np.uint8)
        return save_mask, empty_vis
    
    try:
        gd_model = segmenter.get("gd_model")
        sam_predictor = segmenter.get("sam_predictor")
        device = segmenter.get("device", "cuda:0")

        if gd_model is None or sam_predictor is None:
            print("错误: segmenter 中缺少必要模型")
            h, w = rgb_image.shape[:2]
            empty_vis = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            return save_mask, empty_vis

        if strict_coco6:
            if text_prompt is None:
                text_prompt = STRICT_COCO6_TEXT_PROMPT
            if category_mapping is None:
                category_mapping = dict(STRICT_PHRASE_TO_COCO)
        else:
            if text_prompt is None:
                text_prompt = DEFAULT_TEXT_PROMPT
            if category_mapping is None:
                category_mapping = dict(DEFAULT_PHRASE_TO_COCO)

        h, w = rgb_image.shape[:2]

        # DINO 初筛：有 per-class 阈值时用较低全局阈值，再在 SAM 前按类过滤
        dino_box_thr = float(box_threshold)
        if category_thresholds:
            dino_box_thr = min(dino_box_thr, min(t[0] for t in category_thresholds.values()) * 0.85)
            dino_box_thr = max(0.20, dino_box_thr)

        # 1) Grounding DINO
        boxes_norm, logits, phrases = _run_grounding_dino(
            gd_model,
            rgb_image,
            text_prompt,
            dino_box_thr,
            text_threshold,
            device,
        )
        
        # 转换为绝对坐标
        xyxy = _boxes_cxcywh_norm_to_xyxy(boxes_norm, h, w)

        if nms_threshold is not None and xyxy.shape[0] > 0:
            xyxy, logits, phrases = _nms_filter_xyxy(
                xyxy, logits, phrases, float(nms_threshold), max_boxes
            )

        # 2) 按 phrase 映射到 COCO id
        coco_ids: List[Optional[int]] = []
        for ph in phrases:
            coco_ids.append(
                _phrase_to_coco_id(ph, category_mapping, strict=strict_coco6)
            )

        if logits is not None and logits.numel() > 0:
            logits_np = logits.detach().float().cpu().numpy()
        else:
            logits_np = np.zeros(xyxy.shape[0], dtype=np.float32)

        # 3) 按类别收集mask
        lists: Dict[int, List[np.ndarray]] = {k: [] for k in DEFAULT_COCO_KEYS}

        if xyxy.shape[0] > 0:
            try:
                sam_predictor.set_image(rgb_image, image_format="RGB")
                
                for i in range(xyxy.shape[0]):
                    cid = coco_ids[i] if i < len(coco_ids) else None
                    if cid is None or cid not in DEFAULT_COCO_KEYS:
                        continue
                    sc = float(logits_np[i]) if i < len(logits_np) else 0.0
                    if not _box_passes_category_threshold(
                        cid, sc, box_threshold, category_thresholds
                    ):
                        continue
                    
                    box = xyxy[i].astype(np.float32)
                    
                    # 确保框有效
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    
                    try:
                        masks, iou_pred, _ = sam_predictor.predict(
                            box=box,
                            multimask_output=sam_multimask_output,
                            return_logits=False,
                        )
                        
                        if masks is not None and len(masks) > 0:
                            # 选择最佳mask
                            idx = 0
                            if masks.shape[0] > 1 and iou_pred is not None:
                                idx = int(np.argmax(iou_pred))
                            mask = masks[idx]
                            
                            # 确保mask尺寸正确
                            if mask.shape[0] != h or mask.shape[1] != w:
                                mask = cv2.resize(
                                    mask.astype(np.uint8),
                                    (w, h),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                            
                            # 过滤太小的mask（少于50像素）
                            if mask.sum() > 50:
                                lists[cid].append(mask)
                    except Exception as e:
                        print(f"SAM预测单框失败: {e}")
                        continue
                            
            except Exception as e:
                print(f"SAM批量预测失败: {e}")

        # 4) 合并同一类别的mask
        for k in DEFAULT_COCO_KEYS:
            if lists[k]:
                # 合并所有mask
                combined_mask = np.zeros((h, w), dtype=bool)
                for mask in lists[k]:
                    combined_mask = np.logical_or(combined_mask, mask)
                save_mask[k] = [combined_mask]
            else:
                save_mask[k] = None

        # 5) 可视化
        vis_image = _draw_vis(rgb_image, xyxy, phrases, logits, save_mask, coco_ids)
        
        return save_mask, vis_image
        
    except Exception as e:
        print(f"segment_frame 处理失败: {e}")
        traceback.print_exc()
        
        # 出错时返回空mask和原图
        h, w = rgb_image.shape[:2]
        empty_vis = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return save_mask, empty_vis


_VIS_COLORS = {
    56: [0.94, 0.78, 0.66],  # 椅子
    57: [0.89, 0.94, 0.66],  # 沙发
    58: [0.66, 0.94, 0.85],  # 植物/柜子
    59: [0.71, 0.66, 0.94],  # 床
    61: [0.92, 0.66, 0.94],  # 马桶
    62: [0.94, 0.66, 0.75],  # 电视/灯
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
    try:
        # 转换为BGR用于显示
        out = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        
        # 首先绘制mask（半透明）
        for cid, masks in save_mask.items():
            if cid is None or masks is None or len(masks) == 0:
                continue
            color = _VIS_COLORS.get(cid, [0.5, 0.5, 0.5])
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            for m in masks:
                if m is not None and m.shape[:2] == (out.shape[0], out.shape[1]):
                    # 创建彩色mask
                    colored_mask = np.zeros_like(out)
                    colored_mask[m] = color_bgr
                    # 半透明叠加
                    out = cv2.addWeighted(out, 1.0, colored_mask, 0.5, 0)
        
        # 然后绘制边界框
        if xyxy is not None and xyxy.shape[0] > 0:
            if logits is not None and logits.numel() > 0:
                logits_np = logits.cpu().numpy()
            else:
                logits_np = np.zeros(xyxy.shape[0])
            
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cid = coco_ids[i] if i < len(coco_ids) else None
                label = phrases[i] if i < len(phrases) else "unknown"
                conf = float(logits_np[i]) if i < len(logits_np) else 0.0
                
                color = _VIS_COLORS.get(cid, [0.5, 0.5, 0.5])
                color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                
                # 绘制矩形
                cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
                
                # 绘制标签背景
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(out, (x1, max(0, y1 - text_h - 4)), (x1 + text_w, y1), color_bgr, -1)
                
                # 绘制标签文字
                cv2.putText(
                    out,
                    label_text,
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
        
        return out
        
    except Exception as e:
        print(f"可视化失败: {e}")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)