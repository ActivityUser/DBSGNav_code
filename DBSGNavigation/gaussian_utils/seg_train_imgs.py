"""
批量生成 seg_semantic/*.npy 与可视化 .png，使用开放词汇分割（Grounding DINO + SAM）。
替代原 Mask-RCNN / InternImage：支持文本提示、零样本识别、与 CLIP 特征空间兼容。
输出与 transform_pointcloud、HabitatDataset、DBSGConstruction 完全兼容。

断点续跑：默认开启帧级 resume（已有有效 .npy 的帧会跳过）。SSH 断线后进程会退出，请用 tmux/screen/nohup
保持任务在服务器上运行；连上后重新执行同一条命令即可从未完成帧继续。强制全量重跑请加 --no-resume。
"""
from __future__ import annotations
# 使用 HuggingFace 镜像，避免 bert-base-uncased 等从 huggingface.co 下载时网络不可达
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import os
import sys
from glob import glob
from natsort import natsorted
from tqdm import tqdm

import cv2
import numpy as np
import torch

# 保证从任意 cwd 运行都能找到同目录的 open_vocab_segmenter
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from open_vocab_segmenter import (
    DEFAULT_CATEGORY_THRESHOLDS,
    init_segmenter,
    segment_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="开放词汇分割（Grounding DINO + SAM）：遍历场景 rgb/*.png，写 seg_semantic/*.npy 与 *.png。替代 Mask-RCNN/InternImage。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选 YAML 配置文件路径，其中的路径与参数会被命令行覆盖",
    )
    parser.add_argument(
        "--scenes_root",
        type=str,
        default=None,
        help="场景根目录，其下每子目录为 scene，含 rgb/ 与将生成的 seg_semantic/。也可用环境变量 SCENES_ROOT",
    )
    parser.add_argument(
        "--grounding_config",
        type=str,
        default=None,
        help="Grounding DINO 配置文件路径（如 .py）",
    )
    parser.add_argument(
        "--grounding_checkpoint",
        type=str,
        default=None,
        help="Grounding DINO 权重路径",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="SAM 权重路径（如 sam_vit_h_4b8939.pth）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备，如 cuda:0",
    )
    parser.add_argument(
        "--scene_index_start",
        type=int,
        default=None,
        help="从第几个场景开始（含），用于断点续跑",
    )
    parser.add_argument(
        "--scene_index_end",
        type=int,
        default=None,
        help="到第几个场景结束（不含）；默认处理全部",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=None,
        help="Grounding DINO 框置信度阈值",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=None,
        help="Grounding DINO 文本阈值",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="chair . couch . potted plant . bed . toilet . television . table . sofa . cabinet . shelf . lamp . pillow . curtain . rug . window . door . wall . floor . ceiling",
        help="检测文本提示，默认包含常见室内物体",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="仅处理指定场景，逗号分隔的目录名，如 HY1NcmCgn3n,LT9Jq6dN3Ea；不设则处理全部（或按 scene_index 切片）",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.8,
        help="非极大值抑制阈值",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若整场景 seg_semantic 已全部有效则跳过该场景",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="关闭帧级断点续跑（每帧重新推理）；默认开启：已有有效 seg_semantic/*.npy 的帧会跳过，SSH 断线后用同命令可接着跑",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="生成可视化图片",
    )
    parser.add_argument(
        "--strict_coco6",
        action="store_true",
        help="仅 6 类 IIN 文本/短语映射，抑制 plant/tv 过检（推荐评测与新建图）",
    )
    parser.add_argument(
        "--plant_box_threshold",
        type=float,
        default=None,
        help="potted plant(58) 的 box 阈值；设则启用按类阈值",
    )
    parser.add_argument(
        "--tv_box_threshold",
        type=float,
        default=None,
        help="tv(62) 的 box 阈值",
    )
    args = parser.parse_args()

    # 从 YAML 加载默认值（若存在）
    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if value is not None and getattr(args, key, None) is None:
                setattr(args, key, value)

    # 优先使用 SCENES_ROOT 环境变量
    if not args.scenes_root and os.environ.get("SCENES_ROOT"):
        args.scenes_root = os.environ.get("SCENES_ROOT")

    # 必填项
    if not args.scenes_root:
        raise ValueError(
            "请指定 --scenes_root、通过 --config 设置 scenes_root，或设置环境变量 SCENES_ROOT"
        )
    if not args.grounding_config:
        raise ValueError("请指定 --grounding_config 或通过 --config 在 YAML 中设置")
    if not args.grounding_checkpoint:
        raise ValueError("请指定 --grounding_checkpoint 或通过 --config 在 YAML 中设置")
    if not args.sam_checkpoint:
        raise ValueError("请指定 --sam_checkpoint 或通过 --config 在 YAML 中设置")

    # 默认值
    if args.device is None:
        args.device = "cuda:0"
    if args.scene_index_start is None:
        args.scene_index_start = 0
    if args.box_threshold is None:
        args.box_threshold = 0.3  # 稍微降低阈值，提高召回率
    if args.text_threshold is None:
        args.text_threshold = 0.25
    # 默认帧级续跑；--no-resume 时整场景重算每帧
    args.resume = not bool(getattr(args, "no_resume", False))
    return args


# open_vocab_segmenter 的 save_mask：key 为 COCO id，value 为 list[np.ndarray] 或 None
_VALID_COCO_SEG_KEYS = frozenset({56, 57, 58, 59, 61, 62})


def _mask_value_has_pixels(v) -> bool:
    if v is None:
        return False
    if isinstance(v, np.ndarray):
        return bool(v.sum() > 0)
    if isinstance(v, (list, tuple)):
        for m in v:
            if isinstance(m, np.ndarray) and m.sum() > 0:
                return True
    return False


def save_mask_has_valid_detection(save_mask) -> bool:
    """与 open_vocab_segmenter 输出格式一致；忽略占位键 background。"""
    if not isinstance(save_mask, dict):
        return False
    for k, v in save_mask.items():
        if k == "background" or (isinstance(k, str) and k == "background"):
            continue
        try:
            kid = int(k)
        except (TypeError, ValueError):
            continue
        if kid not in _VALID_COCO_SEG_KEYS:
            continue
        if _mask_value_has_pixels(v):
            return True
    return False


def verify_segmentation_results(save_dir, rgb_paths):
    """验证分割结果是否完整且有效"""
    required_npy = [os.path.join(save_dir, os.path.splitext(os.path.basename(p))[0] + ".npy") 
                    for p in rgb_paths]
    
    # 检查所有文件是否存在且非空
    for npy_path in required_npy:
        if not os.path.exists(npy_path):
            return False
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            if not save_mask_has_valid_detection(data):
                return False
        except:
            return False
    return True


def main():
    args = parse_args()
    device = torch.device(args.device)

    scenes_list = natsorted(glob(os.path.join(args.scenes_root, "*")))
    scenes_list = [d for d in scenes_list if os.path.isdir(d)]
    
    if getattr(args, "scenes", None) and args.scenes.strip():
        names = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes_list = [d for d in scenes_list if os.path.basename(d.rstrip(os.sep)) in names]
        if len(scenes_list) < len(names):
            found = {os.path.basename(d.rstrip(os.sep)) for d in scenes_list}
            print("Warning: 未找到的场景目录:", names - found)
    elif args.scene_index_end is not None:
        start = args.scene_index_start if args.scene_index_start is not None else 0
        scenes_list = scenes_list[start : args.scene_index_end]
    elif args.scene_index_start is not None:
        scenes_list = scenes_list[args.scene_index_start :]

    if not scenes_list:
        print("Error: 未找到任何场景目录，请检查 --scenes_root 或 SCENES_ROOT。")
        return 1

    print(f"[Grounding DINO + SAM] 加载模型...")
    print(f"[配置] 设备: {args.device}")
    print(f"[配置] 文本提示: {args.text_prompt[:100]}...")
    print(f"[配置] box_threshold: {args.box_threshold}, text_threshold: {args.text_threshold}")
    print(f"[配置] 帧级断点续跑 resume={args.resume}（断线后重跑同命令可跳过已完成帧；强制全量请加 --no-resume）")
    
    segmenter = init_segmenter(
        device=device,
        sam_checkpoint_path=args.sam_checkpoint,
        grounding_config_path=args.grounding_config,
        grounding_checkpoint_path=args.grounding_checkpoint,
    )
    cat_thr = None
    if args.plant_box_threshold is not None or args.tv_box_threshold is not None:
        cat_thr = {
            56: (args.box_threshold, args.text_threshold),
            57: (args.box_threshold, args.text_threshold),
            59: (args.box_threshold, args.text_threshold),
            61: (args.box_threshold, args.text_threshold),
        }
        cat_thr[58] = (
            args.plant_box_threshold
            if args.plant_box_threshold is not None
            else DEFAULT_CATEGORY_THRESHOLDS[58][0],
            DEFAULT_CATEGORY_THRESHOLDS[58][1],
        )
        cat_thr[62] = (
            args.tv_box_threshold
            if args.tv_box_threshold is not None
            else DEFAULT_CATEGORY_THRESHOLDS[62][0],
            DEFAULT_CATEGORY_THRESHOLDS[62][1],
        )
    print(f"[Grounding DINO + SAM] 处理 {len(scenes_list)} 个场景")
    if args.strict_coco6:
        print("[配置] strict_coco6=True（6 类短语表）")
    if cat_thr:
        print(f"[配置] category_thresholds={cat_thr}")

    # 统计信息
    total_frames = 0
    success_frames = 0
    failed_frames = 0
    skipped_resume_frames = 0

    for scene_idx, scene in enumerate(tqdm(scenes_list, desc="scenes")):
        rgbs_list = glob(os.path.join(scene, "rgb", "*.png"))
        rgbs_list = natsorted(rgbs_list)
        
        if not rgbs_list:
            print(f"警告: 场景 {scene} 中没有找到RGB图片")
            continue
            
        save_dir = os.path.join(scene, "seg_semantic")
        
        # 如果启用跳过已处理且场景已完成，则跳过
        if args.skip_existing and os.path.exists(save_dir):
            if verify_segmentation_results(save_dir, rgbs_list):
                print(f"场景 {os.path.basename(scene)} 已处理完成，跳过")
                continue
        
        os.makedirs(save_dir, exist_ok=True)
        
        for rgb_path in tqdm(rgbs_list, desc=f"Scene {scene_idx+1}/{len(scenes_list)}", leave=False):
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                print(f"警告: 无法读取图片 {rgb_path}")
                failed_frames += 1
                continue
                
            rgb = rgb[:, :, ::-1]  # BGR -> RGB
            save_file_name = os.path.splitext(os.path.basename(rgb_path))[0]
            npy_out = os.path.join(save_dir, save_file_name + ".npy")

            if args.resume and os.path.isfile(npy_out):
                try:
                    existing = np.load(npy_out, allow_pickle=True).item()
                    if save_mask_has_valid_detection(existing):
                        skipped_resume_frames += 1
                        continue
                except Exception:
                    pass

            try:
                save_mask, save_bgr_vis = segment_frame(
                    segmenter,
                    rgb,
                    text_prompt=args.text_prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    nms_threshold=args.nms_threshold,
                    strict_coco6=bool(args.strict_coco6),
                    category_thresholds=cat_thr,
                )
                
                total_frames += 1
                
                has_valid_detection = save_mask_has_valid_detection(save_mask)

                if has_valid_detection:
                    # 保存npy文件
                    np.save(os.path.join(save_dir, save_file_name + ".npy"), save_mask)
                    
                    # 保存可视化图片
                    if args.visualize and save_bgr_vis is not None:
                        cv2.imwrite(os.path.join(save_dir, save_file_name + ".png"), save_bgr_vis)
                    
                    success_frames += 1
                else:
                    # 如果没有检测到任何物体，创建一个空的mask文件
                    empty_mask = {"background": np.ones_like(rgb[:,:,0], dtype=np.uint8) * 255}
                    np.save(os.path.join(save_dir, save_file_name + ".npy"), empty_mask)
                    
                    # 保存原始图片作为可视化
                    if args.visualize:
                        cv2.imwrite(os.path.join(save_dir, save_file_name + ".png"), rgb[:,:,::-1])
                    
                    print(f"警告: 图片 {rgb_path} 未检测到物体，已创建空mask")
                    failed_frames += 1
                    
            except Exception as e:
                print(f"错误: 处理 {rgb_path} 时发生异常: {str(e)}")
                failed_frames += 1
                
                # 创建空mask作为后备
                empty_mask = {"background": np.ones_like(rgb[:,:,0], dtype=np.uint8) * 255}
                np.save(os.path.join(save_dir, save_file_name + ".npy"), empty_mask)
                
                if args.visualize:
                    cv2.imwrite(os.path.join(save_dir, save_file_name + ".png"), rgb[:,:,::-1])
                
        # 每个场景处理完后打印统计
        scene_success = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
        print(f"场景 {os.path.basename(scene)} 完成: {scene_success}/{len(rgbs_list)} 帧成功")

    # 最终统计
    print("\n" + "="*50)
    print(f"处理完成！本趟实际推理帧数: {total_frames}")
    if skipped_resume_frames:
        print(f"断点跳过（已有有效 npy）: {skipped_resume_frames} 帧")
    if total_frames > 0:
        print(f"成功帧数: {success_frames} ({success_frames/total_frames*100:.1f}%)")
        print(f"失败帧数: {failed_frames} ({failed_frames/total_frames*100:.1f}%)")
    else:
        print(f"成功帧数: {success_frames}；失败帧数: {failed_frames}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())