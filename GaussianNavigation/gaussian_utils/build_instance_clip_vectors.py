#!/usr/bin/env python3
"""
Precompute per-instance CLIP vectors for OpenVocabRetriever.
Reads instance_retrieval.json + rgb/frame_*.png per scene, writes instance_clip_vectors.json.
保持与原脚本完全相同的输入输出接口，但内部实现优化
"""
import sys
import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional, Any, Tuple
import logging

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

COCO_IDS = [56, 57, 58, 59, 61, 62]
CLIP_DIM = 512
CLIP_MODEL_NAME = "ViT-B-32"


class CLIPEncoder:
    """CLIP图像编码器（单张处理，保持与原脚本行为一致）"""
    
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 延迟加载模型，只在需要时加载
        self.model = None
        self.preprocess = None
        self.model_name = model_name
        self.model_loaded = False
    
    def _load_model(self):
        """加载CLIP模型（懒加载）"""
        if self.model_loaded:
            return
            
        try:
            import open_clip
            # open_clip.create_model_and_transforms 返回 (model, preprocess_train, preprocess_eval)
            # 不同版本输出数量略有差异，但我们这里需要 eval preprocess。
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai"
            )
            self.model = self.model.to(self.device).eval()
            self.model_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def encode_image(self, rgb_path: str) -> Optional[List[float]]:
        """
        编码单张图像（与原函数完全相同的输入输出）
        
        Args:
            rgb_path: 图像路径
        
        Returns:
            512维归一化向量，失败时返回None
        """
        try:
            # 懒加载模型
            self._load_model()
            
            # 检查文件是否存在
            if not os.path.exists(rgb_path):
                return None
            
            # 读取图像
            img = cv2.imread(rgb_path)
            if img is None:
                return None
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_img = Image.fromarray(img)
            
            # 预处理
            x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            # 编码
            with torch.no_grad():
                feat = self.model.encode_image(x)
                feat = feat.cpu().numpy().astype(np.float32).squeeze(0)
            
            # L2归一化
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm
            else:
                feat = np.zeros(CLIP_DIM, dtype=np.float32)
            
            return feat.tolist()
            
        except Exception as e:
            # 只在调试时打印详细错误，避免刷屏
            return None


def _instance_clip_vectors_valid(path: str) -> bool:
    """已有输出且结构完整则视为可断点跳过。"""
    if not os.path.isfile(path) or os.path.getsize(path) < 4:
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        for k in COCO_IDS:
            sk = str(k)
            if sk not in data or not isinstance(data[sk], list):
                return False
        return True
    except Exception:
        return False


def load_instance_retrieval(ir_path: str) -> Tuple[Optional[Dict], bool]:
    """
    安全加载instance_retrieval.json
    
    Returns:
        (data, success_flag)
    """
    if not os.path.isfile(ir_path):
        return None, False
    
    try:
        with open(ir_path, "r") as f:
            data = json.load(f)
        return data, True
    except Exception as e:
        print(f"警告：加载 {ir_path} 失败: {e}")
        return None, False


def _resolve_rgb_path(scene_path: str, frame_id) -> str:
    """
    解析 rgb 图像路径。export 使用 5 位命名（frame_00000.png），兼容 5/6 位 frame_id。
    """
    base = os.path.join(scene_path, "rgb")
    # 1) 先尝试原始 frame_id
    path = os.path.join(base, f"frame_{frame_id}.png")
    if os.path.isfile(path):
        return path
    # 2) 数字时尝试 5 位、6 位补零（兼容旧数据）
    if isinstance(frame_id, str) and frame_id.isdigit():
        n = int(frame_id)
        for width in (5, 6):
            padded = str(n).zfill(width)
            path = os.path.join(base, f"frame_{padded}.png")
            if os.path.isfile(path):
                return path
    return os.path.join(base, f"frame_{frame_id}.png")


def process_scene(scene_path: str, encoder: CLIPEncoder, verbose: bool = False) -> Tuple[Optional[Dict[str, List]], int]:
    """
    处理单个场景
    
    Args:
        scene_path: 场景路径
        encoder: CLIP编码器
    
    Returns:
        (输出字典, 实例数量) 如果失败则返回 (None, 0)
    """
    # 初始化输出字典
    out = {str(coco_id): [] for coco_id in COCO_IDS}
    total_instances = 0
    
    # 检查instance_retrieval.json是否存在并加载
    ir_path = os.path.join(scene_path, "instance_retrieval.json")
    instance_retrieval, success = load_instance_retrieval(ir_path)
    
    if not success:
        return None, 0
    
    # 处理每个类别
    for coco_id in COCO_IDS:
        key = str(coco_id)
        
        # 如果类别不存在，返回空列表
        if key not in instance_retrieval:
            out[key] = []
            continue
        
        entries = instance_retrieval[key]
        vectors = []
        
        # 处理该类别的每个实例
        for entry in entries:
            try:
                # 解析entry（与原脚本完全一致）
                # entry格式: [bbox_list, frame_id, navigable_points]
                if len(entry) >= 2:
                    frame_id = entry[1]
                    
                    # 构建图像路径（export 使用 5 位 frame_00000.png，_resolve_rgb_path 兼容 5/6 位）
                    rgb_path = _resolve_rgb_path(scene_path, frame_id)
                    
                    # 编码图像
                    vec = encoder.encode_image(rgb_path)
                    
                    if vec is None:
                        if verbose:
                            exists = os.path.isfile(rgb_path)
                            print(f"    [失败] frame_id={frame_id} path={rgb_path} exists={exists}")
                        vec = [0.0] * CLIP_DIM
                    
                    vectors.append(vec)
                    total_instances += 1
                else:
                    # entry格式不对，添加零向量
                    vectors.append([0.0] * CLIP_DIM)
                    total_instances += 1
                    
            except Exception as e:
                # 单个实例失败，用零向量代替
                vectors.append([0.0] * CLIP_DIM)
                total_instances += 1
        
        out[key] = vectors
    
    return out, total_instances


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build instance_clip_vectors.json from instance_retrieval + rgb for each scene."
    )
    parser.add_argument(
        "--scenes_root",
        type=str,
        default="../GaussianConstruction/configs/data/habitat_val",
        help="Root dir of scenes; each subdir has instance_retrieval.json and rgb/frame_*.png.",
    )
    parser.add_argument(
        "--scenes_file",
        type=str,
        default=None,
        help="Optional. One scene dir name per line; only process these scenes.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=CLIP_MODEL_NAME,
        help="OpenCLIP model name",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Print per-scene information")
    parser.add_argument(
        "--print_scene",
        action="store_true",
        help="在开始处理每个场景时打印场景序号/名称（便于定位进度）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="即使已有 instance_clip_vectors.json 也重新计算并覆盖",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not args.scenes_root or not os.path.isdir(args.scenes_root):
        print("Error: --scenes_root 未指定或目录不存在，请传入有效路径。")
        print("  其下每个子目录为一场景，需包含 instance_retrieval.json 和 rgb/frame_*.png")
        return
    
    # 获取场景列表
    if args.scenes_file and os.path.isfile(args.scenes_file):
        with open(args.scenes_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        scenes_list = [os.path.join(args.scenes_root, n) for n in names]
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"从 --scenes_file 加载 {len(names)} 个场景，有效 {len(scenes_list)} 个")
    else:
        scenes_list = glob(os.path.join(args.scenes_root, "*"))
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"找到 {len(scenes_list)} 个场景")
    
    # 初始化编码器
    try:
        encoder = CLIPEncoder(
            model_name=args.clip_model,
            device=args.device
        )
    except Exception as e:
        print(f"Error: 初始化CLIP编码器失败: {e}")
        return
    
    # 处理每个场景
    processed_count = 0
    skipped_resume = 0
    total_instances_all = 0
    failed_scenes = []
    
    total_scenes = len(scenes_list)
    for scene_idx, scene in enumerate(tqdm(scenes_list, desc="Processing scenes", total=total_scenes)):
        try:
            if args.print_scene:
                print(f"[{scene_idx + 1}/{total_scenes}] {os.path.basename(scene)}")

            # 检查是否有instance_retrieval.json
            ir_path = os.path.join(scene, "instance_retrieval.json")
            if not os.path.isfile(ir_path):
                if args.verbose:
                    print(f"  跳过 {os.path.basename(scene)}: 无 instance_retrieval.json")
                continue

            out_path = os.path.join(scene, "instance_clip_vectors.json")
            if not args.no_resume and _instance_clip_vectors_valid(out_path):
                skipped_resume += 1
                if args.verbose:
                    print(f"  断点续跑跳过 {os.path.basename(scene)}: 已有 {out_path}")
                continue
            
            # 处理场景
            out, num_instances = process_scene(scene, encoder, verbose=args.verbose)
            
            # 只有在成功加载并处理后才写入
            if out is not None:
                # 写入前检查：如果输出文件已存在，先备份？或者直接覆盖（与原脚本一致）
                with open(out_path, "w") as f:
                    json.dump(out, f)
                
                processed_count += 1
                total_instances_all += num_instances
                
                if args.verbose and num_instances > 0:
                    print(f"  场景 {os.path.basename(scene)}: {num_instances} 个实例向量")
            else:
                failed_scenes.append(os.path.basename(scene))
                if args.verbose:
                    print(f"  跳过 {os.path.basename(scene)}: 加载 instance_retrieval.json 失败")
            
        except Exception as e:
            failed_scenes.append(os.path.basename(scene))
            if args.verbose:
                print(f"  错误：处理 {os.path.basename(scene)} 时出错: {e}")
            continue
    
    # 打印汇总信息
    print(f"\n处理完成！")
    print(f"  成功处理: {processed_count} 个场景")
    if skipped_resume:
        print(f"  断点跳过（已有 instance_clip_vectors.json）: {skipped_resume} 个场景")
    print(f"  总实例向量: {total_instances_all} 个")
    if failed_scenes:
        print(f"  失败场景: {len(failed_scenes)} 个")
        if args.verbose:
            for scene in failed_scenes[:10]:  # 只显示前10个失败场景
                print(f"    - {scene}")
            if len(failed_scenes) > 10:
                print(f"    ... 还有 {len(failed_scenes) - 10} 个")
    print("Done.")
    
    # 非零退出码表示有失败
    if failed_scenes:
        sys.exit(1)


if __name__ == "__main__":
    main()