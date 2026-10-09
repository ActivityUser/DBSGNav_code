#!/usr/bin/env python3
"""
Fixed version: Precompute per-instance CLIP vectors for OpenVocabRetriever.
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

logging.basicConfig(level=logging.INFO, format='%(message)s')

COCO_IDS = [56, 57, 58, 59, 61, 62]
CLIP_DIM = 512
CLIP_MODEL_NAME = "ViT-B-32"


class CLIPEncoder:
    """CLIP图像编码器"""
    
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.model_name = model_name
        self.model_loaded = False
        self.total_processed = 0
        self.total_failed = 0
    
    def _load_model(self):
        if self.model_loaded:
            return
        try:
            import open_clip
            print(f"正在加载CLIP模型 {self.model_name} 到设备 {self.device}...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai"
            )
            self.model = self.model.to(self.device).eval()
            self.model_loaded = True
            print("CLIP模型加载成功")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def encode_image(self, rgb_path: str) -> Optional[List[float]]:
        """编码单张图像"""
        try:
            self._load_model()
            
            if not os.path.exists(rgb_path):
                self.total_failed += 1
                return None
            
            # 使用PIL读取（已经测试过可以工作）
            pil_img = Image.open(rgb_path).convert('RGB')
            
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
            
            self.total_processed += 1
            return feat.tolist()
            
        except Exception as e:
            self.total_failed += 1
            return None


def get_frame_path(scene_path: str, frame_id) -> str:
    """
    根据frame_id获取正确的图像路径
    """
    rgb_dir = os.path.join(scene_path, "rgb")
    
    # 如果frame_id已经是完整的文件名
    if isinstance(frame_id, str) and frame_id.startswith("frame_") and frame_id.endswith(".png"):
        return os.path.join(rgb_dir, frame_id)
    
    # 将frame_id转换为字符串
    frame_str = str(frame_id)
    
    # 尝试不同的格式
    possible_paths = [
        os.path.join(rgb_dir, f"frame_{frame_str}.png"),  # 原始格式
        os.path.join(rgb_dir, f"frame_{frame_str.zfill(5)}.png"),  # 5位补零
        os.path.join(rgb_dir, f"frame_{frame_str.zfill(6)}.png"),  # 6位补零
    ]
    
    # 如果frame_str本身是数字，也尝试不带frame_前缀的格式
    if frame_str.isdigit():
        possible_paths.extend([
            os.path.join(rgb_dir, f"{frame_str}.png"),
            os.path.join(rgb_dir, f"{frame_str.zfill(5)}.png"),
            os.path.join(rgb_dir, f"{frame_str.zfill(6)}.png"),
        ])
    
    # 返回第一个存在的路径
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果都不存在，返回默认路径（用于错误提示）
    return os.path.join(rgb_dir, f"frame_{frame_str}.png")


def process_scene(scene_path: str, encoder: CLIPEncoder, verbose: bool = False) -> Tuple[Optional[Dict[str, List]], int]:
    """处理单个场景"""
    scene_name = os.path.basename(scene_path)
    
    # 初始化输出字典
    out = {str(coco_id): [] for coco_id in COCO_IDS}
    total_instances = 0
    failed_instances = 0
    
    # 加载instance_retrieval.json
    ir_path = os.path.join(scene_path, "instance_retrieval.json")
    if not os.path.isfile(ir_path):
        if verbose:
            print(f"  场景 {scene_name}: 无 instance_retrieval.json")
        return None, 0
    
    try:
        with open(ir_path, "r") as f:
            instance_retrieval = json.load(f)
    except Exception as e:
        if verbose:
            print(f"  场景 {scene_name}: 加载失败: {e}")
        return None, 0
    
    if verbose:
        print(f"\n处理场景: {scene_name}")
    
    # 检查rgb目录
    rgb_dir = os.path.join(scene_path, "rgb")
    if not os.path.exists(rgb_dir):
        if verbose:
            print(f"  警告: rgb目录不存在 {rgb_dir}")
        # 返回空向量
        for coco_id in COCO_IDS:
            key = str(coco_id)
            if key in instance_retrieval:
                entries = instance_retrieval[key]
                out[key] = [[0.0] * CLIP_DIM for _ in entries]
                total_instances += len(entries)
        return out, total_instances
    
    # 处理每个类别
    for coco_id in COCO_IDS:
        key = str(coco_id)
        
        if key not in instance_retrieval:
            out[key] = []
            continue
        
        entries = instance_retrieval[key]
        vectors = []
        
        if verbose:
            print(f"  类别 {coco_id}: {len(entries)} 个实例")
        
        for idx, entry in enumerate(entries):
            try:
                # 解析frame_id
                frame_id = None
                
                # entry格式可能是 [bbox, frame_id, ...] 或 [bboxes, frame_id, ...]
                if len(entry) >= 2:
                    # 尝试获取frame_id（通常在第二个位置）
                    if isinstance(entry[1], (int, str)):
                        frame_id = entry[1]
                    elif len(entry) > 2 and isinstance(entry[2], (int, str)):
                        frame_id = entry[2]
                    else:
                        # 遍历查找第一个看起来像frame_id的
                        for item in entry:
                            if isinstance(item, (int, str)):
                                frame_id = item
                                break
                
                if frame_id is None:
                    if verbose:
                        print(f"    [警告] 无法解析frame_id: {entry[:3]}...")
                    vectors.append([0.0] * CLIP_DIM)
                    failed_instances += 1
                    total_instances += 1
                    continue
                
                # 获取正确的图像路径
                rgb_path = get_frame_path(scene_path, frame_id)
                
                # 编码图像
                vec = encoder.encode_image(rgb_path)
                
                if vec is None:
                    if verbose:
                        exists = os.path.isfile(rgb_path)
                        print(f"    [失败] frame_id={frame_id}, 路径={rgb_path}, 存在={exists}")
                    vec = [0.0] * CLIP_DIM
                    failed_instances += 1
                else:
                    if verbose and idx < 3:  # 只显示前3个成功案例
                        print(f"    [成功] frame_id={frame_id}, 非零值")
                
                vectors.append(vec)
                total_instances += 1
                
            except Exception as e:
                if verbose:
                    print(f"    [错误] {e}")
                vectors.append([0.0] * CLIP_DIM)
                failed_instances += 1
                total_instances += 1
        
        out[key] = vectors
    
    if verbose:
        print(f"  场景 {scene_name}: 总计 {total_instances} 实例，失败 {failed_instances}")
        print(f"  编码统计: 成功 {encoder.total_processed}, 失败 {encoder.total_failed}")
    
    return out, total_instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes_root",
        type=str,
        default="../DBSGConstruction/configs/data/habitat_val",
        help="Root dir of scenes"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Process a single scene (for testing)"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=CLIP_MODEL_NAME,
        help="OpenCLIP model name"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", help="Print detailed info")
    
    args = parser.parse_args()
    
    # 初始化编码器
    encoder = CLIPEncoder(model_name=args.clip_model, device=args.device)
    
    # 如果是单场景测试
    if args.scene:
        scene_path = args.scene
        if not os.path.isabs(scene_path):
            scene_path = os.path.join(args.scenes_root, scene_path)
        
        if not os.path.isdir(scene_path):
            print(f"场景不存在: {scene_path}")
            return
        
        print(f"\n=== 处理单个场景: {scene_path} ===")
        out, count = process_scene(scene_path, encoder, verbose=True)
        
        if out:
            # 保存结果
            out_path = os.path.join(scene_path, "instance_clip_vectors.json")
            with open(out_path, "w") as f:
                json.dump(out, f)
            print(f"\n结果已保存到: {out_path}")
            
            # 统计非零向量
            total_vecs = 0
            non_zero = 0
            for key in out:
                for vec in out[key]:
                    total_vecs += 1
                    if any(abs(x) > 1e-6 for x in vec):
                        non_zero += 1
            print(f"总向量数: {total_vecs}, 非零向量: {non_zero}")
        return
    
    # 批量处理所有场景
    scenes_list = glob(os.path.join(args.scenes_root, "*"))
    scenes_list = [d for d in scenes_list if os.path.isdir(d)]
    print(f"找到 {len(scenes_list)} 个场景")
    
    processed = 0
    total_instances = 0
    
    for scene in tqdm(scenes_list, desc="Processing scenes"):
        out, count = process_scene(scene, encoder, verbose=False)
        if out:
            out_path = os.path.join(scene, "instance_clip_vectors.json")
            with open(out_path, "w") as f:
                json.dump(out, f)
            processed += 1
            total_instances += count
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed} 个场景")
    print(f"总实例向量: {total_instances} 个")
    print(f"编码统计: 成功 {encoder.total_processed}, 失败 {encoder.total_failed}")


if __name__ == "__main__":
    main()
