#!/usr/bin/env python3
"""
根据 seg_semantic/*.npy 与 depth/*.png 生成 instance_retrieval.json
保持与原脚本完全相同的输入输出接口，但内部实现优化
"""

import sys
import os.path as osp
import torch
import numpy as np
import cv2
from glob import glob
import os
from natsort import natsorted
from tqdm import tqdm
import skimage
import skimage.morphology
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 颜色映射（保持不变）
color_mapping = {
    56: [0.9400000000000001, 0.7818, 0.66],
    57: [0.8882000000000001, 0.9400000000000001, 0.66],
    58: [0.66, 0.9400000000000001, 0.8518000000000001],
    59: [0.7117999999999999, 0.66, 0.9400000000000001],
    61: [0.9218, 0.66, 0.9400000000000001],
    62: [0.9400000000000001, 0.66, 0.748199999999999]
}


def preprocess_mask(depth: np.ndarray, mask: Dict, conf: float = 0.5) -> Dict:
    """
    预处理掩码，与原函数完全相同的功能，但更健壮
    
    Args:
        depth: 深度图
        mask: 掩码字典
        conf: 置信度阈值
    
    Returns:
        处理后的掩码字典
    """
    new_mask = {}
    
    for key in mask:
        if mask[key] is None:
            new_mask[key] = None
            continue
            
        new_curr_mask_list = []
        curr_mask_list = mask[key]
        
        for curr_mask in curr_mask_list:
            if curr_mask is None or not np.any(curr_mask):
                continue
                
            try:
                # 形态学腐蚀
                selem = skimage.morphology.disk(3)
                eroded_mask = skimage.morphology.erosion(curr_mask, selem)
                
                if not np.any(eroded_mask):
                    continue
                
                # 基于深度一致性过滤
                new_depth = depth.copy()
                new_masked_depth = new_depth[eroded_mask]
                
                if len(new_masked_depth) == 0:
                    continue
                
                mean = np.mean(new_masked_depth)
                std = np.std(new_masked_depth)
                
                threshold = conf * std
                curr_mask_copy = curr_mask.copy()
                
                # 只保留深度值在合理范围内的像素
                depth_diff = np.abs(depth - mean)
                curr_mask_filtered = np.where(depth_diff > threshold, False, curr_mask)
                curr_mask_filtered = np.logical_and(curr_mask_copy, curr_mask_filtered)
                
                if np.any(curr_mask_filtered):
                    new_curr_mask_list.append(curr_mask_filtered)
                    
            except Exception as e:
                print(f"警告：预处理掩码时出错: {e}")
                continue
        
        new_mask[key] = new_curr_mask_list if len(new_curr_mask_list) > 0 else None
    
    return new_mask


def get_pointcloud(mask: Dict, depth: np.ndarray, w2c: torch.Tensor, 
                   transform_pts: bool = True, return_bbox: bool = True, 
                   depth_scale: float = 1000.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    从深度图和掩码生成点云
    
    Args:
        mask: 掩码字典
        depth: 深度图
        w2c: 相机到世界的变换矩阵
        transform_pts: 是否转换点云到世界坐标系
        return_bbox: 是否返回包围盒
        depth_scale: 深度缩放因子
    
    Returns:
        pts: 点云坐标
        cols: 点云颜色
        instance_bbox: 实例包围盒字典
    """
    # 确保depth是tensor
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth.astype(float)).cuda()
    
    # 相机内参（从原脚本保持不变）
    width, height = depth.shape[1], depth.shape[0]
    depth = depth / depth_scale
    CX = 239.5
    CY = 319.5
    FX = 625.2213755265124
    FY = 625.2213755265124
    
    # 生成像素坐标网格
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).cuda().float(),
        torch.arange(height).cuda().float(),
        indexing='xy'
    )
    
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)
    
    # 计算相机坐标系下的点
    pts_cam = torch.stack((xx * depth_z, -yy * depth_z, -depth_z), dim=-1)
    
    if transform_pts:
        # Habitat坐标转换矩阵
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ], dtype=float
        ).cuda()
        
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        
        # 转换到世界坐标系
        pts = (w2c @ pts4.T).T[:, :3]
    else:
        pts = pts_cam
    
    # 初始化颜色和包围盒
    cols = torch.ones((width * height, 3), device='cuda:0', dtype=float) * 0.9
    instance_bbox = {}
    
    # 为每个实例上色并计算包围盒
    for key in mask:
        if mask[key] is not None:
            curr_bbox_list = []
            
            for curr_mask in mask[key]:
                if curr_mask is None or not np.any(curr_mask):
                    continue
                    
                curr_cols = torch.from_numpy(curr_mask).cuda()
                curr_cols = curr_cols.reshape(-1)
                
                # 上色
                cols[curr_cols] = torch.tensor(color_mapping[key], dtype=float).cuda()
                
                # 计算包围盒
                if return_bbox:
                    masked_x = torch.masked_select(pts[:, 0], curr_cols)
                    masked_y = torch.masked_select(pts[:, 1], curr_cols)
                    masked_z = torch.masked_select(pts[:, 2], curr_cols)
                    
                    if len(masked_x) > 0:
                        min_x = torch.min(masked_x).item()
                        max_x = torch.max(masked_x).item()
                        min_y = torch.min(masked_y).item()
                        max_y = torch.max(masked_y).item()
                        min_z = torch.min(masked_z).item()
                        max_z = torch.max(masked_z).item()
                        curr_bbox_list.append([min_x, max_x, min_y, max_y, min_z, max_z])
            
            instance_bbox[key] = curr_bbox_list if curr_bbox_list else None
        else:
            instance_bbox[key] = None
    
    return pts, cols, instance_bbox


def voxelize(pointcloud: torch.Tensor, voxel_size: float = 0.05) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    体素化点云
    
    Args:
        pointcloud: 点云 (N, 3)
        voxel_size: 体素大小
    
    Returns:
        voxel_grid: 体素网格
        xyz_min_bounds: 最小边界
        xyz_max_bounds: 最大边界
    """
    xyz_min_bounds = pointcloud.min(0)[0]
    xyz_max_bounds = pointcloud.max(0)[0]
    
    grid_sizes = ((xyz_max_bounds - xyz_min_bounds) / voxel_size).long() + 1
    voxel_grid = torch.zeros(tuple(grid_sizes.tolist())).cuda().float()
    
    indices = (pointcloud - xyz_min_bounds) / voxel_size
    indices = indices.long()
    
    # 确保索引在有效范围内
    for i in range(3):
        indices[:, i] = torch.clamp(indices[:, i], 0, grid_sizes[i] - 1)
    
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] += 1
    voxel_grid = torch.clamp(voxel_grid, min=0, max=1)
    
    return voxel_grid, xyz_min_bounds.cpu().numpy(), xyz_max_bounds.cpu().numpy()


def find_navigable_viewpoints(instance_center: List[float], 
                             voxel_grid: torch.Tensor,
                             voxel_min_bounds: np.ndarray,
                             voxel_size: float,
                             camera_height: float,
                             max_radius: int = 40) -> List[List[float]]:
    """
    寻找实例周围的可导航视角点
    
    Args:
        instance_center: 实例中心 [x, z]
        voxel_grid: 体素网格
        voxel_min_bounds: 体素网格最小边界
        voxel_size: 体素大小
        camera_height: 相机高度
        max_radius: 最大搜索半径
    
    Returns:
        可导航视角点列表
    """
    navigable_viewpoints = []
    
    # 计算相机高度对应的体素范围
    min_z = int((camera_height - 0.4 - voxel_min_bounds[1]) / voxel_size)
    max_z = int((camera_height + 0.2 - voxel_min_bounds[1]) / voxel_size)
    
    # 确保索引有效
    min_z = max(0, min_z)
    max_z = min(voxel_grid.shape[1] - 1, max_z)
    
    if min_z >= max_z:
        return navigable_viewpoints
    
    # 创建2D可通行地图
    grid_map_3d = voxel_grid[:, min_z:max_z, :]
    grid_map = torch.sum(grid_map_3d, dim=1)
    grid_map[grid_map > 0] = 1
    grid_map = grid_map.cpu().numpy()
    
    # 形态学膨胀
    selem = skimage.morphology.disk(3)
    dilated_grid_map = skimage.morphology.dilation(grid_map, selem)
    
    # 将实例中心转换到地图坐标
    map_center_x = int((instance_center[0] - voxel_min_bounds[0]) / voxel_size)
    map_center_z = int((instance_center[1] - voxel_min_bounds[2]) / voxel_size)
    
    h, w = dilated_grid_map.shape
    if not (0 <= map_center_x < h and 0 <= map_center_z < w):
        return navigable_viewpoints
    
    # 从近到远搜索可导航点
    for r in range(3, max_radius):
        try:
            # 生成圆形区域
            navigable_map = np.zeros_like(grid_map)
            rr, cc = skimage.draw.ellipse(map_center_x, map_center_z, r, r)
            
            # 只保留在地图范围内的点
            valid_idx = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            navigable_map[rr[valid_idx], cc[valid_idx]] = 1
            
            # 找到圆形区域内且可通行的点
            temp_map = dilated_grid_map - navigable_map
            
            if np.any(temp_map == -1):
                navigable_y, navigable_x = np.where(temp_map == -1)
                
                # 转换回世界坐标
                world_x = navigable_y[0] * voxel_size + voxel_min_bounds[0]
                world_z = navigable_x[0] * voxel_size + voxel_min_bounds[2]
                navigable_viewpoints.append([float(world_x), float(world_z)])
                break
                
        except Exception as e:
            continue
    
    return navigable_viewpoints


def process_scene(scene_path: str) -> Dict:
    """
    处理单个场景
    
    Args:
        scene_path: 场景路径
    
    Returns:
        实例检索数据字典
    """
    json_dump_data = {56: [], 57: [], 58: [], 59: [], 61: [], 62: []}
    
    # 加载transforms.json
    transforms_path = os.path.join(scene_path, 'transforms.json')
    if not os.path.exists(transforms_path):
        print(f"警告：{transforms_path} 不存在")
        return json_dump_data
    
    with open(transforms_path, 'r') as f:
        frames = json.load(f)['frames']
    
    # 加载所有相机位姿
    all_w2cs = []
    for i in range(len(frames)):
        try:
            w2c = torch.tensor(frames[i]['transform_matrix'], dtype=float).cuda()
            all_w2cs.append(w2c)
        except:
            all_w2cs.append(None)
    
    # 获取所有分割文件
    seg_masks_list = glob(os.path.join(scene_path, 'seg_semantic', "*.npy"))
    seg_masks_list = natsorted(seg_masks_list)
    
    if not seg_masks_list:
        print(f"警告：{scene_path} 中没有分割文件")
        return json_dump_data
    
    # 处理每一帧
    for seg_mask_path in tqdm(seg_masks_list, desc=f"处理 {os.path.basename(scene_path)}"):
        try:
            # 解析帧索引
            filename = os.path.basename(seg_mask_path)
            index = filename.split('_')[-1].split('.')[0]
            
            # 加载掩码
            mask_data = np.load(seg_mask_path, allow_pickle=True).item()
            
            # 加载深度图
            depth_file_path = os.path.join(scene_path, 'depth', f'{index}.png')
            if not os.path.exists(depth_file_path):
                continue
                
            depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            if depth_data is None:
                continue
            
            # 获取相机位姿
            try:
                w2c = all_w2cs[int(index)]
                if w2c is None:
                    continue
            except (IndexError, ValueError):
                continue
            
            # 预处理掩码
            mask_data = preprocess_mask(depth_data, mask_data)
            
            # 生成点云和包围盒
            pts, cols, bbox = get_pointcloud(mask_data, depth_data, w2c)
            
            # 体素化
            voxel_size = 0.05
            voxel_grid, voxel_min_bounds, voxel_max_bounds = voxelize(pts, voxel_size)
            
            # 相机高度
            camera_height = w2c[1, 3].item()
            
            # 处理每个实例
            for key in bbox:
                if bbox[key] is not None:
                    navigable_viewpoint = []
                    
                    for bbox_i in range(len(bbox[key])):
                        curr_bbox = bbox[key][bbox_i]
                        
                        # 计算实例中心
                        curr_center = [
                            (curr_bbox[0] + curr_bbox[1]) / 2,  # x
                            (curr_bbox[4] + curr_bbox[5]) / 2   # z
                        ]
                        
                        # 寻找可导航视角点
                        viewpoints = find_navigable_viewpoints(
                            curr_center, voxel_grid, voxel_min_bounds, 
                            voxel_size, camera_height
                        )
                        
                        if viewpoints:
                            navigable_viewpoint.append(viewpoints[0])  # 只取最近的一个
                        else:
                            navigable_viewpoint.append(None)
                    
                    json_dump_data[key].append([bbox[key], index, navigable_viewpoint])
                    
        except Exception as e:
            print(f"警告：处理 {seg_mask_path} 时出错: {e}")
            continue
    
    return json_dump_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 seg_semantic/*.npy 与 depth/*.png 生成 instance_retrieval.json"
    )
    parser.add_argument(
        "--scenes_root",
        type=str,
        default="../GaussianConstruction/configs/data/habitat",
        help="场景根目录，其下每子目录为一个 scene，包含 depth/、seg_semantic/、transforms.json。",
    )
    parser.add_argument(
        "--scenes_file",
        type=str,
        default=None,
        help="可选。每行一个 scene 目录名，只处理这些场景；不传则处理 scenes_root 下全部子目录。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入
    if not args.scenes_root or not os.path.isdir(args.scenes_root):
        print("Error: --scenes_root 未指定或目录不存在，请传入有效路径。")
        sys.exit(1)
    
    # 获取场景列表
    if args.scenes_file and os.path.isfile(args.scenes_file):
        with open(args.scenes_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        scenes_list = [os.path.join(args.scenes_root, n) for n in names]
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"从 --scenes_file 加载 {len(names)} 个场景，有效 {len(scenes_list)} 个")
    else:
        scenes_list = glob(os.path.join(args.scenes_root, '*'))
        scenes_list = [d for d in scenes_list if os.path.isdir(d)]
        print(f"找到 {len(scenes_list)} 个场景")
    
    # 处理每个场景
    for scene in tqdm(scenes_list, desc="处理场景"):
        try:
            print(f'\n********** {os.path.basename(scene)} **********')
            
            # 处理场景
            json_dump_data = process_scene(scene)
            
            # 保存结果
            json_file_path = os.path.join(scene, 'instance_retrieval.json')
            with open(json_file_path, 'w') as json_file:
                json.dump(json_dump_data, json_file)
                
            # 统计信息
            total_instances = sum(len(v) for v in json_dump_data.values())
            print(f"完成：{total_instances} 个实例")
            
        except Exception as e:
            print(f"错误：处理 {scene} 失败")
            traceback.print_exc()
            continue
    
    print("\n所有场景处理完成！")


if __name__ == "__main__":
    main()