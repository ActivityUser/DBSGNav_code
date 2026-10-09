import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import numpy as np

# ============ 修复matplotlib导入 ============
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
except ImportError as e:
    print(f"警告: matplotlib导入失败: {e}")
    # 创建虚拟的matplotlib模块
    class FakeMatplotlib:
        class cm:
            @staticmethod
            def get_cmap(name='jet'):
                # 简单的颜色映射替代
                class FakeCmap:
                    def __call__(self, data):
                        if isinstance(data, torch.Tensor):
                            data = data.cpu().numpy()
                        # 简单的线性颜色映射
                        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
                        # 模拟jet颜色映射
                        colors = np.zeros((len(data_norm), 3))
                        # 简单实现：蓝->青->黄->红
                        mask1 = data_norm < 0.25
                        mask2 = (data_norm >= 0.25) & (data_norm < 0.5)
                        mask3 = (data_norm >= 0.5) & (data_norm < 0.75)
                        mask4 = data_norm >= 0.75
                        
                        colors[mask1, 2] = 1.0  # 蓝色
                        colors[mask1, 1] = data_norm[mask1] * 4
                        
                        colors[mask2, 1] = 1.0  # 青色
                        colors[mask2, 2] = 1.0 - (data_norm[mask2] - 0.25) * 4
                        
                        colors[mask3, 0] = (data_norm[mask3] - 0.5) * 4  # 黄色
                        colors[mask3, 1] = 1.0
                        
                        colors[mask4, 0] = 1.0  # 红色
                        colors[mask4, 1] = 1.0 - (data_norm[mask4] - 0.75) * 4
                        
                        return colors
                return FakeCmap()
        
        class Normalize:
            def __init__(self, vmin=0, vmax=1):
                self.vmin = vmin
                self.vmax = vmax
        
        class ScalarMappable:
            def __init__(self, norm=None, cmap=None):
                self.norm = norm
                self.cmap = cmap or 'jet'
            
            def to_rgba(self, data):
                cmap_func = FakeMatplotlib.cm.get_cmap(self.cmap)
                if self.norm:
                    data_norm = np.clip((data - self.norm.vmin) / (self.norm.vmax - self.norm.vmin), 0, 1)
                else:
                    data_norm = data
                result = cmap_func(data_norm)
                # 添加alpha通道
                if result.shape[1] == 3:
                    result = np.concatenate([result, np.ones((result.shape[0], 1))], axis=1)
                return result
    
    plt = type('plt', (), {})()
    plt.cm = FakeMatplotlib.cm
    plt.Normalize = FakeMatplotlib.Normalize

# ============ 修复open3d导入 ============
try:
    import open3d as o3d
except ImportError as e:
    print(f"错误: open3d导入失败: {e}")
    print("请安装open3d: pip install open3d")
    sys.exit(1)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation

def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k

def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    color_mapping = {
        0: [0.9400000000000001, 0.7818, 0.66],
        1: [0.8882000000000001, 0.9400000000000001, 0.66],
        2: [0.66, 0.9400000000000001, 0.8518000000000001],
        3: [0.7117999999999999, 0.66, 0.9400000000000001],
        4: [0.9218, 0.66, 0.9400000000000001],
        5: [0.9400000000000001, 0.66, 0.748199999999999]
    }
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    # Check if Gaussians are Isotropic or Anisotropic
    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    color_mapping = {k: torch.tensor(v, device='cuda') for k, v in color_mapping.items()}

    params['rgb_colors'] = params['rgb_colors'] * 0.5 + 0.5 * torch.tensor([0., 0., 0.], device='cuda')
    for i in range(6):
        cur_mask = params['labels'] == i+1
        params['rgb_colors'][cur_mask] = 0.5 * params['rgb_colors'][cur_mask] + 0.5 * color_mapping[i]

    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], first_frame_w2c),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    return rendervar, depth_rendervar, all_w2cs

def save_results_headless(scene_path, cfg, output_dir="./headless_viz"):
    """无头模式保存结果"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("3D高斯语义重建 - 无头模式可视化")
    print("=" * 60)
    
    # 加载数据
    w2c, k = load_camera(cfg, scene_path)
    scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, w2c, k)
    
    # 1. 保存高斯点云
    points = scene_data['means3D'].cpu().numpy()
    colors = scene_data['colors_precomp'].cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    
    ply_path = os.path.join(output_dir, "gaussian_points.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✓ 保存高斯点云: {ply_path}")
    print(f"  点数: {len(points)}")
    print(f"  颜色: RGB (已归一化)")
    
    # 2. 保存相机轨迹
    if all_w2cs:
        cam_positions = []
        for w2c_mat in all_w2cs:
            cam_pos = np.linalg.inv(w2c_mat)[:3, 3]
            cam_positions.append(cam_pos)
        
        cam_positions = np.array(cam_positions)
        traj_path = os.path.join(output_dir, "camera_trajectory.txt")
        np.savetxt(traj_path, cam_positions, 
                  header="相机位置 (x y z)\n帧数: {}".format(len(cam_positions)))
        print(f"✓ 保存相机轨迹: {traj_path}")
        print(f"  相机数: {len(cam_positions)}")
    
    # 3. 渲染并保存图像
    try:
        # 渲染第一视角
        im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
        
        # 保存RGB图像
        im_np = im.cpu().numpy().transpose(1, 2, 0)
        im_np = np.clip(im_np, 0, 1)
        
        # 使用PIL保存图像
        from PIL import Image
        img_path = os.path.join(output_dir, "render_rgb.png")
        Image.fromarray((im_np * 255).astype(np.uint8)).save(img_path)
        print(f"✓ 保存RGB渲染: {img_path}")
        
        # 保存深度图
        depth_np = depth.cpu().numpy()[0]
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        depth_path = os.path.join(output_dir, "render_depth.png")
        Image.fromarray((depth_norm * 255).astype(np.uint8)).save(depth_path)
        print(f"✓ 保存深度图: {depth_path}")
        
    except Exception as e:
        print(f"⚠ 渲染时出错: {e}")
    
    # 4. 创建HTML查看器
    html_path = os.path.join(output_dir, "viewer.html")
    with open(html_path, 'w') as f:
        f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>3D高斯语义重建 - {os.path.basename(scene_path)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                margin-bottom: 20px; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-list li {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .file-list li:last-child {{ border-bottom: none; }}
        .preview {{ max-width: 100%; border-radius: 5px; }}
        .stats {{ font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 3D高斯语义重建结果</h1>
            <p>生成时间: {np.datetime64('now')}</p>
        </div>
        
        <div class="card">
            <h2>📊 重建统计</h2>
            <div class="stats">
                <p>• 高斯点数量: {len(points):,}</p>
                <p>• 相机轨迹帧数: {len(all_w2cs) if all_w2cs else 0}</p>
                <p>• 点云范围: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] 
                               Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] 
                               Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]</p>
                <p>• 数据文件: {os.path.basename(scene_path)}</p>
            </div>
        </div>
        
        <div class="card">
            <h2>📁 生成文件</h2>
            <ul class="file-list">
                <li>📦 <a href="gaussian_points.ply" download>gaussian_points.ply</a> - 3D点云(PLY格式)</li>
                <li>📊 <a href="camera_trajectory.txt" download>camera_trajectory.txt</a> - 相机轨迹</li>
''')
        if os.path.exists(os.path.join(output_dir, "render_rgb.png")):
            f.write('''                <li>🖼️ <a href="render_rgb.png">render_rgb.png</a> - RGB渲染图</li>
                <li>📐 <a href="render_depth.png">render_depth.png</a> - 深度图</li>
''')
        f.write('''            </ul>
        </div>
        
        <div class="card">
            <h2>🔄 使用方法</h2>
            <p>1. 下载 <code>gaussian_points.ply</code> 文件</p>
            <p>2. 使用以下工具查看：</p>
            <ul>
                <li><a href="https://www.meshlab.net/" target="_blank">MeshLab</a> (免费开源)</li>
                <li><a href="https://www.blender.org/" target="_blank">Blender</a> (导入PLY文件)</li>
                <li><a href="https://github.com/facebookresearch/pytorch3d" target="_blank">PyTorch3D</a></li>
                <li>任何支持PLY格式的3D查看器</li>
            </ul>
        </div>
    </div>
</body>
</html>''')
    
    print(f"✓ 创建HTML查看器: {html_path}")
    print(f"\n✅ 所有文件已保存到: {os.path.abspath(output_dir)}")
    print(f"📁 文件列表:")
    for file in os.listdir(output_dir):
        filepath = os.path.join(output_dir, file)
        size = os.path.getsize(filepath)
        print(f"   • {file} ({size:,} bytes)")

# 保留原visualize函数，但添加错误处理
def visualize(scene_path, cfg):
    try:
        # 尝试使用原visualize函数
        from viz_scripts.final_recon_sem import visualize as original_visualize
        print("尝试使用GUI模式...")
        original_visualize(scene_path, cfg)
    except Exception as e:
        print(f"GUI模式失败: {e}")
        print("切换到无头模式...")
        save_results_headless(scene_path, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--headless", action="store_true", help="强制使用无头模式")
    parser.add_argument("--output", type=str, default="./viz_results", help="输出目录")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    if args.headless:
        # 强制无头模式
        save_results_headless(scene_path, viz_cfg, args.output)
    else:
        # 尝试GUI，失败则自动切换到无头模式
        visualize(scene_path, viz_cfg)
