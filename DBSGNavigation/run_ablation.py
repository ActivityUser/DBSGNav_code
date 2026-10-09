#!/usr/bin/env python3
"""
消融实验运行入口

支持：
  1. 全部六个创新点（full）
  2. 基线（baseline，全部关闭）
  3. 单独各创新点（innov2, innov3, innov4, innov5, innov6）

创新点1（开放词汇分割）在建图/预处理阶段，需先运行：
  python gaussian_utils/seg_train_imgs.py --config configs/seg_open_vocab.yaml

用法：
  cd DBSGNavigation
  python run_ablation.py --preset full
  python run_ablation.py --preset innov2
  python run_ablation.py --preset innov3
  ...
  python run_ablation.py --config configs/instance_imagenav.yaml   # 自定义配置文件
"""

from __future__ import annotations

import argparse
import os
import sys

# 确保从 DBSGNavigation 根目录运行
_NAV_ROOT = os.path.dirname(os.path.abspath(__file__))
if _NAV_ROOT != os.getcwd():
    os.chdir(_NAV_ROOT)
    sys.path.insert(0, _NAV_ROOT)

PRESETS = {
    "full": "全部六个创新点",
    "baseline": "基线（全部关闭）",
    "innov2": "仅创新点2（语义嵌入高斯）",
    "innov3": "仅创新点3（跨模态检索，几何优先）",
    "innov4": "仅创新点3+4（跨模态检索+语义几何融合）",
    "innov5": "创新点5（开放词汇基准）",
    "innov6": "创新点6（动态语义更新）",
}

CONFIG_MAP = {
    "full": "configs/instance_imagenav_ablation_full.yaml",
    "baseline": "configs/instance_imagenav_ablation_baseline.yaml",
    "innov2": "configs/instance_imagenav_ablation_innov2.yaml",
    "innov3": "configs/instance_imagenav_ablation_innov3.yaml",
    "innov4": "configs/instance_imagenav_ablation_innov4.yaml",
    "innov5": "configs/instance_imagenav_ablation_innov5.yaml",
    "innov6": "configs/instance_imagenav_ablation_innov6.yaml",
}


def main():
    parser = argparse.ArgumentParser(
        description="消融实验：按预设或自定义配置运行 GaussNav 导航评测"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="预设：full=全开, baseline=全关, innov2~6=单创新点",
    )
    group.add_argument(
        "--config",
        type=str,
        help="自定义配置文件路径（如 configs/instance_imagenav.yaml）",
    )
    args = parser.parse_args()

    if args.preset:
        cfg_path = os.path.join(_NAV_ROOT, CONFIG_MAP[args.preset])
        print(f"[消融] preset={args.preset} ({PRESETS[args.preset]})")
        print(f"[消融] config={cfg_path}")
    else:
        cfg_path = args.config
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(_NAV_ROOT, cfg_path)
        if not os.path.isfile(cfg_path):
            parser.error(f"配置文件不存在: {cfg_path}")
        print(f"[消融] config={cfg_path}")

    from runners.instance_imagenav_runner import Runner

    runner = Runner(cfg_pth=cfg_path)
    runner.train()


if __name__ == "__main__":
    main()
