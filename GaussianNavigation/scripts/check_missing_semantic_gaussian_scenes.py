#!/usr/bin/env python3
"""
检查 HM3D val 中哪些场景在 gaussian_map_root 下缺少 Semantic Gaussian（params.npz）。
非 GT 模式下，缺场景的 episode 会 fallback 或 prediction_failed，影响指标。

用法:
  python scripts/check_missing_semantic_gaussian_scenes.py
  python scripts/check_missing_semantic_gaussian_scenes.py --gaussian_map_root /path/to/GaussianConstruction/experiments/habitat_val_final
"""

import argparse
import os
from glob import glob

# HM3D val 场景 ID（与 scene_dataset_config 中 val/008xx-XXX 一致，取最后一节）
HM3D_VAL_SCENE_IDS = [
    "TEEsavR23oF", "wcojb4TFT35", "k1cupFYWXJ6", "y9hTuugGdiq",
    "CrMo8WxCyVb", "svBbv1Pavdk", "p53SfW6mjZe", "h1zeeAwLh9Z",
    "mL8ThkuaVTM", "eF36g7L6Z9M", "7MXmsvcQjpJ", "Dd4bFSTQ8gi",
    "BAbdmeyTvMZ", "QaLdnwvtxbs", "yr17PDCnDDW", "qyAac8rV8Zk",
    "q3zU7Yy5E5s", "zt1RVoi7PcG", "DYehNKdT76V", "q5QZSEeHe5g",
    "bCPU9suPUw9", "ziup5kvtCCR", "a8BtkwhxdRV", "5cdEh9F2hJL",
    "GLAQ4DNUx5U", "LT9Jq6dN3Ea", "MHPLjHsuG27", "VBzV5z6i1WS",
    "bxsVRursffK", "mv2HUxq3B53", "4ok3usBNeis", "XB4GS9ShBRE",
    "Nfvxx8J5NCo", "6s7QHgap2fW", "cvZr5TUy5C5", "HY1NcmCgn3n",
]


def resolve_root(root: str, nav_root: str) -> str:
    """若为相对路径则相对于 GaussianNavigation 目录解析为绝对路径。"""
    r = (root or "").strip()
    if not r:
        return ""
    if os.path.isabs(r):
        return os.path.normpath(r)
    return os.path.normpath(os.path.join(nav_root, r))


def has_params_npz(gaussian_map_root: str, scene_id: str) -> bool:
    """与 SemanticGaussianGoal 一致：存在 {scene_id}* 目录且其下有 params.npz。"""
    if not gaussian_map_root or not os.path.isdir(gaussian_map_root):
        return False
    pattern = os.path.join(gaussian_map_root, f"{scene_id}*")
    for d in glob(pattern):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "params.npz")):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Check missing Semantic Gaussian (params.npz) for HM3D val scenes.")
    parser.add_argument(
        "--gaussian_map_root",
        type=str,
        default=None,
        help="建图根目录，默认从 configs/instance_imagenav_val_nogt.yaml（若不存在则 instance_imagenav.yaml）的 gaussian_map_root 读取。",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    nav_root = os.path.dirname(script_dir)

    if args.gaussian_map_root is not None:
        root = os.path.abspath(args.gaussian_map_root)
    else:
        try:
            import yaml
            root = ""
            for yaml_name in (
                "instance_imagenav_val_nogt.yaml",
                "instance_imagenav.yaml",
            ):
                cfg_path = os.path.join(nav_root, "configs", yaml_name)
                if not os.path.isfile(cfg_path):
                    continue
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                root = (cfg.get("end2end_imagenav") or {}).get("gaussian_map_root") or ""
                if root:
                    root = resolve_root(root, nav_root)
                    break
            if not root:
                raise ValueError("未在配置中找到 gaussian_map_root")
        except Exception as e:
            print(f"读取 config 失败: {e}，请使用 --gaussian_map_root 指定目录。")
            return 1

    print(f"gaussian_map_root: {root}")
    print(f"val 数据集场景数: {len(HM3D_VAL_SCENE_IDS)}")

    have = []
    missing = []
    for sid in HM3D_VAL_SCENE_IDS:
        if has_params_npz(root, sid):
            have.append(sid)
        else:
            missing.append(sid)

    print(f"已有 params.npz 的场景数: {len(have)}")
    print(f"缺失场景数: {len(missing)}")
    if have:
        print("已有场景:", ", ".join(sorted(have)))
    if missing:
        print("缺失场景列表（这些在非 GT 模式下会被跳过或 fallback）:")
        for s in sorted(missing):
            print(f"  {s}")
        print("\n补全方式:")
        print("  建图数据已存在时，在 GaussianConstruction 目录执行:")
        print("  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python scripts/build_missing_val_semantic_gaussians.py")
        print("  或: python scripts/run_build_val_semantic_gaussians.py --data_dir configs/data/habitat_val --group_name habitat_val_final")
        print("  显存不足可加: --max_frames 100 --stride 2")
    else:
        print("所有 val 场景均已有 params.npz。")
    return 0


if __name__ == "__main__":
    exit(main())
