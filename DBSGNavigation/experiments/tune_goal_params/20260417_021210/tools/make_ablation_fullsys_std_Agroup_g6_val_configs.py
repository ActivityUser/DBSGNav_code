#!/usr/bin/env python3
"""从 train 消融配置生成 val 消融配置（字段对齐版）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PRESETS = [
    "full",
    "wo_dyn",
    "wo_bench",
    "wo_ov",
    "wo_sg_semgeom",
    "wo_sg_fusion_g6",
    "wo_sg_geom",
]


def _update_for_val(
    cfg: dict[str, Any],
    base_scene_path: str,
    gaussian_map_root: str,
    max_episodes: int | None,
) -> dict[str, Any]:
    out = dict(cfg)
    habitat = dict(out.get("habitat", {}))
    dataset = dict(habitat.get("dataset", {}))
    dataset["split"] = "val"
    habitat["dataset"] = dataset
    out["habitat"] = habitat

    nav = dict(out.get("end2end_imagenav", {}))
    nav["split"] = "val"
    nav["base_scene_path"] = base_scene_path
    nav["gaussian_map_root"] = gaussian_map_root
    if max_episodes is not None:
        nav["max_episodes"] = int(max_episodes)
    out["end2end_imagenav"] = nav
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--train-config-dir",
        type=Path,
        default=root / "configs/ablation_fullsys_std_Agroup_g6_train",
    )
    ap.add_argument(
        "--val-config-dir",
        type=Path,
        default=root / "configs/ablation_fullsys_std_Agroup_g6_val",
    )
    ap.add_argument(
        "--base-scene-path",
        default="../DBSGConstruction/configs/data/habitat_val",
    )
    ap.add_argument(
        "--gaussian-map-root",
        default="../DBSGConstruction/experiments/habitat_val_final",
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="写入 val 配置的 max_episodes；传负数表示保持原值。",
    )
    args = ap.parse_args()

    val_ep = None if args.max_episodes is not None and args.max_episodes < 0 else args.max_episodes
    args.val_config_dir.mkdir(parents=True, exist_ok=True)

    for name in PRESETS:
        src = args.train_config_dir / f"{name}.yaml"
        if not src.is_file():
            raise FileNotFoundError(f"缺少 train 配置: {src}")
        dst = args.val_config_dir / f"{name}.yaml"

        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg = _update_for_val(
            cfg=cfg,
            base_scene_path=args.base_scene_path,
            gaussian_map_root=args.gaussian_map_root,
            max_episodes=val_ep,
        )
        dst.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[ok] {dst}")

    print(f"\nDone. Generated {len(PRESETS)} val configs in: {args.val_config_dir}")


if __name__ == "__main__":
    main()

