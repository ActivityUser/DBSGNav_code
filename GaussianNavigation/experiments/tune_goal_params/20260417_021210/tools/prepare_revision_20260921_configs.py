#!/usr/bin/env python3
"""Generate the reviewer-requested one-factor sensitivity configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
BASE = (
    ROOT
    / "experiments/tune_goal_params/20260417_021210/configs"
    / "ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm/full_scene_crop.yaml"
)
OUT = (
    ROOT
    / "experiments/tune_goal_params/20260417_021210/configs"
    / "revision_20260921_sensitivity"
)

VARIANTS = {
    "control_default": {},
    "wo_sg_geometry": {
        "use_semantic_gaussian_goal": False,
        "semantic_gaussian_api_rerank_enable": False,
    },
    "wo_vlm": {
        "semantic_gaussian_api_rerank_enable": False,
    },
    "wo_dyn": {
        "mapper.dynamic_semantic_planner": False,
        "mapper.enable_dynamic_semantic_update": False,
    },
    "wo_ov": {
        "use_open_vocab_retrieval": False,
    },
    "wo_sg_geom": {
        "use_semantic_gaussian_goal": False,
        "semantic_gaussian_api_rerank_enable": False,
        "open_vocab_semantic_weight": 0.0,
        "fusion_geometric_weight": 0.6,
        "open_vocab_fusion_use_semantic_score": False,
        "open_vocab_fusion_use_geometric_score": True,
    },
    "wo_sg_semgeom": {
        "use_semantic_gaussian_goal": False,
        "semantic_gaussian_api_rerank_enable": False,
        "open_vocab_semantic_weight": 0.8,
        "fusion_geometric_weight": 0.0,
        "open_vocab_fusion_use_semantic_score": True,
        "open_vocab_fusion_use_geometric_score": False,
    },
    "ov_scale_080": {"goal_arbitration_merit_ov_scale": 0.8},
    "ov_scale_120": {"goal_arbitration_merit_ov_scale": 1.2},
    "fusion_sem100_geo040": {
        "open_vocab_semantic_weight": 1.0,
        "fusion_geometric_weight": 0.4,
    },
    "fusion_sem060_geo080": {
        "open_vocab_semantic_weight": 0.6,
        "fusion_geometric_weight": 0.8,
    },
    "topk_3": {"open_vocab_top_k": 3},
    "topk_4": {"open_vocab_top_k": 4},
    "topk_5": {"open_vocab_top_k": 5},
    "topk_7": {"open_vocab_top_k": 7},
    "topk_8": {"open_vocab_top_k": 8},
    "topk_9": {"open_vocab_top_k": 9},
    "sg_margin_010": {"semantic_gaussian_api_rerank_sim_margin": 0.01},
    "sg_margin_050": {"semantic_gaussian_api_rerank_sim_margin": 0.05},
    "corr_thr_040": {"mapper.dynamic_semantic_planner_thr": 0.40},
    "corr_thr_060": {"mapper.dynamic_semantic_planner_thr": 0.60},
}


def set_nested(env: dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    current = env
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base)
        env = cfg["end2end_imagenav"]
        env["goal_arbitration_merit_ov_scale"] = 1.0
        for key, value in overrides.items():
            set_nested(env, key, value)
        header = (
            "# Reviewer revision sensitivity; generated from full_scene_crop.yaml.\n"
            f"# Variant: {name}; overrides: {overrides!r}\n"
        )
        (OUT / f"{name}.yaml").write_text(
            header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
    print(f"generated {len(VARIANTS)} configs in {OUT}")


if __name__ == "__main__":
    main()
