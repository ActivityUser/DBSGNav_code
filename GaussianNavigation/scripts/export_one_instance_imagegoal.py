#!/usr/bin/env python3
"""
Export one evaluation input target image (instance_imagegoal) to PNG.

This script is intentionally standalone and does not modify training/eval code paths.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export one Instance-ImageNav target image (instance_imagegoal)."
    )
    parser.add_argument(
        "--config",
        default="configs/instance_imagenav.yaml",
        help="Path to GaussNav config yaml.",
    )
    parser.add_argument(
        "--output",
        default="experiments/debug/instance_imagegoal_sample.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--meta-output",
        default="",
        help="Optional metadata json output path. Default: <output>.json",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode to export (0-based).",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Optional single scene name filter, e.g. CrMo8WxCyVb.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic episode iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import habitat
    from habitat import make_dataset
    from habitat.config.default import get_config

    cfg = get_config(args.config)
    with habitat.config.read_write(cfg):
        split = str(getattr(cfg.end2end_imagenav, "split", cfg.habitat.dataset.split))
        cfg.habitat.dataset.split = split
        cfg.habitat.seed = int(args.seed)
        cfg.habitat.environment.iterator_options.shuffle = False
        if args.scene.strip():
            cfg.habitat.dataset.content_scenes = [args.scene.strip()]

    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    if len(dataset.episodes) == 0:
        raise RuntimeError("No episodes found. Check split/content_scenes/config path.")

    if args.episode_index < 0 or args.episode_index >= len(dataset.episodes):
        raise IndexError(
            f"episode-index out of range: {args.episode_index}, total episodes: {len(dataset.episodes)}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_output) if args.meta_output else out_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with habitat.Env(config=cfg.habitat, dataset=dataset) as env:
        # Move to requested episode by resetting sequentially.
        obs = None
        for _ in range(args.episode_index + 1):
            obs = env.reset()

        if obs is None:
            raise RuntimeError("Failed to reset environment.")
        if "instance_imagegoal" not in obs:
            raise KeyError(
                "Observation does not contain 'instance_imagegoal'. "
                "Ensure instance_imagegoal sensor is enabled in task config."
            )

        goal_img = obs["instance_imagegoal"]
        goal_img = np.asarray(goal_img)
        if goal_img.dtype != np.uint8:
            goal_img = np.clip(goal_img, 0, 255).astype(np.uint8)

        imageio.imwrite(out_path, goal_img)

        ep = env.current_episode
        meta = {
            "config": args.config,
            "split": cfg.habitat.dataset.split,
            "episode_index": int(args.episode_index),
            "episode_id": getattr(ep, "episode_id", ""),
            "scene_id": getattr(ep, "scene_id", ""),
            "object_category": getattr(ep, "object_category", ""),
            "goal_object_id": int(getattr(ep, "goal_object_id", -1)),
            "goal_image_id": int(getattr(ep, "goal_image_id", -1)),
            "image_shape": list(goal_img.shape),
            "output_image": str(out_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] saved image: {out_path}")
    print(f"[ok] saved meta: {meta_path}")


if __name__ == "__main__":
    main()
