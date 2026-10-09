#!/usr/bin/env python3
"""Run full / wo_ov viz for contrast episodes: full SUCCESS & wo_ov FAIL, same target image.

Reads experiments/case_study_ov/selected_contrast_episodes.json and groups runs by scene.
Each scene run uses max_episodes = max(episode_no) among selected episodes in that scene.
After each mode finishes, copies rank/0/{episode_no-1}/ -> case_study_ov/{mode}/{category}/.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = (
    ROOT
    / "experiments/tune_goal_params/20260417_021210/configs"
    / "ablation_08ext_clip_v2_probe/0.8-0.6_inst_v3_vlm"
)
CASE_ROOT = ROOT / "examples/case_study_ov"
TMP_CFG_DIR = ROOT / "experiments/case_study_ov/_tmp_configs"
SELECTED_JSON = ROOT / "experiments/case_study_ov/selected_contrast_episodes.json"


def load_cfg(stem: str) -> dict:
    with open(CFG_DIR / f"{stem}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_selected(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scene_batches(selected: dict) -> list[dict]:
    by_scene: dict[str, list[dict]] = {}
    for cat, info in selected.items():
        if not info.get("scene") or not info.get("episode_no"):
            continue
        by_scene.setdefault(info["scene"], []).append(
            {
                "category": cat,
                "episode_no": int(info["episode_no"]),
                "scene": info["scene"],
            }
        )
    batches = []
    for scene, items in sorted(by_scene.items()):
        # +1 so the highest selected episode is followed by another reset, which
        # is when the env flushes that episode's trajectory video to disk.
        batches.append(
            {
                "scene": scene,
                "max_episodes": max(x["episode_no"] for x in items) + 1,
                "episodes": {x["episode_no"]: x["category"] for x in items},
            }
        )
    return batches


def _read_goal_info(d: Path) -> dict:
    info = {}
    gi = d / "goal_info.txt"
    if not gi.is_file():
        return info
    for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def find_episode_dir(mode: str, episode_no: int, scene: str) -> Path | None:
    """Locate the viz output dir whose goal_info.txt matches scene + episode_no.

    The dir name normally equals episode_no (1-based), but the video filename
    counter can be off-by-one across code paths, so match on goal_info content.
    """
    rank_dir = CASE_ROOT / mode / "0"
    if not rank_dir.is_dir():
        return None
    # Prefer exact goal_info match.
    for d in sorted(rank_dir.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        info = _read_goal_info(d)
        if (
            info.get("scene") == scene
            and str(info.get("episode_no")) == str(episode_no)
        ):
            return d
    # Fallback: dir literally named episode_no.
    cand = rank_dir / str(episode_no)
    return cand if cand.is_dir() else None


def stage_episode(mode: str, episode_no: int, category: str, scene: str) -> Path | None:
    src = find_episode_dir(mode, episode_no, scene)
    dst = CASE_ROOT / mode / category
    if src is None or not src.is_dir():
        print(f"[warn] missing run output for {mode} {scene} ep={episode_no}")
        return None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    mp4s = list(dst.glob("*-Vis.mp4"))
    (dst / "case_meta.txt").write_text(
        f"mode={mode}\nscene={scene}\nepisode_no={episode_no}\ncategory={category}\n"
        f"src_dir={src.name}\nhas_video={bool(mp4s)}\n",
        encoding="utf-8",
    )
    print(f"[stage] {src} -> {dst} (video={'yes' if mp4s else 'NO'})")
    return dst


def run_batch(
    mode: str,
    scene: str,
    max_episodes: int,
    gpu: str,
    dry_run: bool,
    log_dir: Path,
) -> None:
    base = load_cfg(mode)
    cfg = copy.deepcopy(base)
    cfg["habitat"]["dataset"]["content_scenes"] = [scene]
    # Absolute path: env workers may run with a different cwd (e.g. the _online
    # checkout), so a relative dump_location would land in the wrong tree.
    cfg["end2end_imagenav"]["dump_location"] = str((CASE_ROOT / mode).resolve())
    cfg["end2end_imagenav"]["viz_params"]["viz"] = True
    cfg["end2end_imagenav"]["viz_params"]["viz_failure_case"] = False
    cfg["end2end_imagenav"]["viz_params"]["viz_save_goal_assets"] = True

    tmp_cfg = TMP_CFG_DIR / f"{mode}_{scene}_{max_episodes}.yaml"
    write_cfg(cfg, tmp_cfg)

    out_rank = CASE_ROOT / mode / "0"
    if out_rank.exists():
        shutil.rmtree(out_rank)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_{scene}_{max_episodes}ep.log"

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run_val.py"),
        "--config",
        str(tmp_cfg.resolve()),
        "--max_episodes",
        str(max_episodes),
        "--log_file",
        str(log_file.resolve()),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("GAUSSNAV_RUN_ENTRY", "run_val.py")
    env.setdefault("MAGNUM_LOG", "quiet")
    env.setdefault("HABITAT_SIM_LOG", "quiet")
    # Force NVIDIA EGL backend for headless rendering (avoids EGL_BAD_PARAMETER).
    env.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    print(f"[run] mode={mode} scene={scene} max_episodes={max_episodes} gpu={gpu}")
    print(f"[run] log -> {log_file}")
    if dry_run:
        print("      ", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--modes", default="full,wo_ov")
    ap.add_argument("--selected", default=str(SELECTED_JSON))
    ap.add_argument(
        "--log-dir",
        default=str(ROOT / "experiments/case_study_ov/logs"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    selected = load_selected(Path(args.selected))
    batches = scene_batches(selected)
    log_dir = Path(args.log_dir)

    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    print("[plan] scene batches:")
    for b in batches:
        print(
            f"  scene={b['scene']} max_ep={b['max_episodes']} "
            f"categories={list(b['episodes'].values())}"
        )

    for mode in modes:
        for batch in batches:
            run_batch(
                mode,
                batch["scene"],
                batch["max_episodes"],
                args.gpu,
                args.dry_run,
                log_dir,
            )
            if args.dry_run:
                continue
            for ep_no, category in batch["episodes"].items():
                stage_episode(mode, ep_no, category, batch["scene"])


if __name__ == "__main__":
    main()
