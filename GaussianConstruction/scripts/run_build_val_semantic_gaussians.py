#!/usr/bin/env python3
"""
为 HM3D val 场景批量建带 CLIP 的 Semantic Gaussian（方案三）。

用法:
  # 自动发现：在 data_dir 下每个含 transforms.json 的子目录视为一个场景
  python run_build_val_semantic_gaussians.py --data_dir /path/to/configs/data/habitat_val --group_name habitat_val_final

  # 从文件读场景列表（每行一个 scene_id）
  python run_build_val_semantic_gaussians.py --scenes_file val_scenes.txt --group_name habitat_val_final

  # 只跑指定场景
  python run_build_val_semantic_gaussians.py --sequence CrMo8WxCyVb --group_name habitat_val_final

  # 跳过已存在 params.npz 的场景（默认跳过）
  python run_build_val_semantic_gaussians.py --data_dir ... --no_skip_existing  # 全部重跑

  # 批跑中断后：无 params.npz 但有 params{N}.npz 的场景自动续跑（不必改 config 里 load_checkpoint）
  python run_build_val_semantic_gaussians.py --data_dir ... --group_name habitat_val_final \
    --config configs/habitat/habitat_splatam_val.py --auto_resume

输出目录: experiments/{group_name}/{scene_id}_seed{N}_clip/params.npz
在 GaussianNavigation 的 configs/instance_imagenav_val_nogt.yaml 中设置 gaussian_map_root 指向 experiments/{group_name}（默认 habitat_val_final）即可。
"""

import argparse
import glob
import os
import re
import subprocess
import sys


def max_resume_checkpoint_n(run_dir):
    """若存在成对的 params{N}.npz 与 keyframe_time_indices{N}.npy，返回最大 N；否则 None。"""
    if not os.path.isdir(run_dir):
        return None
    best = []
    for p in glob.glob(os.path.join(run_dir, "params*.npz")):
        base = os.path.basename(p)
        if base == "params.npz":
            continue
        m = re.match(r"params(\d+)\.npz$", base)
        if not m:
            continue
        n = int(m.group(1))
        kf = os.path.join(run_dir, f"keyframe_time_indices{n}.npy")
        if os.path.isfile(kf):
            best.append(n)
    return max(best) if best else None


def main():
    parser = argparse.ArgumentParser(description="Batch build Semantic Gaussians (with CLIP) for val scenes.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Habitat construction data root (e.g. configs/data/habitat). Scenes = subdirs with transforms.json.")
    parser.add_argument("--scenes_file", type=str, default=None,
                        help="Text file with one scene_id per line (overrides --data_dir discovery).")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Single scene_id to build (overrides --data_dir and --scenes_file).")
    parser.add_argument("--group_name", type=str, default="habitat_val_final",
                        help="Output group: workdir = experiments/<group_name> (default matches val SplaTAM output).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to habitat_splatam config (default: configs/habitat/habitat_splatam_val.py).")
    parser.add_argument("--script", type=str, default=None,
                        help="Path to habitat_splatam.py script (default: scripts/habitat_splatam.py).")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Override data.num_frames passed to habitat_splatam.py (e.g. 1000).")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override data.stride passed to habitat_splatam.py (e.g. 1/2/3).")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip scene if params.npz already exists (default: True).")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="Do not skip; rebuild all.")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not run.")
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Per scene: if params.npz missing but params{N}.npz + keyframe_time_indices{N}.npy exist, pass --checkpoint_time_idx N (no need to edit config). Scenes without checkpoints start fresh.",
    )
    parser.add_argument(
        "--no_mid_checkpoints",
        action="store_true",
        help="传给 habitat_splatam.py --no_save_checkpoints：不周期性写 params{N}.npz（极省磁盘；不能与依赖中间 ckpt 的续跑策略混用）。",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    if args.script is None:
        args.script = os.path.join(script_dir, "habitat_splatam.py")
    if args.config is None:
        args.config = os.path.join(base_dir, "configs", "habitat", "habitat_splatam_val.py")

    if not os.path.isfile(args.script):
        print(f"Error: script not found: {args.script}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    scenes = []
    if args.sequence is not None:
        scenes = [args.sequence.strip()]
    elif args.scenes_file is not None:
        if not os.path.isfile(args.scenes_file):
            print(f"Error: scenes file not found: {args.scenes_file}", file=sys.stderr)
            sys.exit(1)
        with open(args.scenes_file, "r") as f:
            scenes = [line.strip() for line in f if line.strip()]
    elif args.data_dir is not None and os.path.isdir(args.data_dir):
        for name in sorted(os.listdir(args.data_dir)):
            path = os.path.join(args.data_dir, name)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "transforms.json")):
                scenes.append(name)
    else:
        print("Error: provide one of --sequence, --scenes_file, or --data_dir.", file=sys.stderr)
        sys.exit(1)

    if not scenes:
        print("No scenes to build.", file=sys.stderr)
        sys.exit(0)

    workdir = os.path.join(base_dir, "experiments", args.group_name)
    os.makedirs(workdir, exist_ok=True)

    to_run = []
    for scene in scenes:
        # run_name is {scene}_seed{N}_clip (must match habitat_splatam --sequence logic)
        run_name = f"{scene}_seed4_clip"
        params_npz = os.path.join(workdir, run_name, "params.npz")
        if args.skip_existing and os.path.isfile(params_npz):
            print(f"[skip] {scene} (exists: {params_npz})")
            continue
        to_run.append(scene)

    print(f"Scenes to build: {len(to_run)} / {len(scenes)}")
    for scene in to_run:
        cmd = [
            sys.executable,
            args.script,
            os.path.abspath(args.config),
            "--sequence", scene,
            "--group_name", args.group_name,
        ]
        if args.auto_resume:
            run_dir = os.path.join(workdir, f"{scene}_seed4_clip")
            n = max_resume_checkpoint_n(run_dir)
            if n is not None:
                cmd.extend(["--checkpoint_time_idx", str(n)])
                print(f"[auto_resume] {scene} from N={n}")
        if args.no_mid_checkpoints:
            cmd.append("--no_save_checkpoints")
        if args.num_frames is not None:
            cmd.extend(["--num_frames", str(args.num_frames)])
        if args.stride is not None:
            cmd.extend(["--stride", str(args.stride)])
        print("Run:", " ".join(cmd))
        if not args.dry_run:
            ret = subprocess.run(cmd, cwd=base_dir)
            if ret.returncode != 0:
                print(f"Failed: {scene} (exit {ret.returncode})", file=sys.stderr)
                sys.exit(ret.returncode)
    print("Done.")


if __name__ == "__main__":
    main()
