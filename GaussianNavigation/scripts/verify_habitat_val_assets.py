#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 HM3D val 评测前「建图/数据层」(A) 是否齐全，无需改算法。

检查项:
  1) gaussian_map_root: 每场景是否存在与 SemanticGaussianGoal 一致规则的 params.npz
     （目录名 {scene}* 且含 _clip 优先；含 means3D + clip_vectors）
  2) base_scene_path: 每场景 instance_retrieval.json、instance_clip_vectors.json
  3) instance_retrieval 中各 entry 的 frame_id 对应 rgb 图是否存在
     （与 build_instance_clip_vectors._resolve_rgb_path 相同解析规则）
  4) 可选: --other-gaussian-root 与 val 的 params 场景集合对比，提示建图是否「非同源」划分

默认路径与 configs/instance_imagenav_val_nogt.yaml 一致（相对 GaussianNavigation/）。

用法:
  cd GaussianNavigation
  python scripts/verify_habitat_val_assets.py
  python scripts/verify_habitat_val_assets.py --gaussian-map-root /abs/path/habitat_val_final \\
      --base-scene-path /abs/path/habitat_val
  python scripts/verify_habitat_val_assets.py --other-gaussian-root ../GaussianConstruction/experiments/habitat_train_final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# 与 open_vocab_retriever / build_instance_clip_vectors 一致
COCO_KEYS = [str(x) for x in (56, 57, 58, 59, 61, 62)]


def _gn_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(_gn_root(), p))


def _resolve_params_path(gaussian_map_root: str, scene_name: str) -> Optional[str]:
    if not gaussian_map_root or not os.path.isdir(gaussian_map_root):
        return None
    candidates = [
        d
        for d in glob(os.path.join(gaussian_map_root, f"{scene_name}*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "params.npz"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: ("_clip" not in os.path.basename(d), d))
    return os.path.join(candidates[0], "params.npz")


def _check_params_npz(path: str) -> Tuple[bool, str]:
    try:
        data = dict(np.load(path, allow_pickle=True))
    except Exception as e:
        return False, f"load 失败: {e}"
    if "means3D" not in data:
        return False, "无 means3D"
    m = np.array(data["means3D"], dtype=np.float64)
    if m.ndim == 3:
        m = m.reshape(-1, 3)
    n = m.shape[0]
    if "clip_vectors" not in data:
        return False, "无 clip_vectors (需 use_clip 建图)"
    cv = np.array(data["clip_vectors"], dtype=np.float32)
    if cv.ndim == 1:
        cv = cv.reshape(1, -1)
    if cv.shape[0] != n:
        return False, f"clip_vectors 行数 {cv.shape[0]} != means3D 行数 {n}"
    if float(np.abs(cv).max()) < 1e-6:
        return False, "clip_vectors 全零"
    return True, f"ok N={n}"


def _resolve_rgb_path(scene_path: str, frame_id) -> str:
    base = os.path.join(scene_path, "rgb")
    path = os.path.join(base, f"frame_{frame_id}.png")
    if os.path.isfile(path):
        return path
    if isinstance(frame_id, str) and frame_id.isdigit():
        n = int(frame_id)
        for width in (5, 6):
            padded = str(n).zfill(width)
            p2 = os.path.join(base, f"frame_{padded}.png")
            if os.path.isfile(p2):
                return p2
    return path


def _list_scene_names(base_scene_path: str) -> List[str]:
    if not os.path.isdir(base_scene_path):
        return []
    out = []
    for name in sorted(os.listdir(base_scene_path)):
        p = os.path.join(base_scene_path, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return out


def _collect_scenes_from_gaussian_root(root: str) -> Set[str]:
    """用目录名前缀做 scene_name（与 params 的父目录 basename 在去掉 _seed*_clip 前的前缀可人工对照）。"""
    if not os.path.isdir(root):
        return set()
    scenes: Set[str] = set()
    for d in glob(os.path.join(root, "*")):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        if os.path.isfile(os.path.join(d, "params.npz")):
            # 常见: 4sIZ4gqNPw8_seed4_clip -> 以 HM3D id 为前缀
            scenes.add(base)
    return scenes


def check_scene(
    scene_name: str,
    gaussian_map_root: str,
    base_scene_path: str,
    max_missing_rgb: int,
) -> Dict:
    row: Dict = {"scene": scene_name, "params": "", "ir": "", "clip_json": "", "rgb_missing": 0, "rgb_checked": 0}
    pparams = _resolve_params_path(gaussian_map_root, scene_name)
    if pparams is None:
        row["params"] = "MISSING"
    else:
        ok, msg = _check_params_npz(pparams)
        row["params"] = msg if ok else f"BAD: {msg}"

    sdir = os.path.join(base_scene_path, scene_name)
    ir = os.path.join(sdir, "instance_retrieval.json")
    cj = os.path.join(sdir, "instance_clip_vectors.json")
    row["ir"] = "ok" if os.path.isfile(ir) else "MISSING"
    row["clip_json"] = "ok" if os.path.isfile(cj) else "MISSING"

    missing = 0
    checked = 0
    if os.path.isfile(ir):
        try:
            with open(ir, "r") as f:
                data = json.load(f)
        except Exception as e:
            row["ir"] = f"BAD: {e}"
            return row
        for key in COCO_KEYS:
            if key not in data:
                continue
            for ent in data[key]:
                if not isinstance(ent, (list, tuple)) or len(ent) < 2:
                    continue
                frame_id = ent[1]
                checked += 1
                rp = _resolve_rgb_path(sdir, frame_id)
                if not os.path.isfile(rp):
                    missing += 1
                    if missing <= 3:
                        print(f"  [rgb missing] {scene_name} key={key} frame_id={frame_id} -> {rp}", file=sys.stderr)
                if max_missing_rgb > 0 and missing >= max_missing_rgb:
                    break
            if max_missing_rgb > 0 and missing >= max_missing_rgb:
                break

    row["rgb_missing"] = missing
    row["rgb_checked"] = checked
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="检查 val 的 params.npz + instance_retrieval + rgb")
    ap.add_argument(
        "--gaussian-map-root",
        default="../GaussianConstruction/experiments/habitat_val_final",
        help="对应 end2end_imagenav.gaussian_map_root",
    )
    ap.add_argument(
        "--base-scene-path",
        default="../GaussianConstruction/configs/data/habitat_val",
        help="对应 end2end_imagenav.base_scene_path",
    )
    ap.add_argument(
        "--other-gaussian-root",
        default="",
        help="可选: 如 train 的建图根目录，用于与 val 场景集合对比（见 stdout 说明）",
    )
    ap.add_argument(
        "--max-missing-rgb-stop",
        type=int,
        default=0,
        help="单场景内累计缺图达到该数则不再继续数（0=不提前停，全量数）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只检查前 N 个场景 (0=全部)，大数据集下可先抽检",
    )
    args = ap.parse_args()

    gm = _resolve(args.gaussian_map_root)
    bs = _resolve(args.base_scene_path)
    other = _resolve(args.other_gaussian_root) if args.other_gaussian_root else ""

    print("== 路径 ==")
    print(f"  gaussian_map_root: {gm}  (exists={os.path.isdir(gm)})")
    print(f"  base_scene_path:   {bs}  (exists={os.path.isdir(bs)})")
    if other:
        print(f"  other (compare):  {other}  (exists={os.path.isdir(other)})")
    print()

    if not os.path.isdir(bs):
        print("错误: base_scene_path 不存在，请检查数据是否已解压到本机或修正路径。", file=sys.stderr)
        return 2
    if not os.path.isdir(gm):
        print("警告: gaussian_map_root 不存在，将仅检查 instance_retrieval/rgb（若存在）。", file=sys.stderr)

    scenes = _list_scene_names(bs)
    if not scenes:
        print("错误: base_scene_path 下无子目录，无法枚举场景。", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        scenes = scenes[: args.limit]

    print(f"== 逐场景检查 (共 {len(scenes)} 个目录于 base_scene_path) ==")
    bad_params: List[str] = []
    bad_ir: List[str] = []
    bad_rgb: List[str] = []

    for sc in scenes:
        row = check_scene(
            sc,
            gm,
            bs,
            max_missing_rgb=args.max_missing_rgb_stop,
        )
        line = (
            f"{sc}: params={row['params']!s} | ir={row['ir']} | "
            f"instance_clip={row['clip_json']} | rgb_missing={row['rgb_missing']}/{row['rgb_checked']}"
        )
        if "MISSING" in row["params"] or row["params"].startswith("BAD"):
            bad_params.append(sc)
        if row["ir"] != "ok":
            bad_ir.append(sc)
        if row["rgb_checked"] > 0 and row["rgb_missing"] > 0:
            bad_rgb.append(sc)
        if "MISSING" in row["params"] or row["ir"] != "ok" or row["rgb_missing"] > 0:
            print(line)

    ok_all = all(
        s not in bad_params and s not in bad_ir and s not in bad_rgb
        for s in scenes
    )
    # 再打印健康场景数量
    n_ok = len(scenes) - len(set(bad_params) | set(bad_ir) | set(bad_rgb))
    print()
    print("== 汇总 ==")
    print(f"  场景目录数: {len(scenes)}")
    print(f"  params 异常: {len(set(bad_params))}  {sorted(set(bad_params))[:20]}{'...' if len(set(bad_params))>20 else ''}")
    print(f"  instance_retrieval 异常: {len(set(bad_ir))}")
    print(f"  缺 rgb 图 (任一 frame): {len(set(bad_rgb))} 场景 {sorted(set(bad_rgb))[:15]}{'...' if len(set(bad_rgb))>15 else ''}")

    if other and os.path.isdir(other):
        val_names = set(scenes)
        # other root 的目录名与 base_scene 的 scene id 常不一致：用「能在 val base 里找到子目录的 scene id」
        # 只报告：val 的 scene 在 other 下是否有 *任意* 含 params 的目录名以 scene 为前缀
        in_other: List[str] = []
        for sc in sorted(val_names):
            if _resolve_params_path(other, sc) is not None:
                in_other.append(sc)
        missing_in_other = sorted(val_names - set(in_other))
        print()
        print("== 与 --other-gaussian-root 对比（建图是否同一批场景/划分） ==")
        print(f"  val 场景在 other 中能找到 params.npz(按 scene 前缀规则): {len(in_other)}/{len(val_names)}")
        if missing_in_other:
            print(
                f"  在 other 中无匹配 params 的 val 场景 (前20): {missing_in_other[:20]}{'...' if len(missing_in_other)>20 else ''}"
            )
        s_val = _collect_scenes_from_gaussian_root(gm)
        s_oth = _collect_scenes_from_gaussian_root(other)
        print(
            f"  (粗) val_root 下含 params 的目录数: {len(s_val)}, other_root: {len(s_oth)}，"
            "仅作体量对比；**同源**还需确认建图脚本与 split 与 habitat_val 配置一致。"
        )

    if ok_all and not bad_params and not bad_ir and not bad_rgb:
        print()
        print("结论: 上述检查下未发现缺 params / 缺 ir / 缺 rgb 的场景（逐行未打印即通常表示该项正常）。")
        print("若仍要确认「非 A 问题」，可再开 log_episode_goal_summary 看 goal_source 与 dist 分布。")
        return 0
    print()
    print(
        "结论: 若缺 params/clip 或缺图比例高，应优先补全 GaussianConstruction 数据与重建图，再调 α/β。"
    )
    return 1 if (bad_params or bad_ir or bad_rgb) else 0


if __name__ == "__main__":
    raise SystemExit(main())
