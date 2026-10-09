#!/usr/bin/env python3
"""汇总 HM3D val 建图验证与离线耗时，输出 markdown + LaTeX 表6。

用法 (CWD = DBSGNavigation):
  python scripts/summarize_mapping_verify.py
  python scripts/summarize_mapping_verify.py --write-tex
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GC_ROOT = os.path.normpath(os.path.join(GN_ROOT, "..", "DBSGConstruction"))
DOCS = os.path.normpath(os.path.join(GN_ROOT, "..", "docs"))


def _resolve(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(GN_ROOT, p))


def _scene_from_run_dir(name: str) -> str:
    return re.sub(r"_seed\d+(_clip)?$", "", name)


def _load_frame_count(scene: str, data_root: str) -> Optional[int]:
    tf = os.path.join(data_root, scene, "transforms.json")
    if not os.path.isfile(tf):
        return None
    try:
        with open(tf, "r") as f:
            data = json.load(f)
        return len(data.get("frames", []))
    except Exception:
        return None


def _read_metric_file(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        arr = np.loadtxt(path)
        return float(np.mean(arr))
    except Exception:
        return None


def _params_stats(npz_path: str) -> Dict[str, Any]:
    """只读 means3D 形状；clip 仅检查 key 存在（避免加载 ~2GB clip_vectors）。"""
    out: Dict[str, Any] = {"ok": False}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "means3D" not in data.files:
                out["error"] = "no means3D"
                return out
            means = data["means3D"]
            n = int(means.shape[0]) if means.ndim >= 2 else 1
            out["n_gaussians"] = n
            out["has_clip"] = "clip_vectors" in data.files
    except Exception as e:
        out["error"] = str(e)
        return out
    out["ok"] = True
    return out


def collect_mapping_rows(
    map_root: str, data_root: str
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(glob(os.path.join(map_root, "*_seed4_clip"))):
        scene = _scene_from_run_dir(os.path.basename(run_dir))
        eval_dir = os.path.join(run_dir, "eval")
        npz = os.path.join(run_dir, "params.npz")
        pst = _params_stats(npz) if os.path.isfile(npz) else {"ok": False, "error": "missing npz"}
        row = {
            "scene": scene,
            "frames": _load_frame_count(scene, data_root),
            "psnr": _read_metric_file(os.path.join(eval_dir, "psnr.txt")),
            "rmse": _read_metric_file(os.path.join(eval_dir, "rmse.txt")),
            "ssim": _read_metric_file(os.path.join(eval_dir, "ssim.txt")),
            "lpips": _read_metric_file(os.path.join(eval_dir, "lpips.txt")),
            **{k: pst.get(k) for k in ("n_gaussians", "has_clip", "ok", "error")},
        }
        rows.append(row)

    def _agg(key: str, filt=None) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None and (filt is None or filt(r))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_scenes": len(rows),
        "n_with_eval": sum(1 for r in rows if r.get("psnr") is not None),
        "n_params_ok": sum(1 for r in rows if r.get("ok")),
        "mean_frames": _agg("frames"),
        "mean_gaussians": _agg("n_gaussians", lambda r: r.get("ok")),
        "n_with_clip": sum(1 for r in rows if r.get("has_clip")),
        "mean_psnr": _agg("psnr"),
        "median_psnr": float(np.median([r["psnr"] for r in rows if r.get("psnr") is not None]))
        if any(r.get("psnr") is not None for r in rows)
        else float("nan"),
        "mean_rmse": _agg("rmse"),
        "median_rmse": float(np.median([r["rmse"] for r in rows if r.get("rmse") is not None]))
        if any(r.get("rmse") is not None for r in rows)
        else float("nan"),
        "mean_ssim": _agg("ssim"),
        "mean_lpips": _agg("lpips"),
    }
    return rows, summary


def parse_build_log(log_path: str) -> List[Dict[str, Any]]:
    """从 habitat_splatam 建图 log 解析每场景 timing（支持多 log 合并）。"""
    paths: List[str] = []
    if os.path.isfile(log_path):
        paths = [log_path]
    elif "*" in log_path:
        paths = sorted(glob(log_path))

    entries: List[Dict[str, Any]] = []
    timing_block_re = re.compile(
        r"Average Tracking/Frame Time:\s*([\d.]+)\s*s.*?Average Mapping/Frame Time:\s*([\d.]+)\s*s",
        re.DOTALL,
    )
    map_iter_re = re.compile(r"Average Mapping/Iteration Time:\s*([\d.]+)\s*ms")
    seq_re = re.compile(r"--sequence\s+(\S+)")
    run_re = re.compile(r"'run_name':\s*'([^']+_seed4_clip)'")

    for lp in paths:
        if not os.path.isfile(lp):
            continue
        with open(lp, "r", errors="replace") as f:
            text = f.read()
        for m in timing_block_re.finditer(text):
            prefix = text[: m.start()]
            scene = ""
            runs = run_re.findall(prefix)
            if runs:
                scene = _scene_from_run_dir(runs[-1])
            else:
                seqs = seq_re.findall(prefix)
                if seqs:
                    scene = seqs[-1]
            if not scene:
                continue
            suffix = text[m.start() : m.end() + 200]
            mi = map_iter_re.search(suffix)
            entries.append(
                {
                    "scene": scene,
                    "track_frame_s": float(m.group(1)),
                    "map_frame_s": float(m.group(2)),
                    "map_iter_ms": float(mi.group(1)) if mi else None,
                    "log": os.path.basename(lp),
                }
            )

    by_scene: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        by_scene[e["scene"]] = e
    return list(by_scene.values())


def _fmt(x: Optional[float], nd: int = 2, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}{suffix}"


def write_markdown(
    path: str,
    summary: Dict[str, float],
    rows: List[Dict[str, Any]],
    timing: List[Dict[str, Any]],
    asset_summary: Optional[str],
) -> None:
    lines = [
        "# 建图验证与离线耗时汇总（表6 数据源）",
        "",
        "> 生成：`DBSGNavigation/scripts/summarize_mapping_verify.py`",
        "",
        "## 资产完整性",
        "",
    ]
    if asset_summary:
        lines.append("```")
        lines.append(asset_summary.strip())
        lines.append("```")
    else:
        lines.append("（未运行 `verify_habitat_val_assets.py`）")

    lines += [
        "",
        "## 建图质量（36 场景 · `habitat_val_final`）",
        "",
        f"- 含 `params.npz` 且结构正常：**{int(summary['n_params_ok'])}** / {int(summary['n_scenes'])}",
        f"- 含 eval 指标：**{int(summary['n_with_eval'])}** / {int(summary['n_scenes'])}",
        f"- 平均输入帧数：**{_fmt(summary['mean_frames'], 0)}**",
        f"- 平均高斯数：**{_fmt(summary['mean_gaussians'], 0)}**",
        f"- 含 CLIP 向量 key：**{int(summary.get('n_with_clip', 0))}** / {int(summary['n_scenes'])}",
        "",
        "### 渲染重建质量（eval 全帧平均）",
        "",
        "| 指标 | Mean | Median |",
        "|------|-----:|-------:|",
        f"| PSNR (dB) ↑ | {_fmt(summary['mean_psnr'])} | {_fmt(summary['median_psnr'])} |",
        f"| Depth L1 (m) ↓ | {_fmt(summary['mean_rmse'], 3)} | {_fmt(summary['median_rmse'], 3)} |",
        f"| SSIM ↑ | {_fmt(summary['mean_ssim'], 3)} | — |",
        f"| LPIPS ↓ | {_fmt(summary['mean_lpips'], 3)} | — |",
        "",
    ]

    if timing:
        tf = np.mean([t["track_frame_s"] for t in timing if "track_frame_s" in t])
        mf = np.mean([t["map_frame_s"] for t in timing if "map_frame_s" in t])
        mi = np.mean([t["map_iter_ms"] for t in timing if "map_iter_ms" in t])
        lines += [
            "## 离线 SplaTAM 耗时（build log 解析）",
            "",
            f"- 解析场景数：**{len(timing)}**",
            f"- Tracking / frame：**{_fmt(tf, 2)} s**（GT pose 时≈0 iter）",
            f"- Mapping / frame：**{_fmt(mf, 2)} s**",
            f"- Mapping / iter：**{_fmt(mi, 1)} ms**",
            f"- 粗估 per-scene wall time（400 frames, stride=2）：**~{int((tf + mf) * 200 / 60)} min**（≈200 处理帧 × (track+map)/frame）",
            "",
            "| Scene | Track/frame (s) | Map/frame (s) | Map/iter (ms) |",
            "|-------|----------------:|--------------:|--------------:|",
        ]
        for t in sorted(timing, key=lambda x: x["scene"])[:10]:
            lines.append(
                f"| {t['scene']} | {_fmt(t.get('track_frame_s'))} | "
                f"{_fmt(t.get('map_frame_s'))} | {_fmt(t.get('map_iter_ms'), 1)} |"
            )
        if len(timing) > 10:
            lines.append(f"| … | （共 {len(timing)} 场景，见 JSON） | | |")

    lines += [
        "",
        "## 论文用法",
        "",
        "- **表6a** `table6_mapping_verify.tex`：完整性 + 重建质量汇总",
        "- **表6b** `table6_offline_timing.tex`：SplaTAM 离线耗时",
        "- 正文可写：scene-specific semantic Gaussian maps built offline; median PSNR XX dB, depth L1 XX m on subsampled mapping frames.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_tex_mapping_verify(path: str, summary: Dict[str, float]) -> None:
    tex = f"""% Table VIa: Offline mapping verification (HM3D val)
\\begin{{table}}[t]
  \\centering
  \\small
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\caption{{Semantic Gaussian mapping verification on HM3D validation scenes (\\texttt{{habitat\\_val\\_final}}). Each scene: RGB-D exploration frames $\\rightarrow$ SplaTAM with CLIP (\\texttt{{params.npz}}). Metrics: per-frame render vs.\\ input RGB-D on subsampled mapping frames; PSNR/depth error report median over scenes. Depth L1 is mean absolute depth error (m), identical to \\texttt{{l1.txt}}/\\texttt{{rmse.txt}} in our SplaTAM eval logs.}}
  \\label{{tab:mapping_verify}}
  \\begin{{tabular}}{{@{{}}lc@{{}}}}
    \\toprule
    Item & Value \\\\
    \\midrule
    Scenes with valid \\texttt{{params.npz}} & {int(summary['n_params_ok'])} / {int(summary['n_scenes'])} \\\\
    Scenes with eval metrics & {int(summary['n_with_eval'])} / {int(summary['n_scenes'])} \\\\
    Mean input frames / scene & {summary['mean_frames']:.0f} \\\\
    Mean \\#Gaussians / scene & {summary['mean_gaussians']:.0f} \\\\
    Scenes with \\texttt{{clip\\_vectors}} & {int(summary.get('n_with_clip', 0))} / {int(summary['n_scenes'])} \\\\
    \\midrule
    PSNR (dB) $\\uparrow$ & {summary['median_psnr']:.2f} \\\\
    Depth L1 (m) $\\downarrow$ & {summary['median_rmse']:.3f} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)


def write_tex_offline_timing(path: str, timing: List[Dict[str, Any]]) -> None:
    if not timing:
        return
    tf = float(np.mean([t["track_frame_s"] for t in timing if "track_frame_s" in t]))
    mf = float(np.mean([t["map_frame_s"] for t in timing if "map_frame_s" in t]))
    mi = float(np.mean([t["map_iter_ms"] for t in timing if "map_iter_ms" in t]))
    wall_min = (tf + mf) * 200 / 60.0

    tex = f"""% Table VIb: Offline pipeline timing (SplaTAM construction)
\\begin{{table}}[t]
  \\centering
  \\small
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\caption{{Offline mapping time per processed frame (SplaTAM + semantic/CLIP densification, single GPU). Averages over {len(timing)} HM3D val scenes parsed from construction logs; GT camera poses (tracking iterations $\\approx 0$). Typical export: $\\leq$400 RGB-D frames, stride~2 ($\\sim$200 mapping steps).}}
  \\label{{tab:offline_timing}}
  \\begin{{tabular}}{{@{{}}lc@{{}}}}
    \\toprule
    Stage & Time \\\\
    \\midrule
    Camera tracking / frame & {tf:.2f}\\,s \\\\
    Gaussian mapping / frame & {mf:.2f}\\,s \\\\
    Mapping optimization / iter & {mi:.1f}\\,ms \\\\
    \\midrule
    Est.\\ wall time / scene & $\\sim${wall_min:.0f}\\,min \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gaussian-map-root",
        default="../DBSGConstruction/experiments/habitat_val_final",
    )
    ap.add_argument(
        "--data-root",
        default="../DBSGConstruction/configs/data/habitat_val",
    )
    ap.add_argument(
        "--build-log",
        default="",
        help="单个建图 log；留空则扫描 DBSGConstruction 下含 timing 的 log",
    )
    ap.add_argument("--write-tex", action="store_true", help="写入 docs/tables/table6_*.tex")
    ap.add_argument("--asset-log", default="", help="verify_habitat_val_assets.py 输出文本")
    args = ap.parse_args()

    map_root = _resolve(args.gaussian_map_root)
    data_root = _resolve(args.data_root)
    rows, summary = collect_mapping_rows(map_root, data_root)

    build_log = _resolve(args.build_log) if args.build_log else ""
    if build_log:
        timing = parse_build_log(build_log)
    else:
        gc = GC_ROOT
        patterns = [
            os.path.join(gc, "experiments", "habitat_val_final_remaining_build.log"),
            os.path.join(gc, "logs", "multi_gpu_*", "gpu*.log"),
            os.path.join(gc, "run_gpu*.log"),
            os.path.join(gc, "rerun_*.log"),
        ]
        timing = []
        seen: set = set()
        for pat in patterns:
            for e in parse_build_log(pat):
                if e["scene"] not in seen:
                    timing.append(e)
                    seen.add(e["scene"])
        timing.sort(key=lambda x: x["scene"])
        val_scenes = {r["scene"] for r in rows}
        timing = [t for t in timing if t["scene"] in val_scenes]

    asset_summary = ""
    if args.asset_log and os.path.isfile(args.asset_log):
        with open(args.asset_log, "r") as f:
            asset_summary = f.read()

    md_path = os.path.join(DOCS, "mapping_verify_summary.md")
    write_markdown(md_path, summary, rows, timing, asset_summary)

    json_path = os.path.join(DOCS, "mapping_verify_detail.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "rows": rows, "timing": timing}, f, indent=2)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(
        f"Scenes={summary['n_scenes']}  PSNR mean={summary['mean_psnr']:.2f}  "
        f"Depth L1 mean={summary['mean_rmse']:.3f} median={summary['median_rmse']:.3f}  timing scenes={len(timing)}"
    )

    if args.write_tex:
        write_tex_mapping_verify(
            os.path.join(DOCS, "tables", "table6_mapping_verify.tex"), summary
        )
        write_tex_offline_timing(
            os.path.join(DOCS, "tables", "table6_offline_timing.tex"), timing
        )
        print(f"Wrote docs/tables/table6_mapping_verify.tex")
        print(f"Wrote docs/tables/table6_offline_timing.tex")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
