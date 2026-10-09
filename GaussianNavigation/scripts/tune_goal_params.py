#!/usr/bin/env python3
"""
批量调参：围绕 SG/OV 目标选择参数自动生成配置、执行评测并汇总指标。

套件 (--suite)：
  classic  — 8 组：min_sim / high_conf / OV·几何权重（与早期实验一致）
  extended — 基于当前 base yaml 的默认值，扫：
             open_vocab_min_geometric_matches、open_vocab_top_k、
             semantic_gaussian_ov_disagree_dist / _force_dist
  all      — classic + extended

示例：
  # 推荐：先跑 extended + 200 ep，看 summary.tsv 再对胜者拉满 1000 ep
  python scripts/tune_goal_params.py --suite extended --max_episodes 200 --cuda_visible_devices 0

  # 仅生成配置不跑
  python scripts/tune_goal_params.py --suite extended --dry_run

  # 完整终端输出默认写入 <output_dir>/session_terminal.log；关闭: --no_session_log

  # 多卡并行（同一输出目录、不同 GPU）：各终端使用相同 --resume_dir，不同
  # --cuda_visible_devices 与互补的 --only_trials，例如：
  #   CUDA_VISIBLE_DEVICES=0 python scripts/tune_goal_params.py --suite extended \\
  #     --max_episodes 200 --resume_dir experiments/tune_goal_params/20260420_120000 \\
  #     --only_trials 1-4
  #   CUDA_VISIBLE_DEVICES=1 python ... --only_trials 5-8
  # 并行时建议为各进程指定不同 --session_log，避免会话日志交错。
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    import fcntl  # type: ignore
except ImportError:  # non-Unix
    fcntl = None  # type: ignore


METRIC_RE = re.compile(
    r"succ / spl / dist / nonGT_cov / GT_fb_rate : "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\(([0-9]+)\),"
)


def parse_last_metrics(text: str) -> Optional[Dict[str, float]]:
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "succ": float(m.group(1)),
        "spl": float(m.group(2)),
        "dist": float(m.group(3)),
        "nonGT_cov": float(m.group(4)),
        "GT_fb_rate": float(m.group(5)),
        "count": int(m.group(6)),
    }


def load_existing_log_result(log_path: Path) -> Dict:
    if not log_path.is_file():
        return {"exists": False, "completed": False, "metrics": None}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = parse_last_metrics(text)
    completed = "'all episode over'" in text
    return {"exists": True, "completed": completed, "metrics": metrics}


def is_effectively_completed(existing: Dict[str, Any], target_episodes: Optional[int]) -> bool:
    """断点续跑判定：显式结束标记，或已解析 count 达到目标 episode。"""
    if not existing.get("exists"):
        return False
    if existing.get("completed"):
        return True
    metrics = existing.get("metrics")
    if not isinstance(metrics, dict) or target_episodes is None:
        return False
    try:
        return int(metrics.get("count", 0)) >= int(target_episodes)
    except (TypeError, ValueError):
        return False


SUMMARY_HEADER = (
    "trial\tmin_sim\thigh_conf\tov_w\tgeo_w\tmin_geom\ttop_k\tdisagree_dist\tforce_dist\t"
    "exit_code\tsucc\tspl\tdist\tnonGT_cov\tGT_fb_rate\tcount\tcfg_path\tlog_path"
)


def parse_only_trials(spec: Optional[str]) -> Optional[Set[int]]:
    """解析 --only_trials：'1,3,5' 或 '4-8' 或 '1,4-6'；返回 1-based 序号集合。"""
    if spec is None or not str(spec).strip():
        return None
    out: Set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("--only_trials 解析后为空；示例: '1,3,5' 或 '4-8'")
    return out


def _trial_row_fields(
    trial: Dict[str, Any],
    idx: int,
    cfg_dir: Path,
    log_dir: Path,
    exit_code: Any,
    metrics: Optional[Dict[str, Any]],
) -> List[str]:
    t = trial
    m = metrics or {}
    cfg_path = cfg_dir / f"{idx:02d}_{t['name']}.yaml"
    if exit_code is None and metrics is None:
        log_cell = ""
    else:
        log_cell = str(log_dir / f"{idx:02d}_{t['name']}.log")
    return [
        t["name"],
        str(t["semantic_gaussian_min_sim"]),
        str(t["semantic_gaussian_high_conf_sim"]),
        str(t["open_vocab_semantic_weight"]),
        str(t["fusion_geometric_weight"]),
        str(t.get("open_vocab_min_geometric_matches", "")),
        str(t.get("open_vocab_top_k", "")),
        str(t.get("semantic_gaussian_ov_disagree_dist", "")),
        str(t.get("semantic_gaussian_ov_disagree_force_dist", "")),
        "" if exit_code is None else str(exit_code),
        "" if m.get("succ") is None else str(m["succ"]),
        "" if m.get("spl") is None else str(m["spl"]),
        "" if m.get("dist") is None else str(m["dist"]),
        "" if m.get("nonGT_cov") is None else str(m["nonGT_cov"]),
        "" if m.get("GT_fb_rate") is None else str(m["GT_fb_rate"]),
        "" if m.get("count") is None else str(m["count"]),
        str(cfg_path),
        log_cell,
    ]


def _result_dict_to_row(
    r: Dict[str, Any],
    idx: int,
    cfg_dir: Path,
    log_dir: Path,
) -> List[str]:
    t = r["trial"]
    raw_m = r.get("metrics")
    m: Optional[Dict[str, Any]] = raw_m if isinstance(raw_m, dict) else None
    row = _trial_row_fields(t, idx, cfg_dir, log_dir, r.get("exit_code"), m)
    if r.get("cfg_path"):
        row[-2] = str(r["cfg_path"])
    if "log_path" in r:
        row[-1] = str(r["log_path"])
    return row


def _read_summary_rows(summary_path: Path) -> Dict[str, List[str]]:
    if not summary_path.is_file():
        return {}
    rows: Dict[str, List[str]] = {}
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts:
            rows[parts[0]] = parts
    return rows


def merge_summary_tsv(
    summary_path: Path,
    lock_path: Path,
    trials: List[Dict[str, Any]],
    cfg_dir: Path,
    log_dir: Path,
    update_results: List[Dict[str, Any]],
) -> None:
    """在锁内读旧表、用 update_results 覆盖对应 trial 行，再按 trials 顺序写回。"""
    by_name = {t["name"]: (i + 1, t) for i, t in enumerate(trials)}

    def do_merge() -> None:
        existing = _read_summary_rows(summary_path)
        for r in update_results:
            t = r["trial"]
            idx, _ = by_name[t["name"]]
            existing[t["name"]] = _result_dict_to_row(r, idx, cfg_dir, log_dir)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(SUMMARY_HEADER + "\n")
            for i, t in enumerate(trials, 1):
                row = existing.get(t["name"])
                if not row:
                    row = _trial_row_fields(t, i, cfg_dir, log_dir, None, None)
                f.write("\t".join(row) + "\n")

    if fcntl is not None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            try:
                do_merge()
            finally:
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
    else:
        do_merge()


def load_scored_from_summary(summary_path: Path) -> List[Dict[str, Any]]:
    """从 summary.tsv 解析带 succ 的行，供打印排序（多进程合并后统一读盘）。"""
    if not summary_path.is_file():
        return []
    lines = summary_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    try:
        i_succ = header.index("succ")
        i_spl = header.index("spl")
        i_dist = header.index("dist")
        i_count = header.index("count")
        i_trial = header.index("trial")
    except ValueError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) <= max(i_succ, i_spl, i_dist, i_count, i_trial):
            continue
        if p[i_succ] == "":
            continue
        try:
            out.append(
                {
                    "name": p[i_trial],
                    "succ": float(p[i_succ]),
                    "spl": float(p[i_spl]),
                    "dist": float(p[i_dist]),
                    "count": int(float(p[i_count])) if p[i_count] else 0,
                }
            )
        except (ValueError, TypeError):
            continue
    return out


def ensure_plan_file_once(
    plan_path: Path,
    lock_path: Path,
    write_fn: Any,
) -> None:
    """并行时仅第一个进程创建 plan.txt；全量跑则直接覆盖写入。"""
    if fcntl is not None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            try:
                if not plan_path.is_file():
                    with plan_path.open("w", encoding="utf-8") as f:
                        write_fn(f)
            finally:
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
    elif not plan_path.is_file():
        with plan_path.open("w", encoding="utf-8") as f:
            write_fn(f)


def make_trials(base_cfg: Dict) -> List[Dict]:
    e2e = base_cfg.get("end2end_imagenav", {})
    base_min = float(e2e.get("semantic_gaussian_min_sim", 0.2))
    base_hc = float(e2e.get("semantic_gaussian_high_conf_sim", 0.96))
    base_ow = float(e2e.get("open_vocab_semantic_weight", 0.8))

    def trial(
        name: str,
        min_sim: float = base_min,
        high_conf: float = base_hc,
        ov_w: float = base_ow,
    ) -> Dict:
        return {
            "name": name,
            "semantic_gaussian_min_sim": float(min_sim),
            "semantic_gaussian_high_conf_sim": float(high_conf),
            "open_vocab_semantic_weight": float(ov_w),
            "fusion_geometric_weight": float(1.0 - ov_w),
        }

    return [
        trial("baseline"),
        trial("hc95", high_conf=0.95),
        trial("hc98", high_conf=0.98),
        trial("hc99", high_conf=0.99),
        trial("min15", min_sim=0.15),
        trial("min25", min_sim=0.25),
        trial("ovw70", ov_w=0.70),
        trial("ovw90", ov_w=0.90),
    ]


def make_trials_extended(base_cfg: Dict) -> List[Dict[str, Any]]:
    """在当前 base 配置的 SG/OV 默认值上，扫几何匹配数、top_k 与 disagree 距离。"""
    e2e = base_cfg.get("end2end_imagenav", {})
    base_min = float(e2e.get("semantic_gaussian_min_sim", 0.3))
    base_hc = float(e2e.get("semantic_gaussian_high_conf_sim", 0.995))
    base_ow = float(e2e.get("open_vocab_semantic_weight", 0.8))
    base_gm = int(e2e.get("open_vocab_min_geometric_matches", 0))
    base_k = int(e2e.get("open_vocab_top_k", 5))
    base_dd = float(e2e.get("semantic_gaussian_ov_disagree_dist", 3.0))
    base_fd = float(e2e.get("semantic_gaussian_ov_disagree_force_dist", 5.0))

    def ext(
        name: str,
        gm: Optional[int] = None,
        k: Optional[int] = None,
        dd: Optional[float] = None,
        fd: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "semantic_gaussian_min_sim": base_min,
            "semantic_gaussian_high_conf_sim": base_hc,
            "open_vocab_semantic_weight": base_ow,
            "fusion_geometric_weight": float(1.0 - base_ow),
            "open_vocab_min_geometric_matches": base_gm if gm is None else int(gm),
            "open_vocab_top_k": base_k if k is None else int(k),
            "semantic_gaussian_ov_disagree_dist": base_dd if dd is None else float(dd),
            "semantic_gaussian_ov_disagree_force_dist": base_fd if fd is None else float(fd),
        }

    return [
        ext("ext_baseline"),
        ext("ext_k8", k=8),
        ext("ext_gm20_k5", gm=20, k=5),
        ext("ext_gm20_k8", gm=20, k=8),
        ext("ext_gm30_k5", gm=30, k=5),
        ext("ext_gm30_k8", gm=30, k=8),
        ext("ext_gm40_k5", gm=40, k=5),
        ext("ext_gm40_k8", gm=40, k=8),
        ext("ext_dd25_fd40", dd=2.5, fd=4.0),
        ext("ext_dd35_fd55", dd=3.5, fd=5.5),
        ext("ext_gm25_k8_dd30_fd50", gm=25, k=8, dd=3.0, fd=5.0),
    ]


def make_trials_success_push(base_cfg: Dict) -> List[Dict[str, Any]]:
    """围绕 gm40_k8 局部网格，优先提升 Success。"""
    e2e = base_cfg.get("end2end_imagenav", {})
    base_min = float(e2e.get("semantic_gaussian_min_sim", 0.3))
    base_hc = float(e2e.get("semantic_gaussian_high_conf_sim", 0.995))
    base_ow = float(e2e.get("open_vocab_semantic_weight", 0.8))
    base_dd = float(e2e.get("semantic_gaussian_ov_disagree_dist", 3.0))
    base_fd = float(e2e.get("semantic_gaussian_ov_disagree_force_dist", 5.0))

    def sp(
        name: str,
        gm: int = 40,
        k: int = 8,
        dd: float = base_dd,
        fd: float = base_fd,
        min_sim: float = base_min,
        high_conf: float = base_hc,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "semantic_gaussian_min_sim": float(min_sim),
            "semantic_gaussian_high_conf_sim": float(high_conf),
            "open_vocab_semantic_weight": float(base_ow),
            "fusion_geometric_weight": float(1.0 - base_ow),
            "open_vocab_min_geometric_matches": int(gm),
            "open_vocab_top_k": int(k),
            "semantic_gaussian_ov_disagree_dist": float(dd),
            "semantic_gaussian_ov_disagree_force_dist": float(fd),
        }

    return [
        sp("sp_gm40_k8_base"),
        sp("sp_gm35_k8", gm=35),
        sp("sp_gm45_k8", gm=45),
        sp("sp_gm50_k8", gm=50),
        sp("sp_gm40_k10", k=10),
        sp("sp_gm40_k12", k=12),
        sp("sp_gm40_k8_dd28_fd48", dd=2.8, fd=4.8),
        sp("sp_gm40_k8_dd32_fd52", dd=3.2, fd=5.2),
        sp("sp_gm40_k10_dd28_fd48", k=10, dd=2.8, fd=4.8),
        sp("sp_gm45_k10_dd32_fd52", gm=45, k=10, dd=3.2, fd=5.2),
        sp("sp_gm40_k8_hc992", high_conf=0.992),
        sp("sp_gm40_k8_hc997", high_conf=0.997),
        sp("sp_gm40_k8_min28", min_sim=0.28),
        sp("sp_gm40_k8_min33", min_sim=0.33),
    ]


_TRIAL_APPLY_KEYS = (
    "semantic_gaussian_min_sim",
    "semantic_gaussian_high_conf_sim",
    "open_vocab_semantic_weight",
    "fusion_geometric_weight",
    "open_vocab_min_geometric_matches",
    "open_vocab_top_k",
    "semantic_gaussian_ov_disagree_dist",
    "semantic_gaussian_ov_disagree_force_dist",
)


def apply_trial(base_cfg: Dict, trial: Dict[str, Any], max_episodes: Optional[int]) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    e2e = cfg.setdefault("end2end_imagenav", {})
    for k in _TRIAL_APPLY_KEYS:
        if k in trial:
            e2e[k] = trial[k]
    if max_episodes is not None:
        e2e["max_episodes"] = int(max_episodes)
    return cfg


def run_one(
    nav_root: Path,
    cfg_path: Path,
    run_log_path: Path,
    cuda_visible_devices: Optional[str],
) -> Dict:
    cmd = [sys.executable, "run.py", "--config", str(cfg_path)]
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    proc = subprocess.Popen(
        cmd,
        cwd=str(nav_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: List[str] = []
    assert proc.stdout is not None
    with run_log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line)
    code = proc.wait()
    output = "".join(lines)
    metrics = parse_last_metrics(output)
    return {"exit_code": code, "metrics": metrics}



def run_tune_trials(
    args: argparse.Namespace,
    base_cfg: Dict,
    nav_root: Path,
    base_cfg_path: Path,
    out_dir: Path,
    cfg_dir: Path,
    log_dir: Path,
) -> int:
    if args.suite == "classic":
        trials = make_trials(base_cfg)
    elif args.suite == "extended":
        trials = make_trials_extended(base_cfg)
    elif args.suite == "success_push":
        trials = make_trials_success_push(base_cfg)
    else:
        trials = make_trials(base_cfg) + make_trials_extended(base_cfg) + make_trials_success_push(base_cfg)
    results = []
    only = args.only_trials_set
    if only:
        bad = sorted(x for x in only if x < 1 or x > len(trials))
        if bad:
            raise ValueError(
                f"--only_trials 超出范围 1..{len(trials)}: {bad}（当前套件共 {len(trials)} 组）"
            )

    plan = out_dir / "plan.txt"
    tune_lock = out_dir / ".tune_goal_params.lock"
    summary_lock = out_dir / ".summary.tsv.lock"
    summary = out_dir / args.summary_name

    def write_plan_body(f) -> None:
        f.write(f"base_config={base_cfg_path}\n")
        f.write(f"suite={args.suite}\n")
        f.write(f"max_episodes_override={args.max_episodes}\n")
        f.write(f"cuda_visible_devices={args.cuda_visible_devices}\n")
        f.write(f"resume_dir={args.resume_dir}\n")
        f.write(f"skip_completed={args.skip_completed}\n")
        f.write(f"only_trials={args.only_trials or ''}\n")
        for i, t in enumerate(trials, 1):
            line = (
                f"{i:02d}. {t['name']}: min_sim={t['semantic_gaussian_min_sim']}, "
                f"high_conf={t['semantic_gaussian_high_conf_sim']}, "
                f"ov_w={t['open_vocab_semantic_weight']}, "
                f"geo_w={t['fusion_geometric_weight']}"
            )
            if "open_vocab_min_geometric_matches" in t:
                line += (
                    f", min_geom={t['open_vocab_min_geometric_matches']}, "
                    f"top_k={t['open_vocab_top_k']}, "
                    f"disagree_dist={t['semantic_gaussian_ov_disagree_dist']}, "
                    f"force_dist={t['semantic_gaussian_ov_disagree_force_dist']}"
                )
            f.write(line + "\n")

    if only:
        ensure_plan_file_once(plan, tune_lock, write_plan_body)
    else:
        with plan.open("w", encoding="utf-8") as f:
            write_plan_body(f)

    for idx, trial in enumerate(trials, 1):
        cfg = apply_trial(base_cfg, trial, args.max_episodes)
        cfg_path = cfg_dir / f"{idx:02d}_{trial['name']}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

        if only and idx not in only:
            continue

        msg = (
            f"\n[{idx}/{len(trials)}] {trial['name']} | "
            f"min_sim={trial['semantic_gaussian_min_sim']}, "
            f"high_conf={trial['semantic_gaussian_high_conf_sim']}, "
            f"ov_w={trial['open_vocab_semantic_weight']}, "
            f"geo_w={trial['fusion_geometric_weight']}"
        )
        if "open_vocab_min_geometric_matches" in trial:
            msg += (
                f" | min_geom={trial['open_vocab_min_geometric_matches']} "
                f"top_k={trial['open_vocab_top_k']} "
                f"dd={trial['semantic_gaussian_ov_disagree_dist']} "
                f"fd={trial['semantic_gaussian_ov_disagree_force_dist']}"
            )
        print(msg)

        if args.dry_run:
            results.append(
                {
                    "trial": trial,
                    "cfg_path": str(cfg_path),
                    "exit_code": None,
                    "metrics": None,
                    "log_path": "",
                }
            )
            continue

        run_log_path = log_dir / f"{idx:02d}_{trial['name']}.log"
        if args.skip_completed:
            existing = load_existing_log_result(run_log_path)
            if is_effectively_completed(existing, args.max_episodes) and existing["metrics"] is not None:
                print(f"  -> skip completed trial from existing log: {run_log_path}")
                results.append(
                    {
                        "trial": trial,
                        "cfg_path": str(cfg_path),
                        "log_path": str(run_log_path),
                        "exit_code": 0,
                        "metrics": existing["metrics"],
                    }
                )
                merge_summary_tsv(
                    summary, summary_lock, trials, cfg_dir, log_dir, [results[-1]]
                )
                continue

        r = run_one(nav_root, cfg_path, run_log_path, args.cuda_visible_devices)
        results.append(
            {
                "trial": trial,
                "cfg_path": str(cfg_path),
                "log_path": str(run_log_path),
                "exit_code": r["exit_code"],
                "metrics": r["metrics"],
            }
        )
        merge_summary_tsv(summary, summary_lock, trials, cfg_dir, log_dir, [results[-1]])

    if args.dry_run and results:
        merge_summary_tsv(summary, summary_lock, trials, cfg_dir, log_dir, results)

    by_name = {t["name"]: t for t in trials}
    scored = load_scored_from_summary(summary)
    scored.sort(
        key=lambda x: (x["succ"], x["spl"], -x["dist"]),
        reverse=True,
    )

    print("\n=== Tuning Summary ===")
    print(f"output_dir: {out_dir}")
    if not scored:
        print("No completed trial with parsed metrics.")
        return 0
    for i, row in enumerate(scored, 1):
        t = by_name.get(row["name"])
        tail = ""
        if t is not None:
            tail = (
                f"min={t['semantic_gaussian_min_sim']} hc={t['semantic_gaussian_high_conf_sim']} "
                f"ov_w={t['open_vocab_semantic_weight']} geo_w={t['fusion_geometric_weight']}"
            )
            if "open_vocab_min_geometric_matches" in t:
                tail += (
                    f" gm={t['open_vocab_min_geometric_matches']} k={t['open_vocab_top_k']} "
                    f"dd={t['semantic_gaussian_ov_disagree_dist']} "
                    f"fd={t['semantic_gaussian_ov_disagree_force_dist']}"
                )
        print(
            f"{i:02d}. {row['name']:22s} | succ={row['succ']:.5f} spl={row['spl']:.5f} "
            f"dist={row['dist']:.5f} count={row['count']} | {tail}"
        )
    print(f"summary_tsv: {summary}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune SG/OV goal selection parameters.")
    parser.add_argument(
        "--suite",
        type=str,
        choices=("classic", "extended", "success_push", "all"),
        default="extended",
        help=(
            "classic: min_sim/hc/ov_w 网格；extended: min_geom_matches/top_k/disagree；"
            "success_push: 围绕 gm40_k8 的局部网格；all: 三者串联。"
        ),
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/instance_imagenav_val_nogt.yaml",
        help="Base config path (relative to GaussianNavigation root or absolute).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Override end2end_imagenav.max_episodes for all trials.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES for all runs (e.g. '5').",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only generate configs and summary plan, do not execute run.py.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help=(
            "Resume an existing tuning directory, e.g. "
            "experiments/tune_goal_params/20260413_004830"
        ),
    )
    parser.add_argument(
        "--only_trials",
        type=str,
        default=None,
        help=(
            "仅运行指定 trial 序号（1-based），逗号分隔，支持区间，如 '1,3,5' 或 '4-8'。"
            "多 GPU 并行时对同一 --resume_dir 使用互补的区间并配合不同 "
            "--cuda_visible_devices；建议各进程使用不同 --session_log。"
        ),
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        default=True,
        help="When resuming, skip trials whose log already contains 'all episode over'.",
    )
    parser.add_argument(
        "--no_skip_completed",
        action="store_false",
        dest="skip_completed",
        help="Do not skip completed trials on resume; rerun all trials.",
    )
    parser.add_argument(
        "--no_session_log",
        action="store_true",
        help="不写入完整终端会话日志（默认: <output_dir>/session_terminal.log）。",
    )
    parser.add_argument(
        "--session_log",
        type=str,
        default=None,
        help="会话日志路径；默认 <output_dir>/session_terminal.log。",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="summary.tsv",
        help="汇总文件名（位于 output_dir 下），如 summary_1000ep_success_push.tsv。",
    )
    parser.add_argument(
        "--log_dirname",
        type=str,
        default="logs",
        help="trial 日志子目录名（位于 output_dir 下），如 logs_1000ep_success_push。",
    )
    args = parser.parse_args()
    try:
        args.only_trials_set = parse_only_trials(args.only_trials)
    except ValueError as e:
        parser.error(str(e))

    nav_root = Path(__file__).resolve().parent.parent
    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.is_absolute():
        base_cfg_path = nav_root / base_cfg_path
    if not base_cfg_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.is_absolute():
            out_dir = nav_root / out_dir
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = nav_root / "experiments" / "tune_goal_params" / ts
    if args.suite == "success_push":
        if args.summary_name == "summary.tsv":
            args.summary_name = "summary_1000ep_success_push.tsv"
        if args.log_dirname == "logs":
            args.log_dirname = "logs_1000ep_success_push"
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / args.log_dirname
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.session_log:
        session_path = Path(args.session_log)
        if not session_path.is_absolute():
            session_path = nav_root / session_path
    else:
        session_path = out_dir / "session_terminal.log"

    if str(nav_root) not in sys.path:
        sys.path.insert(0, str(nav_root))
    from utils.tee_log import tee_session_log

    log_cm = tee_session_log(session_path) if not args.no_session_log else nullcontext()
    with log_cm:
        if not args.no_session_log:
            print(
                f"[tune_goal_params] 终端日志同时写入: {session_path.resolve()}",
                flush=True,
            )
        return run_tune_trials(
            args,
            base_cfg,
            nav_root,
            base_cfg_path,
            out_dir,
            cfg_dir,
            log_dir,
        )


if __name__ == "__main__":
    raise SystemExit(main())
