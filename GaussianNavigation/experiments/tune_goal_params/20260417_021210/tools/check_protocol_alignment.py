#!/usr/bin/env python3
"""
Compare two yaml configs on evaluation-protocol-critical keys.

Usage:
  python tools/check_protocol_alignment.py --a path/to/train.yaml --b path/to/val.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")


KEYS = [
    "habitat.dataset.split",
    "habitat.environment.max_episode_seconds",
    "end2end_imagenav.split",
    "end2end_imagenav.allow_gt_fallback",
    "end2end_imagenav.non_gt_fallback_handling",
    "end2end_imagenav.auto_stop_when_within_success_dist",
    "end2end_imagenav.auto_stop_geodesic_max",
    "end2end_imagenav.use_open_vocab_benchmark",
]


def get_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return "<MISSING>"
        cur = cur[p]
    return cur


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping yaml")
    return data


def rows(a_cfg: Dict[str, Any], b_cfg: Dict[str, Any], keys: Iterable[str]) -> Iterable[Tuple[str, Any, Any, bool]]:
    for k in keys:
        av = get_path(a_cfg, k)
        bv = get_path(b_cfg, k)
        yield k, av, bv, av == bv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="first yaml path")
    parser.add_argument("--b", required=True, help="second yaml path")
    args = parser.parse_args()

    a_path = Path(args.a).resolve()
    b_path = Path(args.b).resolve()
    a_cfg = load_yaml(a_path)
    b_cfg = load_yaml(b_path)

    print(f"[protocol-check] A: {a_path}")
    print(f"[protocol-check] B: {b_path}")
    print("")
    print(f"{'key':55s} | {'A':22s} | {'B':22s} | same")
    print("-" * 115)
    mismatch = 0
    for k, av, bv, same in rows(a_cfg, b_cfg, KEYS):
        if not same:
            mismatch += 1
        print(f"{k:55s} | {str(av):22.22s} | {str(bv):22.22s} | {'Y' if same else 'N'}")
    print("-" * 115)
    if mismatch == 0:
        print("RESULT: protocol-aligned on tracked keys.")
    else:
        print(f"RESULT: {mismatch} mismatches on tracked keys.")


if __name__ == "__main__":
    main()

