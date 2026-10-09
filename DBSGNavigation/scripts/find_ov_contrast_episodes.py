#!/usr/bin/env python3
"""After probe runs on a scene, find full-success & wo_ov-fail episodes by category."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ep_cats(data_root: Path) -> dict:
    ep_cat = {}
    for fn in data_root.iterdir():
        if not fn.name.endswith(".json.gz"):
            continue
        scene = fn.name.replace(".json.gz", "")
        with gzip.open(fn, "rt") as f:
            data = json.load(f)
        for i, ep in enumerate(data.get("episodes", []), start=1):
            raw = (ep.get("object_category") or "").lower().strip()
            if raw == "plant":
                raw = "potted_plant"
            ep_cat[(scene, i)] = raw
    return ep_cat


def parse_summary(path: Path) -> dict:
    out = {}
    pat = re.compile(
        r"scene=(\S+) ep=(\d+).*?success=([01]) spl=([\d.]+).*?dist_to_goal=([\d.]+)"
    )
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "episode_goal_summary" not in line:
                continue
            m = pat.search(line)
            if m:
                out[(m.group(1), int(m.group(2)))] = {
                    "success": int(m.group(3)),
                    "spl": float(m.group(4)),
                    "dist": float(m.group(5)),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-log", required=True)
    ap.add_argument("--wo-log", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--categories",
        default="potted_plant,tv_monitor",
    )
    args = ap.parse_args()

    data_root = ROOT / "data/datasets/instance_imagenav/hm3d/v3/val/content"
    ep_cat = load_ep_cats(data_root)
    full = parse_summary(Path(args.full_log))
    wo = parse_summary(Path(args.wo_log))
    want = {c.strip() for c in args.categories.split(",") if c.strip()}

    by_cat = defaultdict(list)
    for key in set(full) & set(wo):
        if key[0] != args.scene:
            continue
        cat = ep_cat.get(key)
        if cat not in want:
            continue
        if full[key]["success"] == 1 and wo[key]["success"] == 0:
            by_cat[cat].append((key, full[key], wo[key]))

    for cat in sorted(want):
        items = by_cat.get(cat, [])
        items.sort(key=lambda x: (-x[1]["spl"], x[2]["dist"]))
        print(f"\n{cat}: {len(items)} contrast episode(s)")
        for (scene, ep), fr, wr in items[:5]:
            print(
                f"  scene={scene} ep={ep} | "
                f"full spl={fr['spl']:.3f} dist={fr['dist']:.3f} | "
                f"wo_ov dist={wr['dist']:.3f}"
            )


if __name__ == "__main__":
    main()
