#!/usr/bin/env python3
"""Generate fig_b1-style PNG/PDF/EPS for OV contrast case study (full vs wo_ov, same target)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "docs" / "paperfile"))

from make_case_figures import make_case_row  # noqa: E402
from paper_matplotlib import setup_matplotlib  # noqa: E402

CASE_ROOT = ROOT / "examples/case_study_ov"
DEFAULT_OUT = ROOT / "docs/paperfile/figs"
SELECTED_JSON = ROOT / "experiments/case_study_ov/selected_contrast_episodes.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-root", default=str(CASE_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--selected", default=str(SELECTED_JSON))
    ap.add_argument("--cleanup-mp4", action="store_true")
    args = ap.parse_args()

    setup_matplotlib()
    case_root = Path(args.case_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.selected, "r", encoding="utf-8") as f:
        selected = json.load(f)

    made = 0
    for cat, info in selected.items():
        for mode in ("full", "wo_ov"):
            case_dir = case_root / mode / cat
            if not case_dir.is_dir():
                print(f"[skip] missing {case_dir}")
                continue
            out_stem = out_dir / f"fig_case_{cat}_{mode}"
            make_case_row(
                str(case_dir),
                str(out_stem),
                end_box_color="#A9D18E" if mode == "full" else None,
            )
            made += 1
            if args.cleanup_mp4:
                for mp4 in case_dir.glob("*-Vis.mp4"):
                    mp4.unlink(missing_ok=True)

    print(f"[done] generated {made} figure sets -> {out_dir}")


if __name__ == "__main__":
    main()
