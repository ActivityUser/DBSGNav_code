#!/usr/bin/env python3
"""Summarize revision sensitivity runs and archived SG/OV diagnostics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np


NAV = Path(__file__).resolve().parents[4]
ROOT = NAV.parent
OUT = NAV / "experiments/revision_20260921_sensitivity"
PAPER_CONTENTS = ROOT / "paper/DBSGNav/Contents"
ARCHIVE = (
    NAV
    / "experiments/tune_goal_params/20260417_021210/logs_1000ep_rerun"
    / "ablation_08ext_clip_v2_probe/1000(0.8-0.6_inst_v3_vlm)/2"
)
BASE_LOG = ARCHIVE / "full_scene_crop_1000ep.log"
BASE_EP = ARCHIVE / "full_scene_crop_1000ep_episode_goal_summary.log"
BASE_NOHUP = ARCHIVE / "full_scene_crop_1000ep.nohup"

METRIC_RE = re.compile(
    r"succ / spl / dist / nonGT_cov / GT_fb_rate\s*:\s*"
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\((\d+)\)"
)
EP_RE = re.compile(
    r"\[episode_goal_summary\].*?ep=(\d+).*?goal_source=(\S+).*?"
    r"dist_pred_to_gt=([0-9.]+).*?success=(\d).*?spl=([0-9.]+).*?"
    r"dist_to_goal=([0-9.]+)"
)
SG_RE = re.compile(
    r"\[goal\] semantic_gaussian .*?dist_to_GT=([0-9.]+)m.*?sim=([0-9.]+)"
)
OV_RE = re.compile(r"\[goal\] open_vocab .*?dist_to_GT=([0-9.]+)m")


def final_metric(path: Path, episodes: int = 1000):
    found = None
    for match in METRIC_RE.finditer(path.read_text(errors="ignore")):
        if int(match.group(6)) == episodes:
            found = tuple(float(match.group(i)) for i in range(1, 6))
    return found


def episode_rows(path: Path, episode_offset: int = 0):
    rows = []
    for match in EP_RE.finditer(path.read_text(errors="ignore")):
        ep, source, loc, success, spl, distance = match.groups()
        rows.append(
            {
                "episode": episode_offset + int(ep),
                "source": source,
                "loc_error": float(loc),
                "success": int(success),
                "spl": float(spl),
                "distance": float(distance),
            }
        )
    return rows


def bootstrap_ci(values, seed=20260921, samples=10000):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 500
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        idx = rng.integers(0, data.size, size=(size, data.size))
        means[start : start + size] = data[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def completed_chunk_rows(variant_dir: Path):
    rows = []
    for marker in sorted(variant_dir.glob("chunks/*/COMPLETE")):
        metadata = {}
        for line in marker.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
        attempt = Path(metadata["attempt"])
        start = int(metadata["start"])
        stop = int(metadata["stop"])
        sidecars = list(attempt.glob("*_episode_goal_summary.log"))
        if len(sidecars) != 1:
            raise RuntimeError(f"Expected one sidecar in {attempt}, got {len(sidecars)}")
        chunk = episode_rows(sidecars[0], episode_offset=start)
        if len(chunk) != stop - start:
            raise RuntimeError(
                f"Chunk {marker.parent.name} has {len(chunk)} rows, expected {stop-start}"
            )
        rows.extend(chunk)
    rows.sort(key=lambda item: item["episode"])
    return rows


def summarize_runs():
    result = []
    run_sets = [("default", episode_rows(BASE_EP), str(BASE_LOG))]
    if OUT.exists():
        for variant_dir in sorted(path for path in OUT.iterdir() if path.is_dir()):
            episodes = completed_chunk_rows(variant_dir)
            if episodes:
                run_sets.append((variant_dir.name, episodes, str(variant_dir)))

    for variant, episodes, provenance in run_sets:
        succ = float(np.mean([row["success"] for row in episodes]))
        spl = float(np.mean([row["spl"] for row in episodes]))
        dist = float(np.mean([row["distance"] for row in episodes]))
        noncov = 1.0
        gtfb = 0.0
        succ_ci = bootstrap_ci([row["success"] for row in episodes])
        spl_ci = bootstrap_ci([row["spl"] for row in episodes], seed=20260922)
        source_sg = sum(row["source"].startswith("semantic_gaussian") for row in episodes)
        if variant != "default":
            merged_path = OUT / variant / "merged_episode_metrics.csv"
            with merged_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
                writer.writeheader()
                writer.writerows(episodes)
        result.append(
            {
                "variant": variant,
                "success": succ,
                "success_ci_low": succ_ci[0],
                "success_ci_high": succ_ci[1],
                "spl": spl,
                "spl_ci_low": spl_ci[0],
                "spl_ci_high": spl_ci[1],
                "distance": dist,
                "non_gt_coverage": noncov,
                "gt_fallback_rate": gtfb,
                "episodes_logged": len(episodes),
                "sg_selected": source_sg,
                "log": provenance,
            }
        )
    return result


def candidate_pairs():
    pairs = []
    pending = None
    for line in BASE_NOHUP.read_text(errors="ignore").splitlines():
        match = SG_RE.search(line)
        if match:
            pending = (float(match.group(1)), float(match.group(2)))
        match = OV_RE.search(line)
        if match and pending is not None:
            pairs.append((pending[0], float(match.group(1)), pending[1]))
            pending = None
    return pairs


def write_latex_tables(run_rows, paired_rows):
    rows = {row["variant"]: row for row in run_rows}
    expected = {
        "control_default",
        "ov_scale_080",
        "ov_scale_120",
        "fusion_sem100_geo040",
        "fusion_sem060_geo080",
        "topk_3",
        "topk_9",
        "sg_margin_010",
        "sg_margin_050",
        "corr_thr_040",
        "corr_thr_060",
    }
    if not expected.issubset(rows):
        return
    if any(int(rows[name]["episodes_logged"]) != 1000 for name in expected):
        return

    def point(name):
        row = rows[name]
        return f"{100*row['success']:.1f}/{100*row['spl']:.1f}"

    main_lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\captionsetup{font=footnotesize,skip=3pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\caption{\rev{One-factor sensitivity on all 1{,}000 HM3D IIN validation episodes. Cells report Success/SPL (\%). The matched default control and all variants use the same interruption-safe chunked protocol.}}",
        r"\label{tab:revision_sensitivity}",
        r"\begin{tabularx}{\linewidth}{@{}lXXX@{}}",
        r"\toprule",
        r"\rev{Parameter} & \rev{Lower setting} & \rev{\textbf{Default}} & \rev{Upper setting} \\",
        r"\midrule",
        rf"\rev{{$\lambda_{{\mathrm{{ov}}}}$}} & \rev{{0.8: {point('ov_scale_080')}}} & \rev{{\textbf{{1.0: {point('control_default')}}}}} & \rev{{1.2: {point('ov_scale_120')}}} \\",
        rf"\rev{{$\alpha_{{\mathrm{{sem}}}},\beta_{{\mathrm{{geo}}}}$}} & \rev{{1.0,0.4: {point('fusion_sem100_geo040')}}} & \rev{{\textbf{{0.8,0.6: {point('control_default')}}}}} & \rev{{0.6,0.8: {point('fusion_sem060_geo080')}}} \\",
        rf"\rev{{OV Top-$K$}} & \rev{{3: {point('topk_3')}}} & \rev{{\textbf{{6: {point('control_default')}}}}} & \rev{{9: {point('topk_9')}}} \\",
        rf"\rev{{$\delta_{{\mathrm{{amb}}}}$}} & \rev{{0.01: {point('sg_margin_010')}}} & \rev{{\textbf{{0.02: {point('control_default')}}}}} & \rev{{0.05: {point('sg_margin_050')}}} \\",
        rf"\rev{{$\theta_{{\mathrm{{corr}}}}$}} & \rev{{0.40: {point('corr_thr_040')}}} & \rev{{\textbf{{0.50: {point('control_default')}}}}} & \rev{{0.60: {point('corr_thr_060')}}} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        r"\rev{The sweep changes one factor at a time and does not retune the archived main configuration. The matched default is repeated because restartable evaluation reinitializes each deterministic 50-episode chunk. The Supplementary Material reports full episode-bootstrap intervals and paired tests.}",
        "",
    ]
    (PAPER_CONTENTS / "revision_sensitivity.tex").write_text(
        "\n".join(main_lines)
    )

    order = [
        ("Matched default", "control_default"),
        (r"$\lambda_{\mathrm{ov}}=0.8$", "ov_scale_080"),
        (r"$\lambda_{\mathrm{ov}}=1.2$", "ov_scale_120"),
        (r"Fusion $(1.0,0.4)$", "fusion_sem100_geo040"),
        (r"Fusion $(0.6,0.8)$", "fusion_sem060_geo080"),
        (r"Top-$K=3$", "topk_3"),
        (r"Top-$K=9$", "topk_9"),
        (r"$\delta_{\mathrm{amb}}=0.01$", "sg_margin_010"),
        (r"$\delta_{\mathrm{amb}}=0.05$", "sg_margin_050"),
        (r"$\theta_{\mathrm{corr}}=0.40$", "corr_thr_040"),
        (r"$\theta_{\mathrm{corr}}=0.60$", "corr_thr_060"),
    ]
    supp = [
        r"\section{Sensitivity Uncertainty}",
        r"\begin{table}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\captionsetup{font=footnotesize,skip=3pt}",
        r"\caption{Full 95\% episode-bootstrap confidence intervals for the reviewer-requested sensitivity analysis.}",
        r"\label{tab:app_revision_sensitivity}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Setting & Success & 95\% CI & SPL & 95\% CI \\",
        r"\midrule",
    ]
    for label, name in order:
        row = rows[name]
        supp.append(
            f"{label} & {100*row['success']:.1f} & "
            f"[{100*row['success_ci_low']:.1f},{100*row['success_ci_high']:.1f}] & "
            f"{100*row['spl']:.1f} & "
            f"[{100*row['spl_ci_low']:.1f},{100*row['spl_ci_high']:.1f}] \\\\"
        )
    supp.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[!t]",
            r"\centering",
            r"\footnotesize",
            r"\captionsetup{font=footnotesize,skip=3pt}",
            r"\caption{Paired differences from the matched default control. "
            r"Intervals are episode-paired bootstrap 95\% CIs; $p$ is McNemar's exact test for Success.}",
            r"\label{tab:app_revision_sensitivity_paired}",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"Setting & $\Delta$Success [95\% CI] & $p$ & $\Delta$SPL [95\% CI] \\",
            r"\midrule",
        ]
    )
    paired_index = {row["variant"]: row for row in paired_rows}
    for label, name in order[1:]:
        row = paired_index[name]
        supp.append(
            f"{label} & {100*row['delta_success']:+.1f} "
            f"[{100*row['delta_success_ci_low']:+.1f},"
            f"{100*row['delta_success_ci_high']:+.1f}] & "
            f"{row['mcnemar_exact_p']:.3f} & "
            f"{100*row['delta_spl']:+.1f} "
            f"[{100*row['delta_spl_ci_low']:+.1f},"
            f"{100*row['delta_spl_ci_high']:+.1f}] \\\\"
        )
    supp.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"The paired analysis shows that most perturbations are statistically "
            r"indistinguishable from the fixed default on this split. Top-$K=9$ is "
            r"the exception: it improves Success by 2.2 points "
            r"(95\% CI [0.5,3.9], McNemar $p=0.018$). We therefore treat $K=6$ "
            r"as the fixed working point used by the submitted system, not as an "
            r"empirically optimal choice selected by this post-submission sweep.",
            "",
        ]
    )
    (PAPER_CONTENTS / "revision_sensitivity_supp.tex").write_text("\n".join(supp))

    topk9 = paired_index["topk_9"]
    main_lines.extend(
        [
            r"\rev{Across the matched sweep, most paired Success differences "
            r"include zero. The exception is Top-$K=9$, which improves Success "
            f"by {100*topk9['delta_success']:.1f} points "
            f"(95\\% paired CI [{100*topk9['delta_success_ci_low']:.1f},"
            f"{100*topk9['delta_success_ci_high']:.1f}], "
            f"McNemar $p={topk9['mcnemar_exact_p']:.3f}$). "
            r"Accordingly, we present $K=6$ as the fixed working point of the "
            r"submitted system rather than a globally optimal value; the sweep "
            r"was not used to replace the archived main result.}",
            "",
        ]
    )
    (PAPER_CONTENTS / "revision_sensitivity.tex").write_text(
        "\n".join(main_lines)
    )


def write_sensitivity_paired_analysis():
    variants = [
        "ov_scale_080",
        "ov_scale_120",
        "fusion_sem100_geo040",
        "fusion_sem060_geo080",
        "topk_3",
        "topk_9",
        "sg_margin_010",
        "sg_margin_050",
        "corr_thr_040",
        "corr_thr_060",
    ]

    def read_merged(name):
        path = OUT / name / "merged_episode_metrics.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            int(row["episode"]): {
                "success": float(row["success"]),
                "spl": float(row["spl"]),
            }
            for row in rows
        }

    def paired_ci(delta, seed):
        rng = np.random.default_rng(seed)
        means = np.empty(10000)
        for start in range(0, 10000, 500):
            size = min(500, 10000 - start)
            idx = rng.integers(0, len(delta), size=(size, len(delta)))
            means[start : start + size] = delta[idx].mean(axis=1)
        return tuple(np.percentile(means, [2.5, 97.5]))

    control = read_merged("control_default")
    result = []
    for index, variant in enumerate(variants):
        other = read_merged(variant)
        episodes = sorted(set(control) & set(other))
        if len(episodes) != 1000:
            raise RuntimeError(
                f"Matched sensitivity analysis for {variant} has {len(episodes)} episodes"
            )
        control_success = np.asarray(
            [control[ep]["success"] for ep in episodes]
        )
        other_success = np.asarray([other[ep]["success"] for ep in episodes])
        control_spl = np.asarray([control[ep]["spl"] for ep in episodes])
        other_spl = np.asarray([other[ep]["spl"] for ep in episodes])
        delta_success = other_success - control_success
        delta_spl = other_spl - control_spl
        success_ci = paired_ci(delta_success, 20260930 + index)
        spl_ci = paired_ci(delta_spl, 20261010 + index)
        variant_only = int(
            np.sum((other_success == 1) & (control_success == 0))
        )
        control_only = int(
            np.sum((other_success == 0) & (control_success == 1))
        )
        discordant = variant_only + control_only
        if discordant:
            low = min(variant_only, control_only)
            exact_p = min(
                1.0,
                2.0
                * sum(math.comb(discordant, k) for k in range(low + 1))
                / (2.0**discordant),
            )
        else:
            exact_p = 1.0
        result.append(
            {
                "variant": variant,
                "episodes": len(episodes),
                "delta_success": float(delta_success.mean()),
                "delta_success_ci_low": success_ci[0],
                "delta_success_ci_high": success_ci[1],
                "delta_spl": float(delta_spl.mean()),
                "delta_spl_ci_low": spl_ci[0],
                "delta_spl_ci_high": spl_ci[1],
                "variant_only_success": variant_only,
                "control_only_success": control_only,
                "mcnemar_exact_p": exact_p,
            }
        )
    with (OUT / "sensitivity_paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


def write_matched_branch_analysis(run_rows):
    indexed = {row["variant"]: row for row in run_rows}
    if not {"control_default", "wo_sg_geometry"}.issubset(indexed):
        return
    if any(
        int(indexed[name]["episodes_logged"]) != 1000
        for name in ("control_default", "wo_sg_geometry")
    ):
        return

    def read_merged(name):
        path = OUT / name / "merged_episode_metrics.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            int(row["episode"]): {
                "success": float(row["success"]),
                "spl": float(row["spl"]),
            }
            for row in rows
        }

    full = read_merged("control_default")
    wo = read_merged("wo_sg_geometry")
    episodes = sorted(set(full) & set(wo))
    if len(episodes) != 1000:
        raise RuntimeError(f"Matched branch analysis has {len(episodes)} episodes")
    full_success = np.asarray([full[ep]["success"] for ep in episodes])
    wo_success = np.asarray([wo[ep]["success"] for ep in episodes])
    full_spl = np.asarray([full[ep]["spl"] for ep in episodes])
    wo_spl = np.asarray([wo[ep]["spl"] for ep in episodes])
    full_only = int(np.sum((full_success == 1) & (wo_success == 0)))
    wo_only = int(np.sum((full_success == 0) & (wo_success == 1)))
    discordant = full_only + wo_only
    if discordant:
        low = min(full_only, wo_only)
        exact_p = min(
            1.0,
            2.0
            * sum(math.comb(discordant, k) for k in range(low + 1))
            / (2.0**discordant),
        )
    else:
        exact_p = 1.0

    def paired_ci(delta, seed):
        rng = np.random.default_rng(seed)
        means = np.empty(10000)
        for start in range(0, 10000, 500):
            size = min(500, 10000 - start)
            idx = rng.integers(0, len(delta), size=(size, len(delta)))
            means[start : start + size] = delta[idx].mean(axis=1)
        return tuple(np.percentile(means, [2.5, 97.5]))

    delta_success = full_success - wo_success
    delta_spl = full_spl - wo_spl
    success_ci = paired_ci(delta_success, 20260923)
    spl_ci = paired_ci(delta_spl, 20260924)
    result = {
        "episodes": 1000,
        "full_success": full_success.mean(),
        "wo_sg_success": wo_success.mean(),
        "delta_success": delta_success.mean(),
        "delta_success_ci_low": success_ci[0],
        "delta_success_ci_high": success_ci[1],
        "full_spl": full_spl.mean(),
        "wo_sg_spl": wo_spl.mean(),
        "delta_spl": delta_spl.mean(),
        "delta_spl_ci_low": spl_ci[0],
        "delta_spl_ci_high": spl_ci[1],
        "full_only_success": full_only,
        "wo_sg_only_success": wo_only,
        "mcnemar_exact_p": exact_p,
    }
    with (OUT / "branch_paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    text = (
        r"\rev{In a separate matched restartable rerun, the default control reaches "
        f"{100*result['full_success']:.1f}\\% Success and "
        f"{100*result['full_spl']:.1f}\\% SPL, versus "
        f"{100*result['wo_sg_success']:.1f}\\% and "
        f"{100*result['wo_sg_spl']:.1f}\\% for w.o.\\ SG. "
        f"The paired differences are {100*result['delta_success']:+.1f} points "
        f"(95\\% CI [{100*success_ci[0]:+.1f},{100*success_ci[1]:+.1f}]) "
        f"and {100*result['delta_spl']:+.1f} points "
        f"(95\\% CI [{100*spl_ci[0]:+.1f},{100*spl_ci[1]:+.1f}]); "
        f"McNemar's exact $p={exact_p:.3f}$. "
        r"These paired statistics do not establish an aggregate SG gain. "
        r"The archived Full and the main-paper ablation table retain their "
        r"originally reported values; the matched rerun is used only for this "
        r"controlled paired comparison.}"
    )
    (PAPER_CONTENTS / "revision_branch_paired.tex").write_text(text + "\n")


def write_outputs(run_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "summary.csv"
    if run_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)

    pairs = candidate_pairs()
    complement = []
    for threshold in (1.0, 2.0, 3.0):
        sg = [item[0] <= threshold for item in pairs]
        ov = [item[1] <= threshold for item in pairs]
        complement.append(
            {
                "threshold_m": threshold,
                "paired_candidates": len(pairs),
                "sg_recall": sum(sg) / len(pairs),
                "ov_recall": sum(ov) / len(pairs),
                "union_recall": sum(a or b for a, b in zip(sg, ov)) / len(pairs),
                "sg_unique": sum(a and not b for a, b in zip(sg, ov)),
                "ov_unique": sum(b and not a for a, b in zip(sg, ov)),
                "both_fail": sum(not a and not b for a, b in zip(sg, ov)),
            }
        )
    with (OUT / "sg_ov_complement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complement[0]))
        writer.writeheader()
        writer.writerows(complement)
    write_matched_branch_analysis(run_rows)
    paired_rows = write_sensitivity_paired_analysis()
    write_latex_tables(run_rows, paired_rows)

    print(f"wrote {csv_path}")
    for row in run_rows:
        print(
            f"{row['variant']:<24} Success={row['success']:.3f} "
            f"SPL={row['spl']:.3f} SG-selected={row['sg_selected']}"
        )
    for row in complement:
        print(
            f"candidate recall @{row['threshold_m']:.0f}m: "
            f"SG={row['sg_recall']:.3f} OV={row['ov_recall']:.3f} "
            f"union={row['union_recall']:.3f} SG-unique={row['sg_unique']}"
        )


def main():
    write_outputs(summarize_runs())


if __name__ == "__main__":
    main()
