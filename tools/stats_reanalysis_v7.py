#!/usr/bin/env python3
"""
stats_reanalysis_v7.py — Paired statistical reanalysis of bench V7
===================================================================
Wilcoxon signed-rank (paired by seed) + Holm-Bonferroni correction.
Sensitivity analysis: 9 pre-registered vs 12 total seeds.
Comparison with Mann-Whitney for documentation.

Usage:
    python tools/stats_reanalysis_v7.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

# ── Config ──
PRE_REGISTERED_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024]
ALL_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8]

COMPARISONS = [
    ("C > E", "C", "E", "greater"),
    ("R > E", "R", "E", "greater"),
    ("D > E", "D", "E", "greater"),
    ("D2 > E", "D2", "E", "greater"),
    ("D3 > E", "D3", "E", "greater"),
    ("C > R", "C", "R", "greater"),
    ("D2 > D", "D2", "D", "greater"),
    ("D2 > D3", "D2", "D3", "greater"),
]

METRICS = ["t15_mean", "t35_mean"]


def load_data(runs_dir: Path) -> pd.DataFrame:
    """Load all results into a DataFrame."""
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        condition = d.get("condition", "?")
        seed = d.get("seed", 0)
        sim = d.get("sim_curves") or {}

        row = {"seed": seed, "condition": condition}

        for tick_key, prefix in [("tick_15", "t15"), ("tick_35", "t35")]:
            if tick_key in sim:
                mean_vals = [v for v in sim[tick_key].get("mean", []) if v is not None]
                sel_vals = [v for v in sim[tick_key].get("selected", []) if v is not None]
                if tick_key == "tick_15":
                    # t15_mean = mean of k=1..19 (ticks 16-34)
                    row[f"{prefix}_mean"] = float(np.mean(mean_vals[:19])) if len(mean_vals) >= 19 else (float(np.mean(mean_vals)) if mean_vals else None)
                    row[f"{prefix}_sel"] = float(np.mean(sel_vals[:19])) if len(sel_vals) >= 19 else (float(np.mean(sel_vals)) if sel_vals else None)
                else:
                    # t35_mean = mean of all post-perturbation values
                    row[f"{prefix}_mean"] = float(np.mean(mean_vals)) if mean_vals else None
                    row[f"{prefix}_sel"] = float(np.mean(sel_vals)) if sel_vals else None
            else:
                row[f"{prefix}_mean"] = None
                row[f"{prefix}_sel"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def bootstrap_ci_median_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap CI for median of paired differences."""
    diffs = x - y
    median_obs = float(np.median(diffs))
    rng = np.random.default_rng(42)
    boot_medians = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_medians.append(np.median(sample))
    boot_medians = np.array(boot_medians)
    alpha = 1 - ci
    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return median_obs, lo, hi


def run_paired_tests(df: pd.DataFrame, seeds: List[int], label: str) -> List[dict]:
    """Run Wilcoxon signed-rank tests for all comparisons on given seeds."""
    df_sub = df[df["seed"].isin(seeds)]
    results = []

    for comp_label, c1, c2, alt in COMPARISONS:
        for metric in METRICS:
            v1_df = df_sub[df_sub["condition"] == c1][["seed", metric]].dropna()
            v2_df = df_sub[df_sub["condition"] == c2][["seed", metric]].dropna()

            merged = pd.merge(v1_df, v2_df, on="seed", suffixes=("_1", "_2"))
            n = len(merged)

            if n < 5:
                results.append({
                    "subset": label, "comparison": comp_label, "metric": metric,
                    "n": n, "W": None, "p_two": None, "p_one": None,
                    "r_effect": None, "median_diff": None, "ci_lo": None, "ci_hi": None,
                    "note": "insufficient pairs"
                })
                continue

            x = merged[f"{metric}_1"].values
            y = merged[f"{metric}_2"].values
            diffs = x - y

            # Wilcoxon two-sided
            try:
                stat_two, p_two = wilcoxon(diffs, alternative="two-sided")
            except ValueError:
                stat_two, p_two = 0, 1.0

            # Wilcoxon one-sided (direction as hypothesized)
            try:
                stat_one, p_one = wilcoxon(diffs, alternative=alt)
            except ValueError:
                stat_one, p_one = 0, 1.0

            # Effect size: matched-pairs rank-biserial correlation
            # r_rb = 1 - (2*W) / (n*(n+1)/2), bounded [-1, 1]
            r_effect = 1.0 - (2.0 * stat_two) / (n * (n + 1) / 2.0) if n > 0 else 0.0

            # Bootstrap CI on median diff
            med, ci_lo, ci_hi = bootstrap_ci_median_diff(x, y)

            results.append({
                "subset": label, "comparison": comp_label, "metric": metric,
                "n": n, "W": float(stat_two), "p_two": float(p_two), "p_one": float(p_one),
                "r_effect": float(r_effect), "median_diff": med, "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

    return results


def run_mannwhitney(df: pd.DataFrame, seeds: List[int], label: str) -> List[dict]:
    """Run Mann-Whitney U tests for comparison with paired tests."""
    df_sub = df[df["seed"].isin(seeds)]
    results = []

    for comp_label, c1, c2, alt in COMPARISONS:
        for metric in METRICS:
            v1 = df_sub[df_sub["condition"] == c1][metric].dropna().values
            v2 = df_sub[df_sub["condition"] == c2][metric].dropna().values

            if len(v1) < 3 or len(v2) < 3:
                results.append({
                    "subset": label, "comparison": comp_label, "metric": metric,
                    "U": None, "mw_p_one": None,
                })
                continue

            stat, p = mannwhitneyu(v1, v2, alternative=alt)
            results.append({
                "subset": label, "comparison": comp_label, "metric": metric,
                "U": float(stat), "mw_p_one": float(p),
            })

    return results


def holm_bonferroni(p_values: List[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni correction on a list of p-values (None = skip)."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = [None] * len(p_values)

    prev_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (m - rank)
        adjusted = max(adjusted, prev_corrected)  # monotonicity
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = adjusted
        prev_corrected = adjusted

    return corrected


def format_report(paired_12: List[dict], paired_9: List[dict],
                  mw_12: List[dict], output_dir: Path) -> str:
    """Format and save the full report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL REANALYSIS — BENCH V7")
    lines.append("=" * 80)

    # ── Section 1: Paired tests (12 seeds) ──
    lines.append("\n--- WILCOXON SIGNED-RANK (12 seeds, paired by seed) ---\n")

    # Holm-Bonferroni correction
    p_ones_12 = [r["p_one"] for r in paired_12]
    p_corrected_12 = holm_bonferroni(p_ones_12)

    lines.append(f"{'Comparison':<12} {'Metric':<10} {'n':>3} {'W':>8} {'p_one':>10} {'p_corr':>10} {'r':>6} {'med_diff':>10} {'95% CI':>20} {'Sig':>5}")
    lines.append("-" * 100)

    for i, r in enumerate(paired_12):
        if r.get("note"):
            lines.append(f"{r['comparison']:<12} {r['metric']:<10} {r['n']:>3}  {'':>8} {'':>10} {'':>10} {'':>6} {'':>10} {r['note']:>20}")
            continue
        p_corr = p_corrected_12[i]
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ci_str = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        lines.append(f"{r['comparison']:<12} {r['metric']:<10} {r['n']:>3} {r['W']:>8.0f} {r['p_one']:>10.4f} {p_corr:>10.4f} {r['r_effect']:>6.3f} {r['median_diff']:>10.4f} {ci_str:>20} {sig:>5}")

    # ── Section 2: Sensitivity 9 vs 12 ──
    lines.append("\n--- SENSITIVITY: 9 PRE-REGISTERED vs 12 SEEDS ---\n")

    p_ones_9 = [r["p_one"] for r in paired_9]
    p_corrected_9 = holm_bonferroni(p_ones_9)

    lines.append(f"{'Comparison':<12} {'Metric':<10} {'p12_corr':>10} {'p9_corr':>10} {'Δ sig?':>8}")
    lines.append("-" * 55)

    for i, (r12, r9) in enumerate(zip(paired_12, paired_9)):
        if r12.get("note") or r9.get("note"):
            continue
        pc12 = p_corrected_12[i]
        pc9 = p_corrected_9[i]
        sig12 = pc12 < 0.05
        sig9 = pc9 < 0.05
        change = "CHANGED" if sig12 != sig9 else ""
        lines.append(f"{r12['comparison']:<12} {r12['metric']:<10} {pc12:>10.4f} {pc9:>10.4f} {change:>8}")

    # ── Section 3: Paired vs Mann-Whitney ──
    lines.append("\n--- PAIRED (Wilcoxon) vs UNPAIRED (Mann-Whitney) — 12 seeds ---\n")

    lines.append(f"{'Comparison':<12} {'Metric':<10} {'Wilcoxon_p':>12} {'MW_p':>12} {'Diff':>10}")
    lines.append("-" * 60)

    for r_w, r_mw in zip(paired_12, mw_12):
        if r_w.get("note") or r_mw.get("U") is None:
            continue
        diff = abs(r_w["p_one"] - r_mw["mw_p_one"])
        lines.append(f"{r_w['comparison']:<12} {r_w['metric']:<10} {r_w['p_one']:>12.4f} {r_mw['mw_p_one']:>12.4f} {diff:>10.4f}")

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = output_dir / "stats_reanalysis_v7_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save CSV
    csv_path = output_dir / "stats_reanalysis_v7.csv"
    all_rows = []
    for i, (r12, r9, rmw) in enumerate(zip(paired_12, paired_9, mw_12)):
        row = {
            "comparison": r12["comparison"],
            "metric": r12["metric"],
            "n_12": r12["n"],
            "W_12": r12.get("W"),
            "p_one_12": r12.get("p_one"),
            "p_corr_12": p_corrected_12[i],
            "r_effect_12": r12.get("r_effect"),
            "median_diff_12": r12.get("median_diff"),
            "ci_lo_12": r12.get("ci_lo"),
            "ci_hi_12": r12.get("ci_hi"),
            "n_9": r9["n"],
            "p_one_9": r9.get("p_one"),
            "p_corr_9": p_corrected_9[i],
            "mw_p_one_12": rmw.get("mw_p_one"),
        }
        all_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nReport saved to {report_path}")
    print(f"CSV saved to {csv_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Paired statistical reanalysis of bench V7")
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # Load data
    df = load_data(runs_dir)
    print(f"Loaded {len(df)} entries: {df['condition'].nunique()} conditions, {df['seed'].nunique()} seeds")

    # Check completeness
    expected = set()
    for s in ALL_SEEDS:
        for c in ["A", "B", "C", "D", "D2", "D3", "E", "R"]:
            expected.add((c, s))
    actual = set(zip(df["condition"], df["seed"]))
    missing = expected - actual
    if missing:
        print(f"WARNING: {len(missing)} missing entries:")
        for c, s in sorted(missing):
            print(f"  {c}_seed{s}")
    else:
        print("All 96 entries present.")

    # Descriptive stats
    print("\n--- DESCRIPTIVE STATISTICS ---\n")
    for metric in METRICS:
        print(f"\n{metric}:")
        for cond in sorted(df["condition"].unique()):
            vals = df[df["condition"] == cond][metric].dropna().values
            if len(vals) > 0:
                print(f"  {cond:<4}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")

    # Run tests
    paired_12 = run_paired_tests(df, ALL_SEEDS, "12 seeds")
    paired_9 = run_paired_tests(df, PRE_REGISTERED_SEEDS, "9 seeds")
    mw_12 = run_mannwhitney(df, ALL_SEEDS, "12 seeds")

    # Format and save
    format_report(paired_12, paired_9, mw_12, runs_dir)


if __name__ == "__main__":
    main()
