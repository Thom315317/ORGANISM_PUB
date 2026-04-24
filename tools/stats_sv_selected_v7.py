#!/usr/bin/env python3
"""
stats_sv_selected_v7.py — Reanalysis on sv_selected (winning draft only)
=========================================================================
Same analysis as stats_reanalysis_v7.py but on the 'selected' sim_curves
instead of 'mean'. No averaging artifact possible — selected is a single
agent's embedding at each tick.

Usage:
    python tools/stats_sv_selected_v7.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, ttest_1samp

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

METRICS = ["t15_sel", "t35_sel"]
CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]

WINDOWS = {
    15: {"k_max": 19, "label": "t15"},
    35: {"k_max": 44, "label": "t35"},
}


def load_data(runs_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load selected sim_curves + per-tick curves for gradient analysis."""
    rows = []
    gradient_data = defaultdict(lambda: {"sv_selected": None, "condition": None})

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
                sel_vals = [v for v in sim[tick_key].get("selected", []) if v is not None]
                if tick_key == "tick_15":
                    row[f"{prefix}_sel"] = float(np.mean(sel_vals[:19])) if len(sel_vals) >= 19 else (float(np.mean(sel_vals)) if sel_vals else None)
                else:
                    row[f"{prefix}_sel"] = float(np.mean(sel_vals)) if sel_vals else None
            else:
                row[f"{prefix}_sel"] = None

        rows.append(row)

        # Store full selected curves for gradient analysis
        if condition in CONDITIONS_MULTI or condition == "E":
            sv_sel = d.get("state_vector_selected", [])
            if sv_sel and sv_sel[0]:
                gradient_data[run_dir.name] = {
                    "sv_selected": np.array(sv_sel, dtype=np.float64),
                    "condition": condition,
                    "seed": seed,
                    "sim_curves_sel": {
                        tk: sim.get(tk, {}).get("selected", [])
                        for tk in ["tick_15", "tick_35"]
                    },
                }

    return pd.DataFrame(rows), dict(gradient_data)


def bootstrap_ci_median_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000) -> Tuple[float, float, float]:
    diffs = x - y
    median_obs = float(np.median(diffs))
    rng = np.random.default_rng(42)
    boot_medians = [np.median(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_medians, 2.5))
    hi = float(np.percentile(boot_medians, 97.5))
    return median_obs, lo, hi


def run_paired_tests(df: pd.DataFrame, seeds: List[int], label: str) -> List[dict]:
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
                    "r_rb": None, "median_diff": None, "ci_lo": None, "ci_hi": None,
                    "note": "insufficient pairs"
                })
                continue

            x = merged[f"{metric}_1"].values
            y = merged[f"{metric}_2"].values
            diffs = x - y

            try:
                stat_two, p_two = wilcoxon(diffs, alternative="two-sided")
            except ValueError:
                stat_two, p_two = 0, 1.0
            try:
                _, p_one = wilcoxon(diffs, alternative=alt)
            except ValueError:
                p_one = 1.0

            # Rank-biserial correlation
            r_rb = 1.0 - (2.0 * stat_two) / (n * (n + 1) / 2.0) if n > 0 else 0.0

            med, ci_lo, ci_hi = bootstrap_ci_median_diff(x, y)

            results.append({
                "subset": label, "comparison": comp_label, "metric": metric,
                "n": n, "W": float(stat_two), "p_two": float(p_two), "p_one": float(p_one),
                "r_rb": float(r_rb), "median_diff": med, "ci_lo": ci_lo, "ci_hi": ci_hi,
            })

    return results


def holm_bonferroni(p_values: List[Optional[float]]) -> List[Optional[float]]:
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = [None] * len(p_values)
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (m - rank), 1.0)
        adj = max(adj, prev)
        corrected[orig_idx] = adj
        prev = adj
    return corrected


def gradient_analysis(gradient_data: Dict, runs_dir: Path) -> str:
    """Spearman rho on selected curves: does selected improve over the window?"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TEMPORAL GRADIENT ON sv_selected")
    lines.append("=" * 80)

    for t_p, wcfg in WINDOWS.items():
        key = wcfg["label"]
        k_max = wcfg["k_max"]
        tk = f"tick_{t_p}"

        by_cond = defaultdict(list)
        all_rhos = []

        for run_id, data in gradient_data.items():
            cond = data["condition"]
            if cond not in CONDITIONS_MULTI and cond != "E":
                continue
            sel_curve = data["sim_curves_sel"].get(tk, [])
            valid = [(k + 1, v) for k, v in enumerate(sel_curve) if v is not None]
            if len(valid) < 5:
                continue

            ranks = [p[0] for p in valid]
            sims = [p[1] for p in valid]
            rho, _ = spearmanr(ranks, sims)

            all_rhos.append(rho)
            by_cond[cond].append(rho)

        if not all_rhos:
            continue

        rhos = np.array(all_rhos)
        t_stat, p_val = ttest_1samp(rhos, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        lines.append(f"\n  {key.upper()} — Selected gradient")
        lines.append(f"  n={len(rhos)} mean_rho={np.mean(rhos):.4f} std={np.std(rhos):.4f} "
                      f"median={np.median(rhos):.4f}")
        lines.append(f"  pos={np.sum(rhos>0)}/{len(rhos)} t={t_stat:.3f} p={p_val:.6f} {sig}")

        lines.append(f"\n  {'Cond':<6} {'n':>4} {'mean_rho':>10} {'%pos':>6} {'p':>10} {'sig':>5}")
        lines.append("  " + "-" * 42)
        for cond in sorted(by_cond):
            cr = np.array(by_cond[cond])
            if len(cr) < 3:
                lines.append(f"  {cond:<6} {len(cr):>4} insufficient")
                continue
            t_c, p_c = ttest_1samp(cr, 0)
            sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
            lines.append(f"  {cond:<6} {len(cr):>4} {np.mean(cr):>10.4f} {100*np.sum(cr>0)/len(cr):>5.0f}% {p_c:>10.6f} {sig_c:>5}")

    gradient_report = "\n".join(lines)

    gpath = runs_dir / "sv_selected_gradient_report.txt"
    with open(gpath, "w") as f:
        f.write(gradient_report)

    return gradient_report


def main():
    parser = argparse.ArgumentParser(description="sv_selected reanalysis V7")
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # Load
    df, gradient_data = load_data(runs_dir)
    print(f"Loaded {len(df)} entries: {df['condition'].nunique()} conditions, {df['seed'].nunique()} seeds")

    # Completeness check
    expected = set()
    for s in ALL_SEEDS:
        for c in ["A", "B", "C", "D", "D2", "D3", "E", "R"]:
            expected.add((c, s))
    actual = set(zip(df["condition"], df["seed"]))
    missing = expected - actual
    if missing:
        print(f"WARNING: {len(missing)} missing")
    else:
        print("All 96 entries present.")

    # Descriptives
    lines = []
    lines.append("=" * 80)
    lines.append("SV_SELECTED REANALYSIS — BENCH V7")
    lines.append("=" * 80)
    lines.append("\n--- DESCRIPTIVE STATISTICS (selected) ---")

    for metric in METRICS:
        lines.append(f"\n{metric}:")
        for cond in sorted(df["condition"].unique()):
            vals = df[df["condition"] == cond][metric].dropna().values
            if len(vals) > 0:
                lines.append(f"  {cond:<4}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")

    # Paired tests (12 seeds)
    paired_12 = run_paired_tests(df, ALL_SEEDS, "12 seeds")
    paired_9 = run_paired_tests(df, PRE_REGISTERED_SEEDS, "9 seeds")

    p_ones_12 = [r["p_one"] for r in paired_12]
    p_corrected_12 = holm_bonferroni(p_ones_12)
    p_ones_9 = [r["p_one"] for r in paired_9]
    p_corrected_9 = holm_bonferroni(p_ones_9)

    lines.append("\n--- WILCOXON SIGNED-RANK on sv_selected (12 seeds) ---\n")
    lines.append(f"{'Comparison':<12} {'Metric':<10} {'n':>3} {'W':>8} {'p_one':>10} {'p_corr':>10} {'r_rb':>6} {'med_diff':>10} {'95% CI':>20} {'Sig':>5}")
    lines.append("-" * 100)

    for i, r in enumerate(paired_12):
        if r.get("note"):
            lines.append(f"{r['comparison']:<12} {r['metric']:<10} {r['n']:>3}  {r['note']}")
            continue
        p_corr = p_corrected_12[i]
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ci_str = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        lines.append(f"{r['comparison']:<12} {r['metric']:<10} {r['n']:>3} {r['W']:>8.0f} {r['p_one']:>10.4f} {p_corr:>10.4f} {r['r_rb']:>6.3f} {r['median_diff']:>10.4f} {ci_str:>20} {sig:>5}")

    # Sensitivity 9 vs 12
    lines.append("\n--- SENSITIVITY: 9 vs 12 SEEDS (selected) ---\n")
    lines.append(f"{'Comparison':<12} {'Metric':<10} {'p12_corr':>10} {'p9_corr':>10} {'Change':>8}")
    lines.append("-" * 55)
    for i, (r12, r9) in enumerate(zip(paired_12, paired_9)):
        if r12.get("note") or r9.get("note"):
            continue
        pc12 = p_corrected_12[i]
        pc9 = p_corrected_9[i]
        change = "CHANGED" if (pc12 < 0.05) != (pc9 < 0.05) else ""
        lines.append(f"{r12['comparison']:<12} {r12['metric']:<10} {pc12:>10.4f} {pc9:>10.4f} {change:>8}")

    # ── Comparison mean vs selected ──
    # Load mean results from stats_reanalysis CSV if available
    lines.append("\n--- MEAN vs SELECTED COMPARISON ---\n")

    mean_csv = runs_dir / "stats_reanalysis_v7.csv"
    if mean_csv.exists():
        mean_df = pd.read_csv(mean_csv)
        lines.append(f"{'Comparison':<12} {'Metric_mean':<12} {'p_mean':>10} {'Metric_sel':<12} {'p_sel':>10} {'Note':>10}")
        lines.append("-" * 70)

        for i, r_sel in enumerate(paired_12):
            if r_sel.get("note"):
                continue
            comp = r_sel["comparison"]
            met_sel = r_sel["metric"]
            met_mean = met_sel.replace("_sel", "_mean")
            pc_sel = p_corrected_12[i]

            # Find matching mean row
            match = mean_df[(mean_df["comparison"] == comp) & (mean_df["metric"] == met_mean)]
            if not match.empty:
                pc_mean = match.iloc[0]["p_corr_12"]
                sig_mean = pc_mean < 0.05
                sig_sel = pc_sel < 0.05
                note = ""
                if sig_sel and not sig_mean:
                    note = "SEL ONLY"
                elif sig_mean and not sig_sel:
                    note = "MEAN ONLY"
                elif sig_sel and sig_mean:
                    note = "BOTH"
                lines.append(f"{comp:<12} {met_mean:<12} {pc_mean:>10.4f} {met_sel:<12} {pc_sel:>10.4f} {note:>10}")
            else:
                lines.append(f"{comp:<12} {'—':<12} {'—':>10} {met_sel:<12} {pc_sel:>10.4f}")
    else:
        lines.append("  (stats_reanalysis_v7.csv not found — run stats_reanalysis_v7.py first)")

    report = "\n".join(lines)
    print(report)

    # Gradient analysis
    grad_report = gradient_analysis(gradient_data, runs_dir)
    print(grad_report)
    report += grad_report

    # Save report
    report_path = runs_dir / "stats_sv_selected_v7_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save CSV
    csv_path = runs_dir / "stats_sv_selected_v7.csv"
    csv_rows = []
    for i, (r12, r9) in enumerate(zip(paired_12, paired_9)):
        csv_rows.append({
            "comparison": r12["comparison"],
            "metric": r12["metric"],
            "n_12": r12["n"],
            "W_12": r12.get("W"),
            "p_one_12": r12.get("p_one"),
            "p_corr_12": p_corrected_12[i],
            "r_rb_12": r12.get("r_rb"),
            "median_diff_12": r12.get("median_diff"),
            "ci_lo_12": r12.get("ci_lo"),
            "ci_hi_12": r12.get("ci_hi"),
            "n_9": r9["n"],
            "p_one_9": r9.get("p_one"),
            "p_corr_9": p_corrected_9[i],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
