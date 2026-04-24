#!/usr/bin/env python3
"""
temporal_gradient_test.py — Tick-by-tick recovery gradient test
================================================================
For each run, compute Spearman correlation between tick rank in the
post-perturbation window and cosine similarity to pre-perturbation
reference. rho > 0 = progressive recovery. rho ~ 0 = plateau.

Usage:
    python tools/temporal_gradient_test.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]

WINDOWS = {
    15: {"k_max": 19, "label": "t15"},
    35: {"k_max": 44, "label": "t35"},
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_reference(sv: np.ndarray, t_p: int) -> Optional[np.ndarray]:
    pre_indices = [t_p - 3, t_p - 2, t_p - 1]
    valid = [sv[i] for i in pre_indices if 0 <= i < len(sv) and np.linalg.norm(sv[i]) > 0]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def compute_per_tick_sims(sv: np.ndarray, r_pre: np.ndarray, t_p: int, k_max: int) -> List[Optional[float]]:
    """Cosine sim between reference and each tick in post-perturbation window."""
    sims = []
    for k in range(1, k_max + 1):
        idx = t_p + k
        if idx < len(sv) and np.linalg.norm(sv[idx]) > 0:
            sims.append(cosine_sim(r_pre, sv[idx]))
        else:
            sims.append(None)
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # Load runs
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        if cond not in CONDITIONS_MULTI:
            continue
        sv = d.get("state_vector_mean", [])
        if not sv or not sv[0]:
            continue
        runs.append({
            "name": run_dir.name,
            "condition": cond,
            "seed": d.get("seed", 0),
            "sv": np.array(sv, dtype=np.float64),
        })

    print(f"Loaded {len(runs)} multi-agent runs")

    # ── Compute per-run Spearman rho and per-tick curves ──
    results = {}  # key -> {rhos: [], by_cond: {cond: {rhos, curves}}, global_curve}

    for t_p, wcfg in WINDOWS.items():
        k_max = wcfg["k_max"]
        key = wcfg["label"]

        all_rhos = []
        by_cond = defaultdict(lambda: {"rhos": [], "curves": []})
        all_curves = []

        for run in runs:
            sv = run["sv"]
            cond = run["condition"]

            r_pre = compute_reference(sv, t_p)
            if r_pre is None:
                continue

            sims = compute_per_tick_sims(sv, r_pre, t_p, k_max)

            # Filter valid ticks for Spearman
            valid_pairs = [(k + 1, s) for k, s in enumerate(sims) if s is not None]
            if len(valid_pairs) < 5:
                continue

            ranks = [p[0] for p in valid_pairs]
            sim_vals = [p[1] for p in valid_pairs]

            rho, p_rho = spearmanr(ranks, sim_vals)

            all_rhos.append(rho)
            by_cond[cond]["rhos"].append(rho)
            by_cond[cond]["curves"].append(sims)
            all_curves.append(sims)

        results[key] = {
            "rhos": np.array(all_rhos),
            "by_cond": {c: {"rhos": np.array(d["rhos"]), "curves": d["curves"]} for c, d in by_cond.items()},
            "curves": all_curves,
            "k_max": k_max,
        }

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("TEMPORAL GRADIENT TEST — BENCH V7")
    lines.append("=" * 80)
    lines.append(f"Multi-agent conditions: {CONDITIONS_MULTI}")
    lines.append(f"Runs: {len(runs)}")

    for key, res in results.items():
        rhos = res["rhos"]
        if len(rhos) == 0:
            continue

        t_stat, p_ttest = ttest_1samp(rhos, 0)
        n_pos = np.sum(rhos > 0)
        n_neg = np.sum(rhos < 0)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {key.upper()} WINDOW — Spearman rho (tick rank vs similarity)")
        lines.append(f"{'=' * 60}")
        lines.append(f"  n runs:  {len(rhos)}")
        lines.append(f"  mean rho: {np.mean(rhos):.4f}")
        lines.append(f"  std rho:  {np.std(rhos):.4f}")
        lines.append(f"  median:   {np.median(rhos):.4f}")
        lines.append(f"  positive: {n_pos}/{len(rhos)} ({100*n_pos/len(rhos):.0f}%)")
        lines.append(f"  negative: {n_neg}/{len(rhos)} ({100*n_neg/len(rhos):.0f}%)")
        lines.append(f"  t-test H0 (rho=0): t={t_stat:.3f} p={p_ttest:.6f}")

        if p_ttest < 0.001:
            sig = "***"
        elif p_ttest < 0.01:
            sig = "**"
        elif p_ttest < 0.05:
            sig = "*"
        else:
            sig = "ns"

        if np.mean(rhos) > 0.1 and p_ttest < 0.05:
            interp = "PROGRESSIVE RECOVERY confirmed"
        elif np.mean(rhos) > 0 and p_ttest < 0.05:
            interp = "Weak positive gradient"
        elif p_ttest >= 0.05:
            interp = "No significant gradient (plateau)"
        else:
            interp = "Negative gradient (degradation)"

        lines.append(f"  Significance: {sig}")
        lines.append(f"  Interpretation: {interp}")

        # Per-condition
        lines.append(f"\n  {'Cond':<6} {'n':>4} {'mean_rho':>10} {'std':>8} {'%pos':>6} {'t':>8} {'p':>10} {'sig':>5}")
        lines.append("  " + "-" * 58)

        for cond in sorted(res["by_cond"]):
            cd = res["by_cond"][cond]
            cr = cd["rhos"]
            if len(cr) < 3:
                lines.append(f"  {cond:<6} {len(cr):>4} {'insufficient':>10}")
                continue
            t_c, p_c = ttest_1samp(cr, 0)
            sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
            pct_pos = 100 * np.sum(cr > 0) / len(cr)
            lines.append(f"  {cond:<6} {len(cr):>4} {np.mean(cr):>10.4f} {np.std(cr):>8.4f} {pct_pos:>5.0f}% {t_c:>8.3f} {p_c:>10.6f} {sig_c:>5}")

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = runs_dir / "temporal_gradient_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # ── Plot ─��
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_windows = len(results)
        n_conds = len(CONDITIONS_MULTI)
        fig, axes = plt.subplots(n_windows, n_conds + 1, figsize=(4 * (n_conds + 1), 4 * n_windows))
        if n_windows == 1:
            axes = [axes]

        for row, (key, res) in enumerate(results.items()):
            k_max = res["k_max"]

            # Global curve (first column)
            ax = axes[row][0]
            curves = res["curves"]
            if curves:
                arr = np.array([[s if s is not None else np.nan for s in c] for c in curves])
                mean_curve = np.nanmean(arr, axis=0)
                std_curve = np.nanstd(arr, axis=0)
                n_valid = np.sum(~np.isnan(arr), axis=0)
                ci = 1.96 * std_curve / np.sqrt(np.maximum(n_valid, 1))

                x = np.arange(1, k_max + 1)
                ax.plot(x, mean_curve, color="red", linewidth=2, label="mean")
                ax.fill_between(x, mean_curve - ci, mean_curve + ci, alpha=0.2, color="red")
                ax.set_title(f"{key.upper()} — ALL (n={len(curves)})")
                ax.set_xlabel("k (ticks after perturbation)")
                ax.set_ylabel("Cosine similarity to reference")
                ax.set_ylim(0.5, 1.0)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            # Per-condition (remaining columns)
            colors = {"C": "green", "R": "purple", "D": "brown", "D2": "cyan", "D3": "pink"}
            for col, cond in enumerate(sorted(CONDITIONS_MULTI)):
                ax = axes[row][col + 1]
                if cond in res["by_cond"]:
                    cd = res["by_cond"][cond]
                    curves_c = cd["curves"]
                    if curves_c:
                        arr = np.array([[s if s is not None else np.nan for s in c] for c in curves_c])
                        mean_c = np.nanmean(arr, axis=0)
                        std_c = np.nanstd(arr, axis=0)
                        n_v = np.sum(~np.isnan(arr), axis=0)
                        ci_c = 1.96 * std_c / np.sqrt(np.maximum(n_v, 1))

                        x = np.arange(1, k_max + 1)
                        ax.plot(x, mean_c, color=colors.get(cond, "black"), linewidth=2)
                        ax.fill_between(x, mean_c - ci_c, mean_c + ci_c, alpha=0.2, color=colors.get(cond, "gray"))

                        rho_mean = np.mean(cd["rhos"])
                        ax.set_title(f"{key.upper()} — {cond} (n={len(curves_c)}, rho={rho_mean:.3f})")
                else:
                    ax.set_title(f"{key.upper()} — {cond} (no data)")

                ax.set_xlabel("k")
                ax.set_ylabel("Cosine sim")
                ax.set_ylim(0.5, 1.0)
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = runs_dir / "temporal_gradient_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
