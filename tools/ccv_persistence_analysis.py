#!/usr/bin/env python3
"""
ccv_persistence_analysis.py — CCV→persistence, distance→persistence, lag analysis
===================================================================================
1. Does pre-perturbation diversity (CCV) predict post-perturbation persistence?
2. Does winner-to-barycenter distance correlate with persistence?
3. Lag analysis: does CCV[t] predict similarity[t+1]?

Usage:
    python tools/ccv_persistence_analysis.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 10

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]
COND_COLORS = {"C": "#27ae60", "R": "#8e44ad", "D": "#e67e22", "D2": "#2980b9", "D3": "#9b59b6"}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_runs(runs_dir: Path) -> List[dict]:
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        if cond not in CONDITIONS_MULTI:
            continue
        seed = d.get("seed", 0)
        ccv = d.get("claim_cosine_variance", [])
        psv = d.get("post_selection_variance", [])
        sim = d.get("sim_curves") or {}
        sv_mean = d.get("state_vector_mean", [])
        sv_sel = d.get("state_vector_selected", [])

        # t15/t35 persistence (selected and mean)
        t15_sel = t35_sel = t15_mean = t35_mean = None
        if "tick_15" in sim:
            sel_vals = [v for v in sim["tick_15"].get("selected", []) if v is not None]
            mean_vals = [v for v in sim["tick_15"].get("mean", []) if v is not None]
            t15_sel = float(np.mean(sel_vals[:19])) if len(sel_vals) >= 19 else (float(np.mean(sel_vals)) if sel_vals else None)
            t15_mean = float(np.mean(mean_vals[:19])) if len(mean_vals) >= 19 else (float(np.mean(mean_vals)) if mean_vals else None)
        if "tick_35" in sim:
            sel_vals = [v for v in sim["tick_35"].get("selected", []) if v is not None]
            mean_vals = [v for v in sim["tick_35"].get("mean", []) if v is not None]
            t35_sel = float(np.mean(sel_vals)) if sel_vals else None
            t35_mean = float(np.mean(mean_vals)) if mean_vals else None

        # CCV pre-perturbation
        ccv_pre_t15 = float(np.mean(ccv[10:15])) if len(ccv) >= 15 else None
        ccv_pre_t35 = float(np.mean(ccv[30:35])) if len(ccv) >= 35 else None

        # Post-selection variance (distance gagnant/barycentre) post-perturbation
        dist_post_t15 = float(np.mean(psv[16:31])) if len(psv) >= 31 else None
        dist_post_t35 = float(np.mean(psv[36:51])) if len(psv) >= 51 else None

        runs.append({
            "name": run_dir.name, "condition": cond, "seed": seed,
            "ccv": ccv, "psv": psv,
            "ccv_pre_t15": ccv_pre_t15, "ccv_pre_t35": ccv_pre_t35,
            "t15_sel": t15_sel, "t35_sel": t35_sel,
            "t15_mean": t15_mean, "t35_mean": t35_mean,
            "dist_post_t15": dist_post_t15, "dist_post_t35": dist_post_t35,
            "sv_mean": sv_mean, "sv_sel": sv_sel,
        })

    return runs


def analysis_1(runs: List[dict], lines: List[str]):
    """CCV pre-perturbation → persistence post-perturbation."""
    lines.append("\n" + "=" * 60)
    lines.append("  ANALYSIS 1: CCV pre-perturbation → persistence")
    lines.append("=" * 60)

    pairs = [
        ("ccv_pre_t15", "t15_sel", "CCV_pre→t15_sel"),
        ("ccv_pre_t15", "t15_mean", "CCV_pre→t15_mean"),
        ("ccv_pre_t35", "t35_sel", "CCV_pre→t35_sel"),
        ("ccv_pre_t35", "t35_mean", "CCV_pre→t35_mean"),
    ]

    for x_key, y_key, label in pairs:
        lines.append(f"\n  --- {label} ---")
        lines.append(f"  {'Scope':<12} {'n':>4} {'rho':>8} {'p':>10} {'sig':>5}")
        lines.append("  " + "-" * 42)

        # Global
        x_all = [r[x_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        y_all = [r[y_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        if len(x_all) >= 5:
            rho, p = spearmanr(x_all, y_all)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(f"  {'GLOBAL':<12} {len(x_all):>4} {rho:>8.4f} {p:>10.4f} {sig:>5}")

        # Per-condition
        for cond in CONDITIONS_MULTI:
            xc = [r[x_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            yc = [r[y_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            if len(xc) >= 5:
                rho, p = spearmanr(xc, yc)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                lines.append(f"  {cond:<12} {len(xc):>4} {rho:>8.4f} {p:>10.4f} {sig:>5}")


def analysis_2(runs: List[dict], lines: List[str]):
    """Distance gagnant/barycentre → persistence."""
    lines.append("\n" + "=" * 60)
    lines.append("  ANALYSIS 2: Winner-to-barycenter distance → persistence")
    lines.append("=" * 60)

    pairs = [
        ("dist_post_t15", "t15_sel", "dist_post→t15_sel"),
        ("dist_post_t35", "t35_sel", "dist_post→t35_sel"),
    ]

    for x_key, y_key, label in pairs:
        lines.append(f"\n  --- {label} ---")
        lines.append(f"  {'Scope':<12} {'n':>4} {'rho':>8} {'p':>10} {'sig':>5}")
        lines.append("  " + "-" * 42)

        x_all = [r[x_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        y_all = [r[y_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        if len(x_all) >= 5:
            rho, p = spearmanr(x_all, y_all)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(f"  {'GLOBAL':<12} {len(x_all):>4} {rho:>8.4f} {p:>10.4f} {sig:>5}")

        for cond in CONDITIONS_MULTI:
            xc = [r[x_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            yc = [r[y_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            if len(xc) >= 5:
                rho, p = spearmanr(xc, yc)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                lines.append(f"  {cond:<12} {len(xc):>4} {rho:>8.4f} {p:>10.4f} {sig:>5}")

    # Comparison across conditions
    lines.append(f"\n  --- Mean distance winner/barycenter by condition ---")
    lines.append(f"  {'Cond':<6} {'dist_t15':>10} {'dist_t35':>10} {'t15_sel':>10} {'t35_sel':>10}")
    lines.append("  " + "-" * 48)
    for cond in CONDITIONS_MULTI:
        cr = [r for r in runs if r["condition"] == cond]
        d15 = [r["dist_post_t15"] for r in cr if r["dist_post_t15"] is not None]
        d35 = [r["dist_post_t35"] for r in cr if r["dist_post_t35"] is not None]
        p15 = [r["t15_sel"] for r in cr if r["t15_sel"] is not None]
        p35 = [r["t35_sel"] for r in cr if r["t35_sel"] is not None]
        lines.append(f"  {cond:<6} {np.mean(d15):>10.4f} {np.mean(d35):>10.4f} "
                      f"{np.mean(p15):>10.4f} {np.mean(p35):>10.4f}")


def analysis_3(runs: List[dict], lines: List[str]):
    """Lag analysis: CCV[t] → similarity_selected[t+1]."""
    lines.append("\n" + "=" * 60)
    lines.append("  ANALYSIS 3: Lag analysis — CCV[t] → sim_selected[t+1]")
    lines.append("=" * 60)

    for t_p, window_start, window_end, label in [
        (15, 16, 31, "t15"),
        (35, 36, 51, "t35"),
    ]:
        lines.append(f"\n  --- {label} window (ticks {window_start}-{window_end}) ---")
        lines.append(f"  {'Scope':<12} {'n_runs':>6} {'mean_rho':>10} {'std':>8} {'%pos':>6} {'p (t-test)':>10}")
        lines.append("  " + "-" * 55)

        by_cond = defaultdict(list)
        all_rhos = []

        for run in runs:
            ccv = run["ccv"]
            sv_sel = run["sv_sel"]
            if not sv_sel or not sv_sel[0] or len(ccv) < window_end or len(sv_sel) < window_end:
                continue

            sv_arr = np.array(sv_sel, dtype=np.float64)

            # Reference for similarity
            ref_indices = [t_p - 3, t_p - 2, t_p - 1]
            valid_ref = [sv_arr[i] for i in ref_indices if 0 <= i < len(sv_arr) and np.linalg.norm(sv_arr[i]) > 0]
            if not valid_ref:
                continue
            r_pre = np.mean(valid_ref, axis=0)

            # CCV[t] for t in window_start-1..window_end-2 (shifted back by 1 for lag)
            # sim_selected[t+1] for t+1 in window_start..window_end-1
            ccv_lag = []
            sim_lag = []
            for t in range(window_start - 1, window_end - 1):
                if t < len(ccv) and (t + 1) < len(sv_arr):
                    c = ccv[t]
                    s = cosine_sim(r_pre, sv_arr[t + 1])
                    if c is not None and not np.isnan(c) and np.linalg.norm(sv_arr[t + 1]) > 0:
                        ccv_lag.append(c)
                        sim_lag.append(s)

            if len(ccv_lag) >= 5:
                rho, _ = spearmanr(ccv_lag, sim_lag)
                all_rhos.append(rho)
                by_cond[run["condition"]].append(rho)

        if all_rhos:
            from scipy.stats import ttest_1samp
            arr = np.array(all_rhos)
            t_stat, p_val = ttest_1samp(arr, 0)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            pct_pos = 100 * np.sum(arr > 0) / len(arr)
            lines.append(f"  {'GLOBAL':<12} {len(arr):>6} {np.mean(arr):>10.4f} {np.std(arr):>8.4f} {pct_pos:>5.0f}% {p_val:>10.4f} {sig}")

            for cond in CONDITIONS_MULTI:
                cr = np.array(by_cond.get(cond, []))
                if len(cr) >= 3:
                    t_c, p_c = ttest_1samp(cr, 0)
                    sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 else "*" if p_c < 0.05 else "ns"
                    lines.append(f"  {cond:<12} {len(cr):>6} {np.mean(cr):>10.4f} {np.std(cr):>8.4f} "
                                  f"{100*np.sum(cr>0)/len(cr):>5.0f}% {p_c:>10.4f} {sig_c}")

    lines.append("\n  Interpretation:")
    lines.append("  rho > 0: higher CCV at tick t → higher persistence at tick t+1")
    lines.append("  rho < 0: higher CCV at tick t → lower persistence at tick t+1")


def make_plots(runs: List[dict], out_dir: Path):
    """4-panel scatter plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        (0, 0, "ccv_pre_t15", "t15_sel", "CCV pre-t15 vs t15_sel"),
        (0, 1, "ccv_pre_t35", "t35_sel", "CCV pre-t35 vs t35_sel"),
        (1, 0, "dist_post_t15", "t15_sel", "Dist winner/bary post-t15 vs t15_sel"),
        (1, 1, "dist_post_t35", "t35_sel", "Dist winner/bary post-t35 vs t35_sel"),
    ]

    for row, col, x_key, y_key, title in panels:
        ax = axes[row][col]

        for cond in CONDITIONS_MULTI:
            xc = [r[x_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            yc = [r[y_key] for r in runs if r["condition"] == cond and r[x_key] is not None and r[y_key] is not None]
            if xc:
                ax.scatter(xc, yc, c=COND_COLORS.get(cond, "#333"), s=30, alpha=0.7, label=cond)

        # Global Spearman
        x_all = [r[x_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        y_all = [r[y_key] for r in runs if r[x_key] is not None and r[y_key] is not None]
        if len(x_all) >= 5:
            rho, p = spearmanr(x_all, y_all)
            ax.text(0.02, 0.95, f"rho={rho:.3f} p={p:.3f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_key, fontsize=9)
        ax.set_ylabel(y_key, fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)

    plt.suptitle("CCV & Distance → Persistence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_ccv_persistence.png", dpi=300)
    fig.savefig(out_dir / "fig5_ccv_persistence.pdf")
    plt.close(fig)
    print("  Fig 5 saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    fig_dir = runs_dir.parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Loading data...")
    runs = load_runs(runs_dir)
    print(f"Loaded {len(runs)} multi-agent runs")

    lines = []
    lines.append("=" * 80)
    lines.append("CCV → PERSISTENCE & DISTANCE → PERSISTENCE — BENCH V7")
    lines.append("=" * 80)

    analysis_1(runs, lines)
    analysis_2(runs, lines)
    analysis_3(runs, lines)

    report = "\n".join(lines)
    print(report)

    report_path = runs_dir / "ccv_persistence_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    print("\nGenerating plots...")
    make_plots(runs, fig_dir)


if __name__ == "__main__":
    main()
