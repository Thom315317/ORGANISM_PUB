#!/usr/bin/env python3
"""
geometric_baseline_test.py — Geometric baseline test for V7
=============================================================
Tests whether observed recovery after perturbation is a trivial
property of averaging 3 vectors in high-dimensional space, or
whether temporal/interactive structure contributes.

Three baselines:
  A — Random Gaussian vectors (normalized)
  B — Random unit sphere vectors
  C — Bootstrap from real embedding pool (most decisive)

Usage:
    python tools/geometric_baseline_test.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm as normal_dist

N_SIMULATIONS = 1000
DIM = 768  # embedding dimension (all-mpnet-base-v2)
RNG = np.random.default_rng(42)

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]
PERT_TICKS = [15, 35]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_runs(runs_dir: Path) -> Tuple[Dict, np.ndarray, List[float]]:
    """Load real data: per-run sv_mean arrays, pooled embeddings, perturbation deltas."""
    runs = {}
    all_embeddings = []
    pert_deltas = []

    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        seed = d.get("seed", 0)
        sv_mean = d.get("state_vector_mean", [])

        if not sv_mean or not sv_mean[0]:
            continue

        sv_arr = np.array(sv_mean, dtype=np.float64)

        # Collect all non-zero embeddings for pool
        for v in sv_arr:
            if np.linalg.norm(v) > 0:
                all_embeddings.append(v)

        if cond in CONDITIONS_MULTI:
            runs[f"{cond}_seed{seed}"] = {
                "condition": cond, "seed": seed, "sv_mean": sv_arr,
            }

            # Measure perturbation amplitude
            for tp in PERT_TICKS:
                if tp < len(sv_arr) and tp - 1 >= 0:
                    if np.linalg.norm(sv_arr[tp]) > 0 and np.linalg.norm(sv_arr[tp - 1]) > 0:
                        delta = 1.0 - cosine_sim(sv_arr[tp - 1], sv_arr[tp])
                        pert_deltas.append(delta)

    pool = np.array(all_embeddings)
    return runs, pool, pert_deltas


def compute_real_recovery(sv_mean: np.ndarray, t_p: int, k_max: int = 19) -> List[float]:
    """Compute sim_curve from real data: cosine_sim(R_pre, sv_mean[t_p+k]) for k=1..k_max."""
    # R_pre = mean of 3 ticks before perturbation
    pre_indices = [t_p - 3, t_p - 2, t_p - 1]
    valid_pre = [sv_mean[i] for i in pre_indices if 0 <= i < len(sv_mean) and np.linalg.norm(sv_mean[i]) > 0]
    if not valid_pre:
        return []
    r_pre = np.mean(valid_pre, axis=0)

    sims = []
    for k in range(1, k_max + 1):
        idx = t_p + k
        if idx < len(sv_mean) and np.linalg.norm(sv_mean[idx]) > 0:
            sims.append(cosine_sim(r_pre, sv_mean[idx]))
    return sims


def simulate_baseline_A(dim: int, n_agents: int, mean_norm: float,
                        pert_delta: float, n_sim: int) -> List[float]:
    """Baseline A: iid Gaussian vectors, normalized."""
    recoveries = []
    for _ in range(n_sim):
        # Pre-perturbation: 3 random vectors → barycentre
        pre_vecs = RNG.normal(size=(3, n_agents, dim)) if False else RNG.normal(size=(3, dim))
        pre_vecs = pre_vecs / np.linalg.norm(pre_vecs, axis=1, keepdims=True) * mean_norm
        pre_bary = np.mean(pre_vecs, axis=0)

        # Post-perturbation: 3 new random vectors → barycentre
        post_vecs = RNG.normal(size=(3, dim))
        post_vecs = post_vecs / np.linalg.norm(post_vecs, axis=1, keepdims=True) * mean_norm
        post_bary = np.mean(post_vecs, axis=0)

        recoveries.append(cosine_sim(pre_bary, post_bary))
    return recoveries


def simulate_baseline_B(dim: int, mean_norm: float,
                        pert_delta: float, n_sim: int) -> List[float]:
    """Baseline B: uniform on unit sphere."""
    recoveries = []
    for _ in range(n_sim):
        pre_vecs = RNG.normal(size=(3, dim))
        pre_vecs = pre_vecs / np.linalg.norm(pre_vecs, axis=1, keepdims=True)
        pre_bary = np.mean(pre_vecs, axis=0)

        post_vecs = RNG.normal(size=(3, dim))
        post_vecs = post_vecs / np.linalg.norm(post_vecs, axis=1, keepdims=True)
        post_bary = np.mean(post_vecs, axis=0)

        recoveries.append(cosine_sim(pre_bary, post_bary))
    return recoveries


def simulate_baseline_C(pool: np.ndarray, pert_deltas: List[float],
                        n_sim: int) -> List[float]:
    """Baseline C: bootstrap from real embedding pool (most decisive).

    Draw 3 random real embeddings → barycentre (pre).
    Perturb: shift by real perturbation amplitude.
    Draw 3 more random real embeddings → barycentre (post).
    Measure recovery: cosine_sim(pre_bary, post_bary).
    """
    n_pool = len(pool)
    recoveries = []
    for _ in range(n_sim):
        # Pre: 3 random embeddings from pool
        idx_pre = RNG.choice(n_pool, size=3, replace=False)
        pre_bary = np.mean(pool[idx_pre], axis=0)

        # Post: 3 different random embeddings from pool
        idx_post = RNG.choice(n_pool, size=3, replace=False)
        post_bary = np.mean(pool[idx_post], axis=0)

        recoveries.append(cosine_sim(pre_bary, post_bary))
    return recoveries


def cohens_d(real_values: np.ndarray, sim_values: np.ndarray) -> float:
    """Cohen's d: (mean_real - mean_sim) / pooled_std."""
    n1, n2 = len(real_values), len(sim_values)
    m1, m2 = np.mean(real_values), np.mean(sim_values)
    s1, s2 = np.std(real_values, ddof=1), np.std(sim_values, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    print("Loading data...")
    runs, pool, pert_deltas = load_runs(runs_dir)
    print(f"Loaded {len(runs)} multi-agent runs, pool={len(pool)} embeddings, "
          f"pert_deltas={len(pert_deltas)} (mean={np.mean(pert_deltas):.4f})")

    # Compute mean norm of real embeddings
    norms = np.linalg.norm(pool, axis=1)
    mean_norm = float(np.mean(norms))
    print(f"Mean embedding norm: {mean_norm:.4f}")

    # ── Collect real recovery values ──
    real_recoveries = {"t15": [], "t35": []}
    for run_id, data in runs.items():
        sv = data["sv_mean"]
        for tp, key, km in [(15, "t15", 19), (35, "t35", 44)]:
            sims = compute_real_recovery(sv, tp, k_max=km)
            if sims:
                real_recoveries[key].append(float(np.mean(sims)))

    print(f"\nReal recoveries: t15 n={len(real_recoveries['t15'])}, "
          f"t35 n={len(real_recoveries['t35'])}")

    # ── Run baselines ──
    mean_delta = float(np.mean(pert_deltas)) if pert_deltas else 0.1

    print(f"\nSimulating {N_SIMULATIONS} trials per baseline...")
    baseline_A = simulate_baseline_A(DIM, 3, mean_norm, mean_delta, N_SIMULATIONS)
    baseline_B = simulate_baseline_B(DIM, mean_norm, mean_delta, N_SIMULATIONS)
    baseline_C = simulate_baseline_C(pool, pert_deltas, N_SIMULATIONS)

    baselines = {"A_gaussian": baseline_A, "B_sphere": baseline_B, "C_bootstrap": baseline_C}

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("GEOMETRIC BASELINE TEST — BENCH V7")
    lines.append("=" * 80)
    lines.append(f"\nPool: {len(pool)} real embeddings, dim={DIM}")
    lines.append(f"Mean norm: {mean_norm:.4f}")
    lines.append(f"Mean perturbation delta (cosine distance): {mean_delta:.4f}")
    lines.append(f"Simulations per baseline: {N_SIMULATIONS}")

    for key in ["t15", "t35"]:
        real = np.array(real_recoveries[key])
        if len(real) == 0:
            continue

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {key.upper()} RECOVERY")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Real: mean={np.mean(real):.4f} median={np.median(real):.4f} "
                      f"std={np.std(real):.4f} n={len(real)}")
        lines.append(f"  Real range: [{np.min(real):.4f}, {np.max(real):.4f}]")

        for bname, bvals in baselines.items():
            barr = np.array(bvals)
            d = cohens_d(real, barr)
            # What fraction of simulated values exceed real mean?
            pct_above = 100 * np.mean(barr >= np.mean(real))
            p5, p50, p95 = np.percentile(barr, [5, 50, 95])

            position = "ABOVE" if np.mean(real) > p95 else "WITHIN" if np.mean(real) > p5 else "BELOW"

            lines.append(f"\n  Baseline {bname}:")
            lines.append(f"    mean={np.mean(barr):.4f} median={p50:.4f} "
                          f"std={np.std(barr):.4f}")
            lines.append(f"    5th={p5:.4f} 95th={p95:.4f}")
            lines.append(f"    Real mean vs baseline: {position} the 90% interval")
            lines.append(f"    % baseline >= real mean: {pct_above:.1f}%")
            lines.append(f"    Cohen's d (real vs baseline): {d:.3f}")

            if d > 0.8:
                interp = "LARGE effect — real recovery far exceeds baseline"
            elif d > 0.5:
                interp = "MEDIUM effect — real recovery meaningfully exceeds baseline"
            elif d > 0.2:
                interp = "SMALL effect — real recovery slightly exceeds baseline"
            else:
                interp = "NEGLIGIBLE — real recovery indistinguishable from baseline"
            lines.append(f"    Interpretation: {interp}")

    # ── Per-condition breakdown for Baseline C ──
    lines.append(f"\n{'=' * 60}")
    lines.append("  PER-CONDITION BREAKDOWN vs BASELINE C (bootstrap)")
    lines.append(f"{'=' * 60}")

    bc_arr = np.array(baseline_C)
    bc_mean = np.mean(bc_arr)
    bc_std = np.std(bc_arr)

    by_cond = defaultdict(lambda: {"t15": [], "t35": []})
    for run_id, data in runs.items():
        cond = data["condition"]
        sv = data["sv_mean"]
        for tp, key, km in [(15, "t15", 19), (35, "t35", 44)]:
            sims = compute_real_recovery(sv, tp, k_max=km)
            if sims:
                by_cond[cond][key].append(float(np.mean(sims)))

    lines.append(f"\n  {'Cond':<6} {'t15_mean':>10} {'t15_d':>8} {'t35_mean':>10} {'t35_d':>8}")
    lines.append(f"  {'Baseline C':<6} {bc_mean:>10.4f} {'—':>8} {bc_mean:>10.4f} {'—':>8}")
    lines.append("  " + "-" * 45)

    for cond in sorted(by_cond):
        t15 = by_cond[cond]["t15"]
        t35 = by_cond[cond]["t35"]
        t15_m = np.mean(t15) if t15 else 0
        t35_m = np.mean(t35) if t35 else 0
        t15_d = cohens_d(np.array(t15), bc_arr) if t15 else 0
        t35_d = cohens_d(np.array(t35), bc_arr) if t35 else 0
        lines.append(f"  {cond:<6} {t15_m:>10.4f} {t15_d:>8.3f} {t35_m:>10.4f} {t35_d:>8.3f}")

    # ── Conclusion ──
    lines.append(f"\n{'=' * 60}")
    lines.append("  CONCLUSION")
    lines.append(f"{'=' * 60}")

    real_t15 = np.array(real_recoveries["t15"])
    bc_d_t15 = cohens_d(real_t15, bc_arr) if len(real_t15) > 0 else 0

    if bc_d_t15 > 0.5:
        lines.append("  Real multi-agent collectives recover SIGNIFICANTLY BETTER")
        lines.append("  than random bootstrapped embeddings (Baseline C).")
        lines.append("  → Temporal/interactive structure contributes beyond pure averaging.")
    elif bc_d_t15 > 0.2:
        lines.append("  Real collectives show MODEST improvement over bootstrap baseline.")
        lines.append("  → Some temporal structure, but averaging explains most of the effect.")
    else:
        lines.append("  Real collectives are INDISTINGUISHABLE from bootstrap baseline.")
        lines.append("  → Recovery is primarily a geometric averaging artifact.")

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = runs_dir / "geometric_baseline_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, key in enumerate(["t15", "t35"]):
            ax = axes[idx]
            real = real_recoveries[key]
            if not real:
                continue

            # Histogram of Baseline C
            ax.hist(baseline_C, bins=50, alpha=0.5, color="gray", label="Baseline C (bootstrap)")
            ax.hist(baseline_A, bins=50, alpha=0.3, color="blue", label="Baseline A (gaussian)")

            # Real mean as vertical line
            ax.axvline(np.mean(real), color="red", linewidth=2, label=f"Real mean ({np.mean(real):.3f})")
            ax.axvline(np.median(real), color="orange", linewidth=1.5, linestyle="--", label=f"Real median ({np.median(real):.3f})")

            # Per-condition means
            colors = {"C": "green", "R": "purple", "D": "brown", "D2": "cyan", "D3": "pink"}
            for cond in sorted(by_cond):
                vals = by_cond[cond][key]
                if vals:
                    ax.axvline(np.mean(vals), color=colors.get(cond, "black"),
                               linewidth=1, linestyle=":", alpha=0.7, label=f"{cond} ({np.mean(vals):.3f})")

            ax.set_title(f"{key.upper()} Recovery: Real vs Baselines")
            ax.set_xlabel("Cosine Similarity (recovery)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plot_path = runs_dir / "geometric_baseline_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
