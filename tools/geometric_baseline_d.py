#!/usr/bin/env python3
"""
geometric_baseline_d.py — Intra-seed bootstrap baseline
=========================================================
Tests whether the recovery gap between Baseline C (inter-seed) and
real collectives is explained by thematic coherence (same seed = same topic)
or by temporal/interactive dynamics.

Baseline D: for each seed, pool ALL embeddings from all conditions/ticks,
draw random triplets, measure recovery. Same topic, no temporal structure.

Usage:
    python tools/geometric_baseline_d.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

N_SIMULATIONS = 1000
RNG = np.random.default_rng(42)

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]
PERT_TICKS = [15, 35]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cohens_d(real: np.ndarray, sim: np.ndarray) -> float:
    n1, n2 = len(real), len(sim)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(real), np.mean(sim)
    s1, s2 = np.std(real, ddof=1), np.std(sim, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def compute_real_recovery(sv_mean: np.ndarray, t_p: int, k_max: int = 19) -> List[float]:
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


def load_data(runs_dir: Path):
    """Load embeddings grouped by seed, plus real recovery values and perturbation deltas."""
    # Pool by seed: all embeddings from all conditions for that seed
    by_seed = defaultdict(list)  # seed -> list of 768-dim vectors
    # Real recovery values per condition
    real_by_cond = defaultdict(lambda: {"t15": [], "t35": []})
    # Per-seed perturbation deltas
    pert_deltas_by_seed = defaultdict(list)
    # Inter-seed pool (for Baseline C comparison)
    all_embeddings = []

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

        # Add all non-zero embeddings to seed pool and global pool
        for v in sv_arr:
            if np.linalg.norm(v) > 0:
                by_seed[seed].append(v)
                all_embeddings.append(v)

        # Perturbation deltas for this seed
        if cond in CONDITIONS_MULTI:
            for tp in PERT_TICKS:
                if tp < len(sv_arr) and tp - 1 >= 0:
                    if np.linalg.norm(sv_arr[tp]) > 0 and np.linalg.norm(sv_arr[tp - 1]) > 0:
                        pert_deltas_by_seed[seed].append(1.0 - cosine_sim(sv_arr[tp - 1], sv_arr[tp]))

        # Real recovery
        if cond in CONDITIONS_MULTI:
            for tp, key, km in [(15, "t15", 19), (35, "t35", 44)]:
                sims = compute_real_recovery(sv_arr, tp, k_max=km)
                if sims:
                    real_by_cond[cond][key].append(float(np.mean(sims)))

    # Convert to arrays
    for seed in by_seed:
        by_seed[seed] = np.array(by_seed[seed])

    return dict(by_seed), np.array(all_embeddings), dict(real_by_cond), dict(pert_deltas_by_seed)


def simulate_baseline_C(pool: np.ndarray, n_sim: int) -> List[float]:
    """Baseline C: inter-seed bootstrap (for comparison)."""
    n_pool = len(pool)
    recoveries = []
    for _ in range(n_sim):
        idx_pre = RNG.choice(n_pool, size=3, replace=False)
        pre_bary = np.mean(pool[idx_pre], axis=0)
        idx_post = RNG.choice(n_pool, size=3, replace=False)
        post_bary = np.mean(pool[idx_post], axis=0)
        recoveries.append(cosine_sim(pre_bary, post_bary))
    return recoveries


def simulate_baseline_D(by_seed: Dict[int, np.ndarray], n_sim_per_seed: int) -> Tuple[List[float], Dict[int, List[float]]]:
    """Baseline D: intra-seed bootstrap. Same topic, no temporal structure."""
    all_recoveries = []
    per_seed = {}

    for seed, pool in by_seed.items():
        n = len(pool)
        if n < 6:  # need at least 6 distinct embeddings (3 pre + 3 post)
            continue
        seed_recoveries = []
        for _ in range(n_sim_per_seed):
            # Draw 6 distinct indices
            idx = RNG.choice(n, size=6, replace=False)
            pre_bary = np.mean(pool[idx[:3]], axis=0)
            post_bary = np.mean(pool[idx[3:]], axis=0)
            seed_recoveries.append(cosine_sim(pre_bary, post_bary))

        per_seed[seed] = seed_recoveries
        all_recoveries.extend(seed_recoveries)

    return all_recoveries, per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    print("Loading data...")
    by_seed, pool, real_by_cond, pert_deltas = load_data(runs_dir)
    print(f"Seeds: {sorted(by_seed.keys())}")
    print(f"Pool sizes per seed: {', '.join(str(s) + '=' + str(len(v)) for s, v in sorted(by_seed.items()))}")
    print(f"Total pool: {len(pool)} embeddings")

    # Collect all real recoveries
    real_all = {"t15": [], "t35": []}
    for cond in CONDITIONS_MULTI:
        if cond in real_by_cond:
            for key in ["t15", "t35"]:
                real_all[key].extend(real_by_cond[cond][key])

    print(f"Real recoveries: t15 n={len(real_all['t15'])}, t35 n={len(real_all['t35'])}")

    # Simulate baselines
    print(f"\nSimulating {N_SIMULATIONS} trials...")
    baseline_C = simulate_baseline_C(pool, N_SIMULATIONS)
    baseline_D_all, baseline_D_per_seed = simulate_baseline_D(by_seed, N_SIMULATIONS)
    print(f"Baseline C: {len(baseline_C)} samples")
    print(f"Baseline D: {len(baseline_D_all)} samples ({len(baseline_D_per_seed)} seeds)")

    bc = np.array(baseline_C)
    bd = np.array(baseline_D_all)

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("GEOMETRIC BASELINE D — INTRA-SEED BOOTSTRAP")
    lines.append("=" * 80)

    for key in ["t15", "t35"]:
        real = np.array(real_all[key])
        if len(real) == 0:
            continue

        d_c = cohens_d(real, bc)
        d_d = cohens_d(real, bd)
        d_cd = cohens_d(bd, bc)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {key.upper()} RECOVERY")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Baseline C (inter-seed): mean={np.mean(bc):.4f} std={np.std(bc):.4f}")
        lines.append(f"  Baseline D (intra-seed): mean={np.mean(bd):.4f} std={np.std(bd):.4f}")
        lines.append(f"  Real collectives:        mean={np.mean(real):.4f} std={np.std(real):.4f}")
        lines.append(f"")
        lines.append(f"  Cohen's d (Real vs C): {d_c:.3f}")
        lines.append(f"  Cohen's d (Real vs D): {d_d:.3f}")
        lines.append(f"  Cohen's d (D vs C):    {d_cd:.3f}")

        # Interpretation
        lines.append(f"")
        if d_cd > 0.3 and d_d < 0.3:
            lines.append("  INTERPRETATION: D ~ Real, C < D")
            lines.append("  → Thematic coherence (same seed) explains the gap.")
            lines.append("  → No evidence of temporal/interactive dynamics beyond topic similarity.")
        elif d_cd > 0.3 and d_d > 0.3:
            lines.append("  INTERPRETATION: C < D < Real")
            lines.append("  → Thematic coherence explains PART of the gap (C→D).")
            lines.append("  → Temporal/interactive dynamics add MORE (D→Real).")
        elif d_cd < 0.3 and d_d > 0.3:
            lines.append("  INTERPRETATION: C ~ D < Real")
            lines.append("  → Thematic coherence does NOT contribute.")
            lines.append("  → The gap C→Real is entirely temporal/interactive dynamics.")
        else:
            lines.append("  INTERPRETATION: C ~ D ~ Real")
            lines.append("  → Recovery is geometric averaging. No additional structure detected.")

    # ── Per-condition breakdown ──
    lines.append(f"\n{'=' * 60}")
    lines.append("  PER-CONDITION vs BASELINE D")
    lines.append(f"{'=' * 60}")
    lines.append(f"\n  {'Cond':<6} {'t15_real':>10} {'t15_d_vs_D':>12} {'t35_real':>10} {'t35_d_vs_D':>12}")
    lines.append("  " + "-" * 50)

    for cond in sorted(real_by_cond):
        t15 = real_by_cond[cond]["t15"]
        t35 = real_by_cond[cond]["t35"]
        t15_m = np.mean(t15) if t15 else 0
        t35_m = np.mean(t35) if t35 else 0
        t15_d = cohens_d(np.array(t15), bd) if len(t15) >= 2 else 0
        t35_d = cohens_d(np.array(t35), bd) if len(t35) >= 2 else 0
        lines.append(f"  {cond:<6} {t15_m:>10.4f} {t15_d:>12.3f} {t35_m:>10.4f} {t35_d:>12.3f}")

    report = "\n".join(lines)
    print(report)

    # Save report
    report_path = runs_dir / "geometric_baseline_d_report.txt"
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
            real = real_all[key]
            if not real:
                continue

            ax.hist(baseline_C, bins=50, alpha=0.4, color="gray", label=f"C inter-seed ({np.mean(bc):.3f})")
            ax.hist(baseline_D_all, bins=50, alpha=0.4, color="steelblue", label=f"D intra-seed ({np.mean(bd):.3f})")

            ax.axvline(np.mean(real), color="red", linewidth=2, label=f"Real mean ({np.mean(real):.3f})")

            # Per-condition
            colors = {"C": "green", "R": "purple", "D": "brown", "D2": "cyan", "D3": "pink"}
            for cond in sorted(real_by_cond):
                vals = real_by_cond[cond][key]
                if vals:
                    ax.axvline(np.mean(vals), color=colors.get(cond, "black"),
                               linewidth=1, linestyle=":", alpha=0.7, label=f"{cond} ({np.mean(vals):.3f})")

            ax.set_title(f"{key.upper()} Recovery: Baselines C vs D vs Real")
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plot_path = runs_dir / "geometric_baseline_d_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
