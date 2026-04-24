#!/usr/bin/env python3
"""
geometric_scrambled_null.py — Temporal permutation null test
=============================================================
For each multi-agent run, permute the temporal order of post-perturbation
embeddings while keeping the same reference. If recovery depends on temporal
alignment (trajectory back to baseline), scrambling will destroy it.

Two variants:
  1. Full permutation: random shuffle of post-perturbation embeddings
  2. Circular shift: rotate the sequence by random k (preserves local structure)

Usage:
    python tools/geometric_scrambled_null.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

N_PERMUTATIONS = 1000
RNG = np.random.default_rng(42)

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]

# Perturbation windows (matching metrics_v2.py compute_sim_curves)
# R_pre = mean(sv[t_p-3], sv[t_p-2], sv[t_p-1])   (0-indexed)
# sim[k] = cosine_sim(R_pre, sv[t_p + k])  for k=1..k_max
WINDOWS = {
    15: {"k_max": 19},   # t15: recovery ticks 16-34 (k=1..19)
    35: {"k_max": 44},   # t35: recovery ticks 36-79 (k=1..44)
}


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


def compute_reference(sv: np.ndarray, t_p: int) -> Optional[np.ndarray]:
    """R_pre = mean of 3 ticks before perturbation (same as metrics_v2.py)."""
    pre_indices = [t_p - 3, t_p - 2, t_p - 1]
    valid = [sv[i] for i in pre_indices if 0 <= i < len(sv) and np.linalg.norm(sv[i]) > 0]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def compute_recovery(r_pre: np.ndarray, post_vecs: List[np.ndarray]) -> float:
    """Mean cosine similarity between reference and post-perturbation vectors."""
    sims = [cosine_sim(r_pre, v) for v in post_vecs if np.linalg.norm(v) > 0]
    return float(np.mean(sims)) if sims else 0.0


def extract_post_window(sv: np.ndarray, t_p: int, k_max: int) -> List[np.ndarray]:
    """Extract the post-perturbation embedding vectors for k=1..k_max."""
    vecs = []
    for k in range(1, k_max + 1):
        idx = t_p + k
        if idx < len(sv) and np.linalg.norm(sv[idx]) > 0:
            vecs.append(sv[idx])
    return vecs


def scrambled_null(r_pre: np.ndarray, post_vecs: List[np.ndarray], n_perm: int) -> List[float]:
    """Full permutation null: shuffle temporal order of post vectors."""
    recoveries = []
    for _ in range(n_perm):
        shuffled = list(post_vecs)
        RNG.shuffle(shuffled)
        recoveries.append(compute_recovery(r_pre, shuffled))
    return recoveries


def circular_null(r_pre: np.ndarray, post_vecs: List[np.ndarray], n_perm: int) -> List[float]:
    """Circular shift null: rotate sequence by random k."""
    n = len(post_vecs)
    if n < 2:
        return [compute_recovery(r_pre, post_vecs)] * n_perm
    recoveries = []
    for _ in range(n_perm):
        k = RNG.integers(1, n)  # shift by at least 1
        shifted = post_vecs[k:] + post_vecs[:k]
        recoveries.append(compute_recovery(r_pre, shifted))
    return recoveries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # ── Load all multi-agent runs ──
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

    # ── Process each perturbation window ──
    results_by_window = {}

    for t_p, wcfg in WINDOWS.items():
        k_max = wcfg["k_max"]
        key = f"t{t_p}"

        real_recoveries = []
        scramble_all = []
        circular_all = []
        by_cond = defaultdict(lambda: {"real": [], "scramble": [], "circular": []})

        for run in runs:
            sv = run["sv"]
            cond = run["condition"]

            r_pre = compute_reference(sv, t_p)
            if r_pre is None:
                continue

            post_vecs = extract_post_window(sv, t_p, k_max)
            if len(post_vecs) < 3:
                continue

            # Real recovery
            real_rec = compute_recovery(r_pre, post_vecs)
            real_recoveries.append(real_rec)
            by_cond[cond]["real"].append(real_rec)

            # Scrambled null (this run)
            scr = scrambled_null(r_pre, post_vecs, N_PERMUTATIONS)
            scramble_all.extend(scr)
            by_cond[cond]["scramble"].extend(scr)

            # Circular null (this run)
            circ = circular_null(r_pre, post_vecs, N_PERMUTATIONS)
            circular_all.extend(circ)
            by_cond[cond]["circular"].extend(circ)

        results_by_window[key] = {
            "real": np.array(real_recoveries),
            "scramble": np.array(scramble_all),
            "circular": np.array(circular_all),
            "by_cond": {c: {k: np.array(v) for k, v in d.items()} for c, d in by_cond.items()},
        }

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("SCRAMBLED TEMPORAL NULL TEST — BENCH V7")
    lines.append("=" * 80)
    lines.append(f"Permutations per run: {N_PERMUTATIONS}")
    lines.append(f"Multi-agent conditions: {CONDITIONS_MULTI}")
    lines.append(f"Runs: {len(runs)}")

    for key, res in results_by_window.items():
        real = res["real"]
        scr = res["scramble"]
        circ = res["circular"]

        if len(real) == 0:
            continue

        d_scr = cohens_d(real, scr)
        d_circ = cohens_d(real, circ)

        # Empirical p-value: proportion of scrambled >= real mean
        p_scr = float(np.mean(scr >= np.mean(real)))
        p_circ = float(np.mean(circ >= np.mean(real)))

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {key.upper()} WINDOW")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Real:      mean={np.mean(real):.4f} std={np.std(real):.4f} n={len(real)}")
        lines.append(f"  Scrambled: mean={np.mean(scr):.4f} std={np.std(scr):.4f} n={len(scr)}")
        lines.append(f"  Circular:  mean={np.mean(circ):.4f} std={np.std(circ):.4f} n={len(circ)}")
        lines.append(f"")
        lines.append(f"  Cohen's d (Real vs Scrambled): {d_scr:.3f}")
        lines.append(f"  Cohen's d (Real vs Circular):  {d_circ:.3f}")
        lines.append(f"  p empirical (scrambled >= real mean): {p_scr:.4f}")
        lines.append(f"  p empirical (circular >= real mean):  {p_circ:.4f}")

        # Interpretation
        lines.append(f"")
        if abs(d_scr) < 0.1:
            lines.append("  INTERPRETATION: Real ~ Scrambled")
            lines.append("  → Recovery does NOT depend on temporal order.")
            lines.append("  → It's the marginal distribution of embeddings, not a trajectory.")
        elif d_scr > 0:
            lines.append("  INTERPRETATION: Real > Scrambled")
            lines.append("  → Temporal alignment matters. There IS a recovery trajectory.")
            lines.append("  → Ticks close to perturbation are farther from baseline,")
            lines.append("    later ticks are closer = genuine return path.")
        else:
            lines.append("  INTERPRETATION: Real < Scrambled (unexpected)")
            lines.append("  → Temporal order degrades recovery vs random order.")

        # Per-condition
        lines.append(f"\n  {'Cond':<6} {'Real':>8} {'Scrambl':>8} {'d_scr':>8} {'Circ':>8} {'d_circ':>8}")
        lines.append("  " + "-" * 48)

        for cond in sorted(res["by_cond"]):
            cd = res["by_cond"][cond]
            r = cd["real"]
            s = cd["scramble"]
            c = cd["circular"]
            ds = cohens_d(r, s) if len(r) >= 2 else 0
            dc = cohens_d(r, c) if len(r) >= 2 else 0
            lines.append(f"  {cond:<6} {np.mean(r):>8.4f} {np.mean(s):>8.4f} {ds:>8.3f} {np.mean(c):>8.4f} {dc:>8.3f}")

    # ── Global conclusion ──
    lines.append(f"\n{'=' * 60}")
    lines.append("  CONCLUSION")
    lines.append(f"{'=' * 60}")

    all_d = []
    for key, res in results_by_window.items():
        if len(res["real"]) > 0 and len(res["scramble"]) > 0:
            all_d.append(cohens_d(res["real"], res["scramble"]))

    if all_d:
        mean_d = np.mean(all_d)
        if mean_d > 0.3:
            lines.append("  Temporal structure CONFIRMED (mean d > 0.3).")
            lines.append("  Recovery is not just marginal embedding distribution —")
            lines.append("  the sequential order of ticks contributes to the return to baseline.")
        elif mean_d > 0.1:
            lines.append("  Weak temporal structure detected (0.1 < d < 0.3).")
            lines.append("  Some sequential ordering effect, but marginal distribution dominates.")
        else:
            lines.append("  NO temporal structure detected (d < 0.1).")
            lines.append("  Recovery is entirely explained by the marginal distribution of embeddings.")
            lines.append("  The 'trajectory' is an artifact of averaging.")

    report = "\n".join(lines)
    print(report)

    # Save
    report_path = runs_dir / "geometric_scrambled_null_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, (key, res) in enumerate(results_by_window.items()):
            ax = axes[idx]
            real = res["real"]
            scr = res["scramble"]
            circ = res["circular"]

            if len(real) == 0:
                continue

            ax.hist(scr, bins=80, alpha=0.4, color="gray", density=True, label=f"Scrambled ({np.mean(scr):.3f})")
            ax.hist(circ, bins=80, alpha=0.3, color="steelblue", density=True, label=f"Circular ({np.mean(circ):.3f})")

            ax.axvline(np.mean(real), color="red", linewidth=2, label=f"Real mean ({np.mean(real):.3f})")

            # Per-condition real means
            colors = {"C": "green", "R": "purple", "D": "brown", "D2": "cyan", "D3": "pink"}
            for cond in sorted(res["by_cond"]):
                r = res["by_cond"][cond]["real"]
                if len(r) > 0:
                    ax.axvline(np.mean(r), color=colors.get(cond, "black"),
                               linewidth=1, linestyle=":", alpha=0.7, label=f"{cond} ({np.mean(r):.3f})")

            ax.set_title(f"{key.upper()}: Real vs Scrambled Temporal Null")
            ax.set_xlabel("Mean Cosine Similarity (recovery)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plot_path = runs_dir / "geometric_scrambled_null_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
