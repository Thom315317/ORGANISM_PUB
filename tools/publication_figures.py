#!/usr/bin/env python3
"""
publication_figures.py — Generate 4 publication-ready figures for CRISTAL V7
=============================================================================
Usage:
    python tools/publication_figures.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10

CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]
COND_COLORS = {
    "E": "#e74c3c", "D3": "#9b59b6", "R": "#8e44ad",
    "C": "#27ae60", "B": "#3498db", "D2": "#2980b9",
    "D": "#e67e22", "A": "#95a5a6",
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


def load_all(runs_dir: Path):
    """Load recovery metrics and sim_curves from all runs."""
    by_cond = defaultdict(lambda: {"t15": [], "t35": [], "sim15": [], "sim35": []})
    all_embeddings = []
    by_seed_pool = defaultdict(list)

    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        seed = d.get("seed", 0)
        sim = d.get("sim_curves") or {}
        sv = d.get("state_vector_mean", [])

        # Collect embeddings for baselines
        if sv and sv[0]:
            for v in sv:
                arr = np.array(v)
                if np.linalg.norm(arr) > 0:
                    all_embeddings.append(arr)
                    by_seed_pool[seed].append(arr)

        # t15_mean (k=1..19) and t35_mean (k=1..44)
        if "tick_15" in sim:
            vals = [v for v in sim["tick_15"].get("mean", []) if v is not None]
            if vals:
                by_cond[cond]["t15"].append(float(np.mean(vals[:19])))
                by_cond[cond]["sim15"].append(vals)
        if "tick_35" in sim:
            vals = [v for v in sim["tick_35"].get("mean", []) if v is not None]
            if vals:
                by_cond[cond]["t35"].append(float(np.mean(vals)))
                by_cond[cond]["sim35"].append(vals)

    return dict(by_cond), np.array(all_embeddings), {s: np.array(v) for s, v in by_seed_pool.items()}


def compute_baselines(pool: np.ndarray, by_seed: Dict, n_sim: int = 1000):
    """Compute baseline C (inter-seed) and D (intra-seed) recovery distributions."""
    rng = np.random.default_rng(42)

    # Baseline C
    n_pool = len(pool)
    bc = []
    for _ in range(n_sim):
        idx_pre = rng.choice(n_pool, size=3, replace=False)
        idx_post = rng.choice(n_pool, size=3, replace=False)
        bc.append(cosine_sim(np.mean(pool[idx_pre], axis=0), np.mean(pool[idx_post], axis=0)))

    # Baseline D
    bd = []
    for seed, sp in by_seed.items():
        n = len(sp)
        if n < 6:
            continue
        for _ in range(n_sim):
            idx = rng.choice(n, size=6, replace=False)
            bd.append(cosine_sim(np.mean(sp[idx[:3]], axis=0), np.mean(sp[idx[3:]], axis=0)))

    return np.array(bc), np.array(bd)


def fig1_persistence(by_cond: Dict, bc_mean: float, bd_mean: float, out_dir: Path):
    """Figure 1: Bar chart of t15 and t35 persistence by condition."""
    # Order by ascending t15
    conds_with_t15 = [(c, np.mean(d["t15"])) for c, d in by_cond.items() if d["t15"]]
    conds_with_t15.sort(key=lambda x: x[1])
    order = [c for c, _ in conds_with_t15]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    w = 0.35

    t15_means = [np.mean(by_cond[c]["t15"]) for c in order]
    t15_stds = [np.std(by_cond[c]["t15"]) for c in order]
    t35_means = [np.mean(by_cond[c]["t35"]) if by_cond[c]["t35"] else 0 for c in order]
    t35_stds = [np.std(by_cond[c]["t35"]) if by_cond[c]["t35"] else 0 for c in order]

    bars1 = ax.bar(x - w/2, t15_means, w, yerr=t15_stds, label="t15 (post-compression)",
                   color="#3498db", capsize=3, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, t35_means, w, yerr=t35_stds, label="t35 (post-inversion)",
                   color="#e67e22", capsize=3, edgecolor="white", linewidth=0.5)

    # Baseline lines
    ax.axhline(bd_mean, color="#555555", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(bc_mean, color="#999999", linestyle=":", linewidth=1, alpha=0.8)
    ax.text(len(order) - 0.5, bd_mean + 0.005, f"Baseline D (intra-seed): {bd_mean:.3f}",
            fontsize=8, color="#555555", ha="right")
    ax.text(len(order) - 0.5, bc_mean - 0.015, f"Baseline C (inter-seed): {bc_mean:.3f}",
            fontsize=8, color="#999999", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean cosine similarity to pre-perturbation reference", fontsize=10)
    ax.set_title("Semantic Persistence by Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_persistence_by_condition.png", dpi=300)
    fig.savefig(out_dir / "fig1_persistence_by_condition.pdf")
    plt.close(fig)
    print("  Fig 1 saved")


def fig2_gradient(by_cond: Dict, out_dir: Path):
    """Figure 2: Tick-by-tick recovery gradient per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (sim_key, title, k_max) in enumerate([
        ("sim15", "Post-compression (t15)", 19),
        ("sim35", "Post-inversion (t35)", 44),
    ]):
        ax = axes[idx]
        for cond in sorted(CONDITIONS_MULTI):
            if cond not in by_cond or not by_cond[cond][sim_key]:
                continue
            curves = by_cond[cond][sim_key]
            # Pad/truncate to k_max
            arr = np.full((len(curves), k_max), np.nan)
            for i, c in enumerate(curves):
                length = min(len(c), k_max)
                arr[i, :length] = c[:length]

            mean_c = np.nanmean(arr, axis=0)
            std_c = np.nanstd(arr, axis=0)
            n_v = np.sum(~np.isnan(arr), axis=0)
            ci = 1.96 * std_c / np.sqrt(np.maximum(n_v, 1))

            x = np.arange(1, k_max + 1)
            color = COND_COLORS.get(cond, "#333333")
            ax.plot(x, mean_c, color=color, linewidth=1.5, label=cond)
            ax.fill_between(x, mean_c - ci, mean_c + ci, alpha=0.12, color=color)

        # Mark injection at tick 22 (= k=7 in t15 window)
        if idx == 0:
            ax.axvline(7, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(7.3, 0.95, "injection\n(tick 22)", fontsize=7, color="red", alpha=0.7)

        # Mark end of metric window (k_max=15 used for t15_mean/t35_mean)
        ax.axvline(15, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(15.3, 0.57, "metric window\n(k=15)", fontsize=7, color="gray", alpha=0.7)

        ax.set_xlabel("k (ticks after perturbation)", fontsize=10)
        ax.set_ylabel("Cosine similarity to reference", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(0.55, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)

    plt.suptitle("Tick-by-Tick Persistence Gradient", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_temporal_gradient.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_temporal_gradient.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2 saved")


def fig3_nulls(by_cond: Dict, bc: np.ndarray, bd: np.ndarray, out_dir: Path):
    """Figure 3: Scrambled null histograms with real condition means."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, key in enumerate(["t15", "t35"]):
        ax = axes[idx]

        ax.hist(bc, bins=60, alpha=0.35, color="#bdc3c7", density=True, label=f"Baseline C ({np.mean(bc):.3f})")
        ax.hist(bd, bins=60, alpha=0.45, color="#7f8c8d", density=True, label=f"Baseline D ({np.mean(bd):.3f})")

        # Real per-condition means
        all_real_multi = []  # Exclude E (single-agent, not comparable to 3-agent baselines)
        for cond in sorted(by_cond):
            vals = by_cond[cond][key]
            if vals:
                m = np.mean(vals)
                if cond != "E":
                    all_real_multi.extend(vals)
                color = COND_COLORS.get(cond, "#333333")
                ax.axvline(m, color=color, linewidth=1.2, linestyle=":", alpha=0.8, label=f"{cond} ({m:.3f})")

        if all_real_multi:
            real_mean = np.mean(all_real_multi)
            ax.axvline(real_mean, color="black", linewidth=2, label=f"All multi-agent ({real_mean:.3f})")

            # Annotate Cohen's d (excluding E)
            d_c = cohens_d(np.array(all_real_multi), bc)
            d_d = cohens_d(np.array(all_real_multi), bd)
            ax.text(0.02, 0.95, f"d vs C: {d_c:.2f}\nd vs D: {d_d:.2f}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        title = "Post-compression (t15)" if key == "t15" else "Post-inversion (t35)"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cosine similarity (recovery)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Real Recovery vs Bootstrap Null Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_scrambled_nulls.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_scrambled_nulls.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 3 saved")


def fig4_dotplot(by_cond: Dict, bd: np.ndarray, out_dir: Path):
    """Figure 4: Cohen's d per condition vs baseline D (dot plot)."""
    conds = ["D3", "R", "C", "D2", "D"]
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conds))

    t15_ds = []
    t35_ds = []
    for cond in conds:
        v15 = by_cond.get(cond, {}).get("t15", [])
        v35 = by_cond.get(cond, {}).get("t35", [])
        t15_ds.append(cohens_d(np.array(v15), bd) if len(v15) >= 2 else 0)
        t35_ds.append(cohens_d(np.array(v35), bd) if len(v35) >= 2 else 0)

    # Negligible zone
    ax.axhspan(-0.2, 0.2, alpha=0.08, color="gray")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.text(len(conds) - 0.5, 0.15, "negligible", fontsize=7, color="gray", ha="right", alpha=0.6)

    ax.scatter(x - 0.08, t15_ds, s=80, c="#3498db", marker="o", zorder=5, label="t15 (post-compression)")
    ax.scatter(x + 0.08, t35_ds, s=80, edgecolors="#e67e22", marker="o", facecolors="none", linewidths=1.5, zorder=5, label="t35 (post-inversion)")

    # Connect pairs
    for i in range(len(conds)):
        ax.plot([x[i] - 0.08, x[i] + 0.08], [t15_ds[i], t35_ds[i]], color="#cccccc", linewidth=0.5, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=11, fontweight="bold")
    ax.set_ylabel("Cohen's d (vs Baseline D intra-seed)", fontsize=10)
    ax.set_title("Per-Condition Effect Size vs Bootstrap Null", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_per_condition_vs_baseline.png", dpi=300)
    fig.savefig(out_dir / "fig4_per_condition_vs_baseline.pdf")
    plt.close(fig)
    print("  Fig 4 saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    out_dir = runs_dir.parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    print("Loading data...")
    by_cond, pool, by_seed = load_all(runs_dir)
    print(f"Conditions: {sorted(by_cond.keys())}, pool={len(pool)}")

    print("Computing baselines (1000 simulations)...")
    bc, bd = compute_baselines(pool, by_seed)
    print(f"Baseline C: {np.mean(bc):.4f}, Baseline D: {np.mean(bd):.4f}")

    print("\nGenerating figures...")
    fig1_persistence(by_cond, float(np.mean(bc)), float(np.mean(bd)), out_dir)
    fig2_gradient(by_cond, out_dir)
    fig3_nulls(by_cond, bc, bd, out_dir)
    fig4_dotplot(by_cond, bd, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
