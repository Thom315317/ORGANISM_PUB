#!/usr/bin/env python3
"""
slope_comparison_v7.py — Direct slope comparison E vs multi-agent + reference robustness
=========================================================================================
1. Paired Wilcoxon on Spearman rho (t35_sel gradient) between E and each multi-agent condition
2. Reference robustness: recompute t15_sel/t35_sel with single-tick reference vs 3-tick mean
3. Pre-perturbation dispersion of sv_selected between conditions

Usage:
    python tools/slope_comparison_v7.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, wilcoxon, ttest_1samp

ALL_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8]
CONDITIONS_MULTI = ["C", "R", "D", "D2", "D3"]

WINDOWS = {
    15: {"k_max": 19, "label": "t15", "pre_single": 14, "pre_range": [12, 13, 14]},
    35: {"k_max": 44, "label": "t35", "pre_single": 34, "pre_range": [32, 33, 34]},
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def holm_bonferroni(pvals: List[Optional[float]]) -> List[Optional[float]]:
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = [None] * len(pvals)
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(p * (m - rank), 1.0)
        adj = max(adj, prev)
        corrected[idx] = adj
        prev = adj
    return corrected


def load_runs(runs_dir: Path) -> Dict:
    """Load sv_selected for all runs."""
    runs = {}
    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        seed = d.get("seed", 0)
        sv_sel = d.get("state_vector_selected", [])
        if not sv_sel or not sv_sel[0]:
            continue
        runs[(cond, seed)] = np.array(sv_sel, dtype=np.float64)
    return runs


def compute_rho(sv: np.ndarray, t_p: int, k_max: int, ref_indices: List[int]) -> Optional[float]:
    """Spearman rho between tick rank and cosine sim to reference (on sv_selected)."""
    valid_ref = [sv[i] for i in ref_indices if 0 <= i < len(sv) and np.linalg.norm(sv[i]) > 0]
    if not valid_ref:
        return None
    r_pre = np.mean(valid_ref, axis=0)

    pairs = []
    for k in range(1, k_max + 1):
        idx = t_p + k
        if idx < len(sv) and np.linalg.norm(sv[idx]) > 0:
            pairs.append((k, cosine_sim(r_pre, sv[idx])))

    if len(pairs) < 5:
        return None
    ranks = [p[0] for p in pairs]
    sims = [p[1] for p in pairs]
    rho, _ = spearmanr(ranks, sims)
    return float(rho)


def compute_persistence(sv: np.ndarray, t_p: int, k_max: int, ref_indices: List[int]) -> Optional[float]:
    """Mean cosine sim in post-perturbation window to reference."""
    valid_ref = [sv[i] for i in ref_indices if 0 <= i < len(sv) and np.linalg.norm(sv[i]) > 0]
    if not valid_ref:
        return None
    r_pre = np.mean(valid_ref, axis=0)

    sims = []
    for k in range(1, k_max + 1):
        idx = t_p + k
        if idx < len(sv) and np.linalg.norm(sv[idx]) > 0:
            sims.append(cosine_sim(r_pre, sv[idx]))

    return float(np.mean(sims)) if sims else None


def pre_perturbation_dispersion(sv: np.ndarray, ref_indices: List[int]) -> Optional[float]:
    """Pairwise cosine distance among the reference ticks (dispersion of sv_selected pre-pert)."""
    vecs = [sv[i] for i in ref_indices if 0 <= i < len(sv) and np.linalg.norm(sv[i]) > 0]
    if len(vecs) < 2:
        return None
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(1.0 - cosine_sim(vecs[i], vecs[j]))
    return float(np.mean(dists))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    runs = load_runs(runs_dir)
    print(f"Loaded {len(runs)} runs")

    lines = []
    lines.append("=" * 80)
    lines.append("SLOPE COMPARISON + REFERENCE ROBUSTNESS — sv_selected V7")
    lines.append("=" * 80)

    # ════════════════════════════════════════════════════════════════
    # SECTION 1: Direct slope comparison (rho) E vs multi-agent
    # ════════════════════════════════════════════════════════════════
    lines.append("\n" + "=" * 60)
    lines.append("  SECTION 1: Paired Wilcoxon on Spearman rho (E vs multi-agent)")
    lines.append("=" * 60)

    all_pvals = []
    all_results = []

    for t_p, wcfg in WINDOWS.items():
        key = wcfg["label"]
        k_max = wcfg["k_max"]
        ref_idx = wcfg["pre_range"]

        lines.append(f"\n  --- {key.upper()} window (k=1..{k_max}) ---\n")

        # Compute rho per (condition, seed)
        rhos = defaultdict(dict)  # cond -> {seed: rho}
        for (cond, seed), sv in runs.items():
            if cond in CONDITIONS_MULTI or cond == "E":
                rho = compute_rho(sv, t_p, k_max, ref_idx)
                if rho is not None:
                    rhos[cond][seed] = rho

        # Paired tests: each multi-agent condition vs E
        lines.append(f"  {'Comparison':<10} {'n':>3} {'rho_cond':>10} {'rho_E':>10} {'W':>8} {'p_one':>10} {'r_rb':>6}")
        lines.append("  " + "-" * 60)

        for cond in CONDITIONS_MULTI:
            # Match by seed
            common_seeds = set(rhos.get(cond, {}).keys()) & set(rhos.get("E", {}).keys())
            common_seeds = sorted(common_seeds)
            n = len(common_seeds)

            if n < 5:
                lines.append(f"  {cond} vs E   {n:>3}  insufficient pairs")
                all_pvals.append(None)
                all_results.append({"comp": f"{cond}>E", "window": key, "p": None})
                continue

            rho_cond = np.array([rhos[cond][s] for s in common_seeds])
            rho_e = np.array([rhos["E"][s] for s in common_seeds])
            diffs = rho_cond - rho_e  # positive = multi-agent has higher rho (more recovery)

            try:
                stat, p_one = wilcoxon(diffs, alternative="greater")
            except ValueError:
                stat, p_one = 0, 1.0

            r_rb = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0) if n > 0 else 0

            lines.append(f"  {cond} > E   {n:>3} {np.mean(rho_cond):>10.4f} {np.mean(rho_e):>10.4f} "
                          f"{stat:>8.0f} {p_one:>10.4f} {r_rb:>6.3f}")

            all_pvals.append(p_one)
            all_results.append({"comp": f"{cond}>E", "window": key, "p": p_one, "r_rb": r_rb,
                                "rho_cond": float(np.mean(rho_cond)), "rho_e": float(np.mean(rho_e))})

    # Holm-Bonferroni on all slope comparisons
    corrected = holm_bonferroni(all_pvals)
    lines.append(f"\n  Holm-Bonferroni correction ({len([p for p in all_pvals if p is not None])} tests):")
    for i, res in enumerate(all_results):
        if res["p"] is not None:
            sig = "***" if corrected[i] < 0.001 else "**" if corrected[i] < 0.01 else "*" if corrected[i] < 0.05 else "ns"
            lines.append(f"    {res['comp']:<10} {res['window']:<4} p_raw={res['p']:.4f} p_corr={corrected[i]:.4f} {sig}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2: Reference robustness (single tick vs 3-tick mean)
    # ════════════════════════════════════════════════════════════════
    lines.append("\n" + "=" * 60)
    lines.append("  SECTION 2: Reference robustness (single-tick vs 3-tick mean)")
    lines.append("=" * 60)

    for t_p, wcfg in WINDOWS.items():
        key = wcfg["label"]
        k_max = wcfg["k_max"]
        ref_3tick = wcfg["pre_range"]
        ref_single = [wcfg["pre_single"]]

        lines.append(f"\n  --- {key.upper()} ---")
        lines.append(f"  3-tick ref: {ref_3tick}, single ref: {ref_single}")
        lines.append(f"\n  {'Cond':<6} {'pers_3tick':>12} {'pers_single':>12} {'diff':>8} {'rho_3tick':>10} {'rho_single':>10}")
        lines.append("  " + "-" * 62)

        for cond in CONDITIONS_MULTI + ["E"]:
            pers_3 = []
            pers_1 = []
            rho_3 = []
            rho_1 = []

            for seed in ALL_SEEDS:
                if (cond, seed) not in runs:
                    continue
                sv = runs[(cond, seed)]

                p3 = compute_persistence(sv, t_p, k_max, ref_3tick)
                p1 = compute_persistence(sv, t_p, k_max, ref_single)
                r3 = compute_rho(sv, t_p, k_max, ref_3tick)
                r1 = compute_rho(sv, t_p, k_max, ref_single)

                if p3 is not None:
                    pers_3.append(p3)
                if p1 is not None:
                    pers_1.append(p1)
                if r3 is not None:
                    rho_3.append(r3)
                if r1 is not None:
                    rho_1.append(r1)

            if pers_3 and pers_1:
                diff = np.mean(pers_3) - np.mean(pers_1)
                lines.append(f"  {cond:<6} {np.mean(pers_3):>12.4f} {np.mean(pers_1):>12.4f} {diff:>8.4f} "
                              f"{np.mean(rho_3):>10.4f} {np.mean(rho_1):>10.4f}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3: Pre-perturbation dispersion of sv_selected
    # ════════════════════════════════════════════════════════════════
    lines.append("\n" + "=" * 60)
    lines.append("  SECTION 3: Pre-perturbation dispersion of sv_selected")
    lines.append("=" * 60)
    lines.append("  (Mean pairwise cosine distance among reference ticks)")

    for t_p, wcfg in WINDOWS.items():
        key = wcfg["label"]
        ref_idx = wcfg["pre_range"]

        lines.append(f"\n  --- {key.upper()} (reference ticks {ref_idx}) ---")
        lines.append(f"  {'Cond':<6} {'mean_disp':>12} {'std':>8} {'n':>4}")
        lines.append("  " + "-" * 30)

        for cond in CONDITIONS_MULTI + ["E"]:
            disps = []
            for seed in ALL_SEEDS:
                if (cond, seed) not in runs:
                    continue
                sv = runs[(cond, seed)]
                d = pre_perturbation_dispersion(sv, ref_idx)
                if d is not None:
                    disps.append(d)

            if disps:
                lines.append(f"  {cond:<6} {np.mean(disps):>12.4f} {np.std(disps):>8.4f} {len(disps):>4}")

    lines.append("\n  NOTE: Higher dispersion = less stable reference = potential advantage")
    lines.append("  for conditions with tighter pre-perturbation trajectories.")

    report = "\n".join(lines)
    print(report)

    report_path = runs_dir / "slope_comparison_v7_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
