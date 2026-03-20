#!/usr/bin/env python3
"""
convergence_analysis.py — Analyse de convergence multi-theorique
================================================================
Lit les sorties de recompute_dim + theory_constrained.
Calcule le RESULTAT CLE du papier.

Usage:
    python tools/convergence_analysis.py --runs-dir runs/exploratory/
    python tools/convergence_analysis.py \
        --dim-results analysis/recomputed/dim_results.json \
        --theory-results analysis/theory_constrained/results.json \
        --runs-dir runs/exploratory/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SIGNAL_KEYS = [
    "novelty", "conflict", "cohesion", "impl_pressure",
    "cost_pressure", "prediction_error",
]

RUN_ORDER = [
    "single_A", "single_B", "single_C",
    "random_perm0", "random_perm1", "random_perm2",
    "full_perm0", "full_perm1", "full_perm2",
]


def _load_metrics(path: Path) -> List[Dict]:
    ticks = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                ticks.append(json.loads(l))
    return ticks


def _condition_of(run_name: str) -> str:
    if run_name.startswith("full_"):
        return "full"
    elif run_name.startswith("random_"):
        return "random"
    elif run_name.startswith("single_"):
        return "single"
    return "?"


def _discover_runs(runs_dir: Path) -> List[Tuple[str, str, Path]]:
    results = []
    for block_dir in sorted(runs_dir.glob("latin_block_*_seed*")):
        if not block_dir.is_dir():
            continue
        for run_name in RUN_ORDER:
            run_dir = block_dir / run_name
            if (run_dir / "metrics.jsonl").exists():
                results.append((_condition_of(run_name), f"{block_dir.name}/{run_name}", run_dir))
    return results


def _spearman_matrix(data: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    """Compute Spearman correlation matrix between named series."""
    from scipy.stats import spearmanr

    keys = sorted(k for k, v in data.items() if len(v) >= 3)
    n = len(keys)
    mat = np.eye(n)
    p_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x = data[keys[i]]
            y = data[keys[j]]
            min_len = min(len(x), len(y))
            if min_len < 3:
                continue
            r, p = spearmanr(x[:min_len], y[:min_len])
            mat[i, j] = mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p

    return mat, keys


def _count_convergent(mat: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
    """Count pairs with |r| > threshold (upper triangle)."""
    n = mat.shape[0]
    n_conv = 0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if abs(mat[i, j]) > threshold:
                n_conv += 1
    return n_conv, n_pairs


def _count_divergent(mat: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
    """Count pairs with |r| < threshold (upper triangle)."""
    n = mat.shape[0]
    n_div = 0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if abs(mat[i, j]) < threshold:
                n_div += 1
    return n_div, n_pairs


def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for Organism")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/exploratory"))
    parser.add_argument("--dim-results", type=Path, default=None)
    parser.add_argument("--theory-results", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("analysis/convergence"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        sys.exit(1)

    print(f"Found {len(runs)} runs")

    # ── 1. Load dim_results ──────────────────────────────────────────
    dim_data = None
    dim_path = args.dim_results or (PROJECT_ROOT / "analysis" / "recomputed" / "dim_results.json")
    if dim_path.exists():
        with open(dim_path) as f:
            dim_data = json.load(f)
        print(f"Loaded dim results: {dim_path}")
    else:
        print(f"No dim results at {dim_path} — will skip dim correlations")

    # ── 2. Load theory_constrained results ───────────────────────────
    theory_data = None
    theory_path = args.theory_results or (PROJECT_ROOT / "analysis" / "theory_constrained" / "results.json")
    if theory_path.exists():
        with open(theory_path) as f:
            theory_data = json.load(f)
        print(f"Loaded theory-constrained results: {theory_path}")
    else:
        print(f"No theory-constrained results at {theory_path}")
        print("  Run: python tools/theory_constrained.py --phase 1 first")

    # ── 3. Original proxy Spearman matrix (from metrics.jsonl) ───────
    print("\n" + "=" * 70)
    print("  ORIGINAL PROXY CORRELATION MATRIX")
    print("=" * 70)

    # Per-run average theory scores
    run_theory_avgs: Dict[str, Dict[str, float]] = {}
    run_conditions: Dict[str, str] = {}
    run_debate_ticks: Dict[str, int] = {}

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        theory_sums: Dict[str, List[float]] = defaultdict(list)
        n_debate = 0
        for t in ticks:
            ts = t.get("theory_scores", {})
            for th, val in ts.items():
                if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                    theory_sums[th].append(val)
            if t.get("mode") == "Debate":
                n_debate += 1

        avgs = {th: float(np.mean(vals)) for th, vals in theory_sums.items() if vals}
        run_theory_avgs[rid] = avgs
        run_conditions[rid] = cond
        run_debate_ticks[rid] = n_debate

    # Build matrix data: theory -> list of per-run averages
    all_theories = set()
    for avgs in run_theory_avgs.values():
        all_theories.update(avgs.keys())
    all_theories = sorted(all_theories)

    proxy_data: Dict[str, List[float]] = {th: [] for th in all_theories}
    run_ids_ordered = sorted(run_theory_avgs.keys())
    for rid in run_ids_ordered:
        avgs = run_theory_avgs[rid]
        for th in all_theories:
            proxy_data[th].append(avgs.get(th, 0.0))

    proxy_mat, proxy_keys = _spearman_matrix(proxy_data)
    n_div_proxy, n_pairs_proxy = _count_divergent(proxy_mat)

    # Print matrix
    print(f"\nTheories: {', '.join(proxy_keys)}")
    print(f"  {'':<8}", end="")
    for k in proxy_keys:
        print(f"  {k:>6}", end="")
    print()
    for i, k in enumerate(proxy_keys):
        print(f"  {k:<8}", end="")
        for j in range(len(proxy_keys)):
            r = proxy_mat[i, j]
            if i == j:
                print(f"  {'1.00':>6}", end="")
            else:
                marker = " " if abs(r) >= 0.5 else "*"
                print(f"  {r:>5.2f}{marker}", end="")
        print()

    print(f"\n  Divergent pairs (|r| < 0.5): {n_div_proxy}/{n_pairs_proxy}")

    # ── 4. Constrained measures matrix ───────────────────────────────
    n_conv_constrained = None
    n_pairs_constrained = None

    if theory_data and theory_data.get("runs"):
        print("\n" + "=" * 70)
        print("  CONSTRAINED MEASURES CORRELATION MATRIX")
        print("=" * 70)

        # Build data from theory_constrained results
        constrained_data: Dict[str, List[float]] = defaultdict(list)
        constrained_run_ids = []

        for r in theory_data["runs"]:
            rid = f"{r['block']}/{r['run_id']}"
            constrained_run_ids.append(rid)
            for theory, m in r["measures"].items():
                constrained_data[theory].append(m.get("value", 0.0))

        const_mat, const_keys = _spearman_matrix(constrained_data)
        n_conv_constrained, n_pairs_constrained = _count_convergent(const_mat)

        print(f"\nTheories: {', '.join(const_keys)}")
        print(f"  {'':<8}", end="")
        for k in const_keys:
            print(f"  {k:>6}", end="")
        print()
        for i, k in enumerate(const_keys):
            print(f"  {k:<8}", end="")
            for j in range(len(const_keys)):
                r = const_mat[i, j]
                if i == j:
                    print(f"  {'1.00':>6}", end="")
                else:
                    marker = "+" if abs(r) >= 0.5 else " "
                    print(f"  {r:>5.2f}{marker}", end="")
            print()

        print(f"\n  Convergent pairs (|r| > 0.5): {n_conv_constrained}/{n_pairs_constrained}")

    # ── 5. Correlation with dim_effective ────────────────────────────
    print("\n" + "=" * 70)
    print("  CORRELATION WITH dim_effective")
    print("=" * 70)

    if dim_data and dim_data.get("runs"):
        # Build dim lookup
        dim_lookup: Dict[str, float] = {}
        for r in dim_data["runs"]:
            key = f"{r['block']}/{r['run_id']}"
            dim_lookup[key] = r.get("dim_participation", 0.0)

        # Correlate each proxy theory with dim
        from scipy.stats import spearmanr

        dim_vals = []
        for rid in run_ids_ordered:
            dim_vals.append(dim_lookup.get(rid, 0.0))

        print(f"\n  {'Theory':<10} {'r':>8} {'p':>10}")
        print("  " + "-" * 30)
        for th in proxy_keys:
            vals = proxy_data[th]
            if len(vals) >= 3 and len(dim_vals) >= 3:
                r, p = spearmanr(vals[:len(dim_vals)], dim_vals[:len(vals)])
                sig = "*" if p < 0.05 else ""
                print(f"  {th:<10} {r:>8.4f} {p:>9.4f}{sig}")

    # ── 6. Trivial convergence test ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRIVIAL CONVERGENCE TEST (correlation with n_debate)")
    print("=" * 70)

    debate_vals = [run_debate_ticks.get(rid, 0) for rid in run_ids_ordered]

    if any(d > 0 for d in debate_vals):
        from scipy.stats import spearmanr

        n_trivial = 0
        n_tested = 0
        print(f"\n  {'Theory':<10} {'r':>8} {'p':>10} {'Trivial?':>10}")
        print("  " + "-" * 40)
        for th in proxy_keys:
            vals = proxy_data[th]
            if len(vals) >= 3:
                r, p = spearmanr(vals[:len(debate_vals)], debate_vals[:len(vals)])
                trivial = abs(r) > 0.7 and p < 0.05
                n_tested += 1
                if trivial:
                    n_trivial += 1
                tag = "YES" if trivial else "no"
                print(f"  {th:<10} {r:>8.4f} {p:>9.4f} {tag:>10}")

        print(f"\n  {n_trivial}/{n_tested} theories trivially correlated with debate count")
    else:
        print("  No debate ticks found")

    # ── 7. KEY RESULT ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTAT CLE")
    print("=" * 70)

    print(f"\n  Proxys originaux:     {n_div_proxy}/{n_pairs_proxy} paires divergentes (|r| < 0.5)")
    if n_conv_constrained is not None:
        print(f"  Mesures contraintes:  {n_conv_constrained}/{n_pairs_constrained} paires convergentes (|r| > 0.5)")
        print()
        if n_conv_constrained > n_pairs_constrained // 2:
            print("  => Convergence detected.")
            if any(d > 0 for d in debate_vals):
                print("     Check trivial convergence test above for interpretation.")
        else:
            print("  => Theories genuinely measure different aspects.")
    else:
        print("  Mesures contraintes:  PAS ENCORE CALCULEES")
        print("    Run: python tools/theory_constrained.py --phase 1")

    print("=" * 70)

    # ── Save report ──────────────────────────────────────────────────
    report = {
        "proxy_divergent": n_div_proxy,
        "proxy_pairs": n_pairs_proxy,
        "proxy_theories": proxy_keys,
        "constrained_convergent": n_conv_constrained,
        "constrained_pairs": n_pairs_constrained,
    }

    with open(args.output / "convergence_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.output / 'convergence_results.json'}")

    # Try matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2 if n_conv_constrained is not None else 1,
                                 figsize=(12, 5))
        if n_conv_constrained is None:
            axes = [axes]

        # Proxy matrix
        ax = axes[0]
        im = ax.imshow(proxy_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(proxy_keys)))
        ax.set_yticks(range(len(proxy_keys)))
        ax.set_xticklabels(proxy_keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(proxy_keys, fontsize=8)
        ax.set_title(f"Original Proxys\n{n_div_proxy}/{n_pairs_proxy} divergent")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Constrained matrix
        if n_conv_constrained is not None and len(axes) > 1:
            ax = axes[1]
            im = ax.imshow(const_mat, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(const_keys)))
            ax.set_yticks(range(len(const_keys)))
            ax.set_xticklabels(const_keys, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(const_keys, fontsize=8)
            ax.set_title(f"Constrained Measures\n{n_conv_constrained}/{n_pairs_constrained} convergent")
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(args.output / "convergence_matrix.png", dpi=150)
        print(f"Saved: {args.output / 'convergence_matrix.png'}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
