#!/usr/bin/env python3
"""
falsification_tests.py — 12 tests de falsification pour le bench Organism
========================================================================
Lit les donnees de runs existants (metrics.jsonl, events.jsonl).
NE MODIFIE AUCUN fichier dans les dossiers de runs.

Usage:
    python tools/falsification_tests.py --runs-dir runs/exploratory/

Output:
    - Table terminal
    - analysis/falsification/falsification_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Signal keys ──────────────────────────────────────────────────────
SIGNAL_KEYS = [
    "novelty", "conflict", "cohesion", "impl_pressure",
    "cost_pressure", "prediction_error",
]

RUN_ORDER = [
    "single_A", "single_B", "single_C",
    "random_perm0", "random_perm1", "random_perm2",
    "full_perm0", "full_perm1", "full_perm2",
]


# ── Helpers ──────────────────────────────────────────────────────────

def _load_metrics(metrics_path: Path) -> List[Dict]:
    ticks = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ticks.append(json.loads(line))
    return ticks


def _is_judge_failed(tick: dict) -> bool:
    """Return True if this tick had a failed judge verdict."""
    jv = tick.get("judge_verdict", {})
    if not jv:
        return False
    return bool(jv.get("judge_failed", False))


def _signal_matrix(ticks: List[Dict], exclude_judge_failed: bool = False) -> np.ndarray:
    """Extract signal matrix (n_ticks x n_signals), drop constant columns."""
    rows = []
    for t in ticks:
        if exclude_judge_failed and _is_judge_failed(t):
            continue
        sigs = t.get("signals", {})
        rows.append([sigs.get(k, 0.0) for k in SIGNAL_KEYS])
    if not rows:
        return np.zeros((0, len(SIGNAL_KEYS)))
    X = np.array(rows)
    # Drop constant columns
    var = X.var(axis=0)
    active = var > 1e-10
    return X[:, active]


def _dim_effective(X: np.ndarray) -> float:
    """Participation ratio from covariance eigenvalues."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return float("nan")
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    s = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float((s ** 2) / s2) if s2 > 0 else 0.0


def _condition_of(run_name: str) -> str:
    if run_name.startswith("full_"):
        return "full"
    elif run_name.startswith("random_"):
        return "random"
    elif run_name.startswith("single_"):
        return "single"
    return "?"


def _discover_runs(runs_dir: Path) -> List[Tuple[str, str, Path]]:
    """Returns (condition, run_id, run_dir) for all runs."""
    results = []
    for block_dir in sorted(runs_dir.glob("latin_block_*_seed*")):
        if not block_dir.is_dir():
            continue
        for run_name in RUN_ORDER:
            run_dir = block_dir / run_name
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                cond = _condition_of(run_name)
                rid = f"{block_dir.name}/{run_name}"
                results.append((cond, rid, run_dir))
    return results


def _spearman(x, y):
    """Pure-python Spearman rank correlation."""
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return r, p


# ── TEST 1: PCA artifact ────────────────────────────────────────────

def test1_pca_artifact(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Distance to centroid WITHOUT PCA."""
    cond_dists: Dict[str, List[float]] = defaultdict(list)

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        X = _signal_matrix(ticks)
        if X.shape[0] < 5:
            continue
        centroid = X.mean(axis=0)
        dists = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        cond_dists[cond].append(float(np.mean(dists)))

    full_d = cond_dists.get("full", [])
    rand_d = cond_dists.get("random", [])

    result = {
        "full_mean_dist": float(np.mean(full_d)) if full_d else None,
        "random_mean_dist": float(np.mean(rand_d)) if rand_d else None,
    }

    if full_d and rand_d:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(full_d, rand_d, alternative="less")
        result["U"] = float(u)
        result["p"] = float(p)
        result["pass"] = p < 0.05 and np.mean(full_d) < np.mean(rand_d)
    else:
        result["pass"] = None

    return result


# ── TEST 2: Logging bias ────────────────────────────────────────────

def test2_logging_bias(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Recalculate dim_effective excluding first 10 ticks."""
    deltas = []

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        X_full = _signal_matrix(ticks)
        X_trimmed = _signal_matrix(ticks[10:]) if len(ticks) > 20 else X_full

        dim_full = _dim_effective(X_full)
        dim_trim = _dim_effective(X_trimmed)

        if not math.isnan(dim_full) and dim_full > 0:
            delta_pct = abs(dim_trim - dim_full) / dim_full * 100
            deltas.append(delta_pct)

    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    max_delta = float(np.max(deltas)) if deltas else float("nan")

    return {
        "mean_delta_pct": round(mean_delta, 2),
        "max_delta_pct": round(max_delta, 2),
        "n_runs": len(deltas),
        "pass": mean_delta < 10.0 if not math.isnan(mean_delta) else None,
    }


# ── TEST 3: Model confound ──────────────────────────────────────────

def test3_model_confound(runs: List[Tuple[str, str, Path]]) -> Dict:
    """worst_full < best_random?"""
    cond_dims: Dict[str, List[float]] = defaultdict(list)

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        X = _signal_matrix(ticks)
        d = _dim_effective(X)
        if not math.isnan(d):
            cond_dims[cond].append(d)

    full_dims = cond_dims.get("full", [])
    rand_dims = cond_dims.get("random", [])

    if full_dims and rand_dims:
        worst_full = max(full_dims)
        best_random = min(rand_dims)
        return {
            "worst_full": round(worst_full, 4),
            "best_random": round(best_random, 4),
            "pass": worst_full < best_random,
        }
    return {"pass": None}


# ── TEST 4: Proxy circularity ───────────────────────────────────────

def test4_proxy_circularity(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Spearman of theory_scores; compare real vs shuffled."""
    from scipy.stats import spearmanr

    def _count_divergent(matrix_data: Dict[str, List[float]]) -> Tuple[int, int]:
        """Count pairs with |r| < 0.5."""
        theories = sorted(matrix_data.keys())
        n_pairs = 0
        n_divergent = 0
        for i in range(len(theories)):
            for j in range(i + 1, len(theories)):
                x = matrix_data[theories[i]]
                y = matrix_data[theories[j]]
                if len(x) >= 5 and len(y) >= 5:
                    r, _ = spearmanr(x, y)
                    n_pairs += 1
                    if abs(r) < 0.5:
                        n_divergent += 1
        return n_divergent, n_pairs

    # Collect theory scores per run
    cond_theory_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        run_means: Dict[str, float] = {}
        for t in ticks:
            ts = t.get("theory_scores", {})
            for th, val in ts.items():
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    if th not in run_means:
                        run_means[th] = []
                    run_means[th].append(val) if isinstance(run_means[th], list) else None

        # Fix: run_means values are lists, need to average
        run_avgs = {}
        for th, vals in run_means.items():
            if isinstance(vals, list) and vals:
                run_avgs[th] = np.mean(vals)

        for th, val in run_avgs.items():
            cond_theory_data["all"][th].append(val)
            cond_theory_data[cond][th].append(val)

    all_data = cond_theory_data.get("all", {})
    n_div_real, n_pairs = _count_divergent(all_data)

    # Shuffled version
    n_div_shuffled_list = []
    for _ in range(50):
        shuffled_data = {}
        for th, vals in all_data.items():
            shuffled_data[th] = list(np.random.permutation(vals))
        n_div_s, _ = _count_divergent(shuffled_data)
        n_div_shuffled_list.append(n_div_s)

    mean_shuffled = float(np.mean(n_div_shuffled_list)) if n_div_shuffled_list else 0

    # Per condition
    per_cond = {}
    for c in ["full", "random", "single"]:
        cd = cond_theory_data.get(c, {})
        if cd:
            nd, np_ = _count_divergent(cd)
            per_cond[c] = {"divergent": nd, "pairs": np_}

    return {
        "n_divergent_real": n_div_real,
        "n_pairs": n_pairs,
        "mean_divergent_shuffled": round(mean_shuffled, 1),
        "per_condition": per_cond,
        "pass": n_div_real > 0 and mean_shuffled > 0,  # Divergence persists
    }


# ── TEST 5: Scheduler overfit (DEFERRED) ────────────────────────────

def test5_scheduler_overfit() -> Dict:
    return {
        "status": "DEFERRED",
        "note": "Requires mini-bench with perturbed scheduler weights (±30%). Run after confirmatory bench if tests 1-4 pass.",
        "pass": None,
    }


# ── TEST 6: Identity leakage ────────────────────────────────────────

def test6_identity_leakage(runs: List[Tuple[str, str, Path]]) -> Dict:
    """LogisticRegression on TF-IDF to classify agent identity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return {"pass": None, "note": "sklearn not installed"}

    accuracies = []
    for cond, rid, run_dir in runs:
        if cond != "full":
            continue
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue

        # Extract agent drafts
        labels = []
        texts = []
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("type") != "agent_turn":
                    continue
                agent = ev.get("agent", "")
                payload = ev.get("payload", {})
                text = payload.get("text", "") if isinstance(payload, dict) else ""
                if not text:
                    # Try top-level text field
                    text = ev.get("text", "")
                if text and len(text) > 50 and agent in ["A", "B", "C"]:
                    labels.append(agent)
                    texts.append(text[:500])

        if len(texts) < 30 or len(set(labels)) < 2:
            continue

        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words=None)
            X = tfidf.fit_transform(texts)
            clf = LogisticRegression(max_iter=500, C=1.0)
            scores = cross_val_score(clf, X, labels, cv=min(5, len(texts) // 10), scoring="accuracy")
            accuracies.append(float(np.mean(scores)))
        except Exception:
            continue

    if not accuracies:
        return {"pass": None, "note": "no usable full runs for classification"}

    mean_acc = float(np.mean(accuracies))
    return {
        "mean_accuracy": round(mean_acc, 4),
        "n_runs": len(accuracies),
        "pass": mean_acc < 0.50,
    }


# ── TEST 7: Missingness non-random ──────────────────────────────────

def test7_missingness(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Compare signals of ticks with missing/corrected verdict vs clean."""
    from scipy.stats import mannwhitneyu

    all_p_values = []

    for cond, rid, run_dir in runs:
        if cond == "single":
            continue
        ticks = _load_metrics(run_dir / "metrics.jsonl")

        clean_signals: Dict[str, List[float]] = defaultdict(list)
        missing_signals: Dict[str, List[float]] = defaultdict(list)

        for t in ticks:
            jv = t.get("judge_verdict", {})
            is_missing = (not jv) or jv.get("judge_failed", False) or not jv.get("winner")
            sigs = t.get("signals", {})

            target = missing_signals if is_missing else clean_signals
            for k in SIGNAL_KEYS:
                v = sigs.get(k)
                if v is not None:
                    target[k].append(v)

        for k in SIGNAL_KEYS:
            c = clean_signals.get(k, [])
            m = missing_signals.get(k, [])
            if len(c) >= 5 and len(m) >= 3:
                _, p = mannwhitneyu(c, m, alternative="two-sided")
                all_p_values.append(p)

    if not all_p_values:
        return {"pass": None, "note": "insufficient data"}

    min_p = float(np.min(all_p_values))
    n_sig = sum(1 for p in all_p_values if p < 0.05)

    return {
        "n_tests": len(all_p_values),
        "n_significant": n_sig,
        "min_p": round(min_p, 6),
        "pass": n_sig == 0,  # No significant difference
    }


# ── TEST 8: Provider drift ──────────────────────────────────────────

def test8_provider_drift(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Spearman(tick_id, rolling dim) per condition."""
    from scipy.stats import spearmanr

    cond_corrs: Dict[str, List[float]] = defaultdict(list)

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        X = _signal_matrix(ticks)
        n = X.shape[0]
        if n < 60:
            continue

        window = 50
        rolling_dims = []
        tick_ids = []
        for i in range(window, n):
            chunk = X[i - window:i]
            d = _dim_effective(chunk)
            if not math.isnan(d):
                rolling_dims.append(d)
                tick_ids.append(i)

        if len(rolling_dims) >= 10:
            r, _ = spearmanr(tick_ids, rolling_dims)
            cond_corrs[cond].append(abs(r))

    result = {}
    all_pass = True
    for c in ["full", "random", "single"]:
        corrs = cond_corrs.get(c, [])
        if corrs:
            mean_r = float(np.mean(corrs))
            result[c] = {"mean_abs_r": round(mean_r, 4), "n_runs": len(corrs)}
            if mean_r > 0.2:
                all_pass = False
        else:
            result[c] = {"mean_abs_r": None}

    result["pass"] = all_pass
    return result


# ── TEST 10: Cross-judge invariance ────────────────────────────────

def test10_cross_judge_invariance(
    runs_dir_a: Optional[Path],
    runs_dir_b: Optional[Path],
) -> Dict:
    """Compare dim_effective between two benches with different judges.
    Spearman(dim_bench_a, dim_bench_b) on matched run pairs.

    PASS: r > 0.7 AND same gradient full < random < single
    FAIL: gradient disappears with alternate judge."""
    if not runs_dir_a or not runs_dir_b:
        return {"pass": None, "note": "needs --runs-dir-alt for cross-judge comparison"}
    if not runs_dir_a.exists() or not runs_dir_b.exists():
        return {"pass": None, "note": f"missing dir: {runs_dir_a} or {runs_dir_b}"}

    runs_a = _discover_runs(runs_dir_a)
    runs_b = _discover_runs(runs_dir_b)

    if not runs_a or not runs_b:
        return {"pass": None, "note": "one or both dirs have no runs"}

    # Build dim lookup per run_name (ignoring block prefix for matching)
    def _dim_by_run_name(runs):
        dims = {}
        for cond, rid, run_dir in runs:
            # rid is like "latin_block_0_seed42/full_perm0"
            run_name = rid.split("/")[-1] if "/" in rid else rid
            ticks = _load_metrics(run_dir / "metrics.jsonl")
            X = _signal_matrix(ticks, exclude_judge_failed=True)
            d = _dim_effective(X)
            if not math.isnan(d):
                dims[run_name] = {"dim": d, "condition": cond}
        return dims

    dims_a = _dim_by_run_name(runs_a)
    dims_b = _dim_by_run_name(runs_b)

    # Match pairs
    matched_names = sorted(set(dims_a.keys()) & set(dims_b.keys()))
    if len(matched_names) < 5:
        return {"pass": None, "note": f"only {len(matched_names)} matched pairs (need >= 5)"}

    vals_a = [dims_a[n]["dim"] for n in matched_names]
    vals_b = [dims_b[n]["dim"] for n in matched_names]
    conditions = [dims_a[n]["condition"] for n in matched_names]

    from scipy.stats import spearmanr
    r, p = spearmanr(vals_a, vals_b)

    # Check gradient in bench B
    cond_dims_b: Dict[str, List[float]] = defaultdict(list)
    for n in matched_names:
        cond_dims_b[dims_b[n]["condition"]].append(dims_b[n]["dim"])

    gradient_b = True
    full_mean = np.mean(cond_dims_b.get("full", [0]))
    rand_mean = np.mean(cond_dims_b.get("random", [0]))
    single_mean = np.mean(cond_dims_b.get("single", [0]))
    if not (full_mean < rand_mean):
        gradient_b = False

    # Detect judge models from summary.json
    judge_a = judge_b = "unknown"
    for _, _, run_dir in runs_a[:1]:
        sp = run_dir / "summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            judge_a = s.get("judge_model", "unknown")
    for _, _, run_dir in runs_b[:1]:
        sp = run_dir / "summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            judge_b = s.get("judge_model", "unknown")

    return {
        "spearman_r": round(float(r), 4),
        "spearman_p": round(float(p), 6),
        "n_matched": len(matched_names),
        "judge_a": judge_a,
        "judge_b": judge_b,
        "gradient_bench_b": gradient_b,
        "bench_b_means": {
            "full": round(full_mean, 4),
            "random": round(rand_mean, 4),
            "single": round(single_mean, 4),
        },
        "pass": r > 0.7 and gradient_b,
    }


# ── TEST 11: Judge-failed bias ─────────────────────────────────────

def test11_judge_failed_bias(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Compare dim_effective with and without judge_failed ticks.
    PASS: delta < 5% → failed verdicts don't bias dimensionality.
    FAIL: delta >= 5% → compression was artificially inflated."""
    deltas = []
    per_run = []

    for cond, rid, run_dir in runs:
        ticks = _load_metrics(run_dir / "metrics.jsonl")
        n_failed = sum(1 for t in ticks if _is_judge_failed(t))

        if n_failed == 0:
            continue  # No judge failures in this run

        X_all = _signal_matrix(ticks, exclude_judge_failed=False)
        X_clean = _signal_matrix(ticks, exclude_judge_failed=True)

        dim_all = _dim_effective(X_all)
        dim_clean = _dim_effective(X_clean)

        if math.isnan(dim_all) or math.isnan(dim_clean) or dim_clean == 0:
            continue

        delta_pct = 100.0 * (dim_all - dim_clean) / dim_clean
        deltas.append(delta_pct)
        per_run.append({
            "run": rid, "condition": cond,
            "n_failed": n_failed, "n_total": len(ticks),
            "dim_all": round(dim_all, 4), "dim_clean": round(dim_clean, 4),
            "delta_pct": round(delta_pct, 2),
        })

    if not deltas:
        return {"pass": None, "note": "no runs with judge_failed ticks"}

    mean_delta = float(np.mean(deltas))
    max_delta = float(np.max(np.abs(deltas)))

    return {
        "mean_delta_pct": round(mean_delta, 2),
        "max_abs_delta_pct": round(max_delta, 2),
        "n_runs_with_failures": len(deltas),
        "per_run": per_run,
        "pass": abs(mean_delta) < 5.0,
    }


def test12_verbosity_bias(runs: List[Tuple[str, str, Path]]) -> Dict:
    """Spearman(mean_text_len_per_agent, win_rate_per_agent) across multi-agent runs.

    PASS: |r| < 0.3 → text length does not explain judge wins.
    FAIL: |r| >= 0.3 → judge may prefer longer responses (verbosity bias).
    Only uses full and random conditions (multi-agent).
    """
    from scipy.stats import spearmanr

    # Accumulate per-agent stats across all multi-agent runs
    agent_text_lens: Dict[str, List[int]] = defaultdict(list)
    agent_wins: Dict[str, int] = defaultdict(int)
    agent_appearances: Dict[str, int] = defaultdict(int)
    n_ticks_total = 0

    for cond, rid, run_dir in runs:
        if cond == "single":
            continue  # Skip single-agent runs

        ticks = _load_metrics(run_dir / "metrics.jsonl")
        for tick in ticks:
            agents = tick.get("agents", [])
            jv = tick.get("judge_verdict") or {}
            winner = jv.get("winner")

            for a in agents:
                aid = a.get("agent", "?")
                tlen = a.get("text_len", 0)
                agent_text_lens[aid].append(tlen)
                agent_appearances[aid] += 1

            if winner:
                agent_wins[winner] += 1
            n_ticks_total += 1

    # Build paired vectors: (mean_text_len, win_rate) per agent
    common_agents = sorted(set(agent_text_lens.keys()) & set(agent_appearances.keys()))
    if len(common_agents) < 3:
        return {
            "pass": None,
            "reason": f"insufficient_agents ({len(common_agents)})",
        }

    mean_lens = []
    win_rates = []
    per_agent = {}
    for aid in common_agents:
        ml = sum(agent_text_lens[aid]) / len(agent_text_lens[aid]) if agent_text_lens[aid] else 0
        wr = agent_wins.get(aid, 0) / agent_appearances[aid] if agent_appearances[aid] > 0 else 0
        mean_lens.append(ml)
        win_rates.append(wr)
        per_agent[aid] = {
            "mean_text_len": round(ml, 1),
            "win_rate": round(wr, 4),
            "n_wins": agent_wins.get(aid, 0),
            "n_appearances": agent_appearances[aid],
        }

    r, p = spearmanr(mean_lens, win_rates)

    return {
        "spearman_r": round(float(r), 4),
        "spearman_p": round(float(p), 6),
        "n_agents": len(common_agents),
        "n_ticks": n_ticks_total,
        "per_agent": per_agent,
        "pass": abs(float(r)) < 0.3,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Falsification tests for Organism bench")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/exploratory"))
    parser.add_argument("--runs-dir-alt", type=Path, default=None,
                        help="Alternate bench dir for cross-judge invariance test (Test 10)")
    args = parser.parse_args()

    runs = _discover_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        sys.exit(1)

    print(f"\nFound {len(runs)} runs in {args.runs_dir}")
    print(f"  full: {sum(1 for c,_,_ in runs if c=='full')}")
    print(f"  random: {sum(1 for c,_,_ in runs if c=='random')}")
    print(f"  single: {sum(1 for c,_,_ in runs if c=='single')}")

    results = {}

    # TEST 1
    print("\n[1/12] PCA artifact test...")
    results["test1_pca_artifact"] = test1_pca_artifact(runs)

    # TEST 2
    print("[2/12] Logging bias test...")
    results["test2_logging_bias"] = test2_logging_bias(runs)

    # TEST 3
    print("[3/12] Model confound test...")
    results["test3_model_confound"] = test3_model_confound(runs)

    # TEST 4
    print("[4/12] Proxy circularity test...")
    results["test4_proxy_circularity"] = test4_proxy_circularity(runs)

    # TEST 5
    print("[5/12] Scheduler overfit (DEFERRED)...")
    results["test5_scheduler_overfit"] = test5_scheduler_overfit()

    # TEST 6
    print("[6/12] Identity leakage test...")
    results["test6_identity_leakage"] = test6_identity_leakage(runs)

    # TEST 7
    print("[7/12] Missingness non-random test...")
    results["test7_missingness"] = test7_missingness(runs)

    # TEST 8
    print("[8/12] Provider drift test...")
    results["test8_provider_drift"] = test8_provider_drift(runs)

    # TEST 10
    print("[9/12] Cross-judge invariance test...")
    results["test10_cross_judge"] = test10_cross_judge_invariance(args.runs_dir, args.runs_dir_alt)

    # TEST 11
    print("[10/12] Judge-failed bias test...")
    results["test11_judge_failed_bias"] = test11_judge_failed_bias(runs)

    # TEST 12
    print("[11/12] Verbosity bias test...")
    results["test12_verbosity_bias"] = test12_verbosity_bias(runs)

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FALSIFICATION TESTS — Organism")
    print("=" * 60)

    test_labels = {
        "test1_pca_artifact": "PCA artifact",
        "test2_logging_bias": "Logging bias",
        "test3_model_confound": "Model confound",
        "test4_proxy_circularity": "Proxy circularity",
        "test5_scheduler_overfit": "Scheduler overfit",
        "test6_identity_leakage": "Identity leakage",
        "test7_missingness": "Missingness bias",
        "test8_provider_drift": "Provider drift",
        "test10_cross_judge": "Cross-judge invariance",
        "test11_judge_failed_bias": "Judge-failed bias",
        "test12_verbosity_bias": "Verbosity bias",
    }

    n_pass = 0
    n_fail = 0
    n_deferred = 0
    lines = []

    for key, label in test_labels.items():
        r = results[key]
        status = r.get("pass")
        if status is None:
            tag = "DEFERRED"
            n_deferred += 1
        elif status:
            tag = "PASS"
            n_pass += 1
        else:
            tag = "FAIL"
            n_fail += 1

        # Extract test number from key name (test1_xxx -> 1)
        test_num = key.split("_")[0].replace("test", "")

        detail = ""
        if key == "test1_pca_artifact" and r.get("p") is not None:
            detail = f"p={r['p']:.4f}"
        elif key == "test2_logging_bias":
            detail = f"delta={r.get('mean_delta_pct', '?')}%"
        elif key == "test3_model_confound" and r.get("worst_full") is not None:
            detail = f"worst_full={r['worst_full']:.2f} best_random={r['best_random']:.2f}"
        elif key == "test4_proxy_circularity":
            detail = f"{r.get('n_divergent_real', '?')}/{r.get('n_pairs', '?')} divergent"
        elif key == "test6_identity_leakage" and r.get("mean_accuracy") is not None:
            detail = f"acc={r['mean_accuracy']:.1%}"
        elif key == "test7_missingness":
            detail = f"{r.get('n_significant', '?')}/{r.get('n_tests', '?')} significant"
        elif key == "test8_provider_drift":
            for c in ["full", "random", "single"]:
                cd = r.get(c, {})
                if cd.get("mean_abs_r") is not None:
                    detail += f"{c}:|r|={cd['mean_abs_r']:.3f} "
        elif key == "test10_cross_judge":
            if r.get("spearman_r") is not None:
                detail = f"r={r['spearman_r']:.3f} p={r['spearman_p']:.4f} ({r.get('judge_a','?')} vs {r.get('judge_b','?')})"
        elif key == "test11_judge_failed_bias":
            detail = f"mean_delta={r.get('mean_delta_pct', '?')}% max={r.get('max_abs_delta_pct', '?')}%"
        elif key == "test12_verbosity_bias":
            if r.get("spearman_r") is not None:
                detail = f"r={r['spearman_r']:.3f} p={r['spearman_p']:.4f}"
                pa = r.get("per_agent", {})
                for aid in sorted(pa):
                    detail += f" {aid}:len={pa[aid]['mean_text_len']:.0f}/wr={pa[aid]['win_rate']:.2f}"

        line = f"  Test {test_num} ({label}): {tag:>10}  {detail}"
        print(line)
        lines.append(line)

    print()
    print(f"  VERDICT: {n_pass}/{n_pass + n_fail} tests passed "
          f"({n_deferred} deferred)")
    print()
    if n_fail == 0:
        print("  Core result holds.")
    elif n_fail <= 1:
        print("  Minor concern — investigate failing test.")
    else:
        print("  Core result compromised — investigate.")
    print("=" * 60)

    # Save report
    out_dir = PROJECT_ROOT / "analysis" / "falsification"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = ["# Falsification Tests — Organism", "", "```"]
    report.extend(lines)
    report.append("")
    report.append(f"VERDICT: {n_pass}/{n_pass + n_fail} passed ({n_deferred} deferred)")
    report.append("```")
    report.append("")
    report.append("## Raw results")
    report.append("```json")
    report.append(json.dumps(results, indent=2, default=str))
    report.append("```")

    with open(out_dir / "falsification_report.md", "w") as f:
        f.write("\n".join(report))
    print(f"Report saved: {out_dir / 'falsification_report.md'}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
