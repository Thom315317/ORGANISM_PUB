#!/usr/bin/env python3
"""
reanalyze_bgem3.py — Robustness check: recalculate embeddings with BGE-M3
==========================================================================
Reads events.jsonl from bench_v7 runs, recomputes state_vector_mean and
sim_curves using BAAI/bge-m3 instead of all-mpnet-base-v2.

Produces results_bgem3.json alongside existing results.json (no modification).
At the end, prints a comparison table + Mann-Whitney tests.

Usage:
    cd ~/cristal_work/organism
    source .venv/bin/activate
    python tools/reanalyze_bgem3.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bgem3")

# ── Load BGE-M3 ──
def load_model():
    from sentence_transformers import SentenceTransformer
    log.info("Loading BAAI/bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3")
    dim = model.get_sentence_embedding_dimension()
    log.info("BGE-M3 loaded (dim=%d)", dim)
    return model, dim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_drafts_per_tick(events_path: Path) -> Dict[int, Dict[str, str]]:
    """Read events.jsonl, return {tick_id: {"A": text, "B": text, "C": text}}."""
    ticks = defaultdict(dict)
    with open(events_path) as f:
        for line in f:
            e = json.loads(line)
            agent = e.get("agent", "")
            payload = e.get("payload", {})
            tick_id = e.get("tick_id", 0)
            if agent in ("A", "B", "C") and payload.get("text"):
                text = payload["text"].strip()
                if len(text) >= 20:  # skip discarded drafts
                    ticks[tick_id][agent] = text
    return dict(ticks)


def recompute_run(model, run_dir: Path) -> Optional[dict]:
    """Recompute embeddings and sim_curves for a single run using BGE-M3."""
    results_path = run_dir / "results.json"
    events_path = run_dir / "events.jsonl"

    if not results_path.exists() or not events_path.exists():
        return None

    orig = json.load(open(results_path))
    condition = orig.get("condition", "?")
    seed = orig.get("seed", 0)
    total_ticks = orig.get("total_ticks", 80)
    pert_log = orig.get("perturbation_log", [])
    pert_ticks = [p["tick"] for p in pert_log]

    # Extract drafts
    drafts_per_tick = extract_drafts_per_tick(events_path)
    if not drafts_per_tick:
        log.warning("%s: no drafts found", run_dir.name)
        return None

    # Compute embeddings per tick
    sv_mean_list = []
    sv_selected_list = []

    # Get winner sequence from original results or events
    winner_sequence = []
    with open(events_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "tick_end":
                winner_sequence.append(e.get("payload", {}).get("judge_winner"))

    for tick in range(1, total_ticks + 1):
        agents_text = drafts_per_tick.get(tick, {})
        if not agents_text:
            sv_mean_list.append(np.zeros(1024))  # BGE-M3 dim
            sv_selected_list.append(np.zeros(1024))
            continue

        texts = list(agents_text.values())
        agent_ids = list(agents_text.keys())
        embeddings = model.encode(texts)

        # Mean (barycentre)
        sv_mean = np.mean(embeddings, axis=0)
        sv_mean_list.append(sv_mean)

        # Selected (winner's embedding)
        winner = winner_sequence[tick - 1] if tick - 1 < len(winner_sequence) else None
        if winner and winner in agents_text:
            idx = agent_ids.index(winner)
            sv_selected_list.append(embeddings[idx])
        else:
            sv_selected_list.append(sv_mean)

    # Compute sim_curves (same logic as metrics_v2.py)
    k_max = 44
    sim_curves = {}

    for t_p in pert_ticks:
        # R_pre = average of 3 ticks before perturbation (0-indexed: t_p-1, t_p-2, t_p-3)
        pre_indices = [t_p - 3, t_p - 2, t_p - 1]  # 0-indexed
        valid_pre_mean = [sv_mean_list[i] for i in pre_indices
                          if 0 <= i < len(sv_mean_list) and np.linalg.norm(sv_mean_list[i]) > 0]
        valid_pre_sel = [sv_selected_list[i] for i in pre_indices
                         if 0 <= i < len(sv_selected_list) and np.linalg.norm(sv_selected_list[i]) > 0]

        if not valid_pre_mean:
            sim_curves[f"tick_{t_p}"] = {"mean": [], "selected": []}
            continue

        r_pre_mean = np.mean(valid_pre_mean, axis=0)
        r_pre_selected = np.mean(valid_pre_sel, axis=0)

        sim_mean_curve = []
        sim_sel_curve = []

        for k in range(1, k_max + 1):
            idx = t_p + k  # 0-indexed
            if idx < len(sv_mean_list) and np.linalg.norm(sv_mean_list[idx]) > 0:
                sim_mean_curve.append(cosine_similarity(r_pre_mean, sv_mean_list[idx]))
                sim_sel_curve.append(cosine_similarity(r_pre_selected, sv_selected_list[idx]))
            else:
                sim_mean_curve.append(None)
                sim_sel_curve.append(None)

        sim_curves[f"tick_{t_p}"] = {"mean": sim_mean_curve, "selected": sim_sel_curve}

    # Compute t15_mean and t35_mean
    t15_mean = None
    t35_mean = None

    if 15 in pert_ticks and "tick_15" in sim_curves:
        vals = [v for v in sim_curves["tick_15"]["mean"] if v is not None]
        # t15_mean = mean of sim_curves from tick 16 to 34 (k=1 to k=19)
        t15_vals = vals[:19] if len(vals) >= 19 else vals
        t15_mean = float(np.mean(t15_vals)) if t15_vals else None

    if 35 in pert_ticks and "tick_35" in sim_curves:
        vals = [v for v in sim_curves["tick_35"]["mean"] if v is not None]
        # t35_mean = mean of sim_curves from tick 36 to 80 (k=1 to k=44)
        t35_mean = float(np.mean(vals)) if vals else None

    result = {
        "embedding_model": "BAAI/bge-m3",
        "condition": condition,
        "seed": seed,
        "total_ticks": total_ticks,
        "t15_mean": t15_mean,
        "t35_mean": t35_mean,
        "sim_curves": sim_curves,
    }

    # Save
    out_path = run_dir / "results_bgem3.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    return result


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 robustness reanalysis")
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    model, dim = load_model()

    # Process all runs
    all_results = {}
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and (d / "results.json").exists()])

    for i, run_dir in enumerate(run_dirs):
        log.info("[%d/%d] %s", i + 1, len(run_dirs), run_dir.name)
        result = recompute_run(model, run_dir)
        if result:
            all_results[run_dir.name] = result

    # ── Comparison table ──
    print("\n" + "=" * 80)
    print("BGE-M3 vs SBERT COMPARISON")
    print("=" * 80)

    # Aggregate by condition
    by_cond = defaultdict(lambda: {"sbert_t15": [], "sbert_t35": [], "bge_t15": [], "bge_t35": []})

    for run_dir in run_dirs:
        orig = json.load(open(run_dir / "results.json"))
        cond = orig.get("condition", "?")
        bge = all_results.get(run_dir.name)

        # SBERT sim_curves
        sbert_sim = orig.get("sim_curves", {})
        if sbert_sim and "tick_15" in sbert_sim:
            vals = [v for v in sbert_sim["tick_15"].get("mean", []) if v is not None]
            if vals:
                by_cond[cond]["sbert_t15"].append(float(np.mean(vals[:19])))
        if sbert_sim and "tick_35" in sbert_sim:
            vals = [v for v in sbert_sim["tick_35"].get("mean", []) if v is not None]
            if vals:
                by_cond[cond]["sbert_t35"].append(float(np.mean(vals)))

        # BGE
        if bge:
            if bge["t15_mean"] is not None:
                by_cond[cond]["bge_t15"].append(bge["t15_mean"])
            if bge["t35_mean"] is not None:
                by_cond[cond]["bge_t35"].append(bge["t35_mean"])

    print(f"\n{'Cond':<6} | {'SBERT_t15':>10} | {'BGE_t15':>10} | {'SBERT_t35':>10} | {'BGE_t35':>10}")
    print("-" * 60)
    for cond in sorted(by_cond):
        d = by_cond[cond]
        s15 = f"{np.mean(d['sbert_t15']):.3f}" if d["sbert_t15"] else "—"
        b15 = f"{np.mean(d['bge_t15']):.3f}" if d["bge_t15"] else "—"
        s35 = f"{np.mean(d['sbert_t35']):.3f}" if d["sbert_t35"] else "—"
        b35 = f"{np.mean(d['bge_t35']):.3f}" if d["bge_t35"] else "—"
        print(f"{cond:<6} | {s15:>10} | {b15:>10} | {s35:>10} | {b35:>10}")

    # ── Mann-Whitney tests ──
    print("\n" + "=" * 80)
    print("MANN-WHITNEY TESTS (BGE-M3)")
    print("=" * 80)

    from scipy.stats import mannwhitneyu

    comparisons = [
        ("C > E", "C", "E", "bge_t15", "greater"),
        ("C > E", "C", "E", "bge_t35", "greater"),
        ("C vs R", "C", "R", "bge_t15", "two-sided"),
        ("C vs R", "C", "R", "bge_t35", "two-sided"),
        ("D2 > R", "D2", "R", "bge_t15", "greater"),
        ("D2 > D3", "D2", "D3", "bge_t15", "greater"),
    ]

    for label, c1, c2, metric, alt in comparisons:
        v1 = by_cond[c1][metric]
        v2 = by_cond[c2][metric]
        if len(v1) >= 3 and len(v2) >= 3:
            stat, pval = mannwhitneyu(v1, v2, alternative=alt)
            m1, m2 = np.mean(v1), np.mean(v2)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  {label} ({metric}): {c1}={m1:.3f} vs {c2}={m2:.3f}  U={stat:.0f} p={pval:.4f} {sig}")
        else:
            print(f"  {label} ({metric}): insufficient data ({len(v1)}, {len(v2)})")

    # Save summary
    summary_path = runs_dir / "_bgem3_summary.json"
    summary = {}
    for cond, d in by_cond.items():
        summary[cond] = {
            "sbert_t15_mean": float(np.mean(d["sbert_t15"])) if d["sbert_t15"] else None,
            "sbert_t35_mean": float(np.mean(d["sbert_t35"])) if d["sbert_t35"] else None,
            "bge_t15_mean": float(np.mean(d["bge_t15"])) if d["bge_t15"] else None,
            "bge_t35_mean": float(np.mean(d["bge_t35"])) if d["bge_t35"] else None,
            "n_seeds": len(d["bge_t15"]),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
