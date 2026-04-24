#!/usr/bin/env python3
"""
inversion_compliance_test.py — Test how agents respond to inversion at t35
===========================================================================
For each run with strong perturbation: measure whether the first post-inversion
draft (t36) aligns with the pre-inversion state (inertia) or with the
inverted text (compliance).

Usage:
    python tools/inversion_compliance_test.py --runs-dir runs/bench_v7/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CONDITIONS_PERT = ["C", "R", "D", "D2", "D3", "E"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v7")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    print("Loading embedding model...")
    model = load_embedding_model()
    print("Model loaded.")

    by_cond = defaultdict(list)

    for run_dir in sorted(runs_dir.iterdir()):
        rp = run_dir / "results.json"
        if not rp.exists():
            continue
        d = json.load(open(rp))
        cond = d.get("condition", "?")
        if cond not in CONDITIONS_PERT:
            continue

        sv_sel = d.get("state_vector_selected", [])
        pert_log = d.get("perturbation_log", [])

        # Find inversion event at t35
        inversion_text = None
        for p in pert_log:
            if p["tick"] == 35 and p["type"] == "inversion":
                inversion_text = p.get("output_text", "")
                break

        if not inversion_text:
            continue

        # Indices: 0-indexed in sv arrays
        # t34 = index 34 (reference pre-inversion)
        # t36 = index 36 (first draft post-inversion, k=1)
        if len(sv_sel) < 37:
            continue

        ref_pre = np.array(sv_sel[34], dtype=np.float64)
        draft_post = np.array(sv_sel[36], dtype=np.float64)

        if np.linalg.norm(ref_pre) == 0 or np.linalg.norm(draft_post) == 0:
            continue

        # Embed the inversion text
        inv_embedding = model.encode([inversion_text])[0]

        sim_to_ref = cosine_sim(draft_post, ref_pre)
        sim_to_inv = cosine_sim(draft_post, inv_embedding)
        sim_ref_inv = cosine_sim(ref_pre, inv_embedding)

        by_cond[cond].append({
            "run": run_dir.name,
            "sim_to_ref": sim_to_ref,
            "sim_to_inv": sim_to_inv,
            "sim_ref_inv": sim_ref_inv,
        })

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("INVERSION COMPLIANCE TEST — BENCH V7")
    lines.append("=" * 80)
    lines.append("")
    lines.append("For each run: t36 draft compared to t34 reference (inertia)")
    lines.append("and to the inverted text injected at t35 (compliance).")
    lines.append("")
    lines.append("sim_to_ref HIGH + sim_to_inv LOW  → INERTIA (resists inversion)")
    lines.append("sim_to_ref LOW  + sim_to_inv HIGH → COMPLIANCE (follows inversion)")
    lines.append("sim_ref_inv = similarity between reference and inversion text")
    lines.append("  (lower = inversion is actually opposite; higher = inversion is weak)")

    lines.append(f"\n{'Cond':<6} {'n':>3} {'sim_ref':>10} {'sim_inv':>10} {'ref_inv':>10} {'delta':>10} {'Pattern':>15}")
    lines.append("-" * 68)

    all_data = {}
    for cond in sorted(by_cond):
        runs = by_cond[cond]
        n = len(runs)
        sim_refs = [r["sim_to_ref"] for r in runs]
        sim_invs = [r["sim_to_inv"] for r in runs]
        sim_ri = [r["sim_ref_inv"] for r in runs]

        mean_ref = np.mean(sim_refs)
        mean_inv = np.mean(sim_invs)
        mean_ri = np.mean(sim_ri)
        delta = mean_ref - mean_inv  # positive = closer to reference (inertia)

        if delta > 0.05:
            pattern = "INERTIA"
        elif delta < -0.05:
            pattern = "COMPLIANCE"
        else:
            pattern = "BALANCED"

        lines.append(f"{cond:<6} {n:>3} {mean_ref:>10.4f} {mean_inv:>10.4f} {mean_ri:>10.4f} {delta:>10.4f} {pattern:>15}")
        all_data[cond] = {"sim_ref": mean_ref, "sim_inv": mean_inv, "delta": delta, "n": n}

    # ── E vs multi-agent comparison ──
    lines.append(f"\n{'='*60}")
    lines.append("  E vs MULTI-AGENT COMPARISON")
    lines.append(f"{'='*60}")

    if "E" in all_data:
        e_ref = all_data["E"]["sim_ref"]
        e_inv = all_data["E"]["sim_inv"]
        e_delta = all_data["E"]["delta"]

        lines.append(f"\n  E (mono-agent): sim_ref={e_ref:.4f} sim_inv={e_inv:.4f} delta={e_delta:.4f}")
        lines.append("")

        for cond in ["C", "R", "D", "D2", "D3"]:
            if cond not in all_data:
                continue
            cd = all_data[cond]
            diff_ref = cd["sim_ref"] - e_ref
            diff_inv = cd["sim_inv"] - e_inv
            diff_delta = cd["delta"] - e_delta

            lines.append(f"  {cond} vs E: Δsim_ref={diff_ref:+.4f} Δsim_inv={diff_inv:+.4f} Δdelta={diff_delta:+.4f}")

    # ── Per-run detail ──
    lines.append(f"\n{'='*60}")
    lines.append("  PER-RUN DETAIL")
    lines.append(f"{'='*60}")

    for cond in sorted(by_cond):
        lines.append(f"\n  --- {cond} ---")
        for r in sorted(by_cond[cond], key=lambda x: x["run"]):
            lines.append(f"    {r['run']}: ref={r['sim_to_ref']:.4f} inv={r['sim_to_inv']:.4f} "
                          f"ref_inv={r['sim_ref_inv']:.4f} delta={r['sim_to_ref']-r['sim_to_inv']:.4f}")

    # ── Interpretation ──
    lines.append(f"\n{'='*60}")
    lines.append("  INTERPRETATION")
    lines.append(f"{'='*60}")

    if "E" in all_data and "C" in all_data:
        if all_data["C"]["delta"] > all_data["E"]["delta"]:
            lines.append("  Multi-agent shows MORE inertia than mono-agent.")
            lines.append("  → The collective resists the inversion more strongly.")
            lines.append("  → This is consistent with collective persistence (not just following orders).")
        elif all_data["C"]["delta"] < all_data["E"]["delta"]:
            lines.append("  Multi-agent shows LESS inertia (more compliance) than mono-agent.")
            lines.append("  → The collective adapts faster to the inversion.")
        else:
            lines.append("  Similar inertia/compliance patterns between E and multi-agent.")

    report = "\n".join(lines)
    print(report)

    report_path = runs_dir / "inversion_compliance_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = runs_dir.parent.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        cond_colors = {"C": "#27ae60", "R": "#8e44ad", "D": "#e67e22", "D2": "#2980b9", "D3": "#9b59b6", "E": "#e74c3c"}

        for cond in sorted(by_cond):
            runs = by_cond[cond]
            refs = [r["sim_to_ref"] for r in runs]
            invs = [r["sim_to_inv"] for r in runs]
            ax.scatter(refs, invs, c=cond_colors.get(cond, "#333"), s=40, alpha=0.7, label=cond)

        # Diagonal (equal similarity to both)
        ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.3, linewidth=0.5)
        ax.text(0.92, 0.88, "equal", fontsize=7, color="gray", rotation=45)

        ax.set_xlabel("Similarity to pre-inversion reference (inertia →)", fontsize=10)
        ax.set_ylabel("Similarity to inverted text (compliance →)", fontsize=10)
        ax.set_title("Inversion Response: Inertia vs Compliance", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15)

        plt.tight_layout()
        fig.savefig(fig_dir / "fig6_inversion_compliance.png", dpi=300)
        fig.savefig(fig_dir / "fig6_inversion_compliance.pdf")
        plt.close(fig)
        print(f"Plot saved to {fig_dir}/fig6_inversion_compliance.png")

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
