#!/usr/bin/env python3
"""
bench_v2.py — Organism V2 Bench Runner
========================================
Runs 4 conditions × 3 seeds = 12 runs.

Conditions:
  A — standard pipeline, no perturbation
  B — perturbation at ticks 15 and 35 (neutral)
  C — perturbation at ticks 15 and 35 (strong)
  E — single-agent + strong perturbation

Reuses Organism 1 core (orchestrator, agents, judge) without modification.
Computes embedding-based raw time series per tick.

Usage:
    python organism_v2/bench_v2.py                     # Full run (12 runs)
    python organism_v2/bench_v2.py --dry-run            # 10 ticks, dry mode
    python organism_v2/bench_v2.py --conditions A,C     # Subset of conditions
    python organism_v2/bench_v2.py --seeds 42           # Single seed

TICK INDEXING CONVENTION
    bench loop: tick in range(total_ticks) → 0-indexed (0 to 49)
    orchestrator: self._tick_id → 1-indexed (1 to 50)
    perturbation_log["tick"] → 0-indexed (bench loop)
    events.jsonl tick_id → 1-indexed (orchestrator)
    sim_curves keys ("tick_15", "tick_35") → 0-indexed (bench loop)
    All analysis code must use bench loop index as reference.

ANALYSIS RULES
    - quality_per_tick and judge_score_dispersion in condition R
      reflect real judge evaluation (not random selection).
      Do NOT use these as discriminators in C vs R comparisons.
    - perturbation_log output_text enables post-hoc verification
      of semantic similarity between C and R perturbations.
    - sole draft ticks (1 judgeable turn) have confidence=0.5
      hardcoded in orchestrator — flag these ticks in analysis.

KNOWN LIMITATIONS
    - condition E: SingleDraftJudge returns confidence=1.0 always
      (architectural artifact, not a quality signal).
    - condition E: sv_mean == sv_selected by construction
      (single agent, no averaging).
    - DEFAULT_INJECTIONS (ticks 2,12,22,32) applied to all
      conditions for comparability.
    - condition F tick 10: R_pre computed from ticks 7,8,9
      (immediately post-warmup — note in Methods).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from organism.types import (
    EventType, AgentId, AgentStatus, Mode, ControlSignals, AgentParams,
)
from organism.mr import RealityMemory
from organism.l0r import L0RRing
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.orchestrator import Orchestrator, AgentTurn, TickResult
from organism.organism_state import JudgeVerdict, CompetitionPattern
from organism.evaluator import Evaluator
from organism.config import ORGANISM_CONFIG

# Reuse from bench_latin.py (no modification)
from tools.bench_latin import (
    _make_single_agent_fn,
    SingleDraftJudge,
    RandomJudge,
    MODEL_CONFIGS,
    MODEL_NAMES,
    check_ollama,
    _format_eta,
    dry_agent_fn,
)

# V2 modules
from organism_v2.perturbation import get_perturbation, set_cache_path
from organism_v2.metrics_v2 import load_embedding_model, TickMetrics

log = logging.getLogger("bench_v2")

# ── Constants ──────────────────────────────────────────────────────

DEFAULT_TICKS = 80
DEFAULT_SEEDS = [42, 123, 456]
CONDITIONS = ["A", "B", "C", "E", "E_B", "E_C", "R"]
WARMUP_TICKS = 5  # Exclude ticks 1-5 from downstream (flagged, not skipped)

# User injections — same as Organism 1 for comparability
DEFAULT_INJECTIONS: Dict[int, str] = {
    2: "Parlez-moi de musique",
    12: "Qu'est-ce que la conscience ?",
    22: "Comparez Bach et Mozart",
    32: "Comment fonctionne la memoire humaine ?",
}

_MAX_CONSECUTIVE_FAILURES = 5

class NoLLMSingleDraftJudge:
    """Single-agent judge: no LLM call, winner assigned automatically with heuristic signals."""
    def evaluate(self, agent_turns, recent_winners=None):
        draft = None
        for t in agent_turns:
            if t.text and t.text.strip():
                draft = t
                break
        if not draft:
            return None
        aid = draft.agent.value if hasattr(draft.agent, "value") else str(draft.agent)
        signals = {
            "novelty": getattr(draft, "novelty", 0.5),
            "conflict": 0.0,
            "cohesion": 1.0,
            "impl_pressure": getattr(draft, "impl_pressure", 0.5),
        }
        return JudgeVerdict(
            winner=aid,
            reason="single_agent_auto",
            confidence=1.0,
            signals=signals,
            claims=(),
            competition=CompetitionPattern(
                ranking=(aid,), margin_1v2=1.0, margin_2v3=0.0,
                counterfactual="single_agent",
            ),
        )

CONDITION_DESCRIPTIONS = {
    "A": "standard multi-agent pipeline, no perturbation — baseline",
    "B": "multi-agent + neutral perturbation (rephrase) at t15/t35 — controls for injection effect",
    "C": "multi-agent + strong perturbation (compression/inversion) at t15/t35 — measures resilience",
    "E": "single-agent (glm-5:cloud, slot A) + strong perturbation + no-LLM judge — controls for multi-agent contribution.",
    "E_B": "single-agent (kimi-k2.5:cloud, slot B) + strong perturbation + no-LLM judge — controls for model size bias.",
    "E_C": "single-agent (qwen3.5:397b-cloud, slot C) + strong perturbation + no-LLM judge — completes single-agent control triad.",
    "R": "multi-agent + strong perturbation + random winner — isolates geometric smoothing (barycentre effect) from competitive selection. If C > R on sim_curves, selection contributes beyond averaging.",
    "F": "multi-agent + competitive judge + 4× strong perturbation (t10/t20/t30/t40) — tests repeated recovery capacity and breaking point under sustained perturbation.",
}


# ── Run a single condition ────────────────────────────────────────


def run_single(
    condition: str,
    seed: int,
    total_ticks: int,
    output_dir: Path,
    dry_mode: bool = False,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single V2 condition. Returns the results dict."""

    run_id = f"{condition}_seed{seed}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # ── Core components (same as Organism 1) ──
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    results_path = run_dir / "results.json"
    if results_path.exists():
        results_path.unlink()
    mr = RealityMemory(path=str(events_path))
    l0r = L0RRing(mr=mr)
    sched = Scheduler()
    wm = WorldModel(mr=mr)

    # ── Agent function ──
    is_single = condition in ("E", "E_B", "E_C")
    if is_single and condition == "E_C":
        single_model = "C"
        target = AgentId[single_model]
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            single_configs = {single_model: dict(MODEL_CONFIGS[single_model])}
            agent_fn = _make_single_agent_fn(
                OllamaAgentFn(agent_configs=single_configs), target
            )
        else:
            agent_fn = _make_single_agent_fn(dry_agent_fn, target)
    elif is_single and condition == "E_B":
        single_model = "B"
        target = AgentId[single_model]
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            single_configs = {single_model: dict(MODEL_CONFIGS[single_model])}
            agent_fn = _make_single_agent_fn(
                OllamaAgentFn(agent_configs=single_configs), target
            )
        else:
            agent_fn = _make_single_agent_fn(dry_agent_fn, target)
    elif is_single:
        single_model = "A"  # Use Agent A for single condition
        target = AgentId[single_model]
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            single_configs = {single_model: dict(MODEL_CONFIGS[single_model])}
            agent_fn = _make_single_agent_fn(
                OllamaAgentFn(agent_configs=single_configs), target
            )
        else:
            agent_fn = _make_single_agent_fn(dry_agent_fn, target)
    else:
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            agent_fn = OllamaAgentFn(agent_configs=dict(MODEL_CONFIGS))
        else:
            agent_fn = dry_agent_fn

    # ── Judge ──
    judge_pipeline = None
    if is_single:
        judge_pipeline = NoLLMSingleDraftJudge()
    elif condition == "R":
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                real_judge = JudgePipeline(
                    judge_model=judge_model,
                    fixed_temperature=0.5,
                    disable_antistagnation=True,
                    disable_summarizer=True,
                )
                judge_pipeline = RandomJudge(real_pipeline=real_judge)
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable for RandomJudge: %s", run_id, exc)
    else:
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                judge_pipeline = JudgePipeline(
                    judge_model=judge_model,
                    fixed_temperature=0.5,
                    disable_antistagnation=True,
                    disable_summarizer=True,
                )
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)

    # ── Theories + STEM ──
    theories = []
    try:
        pass  # consciousness theories removed from publication build
    except Exception:
        pass

    stem = None
    try:
        from organism.stem import StateEvolutionModel
        stem = StateEvolutionModel()
    except Exception:
        pass

    # ── Orchestrator ──
    orch_condition = "single_agent" if is_single else "full"
    orch = Orchestrator(
        mr=mr, l0r=l0r, scheduler=sched, world_model=wm,
        agent_fn=agent_fn, judge_pipeline=judge_pipeline,
        condition=orch_condition, theories=theories, stem=stem,
        bench_mode=True,
    )

    # ── Metrics collector ──
    metrics = TickMetrics()

    # ── Per-tick metadata ──
    tick_modes: List[str] = []
    tick_theories: List[Dict[str, float]] = []

    # ── Run loop ──
    consecutive_failures = 0
    t_run_start = time.time()
    prev_selected_draft = ""
    perturbation_log = []

    for tick in range(total_ticks):
        # ── User injections (same schedule as V1) ──
        if tick in DEFAULT_INJECTIONS:
            orch.inject_user_message(DEFAULT_INJECTIONS[tick])

        # ── Perturbation ──
        pert_fn, pert_name = get_perturbation(condition, tick)
        if pert_fn and prev_selected_draft:
            perturbed_text = pert_fn(prev_selected_draft, condition=condition, tick=tick)
            if perturbed_text:
                orch.inject_user_message(perturbed_text)
                perturbation_log.append({
                    "tick": tick,
                    "type": pert_name,
                    "input_len": len(prev_selected_draft),
                    "output_len": len(perturbed_text),
                    "output_text": perturbed_text,
                })
                log.info("[%s] Perturbation %s at tick %d: %d→%d chars",
                         run_id, pert_name, tick, len(prev_selected_draft),
                         len(perturbed_text))

        # ── Run tick ──
        result = orch.run_tick()

        # ── Extract drafts ──
        agent_drafts = {}
        for turn in result.agent_turns:
            aid = turn.agent.value if hasattr(turn.agent, "value") else str(turn.agent)
            if turn.text and turn.text.strip():
                agent_drafts[aid] = turn.text.strip()

        # ── Winner ──
        winner_id = None
        verdict = result.judge_verdict if hasattr(result, "judge_verdict") else None
        if verdict:
            winner_id = verdict.winner

        # ── Track selected draft for next perturbation ──
        if winner_id and winner_id in agent_drafts:
            prev_selected_draft = agent_drafts[winner_id]
        elif agent_drafts:
            prev_selected_draft = next(iter(agent_drafts.values()))

        # ── Record metrics ──
        metrics.record_tick(agent_drafts, winner_id, verdict)

        # ── Fail-safe ──
        if not dry_mode:
            active_turns = [t for t in result.agent_turns if t.text]
            if not active_turns:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"ABORT: {_MAX_CONSECUTIVE_FAILURES} consecutive empty ticks"
                    )
            else:
                consecutive_failures = 0

        # ── Live flush (every tick) ──
        ticks_done = tick + 1
        if True:
            live_path = run_dir / "results.json"
            live_data = {
                "condition": condition,
                "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
                "seed": seed,
                "total_ticks": total_ticks,
                "ticks_done": ticks_done,
                "warmup_ticks": WARMUP_TICKS,
                "perturbation_log": perturbation_log,
                **metrics.to_dict(),
                "sim_curves": None,  # computed at end
            }
            with open(live_path, "w") as _lf:
                json.dump(live_data, _lf, ensure_ascii=False)

        # ── Progress ──
        elapsed = time.time() - t_run_start
        rate = ticks_done / elapsed if elapsed > 0 else 1
        eta = (total_ticks - ticks_done) / rate if rate > 0 else 0
        mode_str = result.mode.value if hasattr(result.mode, "value") else str(result.mode)
        print(
            f"\r  [{condition}] seed={seed} | Tick {ticks_done}/{total_ticks} "
            f"| {mode_str} | ETA {_format_eta(eta)}    ",
            end="", flush=True,
        )

    print()  # newline

    # ── Compute sim_curves ──
    perturbation_ticks = [p["tick"] for p in perturbation_log]
    sim_curves = metrics.compute_sim_curves(perturbation_ticks, k_max=44) if perturbation_ticks else None

    # ── Build output ──
    metrics_dict = metrics.to_dict()
    judge_temp_hist = None
    if hasattr(orch, '_judge') and orch._judge and hasattr(orch._judge, '_temp_history'):
        judge_temp_hist = orch._judge._temp_history
    elif hasattr(orch, '_judge') and orch._judge and hasattr(orch._judge, '_real'):
        # RandomJudge wraps real pipeline
        if hasattr(orch._judge._real, '_temp_history'):
            judge_temp_hist = orch._judge._real._temp_history
    output = {
        "bench_version": "v4",
        "condition": condition,
        "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "seed": seed,
        "total_ticks": total_ticks,
        "ticks_done": total_ticks,
        "warmup_ticks": WARMUP_TICKS,
        "perturbation_log": perturbation_log,
        "discarded_drafts": getattr(orch, '_discarded_drafts', []),
        **metrics_dict,
        "sim_curves": sim_curves,
        "judge_temp_history": judge_temp_hist,
        "analysis_notes": {
            "draft_velocity_C_vs_R": "NOT directly comparable — reflects random winner switches in R vs coherent winner sequence in C.",
            "PSV_C_vs_R": "Same limitation as draft_velocity.",
            "winner_L0R_feedback": "By design: winning draft propagates at salience=1.0. In C = best draft, in R = random draft. This is the mechanism under study, not a confound.",
            "sv_selected_C_vs_R": "Winner-dependent. Valid for comparison but account for structural difference in winner sequences.",
            "antistagnation": "Disabled for all conditions in bench_v4.",
            "judge_temperature": "Fixed at 0.5 for all conditions in bench_v4.",
            "num_ctx": "65536 agents / 131072 judge in bench_v4.",
            "strip_thinking": "Applied to all agent drafts before judge/embedding/L0R.",
            "num_predict": "1500 for agents, 4000 for judge/summarizer in bench_v4.",
            "discarded_drafts": "Drafts <20 chars post-strip treated as forfeit (thinking_only). Agent skips tick, not sent to judge.",
            "summarizer_model": "nemotron-3-super:cloud (NVIDIA, family distinct from all agents).",
            "perturbation_model": "minimax-m2.7:cloud (MiniMax, family distinct from all agents and judge).",
        },
    }

    # ── Save ──
    output_path = run_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    elapsed_total = time.time() - t_run_start
    log.info("[%s] Done — %d ticks in %.1fs, saved to %s",
             run_id, total_ticks, elapsed_total, output_path)

    return output


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Organism V2 Bench Runner")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help=f"Ticks per run (default {DEFAULT_TICKS})")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,123,456)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions (default: A,B,C,E)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: runs/bench_v2/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="10 ticks, dry mode (no LLM)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip runs whose results.json already has total_ticks reached (default: ON)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force rerun even if results.json already complete (overrides --skip-existing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dry_mode = args.dry_run
    total_ticks = 10 if dry_mode else args.ticks

    # Seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS

    # Conditions
    if args.conditions:
        conditions = [c.strip().upper() for c in args.conditions.split(",")]
    else:
        conditions = CONDITIONS

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "runs" / "bench_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight
    if not dry_mode:
        if not check_ollama():
            print("\n[ABORT] Ollama not accessible.")
            sys.exit(1)

    # Load embedding model
    try:
        dim = load_embedding_model()
        print(f"  Embedding model loaded (dim={dim})")
    except RuntimeError as exc:
        print(f"\n[ABORT] {exc}")
        sys.exit(1)

    # Init perturbation cache
    set_cache_path(output_dir / "_perturbation_cache.json")

    total_runs = len(conditions) * len(seeds)
    completed = 0
    t_global = time.time()

    print(f"\n{'=' * 60}")
    print(f"  ORGANISM V2 BENCH")
    print(f"  {len(conditions)} conditions × {len(seeds)} seeds = {total_runs} runs")
    print(f"  {total_ticks} ticks/run")
    print(f"  Output: {output_dir}")
    print(f"  {'DRY MODE' if dry_mode else 'LIVE (Ollama)'}")
    print(f"{'=' * 60}\n")

    for condition in conditions:
        for seed in seeds:
            run_id = f"{condition}_seed{seed}"

            # --skip-existing: skip runs already completed
            if args.skip_existing and not args.force_rerun:
                existing = output_dir / run_id / "results.json"
                if existing.exists():
                    try:
                        import json as _json
                        _d = _json.load(open(existing))
                        _sv = _d.get("state_vector_mean", [])
                        if len(_sv) >= total_ticks:
                            print(f"\n  >>> {run_id} — SKIP (already {len(_sv)}/{total_ticks} ticks)")
                            completed += 1
                            continue
                    except Exception:
                        pass

            print(f"\n  >>> {run_id}")

            try:
                run_single(
                    condition=condition,
                    seed=seed,
                    total_ticks=total_ticks,
                    output_dir=output_dir,
                    dry_mode=dry_mode,
                    judge_model=args.judge_model,
                )
                completed += 1
            except Exception as exc:
                import traceback
                log.error("[%s] ABORTED (%s): %s", run_id, type(exc).__name__, exc)
                traceback.print_exc()
            except KeyboardInterrupt:
                print(f"\n\n[INTERRUPTED] Completed {completed}/{total_runs} runs.")
                sys.exit(1)

    elapsed = time.time() - t_global
    print(f"\n{'=' * 60}")
    print(f"  V2 BENCH COMPLETE")
    print(f"  {completed}/{total_runs} runs")
    print(f"  Total time: {_format_eta(elapsed)}")
    print(f"  Results in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
