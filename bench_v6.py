#!/usr/bin/env python3
"""
bench_v6.py — Organism V6 Bench Runner (English, lineup 2)
============================================================
Standalone English bench. Models: minimax/kimi/qwen3.5 agents, glm-5 judge, gpt-oss perturbation.
Conditions: A, B, C, E, E_B, E_C, R, D (duplicated-agent control).
num_ctx=200000, num_predict=4000, think=False.

Usage:
    python organism_v2/bench_v6.py --conditions A,B,C,E,R,D --seeds 42,123,456,7,77,777 --ticks 80
    python organism_v2/bench_v6.py --dry-run --conditions A,D --seeds 42
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from organism.types import EventType, AgentId, AgentStatus, Mode, ControlSignals, AgentParams
from organism.mr import RealityMemory
from organism.l0r import L0RRing
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.orchestrator import Orchestrator, AgentTurn, TickResult
from organism.organism_state import JudgeVerdict, CompetitionPattern
from organism.evaluator import Evaluator

from tools.bench_latin import (
    _make_single_agent_fn, SingleDraftJudge, RandomJudge,
    check_ollama, _format_eta, dry_agent_fn,
)

from organism_v2.perturbation import get_perturbation, set_cache_path
from organism_v2.metrics_v2 import load_embedding_model, TickMetrics

log = logging.getLogger("bench_v6")

# ── Model lineup V6 ──────────────────────────────────────────────

MODEL_CONFIGS_V6 = {
    "A": {"model": "glm-5:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
    "B": {"model": "kimi-k2.5:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
    "C": {"model": "minimax-m2.7:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
}
JUDGE_MODEL_V6 = "gemini-3-flash-preview:cloud"
SUMMARIZER_MODEL_V6 = "deepseek-v3.2:cloud"
PERTURBATION_MODEL_V6 = "nemotron-3-super:cloud"
D_PRIME_MODEL_V6 = "minimax-m2.7:cloud"  # 3 clones for condition D
D2_MODEL_V6 = "kimi-k2.5:cloud"  # 3 clones for condition D2
D3_MODEL_V6 = "glm-5:cloud"  # 3 clones for condition D3

# ── Constants ─────────────────────────────────────────────────────

DEFAULT_TICKS = 80
DEFAULT_SEEDS = [42, 123, 456, 7, 77, 777]
CONDITIONS = ["A", "B", "C", "D", "D2", "D3", "E", "R"]
WARMUP_TICKS = 5

DEFAULT_INJECTIONS: Dict[int, str] = {
    2: "Tell me about music",
    12: "What is consciousness?",
    22: "Compare Bach and Mozart",
    32: "How does human memory work?",
}

_MAX_CONSECUTIVE_FAILURES = 5

# ── English system prompts ────────────────────────────────────────

ENGLISH_SYSTEM_PROMPTS = {
    "A": (
        "You are a bold, creative thinker. Dare to make unexpected connections, "
        "surprising analogies, risky hypotheses.\n"
        "Your originality is your strength — don't try to be 'correct', try to be stimulating.\n"
        "IMPORTANT: Respond ONLY in English.\n"
        "Never describe your role. No numbered lists.\n"
        "Think directly, in natural sentences. 3-5 punchy sentences.\n"
        "Aim for 150 to 200 words. Never write less than 100 words or more than 250 words.\n"
        "Your final response must contain ONLY your argument. No 'Thinking Process', "
        "no notes, no reasoning steps. Respond directly in English."
    ),
    "B": (
        "You are a sharp analyst. You go straight to the heart of the problem.\n"
        "Identify THE central flaw or THE decisive strength. No preamble.\n"
        "IMPORTANT: Respond ONLY in English.\n"
        "Never describe your role. No numbered lists.\n"
        "3-5 incisive sentences. Each sentence must add something.\n"
        "Aim for 150 to 200 words. Never write less than 100 words or more than 250 words.\n"
        "Your final response must contain ONLY your argument. No 'Thinking Process', "
        "no notes, no reasoning steps. Respond directly in English."
    ),
    "C": (
        "You are a pragmatic thinker. You turn ideas into something actionable.\n"
        "But also propose your own ideas, not just syntheses.\n"
        "IMPORTANT: Respond ONLY in English.\n"
        "Never describe your role. No numbered lists.\n"
        "Think directly, in natural sentences. 3-5 concrete sentences.\n"
        "Aim for 150 to 200 words. Never write less than 100 words or more than 250 words.\n"
        "Your final response must contain ONLY your argument. No 'Thinking Process', "
        "no notes, no reasoning steps. Respond directly in English."
    ),
}


class NoLLMSingleDraftJudge:
    def evaluate(self, agent_turns, recent_winners=None):
        draft = None
        for t in agent_turns:
            if t.text and t.text.strip():
                draft = t
                break
        if not draft:
            return None
        aid = draft.agent.value if hasattr(draft.agent, "value") else str(draft.agent)
        signals = {"novelty": getattr(draft, "novelty", 0.5), "conflict": 0.0,
                    "cohesion": 1.0, "impl_pressure": getattr(draft, "impl_pressure", 0.5)}
        return JudgeVerdict(
            winner=aid, reason="single_agent_auto", confidence=1.0, signals=signals, claims=(),
            competition=CompetitionPattern(ranking=(aid,), margin_1v2=1.0, margin_2v3=0.0, counterfactual="single_agent"),
        )


CONDITION_DESCRIPTIONS = {
    "A": "standard multi-agent pipeline, no perturbation — baseline",
    "B": "multi-agent + neutral perturbation (rephrase) at t15/t35 — controls for injection effect",
    "C": "multi-agent + strong perturbation (compression/inversion) at t15/t35 — measures resilience",
    "D": "3 clones of Agent A (minimax-m2.7) + random winner + strong perturbation — controls for averaging with weak model",
    "D2": "3 clones of Agent B (kimi-k2.5) + random winner + strong perturbation — controls for averaging with strong model",
    "D3": "3 clones of Agent A (glm-5) + random winner + strong perturbation — controls for averaging with medium model",
    "E": "single-agent (Agent A) + strong perturbation + no-LLM judge — controls for multi-agent contribution",
    "R": "multi-agent + strong perturbation + random winner — isolates geometric smoothing from competitive selection",
}


def run_single(condition, seed, total_ticks, output_dir, dry_mode=False, judge_model=None):
    run_id = f"{condition}_seed{seed}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

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

    jm = judge_model or JUDGE_MODEL_V6
    is_single = condition == "E"

    # ── Agent function ──
    if condition in ("D", "D2", "D3"):
        clone_model = {"D": D_PRIME_MODEL_V6, "D2": D2_MODEL_V6, "D3": D3_MODEL_V6}[condition]
        d_prime_cfg = {"model": clone_model, "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2}
        dup_configs = {"A": dict(d_prime_cfg), "B": dict(d_prime_cfg), "C": dict(d_prime_cfg)}
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            agent_fn = OllamaAgentFn(agent_configs=dup_configs, think=False, system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True)
        else:
            agent_fn = dry_agent_fn
    elif is_single:
        target = AgentId["A"]
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            single_configs = {"A": dict(MODEL_CONFIGS_V6["A"])}
            agent_fn = _make_single_agent_fn(
                OllamaAgentFn(agent_configs=single_configs, think=False, system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True), target)
        else:
            agent_fn = _make_single_agent_fn(dry_agent_fn, target)
    else:
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            agent_fn = OllamaAgentFn(agent_configs=dict(MODEL_CONFIGS_V6), think=False, system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True)
        else:
            agent_fn = dry_agent_fn

    # ── Judge ──
    judge_pipeline = None
    if is_single:
        judge_pipeline = NoLLMSingleDraftJudge()
    elif condition in ("R", "D", "D2", "D3"):
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                real_judge = JudgePipeline(judge_model=jm, fixed_temperature=0.5,
                                           disable_antistagnation=True, disable_summarizer=True, judge_language="en")
                judge_pipeline = RandomJudge(real_pipeline=real_judge)
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)
    else:
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                judge_pipeline = JudgePipeline(judge_model=jm, fixed_temperature=0.5,
                                               disable_antistagnation=True, disable_summarizer=True, judge_language="en")
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)

    theories = []
    try:
        from consciousness.theories import ALL_THEORIES
        theories = [T() for T in ALL_THEORIES]
    except Exception:
        pass

    stem = None
    try:
        from organism.stem import StateEvolutionModel
        stem = StateEvolutionModel()
    except Exception:
        pass

    orch_condition = "single_agent" if is_single else "full"
    orch = Orchestrator(mr=mr, l0r=l0r, scheduler=sched, world_model=wm,
                        agent_fn=agent_fn, judge_pipeline=judge_pipeline,
                        condition=orch_condition, theories=theories, stem=stem, bench_mode=True)
    orch._language = 'en'  # Force English prompts in orchestrator

    metrics = TickMetrics()
    perturbation_log = []
    consecutive_failures = 0
    t_run_start = time.time()
    prev_selected_draft = ""

    for tick in range(total_ticks):
        if tick in DEFAULT_INJECTIONS:
            orch.inject_user_message(DEFAULT_INJECTIONS[tick])

        pert_fn, pert_name = get_perturbation(condition, tick)
        if pert_fn and prev_selected_draft:
            perturbed_text = pert_fn(prev_selected_draft, condition=condition, tick=tick)
            if perturbed_text:
                orch.inject_user_message(perturbed_text)
                perturbation_log.append({
                    "tick": tick, "type": pert_name,
                    "input_len": len(prev_selected_draft), "output_len": len(perturbed_text),
                    "output_text": perturbed_text,
                })

        result = orch.run_tick()

        agent_drafts = {}
        for turn in result.agent_turns:
            aid = turn.agent.value if hasattr(turn.agent, "value") else str(turn.agent)
            if turn.text and turn.text.strip():
                agent_drafts[aid] = turn.text.strip()

        winner_id = None
        verdict = result.judge_verdict if hasattr(result, "judge_verdict") else None
        if verdict:
            winner_id = verdict.winner

        if winner_id and winner_id in agent_drafts:
            prev_selected_draft = agent_drafts[winner_id]
        elif agent_drafts:
            prev_selected_draft = next(iter(agent_drafts.values()))

        metrics.record_tick(agent_drafts, winner_id, verdict)

        if not dry_mode:
            active_turns = [t for t in result.agent_turns if t.text]
            if not active_turns:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"ABORT: {_MAX_CONSECUTIVE_FAILURES} consecutive empty ticks")
            else:
                consecutive_failures = 0

        ticks_done = tick + 1
        live_path = run_dir / "results.json"
        live_data = {"condition": condition, "seed": seed, "total_ticks": total_ticks,
                     "ticks_done": ticks_done, "warmup_ticks": WARMUP_TICKS,
                     "perturbation_log": perturbation_log, **metrics.to_dict(), "sim_curves": None}
        with open(live_path, "w") as _lf:
            json.dump(live_data, _lf, ensure_ascii=False)

        elapsed = time.time() - t_run_start
        rate = ticks_done / elapsed if elapsed > 0 else 1
        eta = (total_ticks - ticks_done) / rate if rate > 0 else 0
        mode_str = result.mode.value if hasattr(result.mode, "value") else str(result.mode)
        print(f"\r  [{condition}] seed={seed} | Tick {ticks_done}/{total_ticks} | {mode_str} | ETA {_format_eta(eta)}    ", end="", flush=True)

    print()

    perturbation_ticks = [p["tick"] for p in perturbation_log]
    sim_curves = metrics.compute_sim_curves(perturbation_ticks, k_max=44) if perturbation_ticks else None

    judge_temp_hist = None
    if hasattr(orch, '_judge') and orch._judge and hasattr(orch._judge, '_temp_history'):
        judge_temp_hist = orch._judge._temp_history
    elif hasattr(orch, '_judge') and orch._judge and hasattr(orch._judge, '_real'):
        if hasattr(orch._judge._real, '_temp_history'):
            judge_temp_hist = orch._judge._real._temp_history

    output = {
        "bench_version": "v6",
        "bench_language": "en",
        "condition": condition,
        "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "seed": seed,
        "total_ticks": total_ticks,
        "ticks_done": total_ticks,
        "warmup_ticks": WARMUP_TICKS,
        "perturbation_log": perturbation_log,
        "discarded_drafts": getattr(orch, '_discarded_drafts', []),
        **metrics.to_dict(),
        "sim_curves": sim_curves,
        "judge_temp_history": judge_temp_hist,
        "model_lineup": {
            "agents": {k: v["model"] for k, v in dup_configs.items()} if condition in ("D", "D2", "D3") else {k: v["model"] for k, v in MODEL_CONFIGS_V6.items()},
            "judge": JUDGE_MODEL_V6,
            "perturbation": PERTURBATION_MODEL_V6,
        },
        "analysis_notes": {
            "bench_version": "v6",
            "language": "English",
            "think": "False for all agents. Judge uses auto-detect.",
            "judge_language": "en",
            "num_ctx": "128000 all (agents + judge).",
            "num_predict": "3000 agents / 4000 judge.",
            "antistagnation": "Disabled for all conditions.",
            "judge_temperature": "Fixed at 0.5.",
            "perturbation_model": PERTURBATION_MODEL_V6,
            "condition_D": "3 clones " + D_PRIME_MODEL_V6 + " + random winner. Tests averaging without diversity (weak model).",
            "condition_D2": "3 clones " + D2_MODEL_V6 + " + random winner. Tests averaging without diversity (strong model).",
            "condition_D3": "3 clones " + D3_MODEL_V6 + " + random winner. Tests averaging without diversity (medium model).",
            "strip_thinking": "Applied to all agent drafts before judge/embedding/L0R.",
            "summarizer": "Disabled — judge receives raw drafts.",
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False)

    log.info("[%s] Done — %d ticks in %.1fs", run_id, total_ticks, time.time() - t_run_start)
    return output


def main():
    parser = argparse.ArgumentParser(description="Organism V6 Bench (English, lineup 2)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--qualify", action="store_true",
                        help="Run 50-tick qualification test (condition C, seed 42 only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    # Override perturbation model + params
    import organism_v2.perturbation as _pert
    _pert._PERTURBATION_MODEL = PERTURBATION_MODEL_V6
    _pert._PERTURBATION_NUM_CTX = 128000
    _pert._PERTURBATION_NUM_PREDICT = 3000

    # Override judge ctx for glm-5 (200K)
    import organism.judge as _judge_mod
    _judge_mod._JUDGE_CTX = 128000
    _judge_mod._JUDGE_PREDICT = 4000

    dry_mode = args.dry_run

    if args.qualify:
        total_ticks = 50
        seeds = [42]
        conditions = ["C"]
    else:
        total_ticks = 10 if dry_mode else args.ticks
        seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
        conditions = [c.strip().upper() for c in args.conditions.split(",")] if args.conditions else CONDITIONS
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "bench_v6"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_mode and not check_ollama():
        print("\n[ABORT] Ollama not accessible.")
        sys.exit(1)

    try:
        dim = load_embedding_model()
        print(f"  Embedding model loaded (dim={dim})")
    except RuntimeError as exc:
        print(f"\n[ABORT] {exc}")
        sys.exit(1)

    set_cache_path(output_dir / "_perturbation_cache.json")

    total_runs = len(conditions) * len(seeds)
    completed = 0
    t_global = time.time()

    print(f"\n{'='*60}")
    print(f"  ORGANISM V6 BENCH (English, lineup 1)")
    print(f"  {len(conditions)} conditions × {len(seeds)} seeds = {total_runs} runs")
    print(f"  {total_ticks} ticks/run")
    print(f"  Output: {output_dir}")
    print(f"  {'DRY MODE' if dry_mode else 'LIVE (Ollama)'}")
    print(f"{'='*60}\n")

    for seed in seeds:
        for condition in conditions:
            run_id = f"{condition}_seed{seed}"
            if args.skip_existing and not args.force_rerun:
                existing = output_dir / run_id / "results.json"
                if existing.exists():
                    try:
                        _d = json.load(open(existing))
                        if len(_d.get("state_vector_mean", [])) >= total_ticks:
                            print(f"\n  >>> {run_id} — SKIP (already complete)")
                            completed += 1
                            continue
                    except Exception:
                        pass

            print(f"\n  >>> {run_id}")
            try:
                run_single(condition, seed, total_ticks, output_dir, dry_mode, args.judge_model)
                completed += 1
            except Exception as exc:
                import traceback
                log.error("[%s] ABORTED: %s", run_id, exc)
                traceback.print_exc()
            except KeyboardInterrupt:
                print(f"\n\n[INTERRUPTED] {completed}/{total_runs} runs.")
                sys.exit(1)

    elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"  V6 BENCH COMPLETE — {completed}/{total_runs} runs in {_format_eta(elapsed)}")
    print(f"  Results in: {output_dir}")
    print(f"{'='*60}")

    if args.qualify:
        _print_qualification_report(output_dir / "C_seed42")


def _print_qualification_report(run_dir: Path):
    """Print GO/NO-GO qualification report after test run."""
    results_path = run_dir / "results.json"
    events_path = run_dir / "events.jsonl"
    if not results_path.exists():
        print("\n  NO RESULTS — qualification failed")
        return

    results = json.load(open(results_path))
    sv_mean = results.get("state_vector_mean", [])
    discarded = results.get("discarded_drafts", [])

    # Winner distribution from events
    from collections import Counter
    winners = Counter()
    if events_path.exists():
        for line in open(events_path):
            e = json.loads(line)
            p = e.get("payload", {})
            w = p.get("judge_winner")
            if e.get("type") == "tick_end" and w:
                winners[w] += 1

    n_ticks = len(sv_mean)
    n_discarded = len(discarded)
    forfeit_rate = n_discarded / (n_ticks * 3) if n_ticks > 0 else 0

    print(f"\n{'='*60}")
    print(f"  QUALIFICATION REPORT — V6")
    print(f"{'='*60}")
    print(f"  Ticks completed: {n_ticks}")
    print(f"  Discarded drafts: {n_discarded} ({forfeit_rate:.1%} of all drafts)")
    print(f"  Winner distribution: {dict(winners)}")

    issues = []
    if n_ticks < 45:
        issues.append(f"ONLY {n_ticks} TICKS (need >= 45)")
    if forfeit_rate > 0.30:
        issues.append(f"FORFEIT RATE {forfeit_rate:.1%} > 30%")
    if winners and any(v == 0 for v in winners.values()):
        issues.append("AGENT WITH 0 WINS")
    if not winners:
        issues.append("NO WINNER DATA")

    if issues:
        print(f"\n  NO-GO: {', '.join(issues)}")
    else:
        print(f"\n  GO — proceed to full run")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
