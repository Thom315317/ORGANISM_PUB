#!/usr/bin/env python3
"""
bench_v8.py — Condition I: Independent multi-sampling (no interaction)
=======================================================================
3 instances of glm-5:cloud, random winner selection, NO shared context.
Each tick: agents see only system prompt + injection/perturbation.
No L0R propagation, no world model accumulation, no "Recent context".

This is the decisive control: does multi-sampling without interaction
still improve persistence over mono-agent (condition E)?

Design:
  - 3× glm-5:cloud (same as D3 and E)
  - Random winner (same as D3)
  - Perturbation: compression t15 + inversion t35 (same schedule)
  - 12 seeds × 80 ticks
  - After each tick: L0R cleared, WM cleared
  - Perturbation at t15/t35: perturb prev_selected_draft (carried internally,
    NOT injected as context for other ticks)

Output: runs/bench_v8/I_seed{N}/

Usage:
    python bench_v8.py --seeds 42,123,456,7,77,777,1,99,2024,314,2025,8 --ticks 80
    python bench_v8.py --dry-run --seeds 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from organism.mr import RealityMemory
from organism.l0r import L0RRing
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.orchestrator import Orchestrator
from organism.types import AgentId
from organism_v2.metrics_v2 import TickMetrics, load_embedding_model
from organism_v2.perturbation import get_perturbation, set_cache_path

try:
    from tools.bench_latin import _make_single_agent_fn, RandomJudge, check_ollama
except ImportError:
    from organism_v2.bench_v2 import check_ollama
    RandomJudge = None

log = logging.getLogger("bench_v8")

# ── Model config (identical to D3: glm-5 ×3) ──
I_MODEL = "glm-5:cloud"
I_CFG = {"model": I_MODEL, "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2}

JUDGE_MODEL = "gemini-3-flash-preview:cloud"
PERTURBATION_MODEL = "nemotron-3-super:cloud"

# ── English system prompts (identical to bench_v6) ──
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
        "no notes, no reasoning steps. Respond directly in English.\n"
    ),
    "B": (
        "You are a sharp analyst. You go straight to the heart of the problem.\n"
        "Identify THE central flaw or THE decisive strength. No preamble.\n"
        "IMPORTANT: Respond ONLY in English.\n"
        "Never describe your role. No numbered lists.\n"
        "3-5 incisive sentences. Each sentence must add something.\n"
        "Aim for 150 to 200 words. Never write less than 100 words or more than 250 words.\n"
        "Your final response must contain ONLY your argument. No 'Thinking Process', "
        "no notes, no reasoning steps. Respond directly in English.\n"
    ),
    "C": (
        "You are a pragmatic thinker. You turn ideas into something actionable.\n"
        "But also propose your own ideas, not just syntheses.\n"
        "IMPORTANT: Respond ONLY in English.\n"
        "Never describe your role. No numbered lists.\n"
        "Think directly, in natural sentences. 3-5 concrete sentences.\n"
        "Aim for 150 to 200 words. Never write less than 100 words or more than 250 words.\n"
        "Your final response must contain ONLY your argument. No 'Thinking Process', "
        "no notes, no reasoning steps. Respond directly in English.\n"
    ),
}

DEFAULT_INJECTIONS = {
    2: "Tell me about music",
    12: "What is consciousness?",
    22: "Compare Bach and Mozart",
    32: "How does human memory work?",
}

DEFAULT_TICKS = 80
DEFAULT_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8]
WARMUP_TICKS = 5
_MAX_CONSECUTIVE_FAILURES = 5


class NoLLMSingleDraftJudge:
    """Placeholder — not used for condition I but needed for compatibility."""
    def evaluate(self, agent_turns, recent_winners=None):
        return None


def dry_agent_fn(agents, mode, prompts):
    from organism.types import AgentTurn
    turns = []
    for aid in agents:
        turns.append(AgentTurn(
            agent=aid, text=f"Dry tick {aid.value}: independent sampling test.",
            novelty=0.5, conflict=0.0, cohesion=0.5, impl_pressure=0.3,
            token_in=50, token_out=20, latency_ms=10, cost=0.0,
        ))
    return turns


def _format_eta(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_single(seed, total_ticks, output_dir, dry_mode=False):
    condition = "I"
    run_id = f"I_seed{seed}"
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

    # 3× glm-5 (identical to D3)
    dup_configs = {"A": dict(I_CFG), "B": dict(I_CFG), "C": dict(I_CFG)}

    if not dry_mode:
        from organism.agent_wrapper import OllamaAgentFn
        agent_fn = OllamaAgentFn(
            agent_configs=dup_configs, think=False,
            system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True)
    else:
        agent_fn = dry_agent_fn

    # Random judge (same as D3)
    judge_pipeline = None
    if not dry_mode:
        try:
            from organism.judge import JudgePipeline
            real_judge = JudgePipeline(
                judge_model=JUDGE_MODEL, fixed_temperature=0.5,
                disable_antistagnation=True, disable_summarizer=True,
                judge_language="en")
            judge_pipeline = RandomJudge(real_pipeline=real_judge)
        except Exception as exc:
            log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)

    stem = None
    try:
        from organism.stem import StateEvolutionModel
        stem = StateEvolutionModel()
    except Exception:
        pass

    orch = Orchestrator(
        mr=mr, l0r=l0r, scheduler=sched, world_model=wm,
        agent_fn=agent_fn, judge_pipeline=judge_pipeline,
        condition="full", theories=[], stem=stem, bench_mode=True)
    orch._language = 'en'

    metrics = TickMetrics()
    perturbation_log = []
    consecutive_failures = 0
    t_run_start = time.time()
    prev_selected_draft = ""

    for tick in range(total_ticks):
        # ── KEY DIFFERENCE: Clear shared context BEFORE each tick ──
        # L0R: clear the ring buffer so agents see no previous drafts
        l0r._ring.clear()
        # WM: clear accumulated facts
        if hasattr(wm, '_claims'):
            wm._claims.clear()
        if hasattr(wm, '_graph'):
            wm._graph.clear()

        # Inject thematic prompt (only thing agents see besides system prompt)
        if tick in DEFAULT_INJECTIONS:
            orch.inject_user_message(DEFAULT_INJECTIONS[tick])

        # Perturbation: operate on prev_selected_draft (carried internally)
        # but agents don't see previous context — only the perturbation text
        pert_fn, pert_name = get_perturbation(condition, tick)
        if pert_fn and prev_selected_draft:
            perturbed_text = pert_fn(prev_selected_draft, condition=condition, tick=tick)
            if perturbed_text:
                orch.inject_user_message(perturbed_text)
                perturbation_log.append({
                    "tick": tick, "type": pert_name,
                    "input_len": len(prev_selected_draft),
                    "output_len": len(perturbed_text),
                    "output_text": perturbed_text,
                })

        result = orch.run_tick()

        # ── After tick: clear user_messages so they don't carry over ──
        orch._user_messages.clear()

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
        print(f"\r  [I] seed={seed} | Tick {ticks_done}/{total_ticks} | {mode_str} | ETA {_format_eta(eta)}    ", end="", flush=True)

    print()

    perturbation_ticks = [p["tick"] for p in perturbation_log]
    sim_curves = metrics.compute_sim_curves(perturbation_ticks, k_max=44) if perturbation_ticks else None

    output = {
        "bench_version": "v8",
        "bench_language": "en",
        "condition": "I",
        "condition_description": (
            "Independent multi-sampling: 3× glm-5:cloud, random winner, NO shared context. "
            "L0R and WM cleared each tick. Agents see only system prompt + injection/perturbation. "
            "Controls for multi-sampling advantage vs interaction advantage."
        ),
        "seed": seed,
        "total_ticks": total_ticks,
        "ticks_done": total_ticks,
        "warmup_ticks": WARMUP_TICKS,
        "perturbation_log": perturbation_log,
        "discarded_drafts": getattr(orch, '_discarded_drafts', []),
        **metrics.to_dict(),
        "sim_curves": sim_curves,
        "model_lineup": {
            "agents": {"A": I_MODEL, "B": I_MODEL, "C": I_MODEL},
            "judge": JUDGE_MODEL,
            "perturbation": PERTURBATION_MODEL,
        },
        "analysis_notes": {
            "bench_version": "v8",
            "language": "English",
            "condition_I_design": (
                "Each tick: L0R cleared, WM cleared, user_messages cleared. "
                "Agents receive system prompt + mode instruction + injection (if tick 2/12/22/32) "
                "or perturbation text (if tick 15/35). NO 'Recent context', NO 'Established facts', "
                "NO propagation of winning draft. Perturbation operates on prev_selected_draft "
                "(tracked internally) but this draft is NOT visible to agents at other ticks."
            ),
            "comparison_targets": {
                "I_vs_E": "Multi-sampling without interaction vs mono-agent",
                "D3_vs_I": "Interaction (D3) vs no interaction (I), same model (glm-5)",
            },
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False)

    log.info("[%s] Done — %d ticks in %.1fs", run_id, total_ticks, time.time() - t_run_start)
    return output


def main():
    parser = argparse.ArgumentParser(description="Bench V8: Condition I (independent multi-sampling)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    # Override perturbation model
    import organism_v2.perturbation as _pert
    _pert._PERTURBATION_MODEL = PERTURBATION_MODEL
    _pert._PERTURBATION_NUM_CTX = 128000
    _pert._PERTURBATION_NUM_PREDICT = 3000

    # Override judge ctx
    import organism.judge as _judge_mod
    _judge_mod._JUDGE_CTX = 128000
    _judge_mod._JUDGE_PREDICT = 4000

    # Add "I" to perturbation schedule if not present
    if "I" not in _pert.PERTURBATION_SCHEDULE:
        from organism_v2.perturbation import compression, inversion
        _pert.PERTURBATION_SCHEDULE["I"] = {15: compression, 35: inversion}

    dry_mode = args.dry_run
    total_ticks = 10 if dry_mode else args.ticks
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "bench_v8"
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

    total_runs = len(seeds)
    completed = 0
    t_global = time.time()

    print(f"\n{'='*60}")
    print(f"  BENCH V8 — CONDITION I (Independent Multi-Sampling)")
    print(f"  Model: {I_MODEL} ×3")
    print(f"  {total_runs} seeds × {total_ticks} ticks")
    print(f"  Output: {output_dir}")
    print(f"  {'DRY MODE' if dry_mode else 'LIVE (Ollama)'}")
    print(f"{'='*60}\n")

    for seed in seeds:
        run_id = f"I_seed{seed}"
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
            run_single(seed, total_ticks, output_dir, dry_mode)
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
    print(f"  V8 BENCH COMPLETE — {completed}/{total_runs} runs in {_format_eta(elapsed)}")
    print(f"  Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
