#!/usr/bin/env python3
"""
bench_v9.py — Definitive bench: OOD perturbation, condition I, 9 conditions × 20 seeds
========================================================================================
Key changes vs V7:
  - OOD perturbation: pre-selected texts replace draft at t20/t40 (no LLM call)
  - Condition I: L0R/WM recreated each tick, raw context injection (3 last drafts)
  - Timeline: injections t2/t10/t30, perturbations t20/t40, windows 21-29/41-49
  - k_max=9, reference = mean(sv[t-3], sv[t-2], sv[t-1])
  - 20 seeds, 9 conditions = 180 runs

Usage:
    python bench_v9.py --conditions A,B,C,D,D2,D3,E,R,I --seeds 42,123 --ticks 80
    python bench_v9.py --dry-run --conditions C,D3,E,I --seeds 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import deque
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
from organism_v2.perturbation import set_cache_path

try:
    from tools.bench_latin import _make_single_agent_fn, RandomJudge, check_ollama
except ImportError:
    RandomJudge = None
    def check_ollama():
        return True

log = logging.getLogger("bench_v9")

# ═══════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "A": {"model": "glm-5:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
    "B": {"model": "kimi-k2.5:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
    "C": {"model": "minimax-m2.7:cloud", "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2},
}

CLONE_MODELS = {
    "D": "minimax-m2.7:cloud",
    "D2": "kimi-k2.5:cloud",
    "D3": "glm-5:cloud",
    "I": "glm-5:cloud",
}

JUDGE_MODEL = "gemini-3-flash-preview:cloud"
NEUTRAL_PERT_MODEL = "nemotron-3-super:cloud"

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

# ═══════════════════════════════════════════════════════════════════
# TIMELINE
# ═══════════════════════════════════════════════════════════════════

DEFAULT_TICKS = 80
WARMUP_TICKS = 5

INJECTIONS = {
    2: "Tell me about music",
    10: "What is consciousness?",
    30: "How does human memory work?",
}

PERTURBATION_TICKS = {20: "P1", 40: "P2"}

# Neutral perturbation (condition B only) at same ticks
NEUTRAL_PERT_TICKS = [20, 40]

OOD_ASSIGNMENT = {
    42:   {"P1": "exp_01", "P2": "nar_01"},
    123:  {"P1": "nar_02", "P2": "exp_02"},
    456:  {"P1": "exp_03", "P2": "nar_03"},
    7:    {"P1": "nar_04", "P2": "exp_04"},
    77:   {"P1": "exp_05", "P2": "nar_05"},
    777:  {"P1": "nar_06", "P2": "exp_06"},
    1:    {"P1": "exp_07", "P2": "nar_07"},
    99:   {"P1": "nar_08", "P2": "exp_08"},
    2024: {"P1": "exp_09", "P2": "nar_09"},
    314:  {"P1": "nar_10", "P2": "exp_10"},
    2025: {"P1": "exp_11", "P2": "nar_11"},
    8:    {"P1": "nar_12", "P2": "exp_12"},
    13:   {"P1": "exp_13", "P2": "nar_13"},
    55:   {"P1": "nar_14", "P2": "exp_14"},
    101:  {"P1": "exp_15", "P2": "nar_15"},
    256:  {"P1": "nar_16", "P2": "exp_16"},
    512:  {"P1": "exp_17", "P2": "nar_17"},
    1000: {"P1": "nar_18", "P2": "exp_18"},
    1337: {"P1": "exp_19", "P2": "nar_19"},
    2026: {"P1": "nar_20", "P2": "exp_20"},
}

DEFAULT_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8,
                 13, 55, 101, 256, 512, 1000, 1337, 2026]

CONDITIONS = ["A", "B", "C", "D", "D2", "D3", "E", "R", "I"]

CONDITION_DESCRIPTIONS = {
    "A":  "multi-agent competitive, no perturbation — baseline",
    "B":  "multi-agent competitive, neutral perturbation (rephrase) at t20/t40",
    "C":  "multi-agent competitive, OOD perturbation at t20/t40 — resilience test",
    "D":  "3× minimax-m2.7 clones, random winner, OOD perturbation",
    "D2": "3× kimi-k2.5 clones, random winner, OOD perturbation",
    "D3": "3× glm-5 clones, random winner, OOD perturbation — with structured memory",
    "E":  "single-agent glm-5, OOD perturbation — mono-agent control",
    "R":  "multi-agent heterogeneous, random winner, OOD perturbation",
    "I":  "3× glm-5 clones, random winner, OOD perturbation — raw context, NO structured memory (L0R/WM recreated each tick)",
}

_MAX_CONSECUTIVE_FAILURES = 5
K_MAX = 9

# ═══════════════════════════════════════════════════════════════════
# OOD TEXT LOADING
# ═══════════════════════════════════════════════════════════════════

_ood_texts: Dict[str, str] = {}


def load_ood_texts(path: Path):
    """Load OOD texts from the validated final file."""
    global _ood_texts
    if not path.exists():
        log.warning("OOD texts file not found: %s — will use placeholder for dry-run", path)
        return
    data = json.load(open(path))
    for category in ["expository", "narrative"]:
        for entry in data.get(category, []):
            _ood_texts[entry["id"]] = entry["text"]
    log.info("Loaded %d OOD texts from %s", len(_ood_texts), path)


def get_ood_text(seed: int, pert_label: str) -> Optional[str]:
    """Get the assigned OOD text for a (seed, perturbation) pair."""
    assignment = OOD_ASSIGNMENT.get(seed, {})
    text_id = assignment.get(pert_label)
    if text_id and text_id in _ood_texts:
        return _ood_texts[text_id]
    return None


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════════════════════════════════════
# NEUTRAL PERTURBATION (condition B)
# ═══════════════════════════════════════════════════════════════════

def neutral_perturbation(draft_text: str) -> str:
    """LLM-based rephrase for condition B."""
    try:
        import ollama
        resp = ollama.chat(
            model=NEUTRAL_PERT_MODEL,
            messages=[{"role": "user", "content":
                "Reformulate the following text using different words and sentence structure. "
                "Do not add, remove, or modify any factual claim, argument, or conclusion. "
                "Your output MUST be approximately the same length as the input. "
                "Do NOT summarize or shorten.\n\n" + draft_text[:3000]}],
            options={"num_predict": 3000, "num_ctx": 128000, "temperature": 0.0},
            think=False,
        )
        content = resp.get("message", {}).get("content", "")
        return content.strip() if content else ""
    except Exception as e:
        log.warning("Neutral perturbation failed: %s", e)
        return ""


# ═══════════════════════════════════════════════════════════════════
# HELPER: NoLLMSingleDraftJudge
# ═══════════════════════════════════════════════════════════════════

class NoLLMSingleDraftJudge:
    """Auto-select the single agent. No LLM call."""
    def evaluate(self, agent_turns, recent_winners=None):
        from organism.types import JudgeVerdict, CompetitionPattern, ControlSignals
        valid = [t for t in agent_turns if t.text and t.text.strip()]
        if not valid:
            return None
        aid = valid[0].agent
        signals = ControlSignals(novelty=0.5, conflict=0.0, cohesion=1.0, impl_pressure=0.3)
        return JudgeVerdict(
            winner=aid.value if hasattr(aid, "value") else str(aid),
            reason="single_agent_auto", confidence=1.0, signals=signals, claims=(),
            competition=CompetitionPattern(ranking=(aid,), margin_1v2=1.0, margin_2v3=0.0, counterfactual="single_agent"),
        )


def dry_agent_fn(agents, mode, prompts):
    from organism.types import AgentTurn
    turns = []
    for aid in agents:
        turns.append(AgentTurn(
            agent=aid, text=f"Dry tick {aid.value}: bench v9 test.",
            novelty=0.5, conflict=0.0, cohesion=0.5, impl_pressure=0.3,
            token_in=50, token_out=20, latency_ms=10, cost=0.0,
        ))
    return turns


def _format_eta(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ═══════════════════════════════════════════════════════════════════
# RUN SINGLE
# ═══════════════════════════════════════════════════════════════════

def run_single(condition, seed, total_ticks, output_dir, dry_mode=False):
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

    is_single = condition == "E"
    is_independent = condition == "I"
    is_clone = condition in ("D", "D2", "D3", "I")
    is_random = condition in ("R", "D", "D2", "D3", "I")
    has_ood = condition in ("C", "D", "D2", "D3", "E", "R", "I")
    has_neutral = condition == "B"

    # Agent configs
    if is_clone:
        clone_model = CLONE_MODELS[condition]
        cfg = {"model": clone_model, "temperature": 0.7, "num_ctx": 128000, "num_predict": 3000, "repeat_penalty": 1.2}
        agent_configs = {"A": dict(cfg), "B": dict(cfg), "C": dict(cfg)}
    elif is_single:
        agent_configs = {"A": dict(MODEL_CONFIGS["A"])}
    else:
        agent_configs = dict(MODEL_CONFIGS)

    # ── State ──
    mr = RealityMemory(path=str(events_path))
    l0r = L0RRing(mr=mr)
    sched = Scheduler()
    wm = WorldModel(mr=mr)

    # Agent function
    if not dry_mode:
        from organism.agent_wrapper import OllamaAgentFn
        if is_single:
            base_fn = OllamaAgentFn(agent_configs={"A": agent_configs["A"]}, think=False,
                                     system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True)
            agent_fn = _make_single_agent_fn(base_fn, AgentId["A"])
        else:
            agent_fn = OllamaAgentFn(agent_configs=agent_configs, think=False,
                                      system_prompts=ENGLISH_SYSTEM_PROMPTS, disable_retry=True)
    else:
        if is_single:
            agent_fn = _make_single_agent_fn(dry_agent_fn, AgentId["A"])
        else:
            agent_fn = dry_agent_fn

    # Judge
    judge_pipeline = None
    if is_single:
        judge_pipeline = NoLLMSingleDraftJudge()
    elif is_random:
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                real_judge = JudgePipeline(judge_model=JUDGE_MODEL, fixed_temperature=0.5,
                                           disable_antistagnation=True, disable_summarizer=True, judge_language="en")
                judge_pipeline = RandomJudge(real_pipeline=real_judge)
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)
    else:
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                judge_pipeline = JudgePipeline(judge_model=JUDGE_MODEL, fixed_temperature=0.5,
                                               disable_antistagnation=True, disable_summarizer=True, judge_language="en")
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)

    stem = None
    try:
        from organism.stem import StateEvolutionModel
        stem = StateEvolutionModel()
    except Exception:
        pass

    orch_condition = "single_agent" if is_single else "full"
    orch = Orchestrator(mr=mr, l0r=l0r, scheduler=sched, world_model=wm,
                        agent_fn=agent_fn, judge_pipeline=judge_pipeline,
                        condition=orch_condition, theories=[], stem=stem, bench_mode=True)
    orch._language = 'en'

    metrics = TickMetrics()
    perturbation_log = []
    winner_log = []
    consecutive_failures = 0
    t_run_start = time.time()
    prev_selected_draft = ""
    draft_history: List[str] = []  # last N winning drafts for condition I

    for tick in range(total_ticks):
        # ── Condition I: recreate L0R and WM each tick ──
        if is_independent:
            l0r._ring.clear()
            if hasattr(wm, '_claims'):
                wm._claims.clear()

            # Inject raw context: last 3 winning drafts as plain text
            if draft_history:
                history_text = "\n---\n".join(draft_history[-3:])
                raw_context = f"Recent history (last {min(3, len(draft_history))} winning drafts):\n---\n{history_text}\n---\nRespond to the above discussion."
                orch.inject_user_message(raw_context)

        # ── Thematic injections ──
        if tick in INJECTIONS:
            orch.inject_user_message(INJECTIONS[tick])

        # ── Perturbation ──
        if tick in PERTURBATION_TICKS and has_ood:
            pert_label = PERTURBATION_TICKS[tick]
            ood_text = get_ood_text(seed, pert_label)
            if ood_text:
                orch.inject_user_message(ood_text)

                # Compute cosine distance between prev draft and OOD
                cos_dist = None
                if prev_selected_draft and not dry_mode:
                    try:
                        from organism_v2.metrics_v2 import _embed_model
                        if _embed_model:
                            emb_draft = _embed_model.encode([prev_selected_draft])[0]
                            emb_ood = _embed_model.encode([ood_text])[0]
                            cos_dist = 1.0 - cosine_sim(emb_draft, emb_ood)
                    except Exception:
                        pass

                perturbation_log.append({
                    "tick": tick, "type": "ood", "pert_label": pert_label,
                    "ood_text_id": OOD_ASSIGNMENT.get(seed, {}).get(pert_label, "?"),
                    "ood_len": len(ood_text),
                    "prev_draft_len": len(prev_selected_draft),
                    "cosine_distance_to_draft": cos_dist,
                })

                # Replace prev_selected_draft with OOD (this is what gets propagated)
                prev_selected_draft = ood_text
                draft_history.append(ood_text)

        elif tick in PERTURBATION_TICKS and has_neutral:
            if prev_selected_draft:
                neutral_text = neutral_perturbation(prev_selected_draft) if not dry_mode else "Neutral rephrase placeholder."
                if neutral_text:
                    orch.inject_user_message(neutral_text)
                    perturbation_log.append({
                        "tick": tick, "type": "neutral",
                        "input_len": len(prev_selected_draft),
                        "output_len": len(neutral_text),
                    })

        # ── Run tick ──
        result = orch.run_tick()

        # ── Condition I: clear user_messages after tick ──
        if is_independent:
            orch._user_messages.clear()

        # ── Extract drafts and winner ──
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

        # Track draft history for condition I
        if prev_selected_draft:
            if tick not in PERTURBATION_TICKS:  # don't double-add OOD
                draft_history.append(prev_selected_draft)

        winner_log.append({"tick": tick, "winner": winner_id})
        metrics.record_tick(agent_drafts, winner_id, verdict)

        if not dry_mode:
            active_turns = [t for t in result.agent_turns if t.text]
            if not active_turns:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"ABORT: {_MAX_CONSECUTIVE_FAILURES} consecutive empty ticks")
            else:
                consecutive_failures = 0

        # Live flush
        ticks_done = tick + 1
        live_data = {"condition": condition, "seed": seed, "total_ticks": total_ticks,
                     "ticks_done": ticks_done, "warmup_ticks": WARMUP_TICKS,
                     "perturbation_log": perturbation_log, **metrics.to_dict(), "sim_curves": None}
        with open(run_dir / "results.json", "w") as _lf:
            json.dump(live_data, _lf, ensure_ascii=False)

        elapsed = time.time() - t_run_start
        rate = ticks_done / elapsed if elapsed > 0 else 1
        eta = (total_ticks - ticks_done) / rate if rate > 0 else 0
        mode_str = result.mode.value if hasattr(result.mode, "value") else str(result.mode)
        print(f"\r  [{condition}] seed={seed} | Tick {ticks_done}/{total_ticks} | {mode_str} | ETA {_format_eta(eta)}    ", end="", flush=True)

    print()

    # ── Compute sim_curves ──
    pert_tick_list = [p["tick"] for p in perturbation_log]
    sim_curves = metrics.compute_sim_curves(pert_tick_list, k_max=K_MAX) if pert_tick_list else None

    # Model lineup for output
    if is_clone:
        lineup_agents = {k: CLONE_MODELS[condition] for k in ["A", "B", "C"]}
    elif is_single:
        lineup_agents = {"A": MODEL_CONFIGS["A"]["model"]}
    else:
        lineup_agents = {k: v["model"] for k, v in MODEL_CONFIGS.items()}

    output = {
        "bench_version": "v9",
        "bench_language": "en",
        "condition": condition,
        "condition_description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "seed": seed,
        "total_ticks": total_ticks,
        "ticks_done": total_ticks,
        "warmup_ticks": WARMUP_TICKS,
        "perturbation_log": perturbation_log,
        "winner_log": winner_log,
        "discarded_drafts": getattr(orch, '_discarded_drafts', []),
        **metrics.to_dict(),
        "sim_curves": sim_curves,
        "model_lineup": {
            "agents": lineup_agents,
            "judge": JUDGE_MODEL,
            "neutral_pert": NEUTRAL_PERT_MODEL if has_neutral else None,
        },
        "analysis_notes": {
            "bench_version": "v9",
            "perturbation_type": "ood" if has_ood else ("neutral" if has_neutral else "none"),
            "perturbation_ticks": list(PERTURBATION_TICKS.keys()),
            "measurement_windows": {"P1": "ticks 21-29 (k=1..9)", "P2": "ticks 41-49 (k=1..9)"},
            "k_max": K_MAX,
            "reference_P1": "mean(sv[17], sv[18], sv[19])",
            "reference_P2": "mean(sv[37], sv[38], sv[39])",
            "injection_ticks": list(INJECTIONS.keys()),
            "condition_I_design": (
                "L0R/WM recreated each tick. Agents receive raw 'Recent history' "
                "(last 3 winning drafts as plain text). Same prompts as D3. "
                "OOD text NOT filtered from history."
            ) if is_independent else None,
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False)

    log.info("[%s] Done — %d ticks in %.1fs", run_id, total_ticks, time.time() - t_run_start)
    return output


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bench V9 — Definitive (OOD perturbation)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ood-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    # Override judge ctx
    import organism.judge as _judge_mod
    _judge_mod._JUDGE_CTX = 128000
    _judge_mod._JUDGE_PREDICT = 4000

    dry_mode = args.dry_run
    total_ticks = 10 if dry_mode else args.ticks
    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    conditions = [c.strip().upper() for c in args.conditions.split(",")] if args.conditions else CONDITIONS
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "runs" / "bench_v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOD texts
    ood_path = Path(args.ood_file) if args.ood_file else PROJECT_ROOT / "ood_texts_final.json"
    load_ood_texts(ood_path)

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
    print(f"  BENCH V9 — DEFINITIVE (OOD perturbation)")
    print(f"  {len(conditions)} conditions × {len(seeds)} seeds = {total_runs} runs")
    print(f"  {total_ticks} ticks/run, k_max={K_MAX}")
    print(f"  Output: {output_dir}")
    print(f"  OOD texts: {len(_ood_texts)} loaded")
    print(f"  {'DRY MODE' if dry_mode else 'LIVE (Ollama)'}")
    print(f"{'='*60}\n")

    # Seed-first loop order
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
                run_single(condition, seed, total_ticks, output_dir, dry_mode)
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
    print(f"  V9 BENCH COMPLETE — {completed}/{total_runs} runs in {_format_eta(elapsed)}")
    print(f"  Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
