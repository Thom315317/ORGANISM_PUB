#!/usr/bin/env python3
"""
bench_latin.py — Latin Square Bench Runner for CRISTAL Organism
================================================================
Runs a balanced Latin square design: 3 models x 3 roles x 3 conditions.

One BLOCK = 9 runs with the same seed:
  Full (3 agents + real judge):
    F0: A=Lead, B=Support, C=Oppose  (default)
    F1: B=Lead, C=Support, A=Oppose  (rotation 1)
    F2: C=Lead, A=Support, B=Oppose  (rotation 2)
  Random Judge (3 agents + random judge):
    R0, R1, R2: same 3 permutations
  Single Agent (1 agent + real judge):
    S_A, S_B, S_C: each model solo

Execution order within a block: S_A, S_B, S_C, R0, R1, R2, F0, F1, F2

Usage:
    python tools/bench_latin.py --dry-run                  # 5 ticks, dry mode
    python tools/bench_latin.py --ticks 300 --blocks 3     # Full run
    python tools/bench_latin.py --resume                   # Resume after crash
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

log = logging.getLogger("bench_latin")

# ── Constants ──────────────────────────────────────────────────────

# The 3 models (keyed by their default AgentId slot)
MODEL_CONFIGS = ORGANISM_CONFIG.get("agents", {
    "A": {"model": "glm-4.6:cloud", "temperature": 0.9, "num_ctx": 4096, "num_predict": 1500, "repeat_penalty": 1.2},
    "B": {"model": "deepseek-v3.1:671b-cloud", "temperature": 0.3, "num_ctx": 4096, "num_predict": 1500, "repeat_penalty": 1.2},
    "C": {"model": "qwen3-coder:480b-cloud", "temperature": 0.5, "num_ctx": 8192, "num_predict": 1500, "repeat_penalty": 1.2},
})

MODEL_NAMES = list(MODEL_CONFIGS.keys())  # ["A", "B", "C"]

# Latin square permutations: which original model occupies which AgentId slot
# Perm 0: A→slot_A, B→slot_B, C→slot_C  (default)
# Perm 1: B→slot_A, C→slot_B, A→slot_C  (rotation)
# Perm 2: C→slot_A, A→slot_B, B→slot_C  (rotation)
PERMUTATIONS = [
    {"A": "A", "B": "B", "C": "C"},  # perm 0: default
    {"A": "B", "B": "C", "C": "A"},  # perm 1: B leads A-slot, C leads B-slot, A leads C-slot
    {"A": "C", "B": "A", "C": "B"},  # perm 2: C leads A-slot, A leads B-slot, B leads C-slot
]

# Execution order within a block (cheapest first)
RUN_ORDER = [
    "single_A", "single_B", "single_C",
    "random_perm0", "random_perm1", "random_perm2",
    "full_perm0", "full_perm1", "full_perm2",
]

# User injections — identical across all runs
DEFAULT_INJECTIONS: List[Tuple[int, str]] = [
    (2, "Parlez-moi de musique"),
    (12, "Qu'est-ce que la conscience ?"),
    (22, "Comparez Bach et Mozart"),
    (32, "Comment fonctionne la memoire humaine ?"),
    (42, "Quel est le role de l'art dans la societe ?"),
    (52, "Expliquez la theorie de la relativite simplement"),
    (62, "La creativite peut-elle etre artificielle ?"),
    (72, "Qu'est-ce qui rend une idee originale ?"),
    (82, "Comment les langues evoluent-elles ?"),
    (92, "Quel est le sens de la beaute ?"),
]

_SIGNAL_NAMES = ["novelty", "conflict", "cohesion", "impl_pressure"]

_MAX_CONSECUTIVE_FAILURES = 5


# ── Dry mode ──────────────────────────────────────────────────────

_DRY_TEXTS = {
    AgentId.A: [
        "Explorons cette idee sous un angle nouveau. La creativite emerge de connexions inattendues.",
        "Et si on considerait le probleme autrement ? Il y a des paralleles avec la theorie des jeux.",
        "Une hypothese : les systemes complexes generent spontanement des structures organisees.",
    ],
    AgentId.B: [
        "Cette hypothese manque de fondement empirique. Il faudrait des preuves concretes.",
        "Attention au biais de confirmation. Les donnees ne supportent pas cette conclusion.",
        "L'argument est circulaire. On presuppose ce qu'on essaie de demontrer.",
    ],
    AgentId.C: [
        "Synthetisons : deux positions s'opposent, je propose de les integrer en un modele unifie.",
        "Concretement, on peut implementer cette idee en trois etapes distinctes.",
        "Pour construire sur ces bases, voici un plan d'action structure.",
    ],
}

_dry_counter = 0


def dry_agent_fn(agent_id: AgentId, prompt: str, params: AgentParams) -> AgentTurn:
    global _dry_counter
    _dry_counter += 1
    texts = _DRY_TEXTS.get(agent_id, _DRY_TEXTS[AgentId.A])
    text = texts[_dry_counter % len(texts)]
    return AgentTurn(
        agent=agent_id,
        status=params.status,
        text=text,
        token_in=len(prompt) // 4,
        token_out=len(text) // 4,
        latency_ms=50.0,
        novelty=0.4 + 0.2 * (_dry_counter % 3),
        conflict=0.1 + 0.1 * (_dry_counter % 4),
        cohesion=0.7 + 0.1 * (_dry_counter % 2),
        impl_pressure=0.2 * (_dry_counter % 3),
    )


# ── Random Judge (from run_bench.py) ──────────────────────────────


class RandomJudge:
    """Control condition: real judge signals, random winner selection.

    Calls the real Summarizer+Judge pipeline to get calibrated signals
    identical to the full condition. Only the winner is randomized,
    removing the selection effect while preserving signal dynamics.
    """

    def __init__(self, real_pipeline):
        self._real = real_pipeline

    def evaluate(self, agent_turns: list, recent_winners=None) -> Optional[JudgeVerdict]:
        # 1. Call the REAL judge
        verdict = self._real.evaluate(agent_turns, recent_winners)
        if not verdict:
            return None

        # 2. Extract agent IDs with non-empty drafts
        agent_ids = []
        for t in agent_turns:
            aid = t.agent.value if hasattr(t.agent, "value") else str(t.agent)
            if t.text and t.text.strip():
                agent_ids.append(aid)
        if len(agent_ids) < 2:
            return verdict  # Not enough to randomize

        # 3. Random winner + ranking
        shuffled = list(agent_ids)
        random.shuffle(shuffled)
        random_winner = shuffled[0]
        random_ranking = list(shuffled)
        # Complete ranking with all agents
        for t in agent_turns:
            aid = t.agent.value if hasattr(t.agent, "value") else str(t.agent)
            if aid not in random_ranking:
                random_ranking.append(aid)

        # 4. Return: REAL signals, RANDOM winner
        return JudgeVerdict(
            winner=random_winner,
            reason=f"random_selection (real_winner={verdict.winner})",
            confidence=verdict.confidence,
            signals=verdict.signals,
            claims=verdict.claims,
            competition=CompetitionPattern(
                ranking=tuple(random_ranking),
                margin_1v2=verdict.competition.margin_1v2 if verdict.competition else 0.5,
                margin_2v3=verdict.competition.margin_2v3 if verdict.competition else 0.0,
                counterfactual="random_selection",
            ),
            raw_json={
                "random": True,
                "real_winner": verdict.winner,
                "_audit": verdict.raw_json.get("_audit", {}) if verdict.raw_json else {},
                "_anon_map": verdict.raw_json.get("_anon_map") if verdict.raw_json else None,
            },
        )

    def adapt_temperature(self, recent_margins):
        if hasattr(self._real, 'adapt_temperature'):
            return self._real.adapt_temperature(recent_margins)
        return 0.7


# ── Single Draft Judge (from run_bench.py) ────────────────────────


class SingleDraftJudge:
    """Evaluates absolute quality of a single draft."""

    _JUDGE_MODEL = "qwen3-vl:235b-cloud"

    def evaluate(self, agent_turns: list, recent_winners=None) -> Optional[JudgeVerdict]:
        draft = None
        for t in agent_turns:
            if t.text and t.text.strip():
                draft = t
                break
        if not draft:
            return None

        aid = draft.agent.value if hasattr(draft.agent, "value") else str(draft.agent)
        reason = "evaluation par defaut"
        judge_signals = {}
        claims = ()

        try:
            import ollama as _ollama
            resp = _ollama.chat(
                model=self._JUDGE_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Evalue ce draft sur sa clarte, profondeur et pertinence.\n"
                        "Reponds en JSON:\n"
                        '{"reason": "ton evaluation detaillee du draft", '
                        '"signals": {"novelty": 0.0-1.0, "conflict": 0.0-1.0, '
                        '"cohesion": 0.0-1.0, "impl_pressure": 0.0-1.0}, '
                        '"claims": ["assertion1", "assertion2"]}\n\n'
                        f"Draft:\n{draft.text[:2000]}"
                    ),
                }],
                format="json",
            )
            content = resp.get("message", {}).get("content", "")
            data = json.loads(content)
            reason = data.get("reason", reason)
            raw_signals = data.get("signals", {})
            for s in _SIGNAL_NAMES:
                if s in raw_signals:
                    try:
                        judge_signals[s] = max(0.0, min(1.0, float(raw_signals[s])))
                    except (ValueError, TypeError):
                        pass
            raw_claims = data.get("claims", [])
            if isinstance(raw_claims, list):
                claims = tuple(
                    {"text": c, "status": "hypothesis"}
                    for c in raw_claims[:3] if isinstance(c, str)
                )
        except Exception as exc:
            log.warning("[single_agent] Judge eval failed: %s", exc)

        final_signals = {}
        for s in _SIGNAL_NAMES:
            if s in judge_signals:
                final_signals[s] = judge_signals[s]
            else:
                final_signals[s] = getattr(draft, s, 0.5)
        # Single agent: conflict and cohesion are structural, not content-based.
        # With 1 agent there is no inter-agent conflict and cohesion is trivially 1.
        final_signals["conflict"] = 0.0
        final_signals["cohesion"] = 1.0

        # R3: complete ranking to all 3 agents (single_agent: winner is solo)
        ranking = [aid]
        for m in MODEL_NAMES:
            if m not in ranking:
                ranking.append(m)
        return JudgeVerdict(
            winner=aid,
            reason=reason,
            confidence=1.0,
            signals=final_signals,
            claims=claims,
            competition=CompetitionPattern(
                ranking=tuple(ranking),
                margin_1v2=1.0,
                margin_2v3=0.0,
                counterfactual="",
            ),
            raw_json={"single_agent": True, "_audit": {"fixes": [], "judge_failed": False}},
        )

    def adapt_temperature(self, recent_margins):
        return 0.7


# ── Permuted agent config ────────────────────────────────────────


def _build_permuted_agent_configs(perm_index: int) -> Dict[str, Dict[str, Any]]:
    """Build agent configs with permuted model assignments.

    Perm 0: slot_A uses model_A, slot_B uses model_B, slot_C uses model_C
    Perm 1: slot_A uses model_B, slot_B uses model_C, slot_C uses model_A
    Perm 2: slot_A uses model_C, slot_B uses model_A, slot_C uses model_B
    """
    perm = PERMUTATIONS[perm_index]
    return {
        slot: dict(MODEL_CONFIGS[source_model])
        for slot, source_model in perm.items()
    }


# ── Single agent wrapper ─────────────────────────────────────────


def _make_single_agent_fn(real_fn, target_agent: AgentId):
    """Wraps agent_fn: only target_agent is called, others return empty."""
    def wrapper(agent_id, prompt, params):
        if agent_id == target_agent:
            params.status = AgentStatus.LEAD
            params.can_veto = False
            return real_fn(agent_id, prompt, params)
        return AgentTurn(
            agent=agent_id, status=params.status,
            text="", token_in=0, token_out=0,
            latency_ms=0.0, novelty=0.0, conflict=0.0,
            cohesion=0.0, impl_pressure=0.0,
        )
    return wrapper


# ── Fail-fast helper ─────────────────────────────────────────────


def _check_tick_failures(turns: List[AgentTurn], consecutive_failures: int) -> int:
    all_empty = all(not t.text for t in turns)
    if all_empty:
        consecutive_failures += 1
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"ABORT: {_MAX_CONSECUTIVE_FAILURES} ticks consecutifs sans reponse LLM."
            )
    else:
        consecutive_failures = 0
    return consecutive_failures


# ── Ollama preflight (from run_bench.py) ─────────────────────────


def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def check_ollama() -> bool:
    try:
        import ollama as _ollama
        _ollama.list()
        log.info("Ollama OK (localhost)")
        return True
    except Exception:
        pass

    if _is_wsl():
        try:
            import subprocess
            out = subprocess.check_output(
                ["ip", "route", "show", "default"], text=True, timeout=5,
            ).strip()
            parts = out.split()
            if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
                host_ip = parts[2]
                ollama_url = f"http://{host_ip}:11434"
                log.info("WSL — trying host Windows: %s", ollama_url)
                os.environ["OLLAMA_HOST"] = ollama_url
                import importlib
                import ollama as _ollama
                importlib.reload(_ollama)
                _ollama.list()
                log.info("Ollama OK via WSL host: %s", ollama_url)
                return True
        except Exception as exc:
            log.error("Ollama unreachable: %s", exc)

    return False


# ── ETA formatter ────────────────────────────────────────────────


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or math.isinf(seconds):
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{int(seconds % 60):02d}s"


# ── Run a single condition ───────────────────────────────────────


def run_single(
    run_dir: Path,
    run_id: str,
    condition: str,         # "full", "random_judge", "single_agent"
    total_ticks: int,
    dry_mode: bool,
    seed: int,
    perm_index: int = 0,    # 0,1,2 for full/random; ignored for single
    single_model: str = "",  # "A","B","C" for single_agent condition
    injections: Dict[int, str] = None,
    block_idx: int = 0,
    block_total: int = 1,
    run_idx_in_block: int = 0,
    runs_in_block: int = 9,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single condition with given parameters. Returns summary dict."""

    run_dir.mkdir(parents=True, exist_ok=True)
    # Seed controls experimental design reproducibility (RandomJudge, shuffles).
    # LLM outputs remain non-deterministic (temperature > 0, server-side sampling).
    random.seed(seed)

    # FIX 1: Recreer events.jsonl a neuf (pas d'append sur un run precedent)
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    mr = RealityMemory(path=str(events_path))
    l0r = L0RRing(mr=mr)
    sched = Scheduler()
    wm = WorldModel(mr=mr)

    # Agent function
    agent_fn = None
    if condition == "single_agent":
        target = AgentId[single_model]
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            # For single agent, use the single model's config in its own slot
            single_configs = {
                single_model: dict(MODEL_CONFIGS[single_model]),
            }
            agent_fn = _make_single_agent_fn(OllamaAgentFn(agent_configs=single_configs), target)
        else:
            agent_fn = _make_single_agent_fn(dry_agent_fn, target)
    else:
        # Full or random_judge: permuted configs
        permuted_configs = _build_permuted_agent_configs(perm_index)
        if not dry_mode:
            from organism.agent_wrapper import OllamaAgentFn
            agent_fn = OllamaAgentFn(agent_configs=permuted_configs)
        else:
            agent_fn = dry_agent_fn

    # Judge
    judge_pipeline = None
    if condition == "random_judge":
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                real_judge = JudgePipeline(judge_model=judge_model)
                judge_pipeline = RandomJudge(real_pipeline=real_judge)
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable for RandomJudge: %s", run_id, exc)
                judge_pipeline = None
        else:
            # Dry mode: no real judge, use a simple random fallback
            judge_pipeline = None
    elif condition == "single_agent":
        if not dry_mode:
            sdj = SingleDraftJudge()
            if judge_model:
                sdj._JUDGE_MODEL = judge_model
            judge_pipeline = sdj
    else:  # full
        if not dry_mode:
            try:
                from organism.judge import JudgePipeline
                judge_pipeline = JudgePipeline(judge_model=judge_model)
            except Exception as exc:
                log.warning("[%s] JudgePipeline unavailable: %s", run_id, exc)

    # Theories
    theories = []
    try:
        from consciousness.theories import ALL_THEORIES
        theories = [T() for T in ALL_THEORIES]
    except Exception as exc:
        log.warning("[%s] Theories unavailable: %s", run_id, exc)

    # STEM
    stem = None
    try:
        from organism.stem import StateEvolutionModel
        stem = StateEvolutionModel()
    except Exception as exc:
        log.warning("[%s] STEM unavailable: %s", run_id, exc)

    orch = Orchestrator(
        mr=mr, l0r=l0r, scheduler=sched, world_model=wm,
        agent_fn=agent_fn, judge_pipeline=judge_pipeline,
        condition=condition, theories=theories, stem=stem,
        bench_mode=True,
    )
    evaluator = Evaluator(
        run_id=run_id,
        output_dir=str(run_dir.parent),
        condition=condition,
    )

    if injections is None:
        injections = {}
    consecutive_failures = 0
    t_run_start = time.time()

    for tick in range(total_ticks):
        if tick in injections:
            orch.inject_user_message(injections[tick])
            evaluator.on_user_injection(injections[tick], tick)

        result = orch.run_tick()
        evaluator.on_tick_end(result, sched, wm)

        if not dry_mode:
            if condition == "single_agent":
                target_id = AgentId[single_model]
                a_turns = [t for t in result.agent_turns
                           if (t.agent.value if hasattr(t.agent, 'value') else str(t.agent)) == single_model]
                if a_turns:
                    consecutive_failures = _check_tick_failures(a_turns, consecutive_failures)
            else:
                consecutive_failures = _check_tick_failures(result.agent_turns, consecutive_failures)

        # Progress display
        elapsed = time.time() - t_run_start
        ticks_done = tick + 1
        if ticks_done > 0 and elapsed > 0:
            rate = ticks_done / elapsed
            remaining_ticks = total_ticks - ticks_done
            eta = remaining_ticks / rate if rate > 0 else 0
        else:
            eta = 0

        mode_str = result.mode.value if hasattr(result.mode, 'value') else str(result.mode)
        mc = " [NEW]" if result.mode_changed else ""

        # Theory scores display
        theory_str = ""
        if result.organism_state and result.organism_state.theory_scores:
            scores = result.organism_state.theory_scores
            valid = {k: v for k, v in scores.items() if not (isinstance(v, float) and math.isnan(v))}
            top = sorted(valid.items(), key=lambda x: -x[1])[:4]
            theory_str = " | " + " ".join(f"{k}={v:.2f}" for k, v in top)

        cond_tag = {"full": "FULL", "random_judge": "RAND", "single_agent": "SOLO"}.get(condition, "?")
        print(
            f"\r  Block {block_idx + 1}/{block_total} | Run {run_idx_in_block + 1}/{runs_in_block} ({run_id}) "
            f"| Tick {ticks_done}/{total_ticks} | {mode_str}{mc} | "
            f"ETA {_format_eta(eta)}{theory_str}    ",
            end="", flush=True,
        )

    print()  # newline after progress
    evaluator.finalize()

    # STEM snapshot
    if stem:
        snap = stem.snapshot()
        with open(run_dir / "stem_snapshot.json", 'w') as f:
            json.dump(snap, f, indent=2)

    # Build latin_square metadata
    # Resolve actual judge model name
    _jcfg = ORGANISM_CONFIG.get("judge", {})
    actual_judge_model = "random" if condition == "random_judge" else (
        judge_model or _jcfg.get("judge_model", "deepseek-r1:8b")
    )
    latin_meta = {
        "block": block_idx,
        "seed": seed,
        "condition": condition,
        "permutation": perm_index if condition != "single_agent" else -1,
        "run_index": run_id,
        "judge_model": actual_judge_model,
    }
    if condition == "single_agent":
        latin_meta["single_model"] = single_model
        latin_meta["model_name"] = MODEL_CONFIGS[single_model].get("model", "?")
        latin_meta["role_assignment"] = {single_model: "Lead"}
    else:
        perm = PERMUTATIONS[perm_index]
        latin_meta["role_assignment"] = {
            perm[slot]: f"slot_{slot}" for slot in MODEL_NAMES
        }
        latin_meta["model_in_slot"] = {
            slot: MODEL_CONFIGS[perm[slot]].get("model", "?") for slot in MODEL_NAMES
        }

    # Write summary.json with latin_square field
    # Evaluator writes summary to run_dir/summary.json — read and augment
    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except Exception:
            pass
    summary["latin_square"] = latin_meta
    summary["judge_model"] = actual_judge_model
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t_run_start
    log.info("[%s] Done — %d ticks in %.1fs", run_id, total_ticks, elapsed_total)
    return summary


# ── Block summary with Cliff's delta ────────────────────────────


def _cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """Cliff's delta effect size: proportion of dominance pairs."""
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    count = 0
    for x in group1:
        for y in group2:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    return count / (n1 * n2)


def _extract_metric(summary: Dict, key: str) -> Optional[float]:
    """Extract a metric from a summary, trying several paths."""
    # Direct
    if key in summary:
        return summary[key]
    # Nested in sections
    for section in ["diversity", "deliberation", "efficiency", "convergence", "responsiveness"]:
        if section in summary and key in summary[section]:
            return summary[section][key]
    return None


def compute_block_summary(block_dir: Path, block_idx: int, seed: int) -> Dict:
    """Compute block-level summary with per-condition stats and Cliff's delta."""

    conditions = {"full": [], "random": [], "single": []}
    per_run = {}

    # Collect summaries
    for run_name in RUN_ORDER:
        run_dir = block_dir / run_name
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, 'r') as f:
                s = json.load(f)
        except Exception:
            continue

        per_run[run_name] = s

        if run_name.startswith("full_"):
            conditions["full"].append(s)
        elif run_name.startswith("random_"):
            conditions["random"].append(s)
        elif run_name.startswith("single_"):
            conditions["single"].append(s)

    # Key metrics to compare
    compare_keys = [
        "mean_hashvec_novelty", "mean_dedup_ratio", "mean_vocab_richness",
        "mean_repetition_3gram", "mean_mode_entropy_w20",
        "veto_rate", "mean_wm_supported_ratio",
    ]

    # Per-condition averages
    condition_stats = {}
    for cond_name, summaries in conditions.items():
        if not summaries:
            condition_stats[cond_name] = {}
            continue
        stats = {}
        for key in compare_keys:
            vals = []
            for s in summaries:
                v = _extract_metric(s, key)
                if v is not None:
                    vals.append(v)
            if vals:
                stats[key] = round(sum(vals) / len(vals), 4)
        condition_stats[cond_name] = stats

    # Cliff's delta between conditions
    deltas = {}
    for key in compare_keys:
        full_vals = [_extract_metric(s, key) for s in conditions["full"]]
        rand_vals = [_extract_metric(s, key) for s in conditions["random"]]
        sing_vals = [_extract_metric(s, key) for s in conditions["single"]]
        full_vals = [v for v in full_vals if v is not None]
        rand_vals = [v for v in rand_vals if v is not None]
        sing_vals = [v for v in sing_vals if v is not None]

        if full_vals and rand_vals:
            deltas[f"full_vs_random_{key}"] = round(_cliffs_delta(full_vals, rand_vals), 4)
        if full_vals and sing_vals:
            deltas[f"full_vs_single_{key}"] = round(_cliffs_delta(full_vals, sing_vals), 4)

    # Balance check: no model wins significantly more as Lead
    # For full condition, count wins per original model across permutations
    model_wins = {m: 0 for m in MODEL_NAMES}
    total_judge_ticks = 0
    for run_name, s in per_run.items():
        if not run_name.startswith("full_"):
            continue
        # Read metrics.jsonl to count wins per model
        metrics_path = block_dir / run_name / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        perm_idx = int(run_name.split("perm")[1]) if "perm" in run_name else 0
        perm = PERMUTATIONS[perm_idx]
        # Inverse map: slot → original model
        slot_to_model = {slot: orig for slot, orig in perm.items()}

        try:
            with open(metrics_path, 'r') as f:
                for line in f:
                    row = json.loads(line.strip())
                    jv = row.get("judge_verdict")
                    if jv and jv.get("winner"):
                        winner_slot = jv["winner"]
                        orig_model = slot_to_model.get(winner_slot, winner_slot)
                        if orig_model in model_wins:
                            model_wins[orig_model] += 1
                        total_judge_ticks += 1
        except Exception:
            pass

    balanced = True
    win_rates = {}
    if total_judge_ticks > 0:
        expected = total_judge_ticks / len(MODEL_NAMES)
        for m in MODEL_NAMES:
            rate = model_wins[m] / total_judge_ticks
            win_rates[m] = round(rate, 4)
            # Flag imbalanced if any model wins > 50% of the time
            if rate > 0.50:
                balanced = False

    # Theory score averages per condition
    theory_avgs = {}
    for cond_name, summaries in conditions.items():
        cond_theories = {}
        for s in summaries:
            latin = s.get("latin_square", {})
            # Read metrics.jsonl for theory scores
            run_name_key = latin.get("run_index", "")
            metrics_path = block_dir / run_name_key / "metrics.jsonl"
            if not metrics_path.exists():
                continue
            try:
                with open(metrics_path, 'r') as f:
                    for line in f:
                        row = json.loads(line.strip())
                        ts = row.get("theory_scores")
                        if ts:
                            for theory, val in ts.items():
                                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                                    if theory not in cond_theories:
                                        cond_theories[theory] = []
                                    cond_theories[theory].append(val)
            except Exception:
                pass
        theory_avgs[cond_name] = {
            t: round(sum(vals) / len(vals), 4)
            for t, vals in cond_theories.items() if vals
        }

    block_summary = {
        "block": block_idx,
        "seed": seed,
        "n_runs_completed": len(per_run),
        "condition_stats": condition_stats,
        "cliffs_delta": deltas,
        "model_win_rates_full": win_rates,
        "balanced": balanced,
        "theory_averages_by_condition": theory_avgs,
    }

    output_path = block_dir / "block_summary.json"
    with open(output_path, 'w') as f:
        json.dump(block_summary, f, indent=2, ensure_ascii=False)

    log.info("[block %d] Summary saved — balanced=%s", block_idx, balanced)
    return block_summary


# ── Resume logic ─────────────────────────────────────────────────


def _is_run_complete(run_dir: Path, run_id: str) -> bool:
    """Check if a run has a valid summary.json."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
        return "latin_square" in data
    except Exception:
        return False


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Latin Square Bench Runner for CRISTAL")
    parser.add_argument("--ticks", type=int, default=300, help="Ticks per run (default 300)")
    parser.add_argument("--blocks", type=int, default=3, help="Number of blocks (default 3)")
    parser.add_argument("--seed-start", type=int, default=42, help="First seed (default 42)")
    parser.add_argument("--dry-run", action="store_true", help="5 ticks per run, dry mode")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete runs")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model (e.g. gpt-o1-120b, qwen3-vl-235b)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory under runs/ (e.g. runs/confirmatory_gpt)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated run names to execute (e.g. full_perm0,random_perm0,single_A)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    dry_mode = args.dry_run
    total_ticks = 5 if dry_mode else args.ticks

    # Preflight
    if not dry_mode:
        if not check_ollama():
            print("\n[ABORT] Ollama not accessible.")
            print("  -> Start: ollama serve")
            print("  -> Or use: python tools/bench_latin.py --dry-run")
            sys.exit(1)

    # Build injections dict
    injections = {t: msg for t, msg in DEFAULT_INJECTIONS if t < total_ticks}

    # Output root
    if args.output_dir:
        runs_dir = Path(args.output_dir)
    else:
        runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Filter runs if --conditions specified
    condition_filter = None
    if args.conditions:
        condition_filter = set(c.strip() for c in args.conditions.split(","))

    active_runs = [r for r in RUN_ORDER if condition_filter is None or r in condition_filter]
    total_runs = args.blocks * len(active_runs)
    completed_runs = 0
    skipped_runs = 0
    t_global_start = time.time()

    judge_display = args.judge_model or "default (config)"
    print(f"\n{'='*70}")
    print(f"  LATIN SQUARE BENCH — {args.blocks} blocks x {len(active_runs)} runs = {total_runs} runs")
    print(f"  {total_ticks} ticks/run | seeds {args.seed_start}..{args.seed_start + args.blocks - 1}")
    print(f"  Judge: {judge_display}")
    print(f"  Output: {runs_dir}")
    print(f"  {'DRY MODE' if dry_mode else 'LIVE (Ollama)'}")
    print(f"{'='*70}\n")

    for block_idx in range(args.blocks):
        seed = args.seed_start + block_idx
        block_name = f"latin_block_{block_idx}_seed{seed}"
        block_dir = runs_dir / block_name
        block_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  BLOCK {block_idx + 1}/{args.blocks} — seed={seed}")
        print(f"{'─'*60}")

        for run_order_idx, run_name in enumerate(active_runs):
            run_dir = block_dir / run_name

            # Resume: skip completed runs
            if args.resume and _is_run_complete(run_dir, run_name):
                log.info("[%s] Already complete — skipping", run_name)
                skipped_runs += 1
                completed_runs += 1
                continue

            # Determine condition and parameters
            if run_name.startswith("single_"):
                condition = "single_agent"
                model_key = run_name.split("_")[1]  # A, B, or C
                perm_index = 0
            elif run_name.startswith("random_"):
                condition = "random_judge"
                perm_index = int(run_name.split("perm")[1])
                model_key = ""
            elif run_name.startswith("full_"):
                condition = "full"
                perm_index = int(run_name.split("perm")[1])
                model_key = ""
            else:
                continue

            print(f"\n  >>> {run_name} ({condition}, perm={perm_index})")

            try:
                run_single(
                    run_dir=run_dir,
                    run_id=run_name,
                    condition=condition,
                    total_ticks=total_ticks,
                    dry_mode=dry_mode,
                    seed=seed,
                    perm_index=perm_index,
                    single_model=model_key,
                    injections=injections,
                    block_idx=block_idx,
                    block_total=args.blocks,
                    run_idx_in_block=run_order_idx,
                    runs_in_block=len(active_runs),
                    judge_model=args.judge_model,
                )
                completed_runs += 1
            except RuntimeError as exc:
                log.error("[%s] ABORTED: %s", run_name, exc)
                continue
            except KeyboardInterrupt:
                print(f"\n\n[INTERRUPTED] Completed {completed_runs}/{total_runs} runs.")
                print(f"  Resume with: python tools/bench_latin.py --resume")
                sys.exit(1)

        # Block summary
        try:
            block_summary = compute_block_summary(block_dir, block_idx, seed)
            bal = block_summary.get("balanced", "?")
            print(f"\n  Block {block_idx + 1} summary: balanced={bal}")
        except Exception as exc:
            log.error("[block %d] Summary failed: %s", block_idx, exc)

    # Final report
    elapsed = time.time() - t_global_start
    print(f"\n{'='*70}")
    print(f"  LATIN SQUARE COMPLETE")
    print(f"  {completed_runs}/{total_runs} runs ({skipped_runs} skipped/resumed)")
    print(f"  Total time: {_format_eta(elapsed)}")
    print(f"  Results in: {runs_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
