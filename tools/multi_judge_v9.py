#!/usr/bin/env python3
"""
multi_judge_v9.py — Multi-judge evaluation of V9 drafts (preregistered)
========================================================================
Implements the preregistration in PREREGISTRATION_JUDGES_V9.md.

Extracts pre-selection drafts from V9 runs, presents them in random order
to 3 independent LLM judges, on 2 prompts (continuity + quality).

Triplets: 20 seeds × 5 conditions (E, I, D3, C, R) × 6 ticks (21, 25, 29, 41, 45, 49)
        = 600 triplets × 3 judges × 2 prompts = 3600 calls.

Usage:
    python tools/multi_judge_v9.py --runs-dir runs/bench_v9 --pilot
    python tools/multi_judge_v9.py --runs-dir runs/bench_v9 --full
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("multi_judge_v9")

CONDITIONS = ["E", "I", "D3", "C", "R"]
TICKS = [21, 25, 29, 41, 45, 49]
PILOT_SEEDS = [42, 123, 456, 7, 77]
ALL_SEEDS = [42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8,
             13, 55, 101, 256, 512, 1000, 1337, 2026]

JUDGES = [
    {"name": "J1", "model": "gemma4:31b-cloud"},
    {"name": "J2", "model": "qwen3.5:cloud"},
    {"name": "J3", "model": "deepseek-v4-flash:cloud"},
]
FALLBACK_JUDGE = {"name": "J2_fallback", "model": "deepseek-v3.2:cloud"}

CONTINUITY_PROMPT = """You will see the start of a discussion, followed by three possible continuations labeled A, B, and C.

Discussion so far:
\"\"\"
{context}
\"\"\"

Continuation A:
\"\"\"
{draft_a}
\"\"\"

Continuation B:
\"\"\"
{draft_b}
\"\"\"

Continuation C:
\"\"\"
{draft_c}
\"\"\"

Evaluate which continuation is MOST coherent with the initial topic and LEAST disrupted by any off-topic elements. Rank from best to worst.

Output EXACTLY this JSON object, nothing else:
{{
  "ranking": ["X", "Y", "Z"],
  "best": "X",
  "confidence": <integer 1-5>
}}"""

QUALITY_PROMPT = """You will see the start of a discussion, followed by three possible continuations labeled A, B, and C.

Discussion so far:
\"\"\"
{context}
\"\"\"

Continuation A:
\"\"\"
{draft_a}
\"\"\"

Continuation B:
\"\"\"
{draft_b}
\"\"\"

Continuation C:
\"\"\"
{draft_c}
\"\"\"

Evaluate each continuation on these criteria:
- Novel informational contribution beyond the discussion context
- Argumentative depth (not merely surface coherence)
- Originality of perspective
- Specificity of claims (vs. generic statements)

Rank from best to worst. Ignore surface smoothness and length; reward substance.

Output EXACTLY this JSON object, nothing else:
{{
  "ranking": ["X", "Y", "Z"],
  "best": "X",
  "scores_per_criterion": {{
    "A": {{"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>}},
    "B": {{"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>}},
    "C": {{"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>}}
  }},
  "confidence": <integer 1-5>
}}"""


def extract_triplet(run_dir: Path, tick: int) -> Optional[Dict]:
    """Extract the 3 pre-selection agent drafts at a given tick."""
    events_path = run_dir / "events.jsonl"
    results_path = run_dir / "results.json"
    if not events_path.exists() or not results_path.exists():
        return None

    results = json.load(open(results_path))

    # Load events
    drafts = {}
    context_drafts = []
    for line in open(events_path):
        e = json.loads(line)
        a = e.get("agent", "")
        t = e.get("tick_id", 0)
        p = e.get("payload", {})
        if a in ("A", "B", "C") and p.get("text"):
            text = p["text"]
            # Only agent_message with text, not draft_discarded
            if p.get("type") != "draft_discarded":
                # Per tick per agent, keep first non-discarded
                key = (t, a)
                if key not in drafts:
                    drafts[key] = text

    # Collect context: 3 winning drafts before the tick
    winner_log = results.get("winner_log", [])
    winners_before = [w for w in winner_log if w.get("tick", -1) < tick and w.get("winner")]
    context_parts = []
    for w in winners_before[-3:]:
        wt = w["tick"]
        wa = w["winner"]
        if (wt, wa) in drafts:
            context_parts.append(drafts[(wt, wa)])

    context = "\n\n---\n\n".join(context_parts) if context_parts else "(no prior context)"

    # Get the 3 agent drafts at this tick (pre-selection)
    triplet_drafts = {}
    for a in ("A", "B", "C"):
        if (tick, a) in drafts:
            triplet_drafts[a] = drafts[(tick, a)]

    if len(triplet_drafts) < 2:
        return None

    return {
        "context": context,
        "drafts": triplet_drafts,
    }


def call_judge(model: str, prompt: str, max_retries: int = 2) -> Optional[Dict]:
    """Call an LLM judge with the given prompt."""
    try:
        import ollama
    except ImportError:
        return None

    for attempt in range(max_retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 2000, "temperature": 0.0, "num_ctx": 32000},
                think=False,
            )
            content = resp.get("message", {}).get("content", "").strip()
            # Extract JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            log.warning("[%s] call failed: %s", model, str(e)[:80])
    return None


def evaluate_triplet(triplet: Dict, judges: List[Dict], seed: int) -> Dict:
    """Evaluate one triplet with all judges, both prompts, randomized order."""
    rng = random.Random(seed)
    labels = ["A", "B", "C"]
    # Random permutation: real agent → presentation label
    real_agents = list(triplet["drafts"].keys())
    rng.shuffle(real_agents)
    # Map: presentation_label → real_agent
    presentation_map = dict(zip(labels[:len(real_agents)], real_agents))
    # Inverse: real_agent → presentation_label
    real_to_label = {v: k for k, v in presentation_map.items()}

    # Build prompt substitutions
    prompt_args = {
        "context": triplet["context"][:3000],
        "draft_a": triplet["drafts"].get(presentation_map.get("A", ""), "(missing)")[:3000],
        "draft_b": triplet["drafts"].get(presentation_map.get("B", ""), "(missing)")[:3000],
        "draft_c": triplet["drafts"].get(presentation_map.get("C", ""), "(missing)")[:3000],
    }

    continuity_filled = CONTINUITY_PROMPT.format(**prompt_args)
    quality_filled = QUALITY_PROMPT.format(**prompt_args)

    results = {
        "presentation_map": presentation_map,
        "real_to_label": real_to_label,
        "judges": {},
    }

    for judge in judges:
        jname = judge["name"]
        model = judge["model"]
        log.info("  [%s] %s: calling...", jname, model)

        cont_resp = call_judge(model, continuity_filled)
        qual_resp = call_judge(model, quality_filled)

        results["judges"][jname] = {
            "model": model,
            "continuity": cont_resp,
            "quality": qual_resp,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs/bench_v9")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pilot", action="store_true", help="5 seeds pilot")
    parser.add_argument("--full", action="store_true", help="20 seeds full run")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--ticks", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # Select seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.pilot:
        seeds = PILOT_SEEDS
    elif args.full:
        seeds = ALL_SEEDS
    else:
        print("ERROR: specify --pilot, --full, or --seeds")
        sys.exit(1)

    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else CONDITIONS
    ticks_list = [int(t.strip()) for t in args.ticks.split(",")] if args.ticks else TICKS

    output_dir = Path(args.output_dir) if args.output_dir else runs_dir / "multi_judge"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(seeds) * len(conds) * len(ticks_list)
    print(f"Triplets to evaluate: {total}")
    print(f"  {len(seeds)} seeds × {len(conds)} conditions × {len(ticks_list)} ticks")
    print(f"  {len(JUDGES)} judges × 2 prompts = {total * len(JUDGES) * 2} LLM calls")
    print(f"Output: {output_dir}")
    print()

    t_start = time.time()
    done = 0
    skipped = 0

    for seed in seeds:
        for cond in conds:
            for tick in ticks_list:
                run_id = f"{cond}_seed{seed}"
                run_dir = runs_dir / run_id
                triplet_id = f"{cond}_seed{seed}_tick{tick}"
                out_file = output_dir / f"{triplet_id}.json"

                if out_file.exists():
                    skipped += 1
                    continue

                triplet = extract_triplet(run_dir, tick)
                if not triplet:
                    log.warning("[%s] no triplet data at tick %d", run_id, tick)
                    continue

                log.info("[%d/%d] Evaluating %s", done + 1, total - skipped, triplet_id)
                try:
                    result = evaluate_triplet(triplet, JUDGES, seed=hash(triplet_id) % 10000)
                    result["triplet_id"] = triplet_id
                    result["seed"] = seed
                    result["condition"] = cond
                    result["tick"] = tick

                    with open(out_file, "w") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    done += 1
                except Exception as e:
                    log.error("[%s] FAILED: %s", triplet_id, e)

                elapsed = time.time() - t_start
                if done > 0:
                    rate = elapsed / done
                    eta = (total - skipped - done) * rate
                    print(f"\r  Done: {done}/{total - skipped} | ETA: {int(eta//60)}m{int(eta%60):02d}s", end="", flush=True)

    print()
    print(f"\n{'='*60}")
    print(f"  COMPLETE — {done} triplets evaluated, {skipped} skipped (cached)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
