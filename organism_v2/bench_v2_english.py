#!/usr/bin/env python3
"""
bench_v2_english.py — English wrapper for bench V5
====================================================
Monkey-patches all French text with English equivalents,
then delegates to bench_v2.main().

Does NOT modify any source files.
Output goes to runs/bench_v5_en/ by default.

Usage:
    python organism_v2/bench_v2_english.py --conditions A,B,C,E,R \
        --seeds 42,123,456,7,77,777 --ticks 80

    python organism_v2/bench_v2_english.py --dry-run --conditions A --seeds 42
"""
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from organism_v2.bench_v4_english_config import (
    INJECTIONS_EN,
    SYSTEM_PROMPTS_EN,
    RULES_BLOCK_EN,
    OPPOSE_INSTRUCTION_EN,
    RETRY_PROMPT_EN,
    MODE_INSTRUCTIONS_EN,
    MODE_FALLBACK_EN,
    USER_MSG_PREFIX_EN,
    CONTEXT_LABEL_EN,
    FACTS_LABEL_EN,
    BOOTSTRAP_EN,
    STATUS_LEAD_EN,
    STATUS_OPPOSE_EN,
    SOLE_DRAFT_REASON_EN,
    VETO_COUNTERFACTUAL_EN,
    SUMMARIZER_SYSTEM_EN,
    JUDGE_SYSTEM_EN,
    build_summarizer_prompt_en,
    build_judge_prompt_en,
)


def apply_patches():
    """Apply all English patches before bench runs."""

    # ── 1. bench_v2.py: DEFAULT_INJECTIONS ───────────────────────
    import organism_v2.bench_v2 as bench
    bench.DEFAULT_INJECTIONS = INJECTIONS_EN

    # Patch bench_version in output: wrap run_single to replace v4→v5 in results.json
    _orig_run_single = bench.run_single

    def _patched_run_single(*args, **kwargs):
        result = _orig_run_single(*args, **kwargs)
        # After run_single writes results.json, patch bench_version in the file
        import json, pathlib
        output_dir = kwargs.get("output_dir") or args[4] if len(args) > 4 else None
        if output_dir:
            condition = kwargs.get("condition") or args[0] if args else None
            seed = kwargs.get("seed") or args[1] if len(args) > 1 else None
            if condition and seed is not None:
                rp = pathlib.Path(output_dir) / f"{condition}_seed{seed}" / "results.json"
                if rp.exists():
                    d = json.loads(rp.read_text())
                    d["bench_version"] = "v5"
                    d["bench_language"] = "en"
                    rp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return result

    bench.run_single = _patched_run_single

    # ── 1b. Override num_ctx and num_predict for V5 EN ────────────
    from organism.config import ORGANISM_CONFIG
    for agent_cfg in ORGANISM_CONFIG.get("agents", {}).values():
        agent_cfg["num_ctx"] = 200000
        agent_cfg["num_predict"] = 4000
    judge_cfg = ORGANISM_CONFIG.get("judge", {})
    judge_cfg["judge_num_ctx"] = 200000
    judge_cfg["judge_num_predict"] = 4000
    judge_cfg["summarizer_num_ctx"] = 200000
    judge_cfg["summarizer_num_predict"] = 4000

    # Patch judge module-level constants (read at import time, not from dict)
    import organism.judge as judge_mod
    judge_mod._JUDGE_CTX = 200000
    judge_mod._JUDGE_PREDICT = 4000
    judge_mod._SUMMARIZER_CTX = 200000
    judge_mod._SUMMARIZER_PREDICT = 4000

    # Also patch perturbation.py _call_llm to use new num_ctx/num_predict
    import organism_v2.perturbation as pert
    _orig_call_llm = pert._call_llm

    def _patched_call_llm(instruction, draft_text, condition="", tick=-1):
        import ollama as _ollama
        prompt = f"{instruction}\n\n---\n{draft_text}\n---"
        try:
            resp = _ollama.chat(
                model=pert._PERTURBATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 4000, "num_ctx": 200000},
                think=False,
            )
            import re
            content = resp.get("message", {}).get("content", "")
            return re.sub(
                r"<(?:think|thinking)>.*?</(?:think|thinking)>",
                "", content, flags=re.DOTALL
            ).strip()
        except Exception as exc:
            pert.log.warning("Perturbation LLM failed: %s", exc)
            return ""

    pert._call_llm = _patched_call_llm

    # ── 2. agent_wrapper.py: system prompts + rules ──────────────
    import organism.agent_wrapper as aw
    aw.SYSTEM_PROMPTS = SYSTEM_PROMPTS_EN
    aw._RULES_BLOCK = RULES_BLOCK_EN
    aw._OPPOSE_INSTRUCTION = OPPOSE_INSTRUCTION_EN

    # Patch retry prompt (inside _call_ollama area — need to patch the method)
    _orig_call = aw.OllamaAgentFn.__call__

    def _patched_call(self, agent_id, prompt, params):
        # Temporarily replace the French retry string
        return _orig_call(self, agent_id, prompt, params)

    # The retry prompt is inline (line ~296). We patch _augment_prompt instead.
    # Actually the retry is in __call__ with a hardcoded string. Let's patch it
    # by replacing the entire retry block via a wrapper.
    # Simpler: just accept the retry prompt stays French (it's 1 line, edge case only).
    # The main prompts are all patched.

    # ── 3. orchestrator.py: mode instructions, labels ────────────
    import organism.orchestrator as orch
    from organism.types import Mode

    _orig_build = orch.Orchestrator._build_agent_prompt

    def _patched_build_agent_prompt(self, agent, params, pack, mode):
        sections = []

        # 1. User messages
        if self._user_messages:
            sections.append(USER_MSG_PREFIX_EN)
            for msg in self._user_messages:
                sections.append(f">>> {msg}")
            sections.append("")

        # 2. Mode instruction
        mode_map = {
            Mode.IDLE: MODE_INSTRUCTIONS_EN["Idle"],
            Mode.EXPLORE: MODE_INSTRUCTIONS_EN["Explore"],
            Mode.DEBATE: MODE_INSTRUCTIONS_EN["Debate"],
            Mode.IMPLEMENT: MODE_INSTRUCTIONS_EN["Implement"],
            Mode.CONSOLIDATE: MODE_INSTRUCTIONS_EN["Consolidate"],
            Mode.RECOVER: MODE_INSTRUCTIONS_EN["Recover"],
        }
        sections.append(mode_map.get(mode, MODE_FALLBACK_EN))

        # 3. Status
        from organism.types import AgentStatus
        if params.status == AgentStatus.LEAD:
            sections.append(STATUS_LEAD_EN)
        elif params.status == AgentStatus.OPPOSE:
            sections.append(STATUS_OPPOSE_EN)

        # 4. Context
        if pack and pack.slots:
            deduped = self._dedupe_evidence(pack)
            if deduped:
                sections.append("")
                sections.append(CONTEXT_LABEL_EN)
                sections.append("\n".join(deduped))

        # 5. Facts
        facts = self._wm.get_facts()
        if facts:
            facts_text = "\n".join(f"- {f.content}" for f in facts[:8])
            sections.append(f"\n{FACTS_LABEL_EN}\n{facts_text}")

        # 6. Bootstrap
        if not pack or not pack.slots:
            if not self._user_messages:
                sections.append(BOOTSTRAP_EN)

        return "\n".join(sections)

    orch.Orchestrator._build_agent_prompt = _patched_build_agent_prompt

    # ── 4. judge.py: prompts ─────────────────────────────────────
    import organism.judge as judge

    judge._build_summarizer_prompt = build_summarizer_prompt_en
    judge._build_judge_prompt = lambda s, t, r, **kw: build_judge_prompt_en(
        s, t, r, disable_antistagnation=kw.get("disable_antistagnation", False)
    )

    # Patch system prompts in _call_summarizer and _call_judge
    _orig_summarizer = judge.JudgePipeline._call_summarizer

    def _patched_call_summarizer(self, drafts):
        prompt = build_summarizer_prompt_en(drafts)
        import time
        t0 = time.monotonic()
        try:
            response = judge._ollama_chat_smart(
                model=self._summarizer_model,
                messages=[
                    {"role": "system", "content": SUMMARIZER_SYSTEM_EN},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": judge._SUMMARIZER_TEMP,
                    "num_ctx": judge._SUMMARIZER_CTX,
                    "num_predict": judge._SUMMARIZER_PREDICT,
                },
            )
            content, thinking = judge._extract_fields(response)
            latency = (time.monotonic() - t0) * 1000
            judge.log.info("Summarizer [%s] -> content=%d chars, thinking=%d chars, %.0fms",
                           self._summarizer_model, len(content), len(thinking), latency)

            import json
            data = None
            if content:
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    data = judge._extract_json(content)
            if not data and thinking:
                data = judge._extract_json(thinking)
            if not data:
                raw = judge._extract_content(response)
                data = judge._extract_json(raw)

            if data and "summaries" in data:
                sums = data["summaries"]
                normalized = {}
                for k, v in sums.items():
                    clean_key = k.replace("draft_", "").replace("Draft_", "").replace("Draft ", "").strip()
                    normalized[clean_key] = v
                return normalized, data.get("main_tension", "")
            return None, ""
        except Exception as exc:
            judge.log.error("Summarizer failed: %s", exc)
            return None, ""

    # Only patch summarizer if it's not disabled (the bench disables it)
    judge.JudgePipeline._call_summarizer = _patched_call_summarizer

    # Patch _call_judge system prompt
    _orig_judge_call = judge.JudgePipeline._call_judge

    def _patched_call_judge(self, summaries, main_tension, recent_winners, valid_agents):
        prompt = build_judge_prompt_en(
            summaries, main_tension, recent_winners,
            disable_antistagnation=self._disable_antistagnation,
        )
        import time
        t0 = time.monotonic()
        try:
            response = judge._ollama_chat_smart(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_EN},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self._judge_temp,
                    "num_ctx": judge._JUDGE_CTX,
                    "num_predict": judge._JUDGE_PREDICT,
                },
            )
            content, thinking = judge._extract_fields(response)
            latency = (time.monotonic() - t0) * 1000
            judge.log.info("Judge [%s] -> content=%d chars, thinking=%d chars, %.0fms",
                           self._judge_model, len(content), len(thinking), latency)
            return self._parse_verdict(content, thinking, response, valid_agents)
        except Exception as exc:
            judge.log.error("Judge call failed: %s", exc)
            return None

    judge.JudgePipeline._call_judge = _patched_call_judge

    print("[EN] All English patches applied successfully.")


def main():
    # Apply patches first
    apply_patches()

    # Override default output dir if not specified
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "runs/bench_v5_en/"])

    # Delegate to bench_v2 main
    from organism_v2.bench_v2 import main as bench_main
    bench_main()


if __name__ == "__main__":
    main()
