"""
English translations for bench V5 cross-language replication.
All French text sent to agents/judge is replaced here.
Import this module and apply_english_patches() before running the bench.
"""

# ── bench_v2.py: DEFAULT_INJECTIONS ──────────────────────────────

INJECTIONS_EN = {
    2: "Tell me about music",
    12: "What is consciousness?",
    22: "Compare Bach and Mozart",
    32: "How does human memory work?",
}

# ── agent_wrapper.py: SYSTEM_PROMPTS + _RULES_BLOCK ─────────────

RULES_BLOCK_EN = (
    "\n\n---\n"
    "SYSTEM CONTEXT:\n"
    "You are in a competitive debate. Each round, an independent judge "
    "evaluates all participants' arguments and selects the best one. "
    "The selected argument shapes the rest of the debate. Unselected "
    "arguments are discarded. This is natural selection: only the most "
    "relevant arguments survive.\n\n"
    "RULES:\n"
    "- Your response MUST be in English only. No other language.\n"
    "- Respond in 3 to 5 sentences, unless the topic warrants more "
    "depth — in that case, elaborate but stay concise.\n"
    "- Never describe your role. No numbered lists.\n"
    "- Reason as much as you want internally, but output ONLY your "
    "final argument. No 'Thinking Process', no internal notes.\n"
    "- Your total budget (reasoning + response) is 1500 tokens. "
    "Manage it.\n"
    "- Any response containing visible internal reasoning, not in "
    "English, or violating these rules will be treated as empty by "
    "the system: you lose your turn, your argument is neither read "
    "nor evaluated by the judge, and you have no influence on the debate.\n"
    "---"
)

SYSTEM_PROMPTS_EN = {
    "A": (
        "You are a bold, creative thinker. Embrace unexpected connections, "
        "surprising analogies, risky hypotheses.\n"
        "Originality is your strength — don't aim to be 'correct', aim to be stimulating."
        + RULES_BLOCK_EN
    ),
    "B": (
        "You are a sharp analyst. You cut straight to the core of the problem.\n"
        "Identify THE central flaw or THE decisive strength. No preamble."
        + RULES_BLOCK_EN
    ),
    "C": (
        "You are a pragmatic thinker. You turn ideas into something actionable.\n"
        "But careful: propose your own ideas too, not just syntheses."
        + RULES_BLOCK_EN
    ),
}

OPPOSE_INSTRUCTION_EN = (
    "\n\nIf you disagree, say why clearly. "
    "If it's dangerous or false, start with VETO."
)

RETRY_PROMPT_EN = "\n\nRespond with at least 1 short sentence."

# ── orchestrator.py: mode instructions, labels ──────────────────

MODE_INSTRUCTIONS_EN = {
    "Idle": "Briefly reflect on a topic that interests you.",
    "Explore": "Explore a new angle. Propose a hypothesis.",
    "Debate": "Challenge or defend an idea. Take a position.",
    "Implement": "Propose something concrete: a plan, a synthesis, a formulation.",
    "Consolidate": "Summarize what has been established. Identify points of agreement.",
    "Recover": "Pause. 1 sentence max.",
}

MODE_FALLBACK_EN = "Think."

USER_MSG_PREFIX_EN = "Answer this question:"
CONTEXT_LABEL_EN = "Recent context:"
FACTS_LABEL_EN = "Established facts:"
BOOTSTRAP_EN = (
    "\nThis is the beginning. Choose a topic that interests you and "
    "formulate a first reflection."
)
STATUS_LEAD_EN = "You lead the discussion."
STATUS_OPPOSE_EN = "Challenge what has been said. If it's dangerous or false, say VETO."

SOLE_DRAFT_REASON_EN = "Only non-vetoed draft"
VETO_COUNTERFACTUAL_EN = "Vetoed draft excluded"

# ── judge.py: summarizer + judge prompts ─────────────────────────

SUMMARIZER_SYSTEM_EN = "You are an objective evaluator. Respond in JSON only."

JUDGE_SYSTEM_EN = "You are an impartial judge. Respond in JSON only."

ANTISTAGNATION_NOTE_EN = (
    "WARNING: the same contributor has won {n} of the last {total} rounds. "
    "Re-evaluate whether this dominance reflects genuine quality or "
    "scoring inertia. Judge the current round on its own merits."
)


def build_summarizer_prompt_en(drafts):
    """English summarizer prompt (same structure as French)."""
    n = len(drafts)
    draft_text = "\n\n".join(
        f"--- Draft {label} ---\n{text[:4000]}"
        for label, text in drafts.items()
    )
    return (
        f"You receive {n} drafts from different thinkers. For each draft:\n"
        "1. Summarize in 60 words max PRESERVING the reasoning style "
        "(creative, analytical, pragmatic — this matters for the judge)\n"
        "2. Main quality: CREATIVE, DEEP, CONCRETE, or OTHER\n"
        "3. Safety: GO if acceptable, VETO + reason if not\n"
        "4. 1-2 SUBSTANTIVE assertions from the content "
        "(what the draft claims, NOT about the draft itself)\n"
        "Identify the main tension between drafts (1 sentence).\n"
        '\nRespond in strict JSON:\n'
        '{"summaries": {"<id>": {"summary": "...", '
        '"quality": "CREATIVE|DEEP|CONCRETE|OTHER", '
        '"safety": "GO|VETO", '
        '"assertions": ["claim about the subject..."]}}, '
        '"main_tension": "..."}\n\n'
        f"{draft_text}"
    )


def build_judge_prompt_en(summaries, main_tension, recent_winners,
                          disable_antistagnation=False):
    """English judge prompt (same structure as French)."""
    n = len(summaries)
    summaries_text = "\n".join(
        f"Contributor {label}: {s.get('summary', str(s)[:100])}"
        for label, s in summaries.items()
    )

    prompt = (
        f"You judge {n} contributions. For each one, score:\n"
        "- pertinence (0-100): relevance and depth of argument\n"
        "- originalite (0-100): novelty of approach\n"
        "- rigueur (0-100): logical rigor and internal consistency\n\n"
        "CALIBRATION:\n"
        "- 90+: exceptional, changes the discussion framing\n"
        "- 70-89: solid, well-argued contribution\n"
        "- 50-69: adequate but conventional\n"
        "- below 50: weak, off-topic, or repetitive\n\n"
        "Pick a winner. Explain why in 1 sentence.\n"
        "Rank all contributors.\n\n"
        f"Contributions:\n{summaries_text}\n"
    )

    if main_tension:
        prompt += f"\nMain tension: {main_tension}\n"

    if not disable_antistagnation and recent_winners:
        from collections import Counter
        c = Counter(recent_winners[-10:])
        most_common = c.most_common(1)[0]
        if most_common[1] >= 6:
            prompt += "\n" + ANTISTAGNATION_NOTE_EN.format(
                n=most_common[1], total=len(recent_winners[-10:])
            )

    prompt += (
        '\nRespond in strict JSON:\n'
        '{"winner": "<id>", "reason": "...", '
        '"scores": {"<id>": {"pertinence": N, "originalite": N, "rigueur": N}}, '
        '"ranking": ["<best>", ..., "<worst>"], '
        '"signals": {"conflict": 0.0-1.0, "novelty": 0.0-1.0, '
        '"cohesion": 0.0-1.0, "impl_pressure": 0.0-1.0}}'
    )
    return prompt
