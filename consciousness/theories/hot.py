"""
HOT - Higher-Order Thought Theory (Rosenthal, 2005)
=====================================================
Consciousness requires meta-cognition: thinking about thinking.
A mental state is conscious when accompanied by a higher-order
representation of that state.

Measures: meta-judgment presence, counterfactual depth, evaluative
density in judge reason, judge divergence from agents.
"""
from __future__ import annotations

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState

# Evaluative keywords: the judge EVALUATES, not just describes
_EVAL_KEYWORDS = [
    "meilleur", "faille", "critique", "insuffisant", "supérieur", "équilibr",
    "profond", "superficiel", "original", "banal", "pertinent", "hors-sujet",
    "better", "flaw", "critical", "superior", "weak", "strong", "relevant",
    "domine", "manque", "excell", "limit", "innov", "concret",
]


class HOTTheory(ConsciousnessTheory):
    """Higher-Order Thought Theory implementation."""

    @property
    def name(self) -> str:
        return "HOT"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}

        if state.judge_verdict:
            # 1. Has meta-judgment: the judge produced a second-order thought
            has_meta_judgment = 1.0
            components["has_meta_judgment"] = 1.0

            # 2. Counterfactual: thinking about what COULD have been conscious
            cf = ""
            if (state.judge_verdict.competition
                    and state.judge_verdict.competition.counterfactual):
                cf = state.judge_verdict.competition.counterfactual
            has_counterfactual = 1.0 if len(cf) > 10 else 0.0
            components["has_counterfactual"] = has_counterfactual

            # 3. Claims richness: judge produces thoughts about agent thoughts
            n_claims = len(state.judge_verdict.claims) if state.judge_verdict.claims else 0
            claims_richness = min(1.0, n_claims / 5.0)
            components["claims_richness"] = round(claims_richness, 4)

            # 4. Evaluative density: judge uses evaluative (not descriptive) words
            reason = state.judge_verdict.reason or ""
            reason_lower = reason.lower()
            eval_hits = sum(1 for k in _EVAL_KEYWORDS if k in reason_lower)
            eval_density = min(1.0, eval_hits / 3.0)
            components["eval_density"] = round(eval_density, 4)

            # 5. Judge divergence: judge has its own perspective (low margin_2v3)
            judge_divergence = 0.5  # default
            if (state.judge_verdict.competition
                    and state.judge_verdict.competition.margin_2v3 is not None):
                judge_divergence = 1.0 - state.judge_verdict.competition.margin_2v3
            components["judge_divergence"] = round(judge_divergence, 4)

            # HOT composite
            value = (
                0.25 * has_meta_judgment
                + 0.20 * has_counterfactual
                + 0.15 * claims_richness
                + 0.20 * eval_density
                + 0.20 * judge_divergence
            )
        else:
            # Degraded: no judge verdict
            diagnostics["degraded"] = True

            # Agent self-monitoring as first-order HOT proxy
            veto_present = any(t.veto for t in state.agent_turns)
            oppose_count = sum(1 for t in state.agent_turns
                               if t.status == "Oppose" and t.text.strip())
            meta_proxy = 0.0
            if veto_present:
                meta_proxy = 0.4
            elif oppose_count > 0:
                meta_proxy = min(0.3, oppose_count * 0.15)

            components["has_meta_judgment"] = 0.0
            components["has_counterfactual"] = 0.0
            components["claims_richness"] = 0.0
            components["eval_density"] = 0.0
            components["judge_divergence"] = 0.5
            components["meta_proxy"] = round(meta_proxy, 4)

            value = meta_proxy

        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )


# Backward compatibility
HigherOrderThought = HOTTheory
