"""
MDM - Multiple Drafts Model (Dennett, 1991)
=============================================
Consciousness arises from the competition between multiple parallel
drafts of content. The winning draft becomes "conscious" through
selection, not through entering a special medium.

Key signals:
- competition_intensity: how close the drafts were (1 - margin_1v2)
- draft_diversity: how different the drafts are (1 - jaccard on 3-grams)
- selection_pressure: confidence * (1 - margin_1v2)

Score = weighted mean of components.
"""
from __future__ import annotations

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


def _ngram_set(text: str, n: int = 3) -> set:
    """Build set of word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return set(tuple(words[i:i + 1]) for i in range(len(words))) if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard_3gram(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word 3-grams."""
    if not text_a or not text_b:
        return 0.0
    sa = _ngram_set(text_a)
    sb = _ngram_set(text_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class MDMTheory(ConsciousnessTheory):
    """Multiple Drafts Model implementation."""

    @property
    def name(self) -> str:
        return "MDM"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}
        degraded = False

        # 1. Competition intensity from judge
        margin_1v2 = 0.5  # default
        if state.judge_verdict and state.judge_verdict.competition:
            margin_1v2 = state.judge_verdict.competition.margin_1v2
        else:
            degraded = True
        competition_intensity = 1.0 - margin_1v2
        components["competition_intensity"] = round(competition_intensity, 4)

        # 2. Draft diversity (1 - mean jaccard 3-gram between all pairs)
        texts = [t.text for t in state.agent_turns if t.text.strip()]
        if len(texts) >= 2:
            jaccards = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    jaccards.append(_jaccard_3gram(texts[i], texts[j]))
            mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
            draft_diversity = 1.0 - mean_jaccard
            diagnostics["mean_jaccard_3gram"] = round(mean_jaccard, 4)
            diagnostics["n_pairs"] = len(jaccards)
        else:
            draft_diversity = 0.0
        components["draft_diversity"] = round(draft_diversity, 4)

        # 3. Selection pressure
        confidence = 0.5
        if state.judge_verdict:
            confidence = state.judge_verdict.confidence
        else:
            degraded = True
        selection_pressure = confidence * (1.0 - margin_1v2)
        components["selection_pressure"] = round(selection_pressure, 4)

        # Weighted composite (w1=0.4, w2=0.3, w3=0.3)
        value = (
            0.40 * competition_intensity
            + 0.30 * draft_diversity
            + 0.30 * selection_pressure
        )
        value = max(0.0, min(1.0, value))

        if degraded:
            diagnostics["degraded"] = True

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )
