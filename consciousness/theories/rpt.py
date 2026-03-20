"""
RPT - Recurrent Processing Theory (Lamme, 2006)
=================================================
Consciousness requires recurrent (feedback) processing, not just
feedforward sweeps. Information must flow back and forth between
processing levels.

Measures: feedback loops - corr(verdict(t), drafts(t+1)).
"""
from __future__ import annotations

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


class RPTTheory(ConsciousnessTheory):
    """Recurrent Processing Theory implementation."""

    def __init__(self):
        self._prev_winner = None
        self._prev_winner_text = ""
        self._feedback_detected = 0
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "RPT"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        self._tick_count += 1

        # 1. Feedback detection: does current draft reference previous winner?
        feedback_strength = 0.0
        if self._prev_winner_text:
            prev_words = set(self._prev_winner_text.lower().split()[:20])
            for t in state.agent_turns:
                if t.text.strip() and prev_words:
                    curr_words = set(t.text.lower().split())
                    overlap = len(prev_words & curr_words)
                    ratio = overlap / len(prev_words) if prev_words else 0.0
                    feedback_strength = max(feedback_strength, ratio)
        if feedback_strength > 0.1:
            self._feedback_detected += 1
        components["feedback_strength"] = round(feedback_strength, 4)

        # 2. Recurrence rate over history
        recurrence_rate = 0.0
        if self._tick_count > 1:
            recurrence_rate = self._feedback_detected / self._tick_count
        components["recurrence_rate"] = round(recurrence_rate, 4)

        # 3. Evidence pack usage = info feeding back into agents
        l0r_slots = state.l0r_stats.get("active_slots", 0)
        evidence_feedback = min(1.0, l0r_slots / 20.0)
        components["evidence_feedback"] = round(evidence_feedback, 4)

        # 4. Winner continuity: same agent winning = stable recurrent loop
        winner_continuity = 0.0
        if state.judge_verdict and state.judge_verdict.winner and self._prev_winner:
            if state.judge_verdict.winner == self._prev_winner:
                winner_continuity = 0.5
            else:
                winner_continuity = 0.3  # Change = new recurrent pattern
        components["winner_continuity"] = round(winner_continuity, 4)

        # 5. Oppose agents = explicit feedback mechanism
        oppose_active = sum(
            1 for t in state.agent_turns
            if t.status == "Oppose" and t.text.strip()
        )
        oppose_feedback = min(1.0, oppose_active / 2.0)
        components["oppose_feedback"] = round(oppose_feedback, 4)

        # Update for next tick
        if state.judge_verdict and state.judge_verdict.winner:
            self._prev_winner = state.judge_verdict.winner
            for t in state.agent_turns:
                aid = t.agent
                if aid == state.judge_verdict.winner:
                    self._prev_winner_text = t.text[:500]
                    break

        # Composite
        value = (
            0.30 * feedback_strength
            + 0.20 * recurrence_rate
            + 0.20 * evidence_feedback
            + 0.15 * winner_continuity
            + 0.15 * oppose_feedback
        )
        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
        )


# Backward compatibility
RecurrentProcessing = RPTTheory
