"""
IIT - Integrated Information Theory (Tononi, 2004/2008)
========================================================
Consciousness = integrated information (Phi). A system is conscious
to the degree that it is both differentiated and integrated.

This is a TOY proxy: we compute correlation-based integration on the
agent/judge/memory dependency graph, not full IIT 4.0.

IMPORTANT: IIT must NOT return 0.0 when judge_verdict is None.
Agent signals and WM are always available.
"""
from __future__ import annotations

import math

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


class IITTheory(ConsciousnessTheory):
    """IIT proxy implementation (toy Phi)."""

    @property
    def name(self) -> str:
        return "IIT"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}

        agents = [t for t in state.agent_turns if t.text.strip()]
        n_agents = len(agents)
        diagnostics["n_active_agents"] = n_agents

        # 1. Integration between agents: cosine similarity of signal vectors
        integration_agents = 0.0
        if n_agents >= 2:
            correlations = []
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    vi = (agents[i].novelty, agents[i].conflict,
                          agents[i].cohesion, agents[i].impl_pressure)
                    vj = (agents[j].novelty, agents[j].conflict,
                          agents[j].cohesion, agents[j].impl_pressure)
                    dot = sum(a * b for a, b in zip(vi, vj))
                    ni = math.sqrt(sum(a * a for a in vi)) + 1e-8
                    nj = math.sqrt(sum(a * a for a in vj)) + 1e-8
                    correlations.append(dot / (ni * nj))
            integration_agents = sum(correlations) / len(correlations)
            integration_agents = (integration_agents + 1.0) / 2.0  # normalize to [0, 1]
        components["integration_agents"] = round(integration_agents, 4)

        # 2. Signal-mode alignment: does the mode match the dominant signal?
        _MODE_EXPECTS = {
            "Explore": "novelty",
            "Debate": "conflict",
            "Implement": "impl_pressure",
            "Consolidate": "cohesion",
        }
        expected_signal = _MODE_EXPECTS.get(state.mode)
        if expected_signal and expected_signal in state.signals:
            signal_mode_alignment = state.signals[expected_signal]
        else:
            signal_mode_alignment = 0.3  # neutral prior
        components["signal_mode_alignment"] = round(signal_mode_alignment, 4)

        # 3. WM integration: contradictions = WM relates claims to each other
        total_claims = state.wm_stats.get("total_claims", 0)
        if total_claims > 0:
            contradiction_ratio = state.wm_stats.get("contradicted", 0) / total_claims
            avg_conf = state.wm_stats.get("avg_confidence", 0.0)
            wm_integration = min(1.0, contradiction_ratio * 10 + avg_conf)
        else:
            wm_integration = 0.0
        components["wm_integration"] = round(wm_integration, 4)

        # 4. Judge integration bonus (when available)
        judge_bonus = 0.0
        if state.judge_verdict:
            judge_bonus = 0.15
            if state.judge_verdict.competition:
                n_ranked = len(state.judge_verdict.competition.ranking)
                judge_bonus += min(0.15, n_ranked * 0.05)
            if state.judge_verdict.signals:
                judge_bonus += 0.1
        else:
            diagnostics["degraded"] = True
        judge_bonus = min(1.0, judge_bonus)
        components["judge_bonus"] = round(judge_bonus, 4)

        # Phi proxy composite
        value = (
            0.35 * integration_agents
            + 0.25 * signal_mode_alignment
            + 0.20 * wm_integration
            + 0.20 * judge_bonus
        )
        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )


# Backward compatibility
IITImplementation = IITTheory
