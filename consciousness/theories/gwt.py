"""
GWT - Global Workspace Theory (Baars, 1988; Dehaene et al., 2003)
==================================================================
Consciousness = global broadcast. Information becomes conscious when
it is broadcast widely across processing modules.

Measures: broadcast to WM (claims this tick), memory (L0R), mode change,
agent participation.
"""
from __future__ import annotations

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


class GWTTheory(ConsciousnessTheory):
    """Global Workspace Theory implementation."""

    @property
    def name(self) -> str:
        return "GWT"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}

        # 1. Broadcast to WM: claims added THIS tick (not cumulative)
        wm_churn = state.wm_stats.get("claims_added_this_tick", 0)
        if wm_churn is None:
            wm_churn = 0
        broadcast_wm = min(1.0, wm_churn / 20.0)
        components["broadcast_wm"] = round(broadcast_wm, 4)

        # 2. Broadcast to memory: L0R active slots (proxy for memory integration)
        l0r_slots = state.l0r_stats.get("active_slots", 0)
        broadcast_memory = min(1.0, l0r_slots / 32.0)
        components["broadcast_memory"] = round(broadcast_memory, 4)

        # 3. Mode transition: broadcast changed the global state
        broadcast_mode = 1.0 if state.mode_changed else 0.0
        components["broadcast_mode"] = round(broadcast_mode, 4)

        # 4. Agent participation: all agents active = broad broadcast
        n_active = sum(1 for t in state.agent_turns if t.text.strip())
        n_total = max(len(state.agent_turns), 1)
        broadcast_agents = n_active / n_total
        components["broadcast_agents"] = round(broadcast_agents, 4)

        # 5. Winner broadcast: judge selected a winner = content entered workspace
        winner_broadcast = 0.0
        if state.judge_verdict and state.judge_verdict.winner:
            winner_broadcast = state.judge_verdict.confidence
        else:
            diagnostics["degraded"] = True
        components["winner_broadcast"] = round(winner_broadcast, 4)

        # Composite
        value = (
            0.35 * broadcast_wm
            + 0.20 * broadcast_memory
            + 0.15 * broadcast_mode
            + 0.15 * broadcast_agents
            + 0.15 * winner_broadcast
        )
        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )


# Backward compatibility alias
GlobalWorkspace = GWTTheory
