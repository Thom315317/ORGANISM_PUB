"""
DYN - Dynamical Systems / Neural Synchrony
============================================
Consciousness correlates with temporal coherence between processing
units. When agents synchronize (similar signal trajectories over time),
the system exhibits higher dynamical complexity.

Measures: Pearson correlation between agent signal time series on
a sliding window. Returns NaN if < 3 ticks of history.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient. Returns 0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-8 or dy < 1e-8:
        return 0.0
    return num / (dx * dy)


class DYNTheory(ConsciousnessTheory):
    """Dynamical synchrony implementation with Pearson on sliding window."""

    def __init__(self):
        self._agent_history: Dict[str, deque] = {}  # agent -> deque of (nov, conf, coh, imp)
        self._window = 20
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "DYN"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}
        self._tick_count += 1

        # Record current tick signals per agent
        for t in state.agent_turns:
            if t.agent not in self._agent_history:
                self._agent_history[t.agent] = deque(maxlen=self._window)
            self._agent_history[t.agent].append(
                (t.novelty, t.conflict, t.cohesion, t.impl_pressure)
            )

        # Guard: need at least 3 ticks of history — return 0.0, not NaN
        min_len = min(
            (len(h) for h in self._agent_history.values()),
            default=0,
        )
        if min_len < 3:
            diagnostics["degraded"] = True
            diagnostics["reason"] = "insufficient_history"
            return TheoryScore(
                theory=self.name,
                value=0.0,
                components={
                    "pairwise_synchrony": 0.0,
                    "mode_coherence": 0.0,
                    "signal_complexity": 0.0,
                },
                diagnostics=diagnostics,
            )

        # 1. Pairwise Pearson synchrony across agents x signal dimensions
        agent_ids = list(self._agent_history.keys())
        synchronies: List[float] = []
        window_size = min(min_len, self._window)
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                hi = list(self._agent_history[agent_ids[i]])[-window_size:]
                hj = list(self._agent_history[agent_ids[j]])[-window_size:]
                for dim in range(4):  # novelty, conflict, cohesion, impl_pressure
                    si = [h[dim] for h in hi]
                    sj = [h[dim] for h in hj]
                    r = _pearson(si, sj)
                    synchronies.append(abs(r))  # abs: anti-corr = synchrony too

        pairwise_synchrony = sum(synchronies) / len(synchronies) if synchronies else 0.0
        components["pairwise_synchrony"] = round(pairwise_synchrony, 4)
        diagnostics["n_pairs_x_dims"] = len(synchronies)

        # 2. Mode coherence: not changing mode = phase locking
        mode_coherence = 0.6 if not state.mode_changed else 0.0
        components["mode_coherence"] = round(mode_coherence, 4)

        # 3. Signal complexity: entropy of agent signals (not too uniform, not too random)
        signal_complexity = 0.0
        all_sigs: List[float] = []
        for t in state.agent_turns:
            all_sigs.extend([t.novelty, t.conflict, t.cohesion, t.impl_pressure])
        if all_sigs:
            bins = [0] * 5
            for s in all_sigs:
                idx = min(4, int(s * 5))
                bins[idx] += 1
            total = len(all_sigs)
            entropy = 0.0
            for b in bins:
                if b > 0:
                    p = b / total
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(5)
            signal_complexity = entropy / max_entropy if max_entropy > 0 else 0.0
        components["signal_complexity"] = round(signal_complexity, 4)

        # Composite
        value = (
            0.50 * pairwise_synchrony
            + 0.20 * mode_coherence
            + 0.30 * signal_complexity
        )
        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )


# Backward compatibility
DynamicalSynchrony = DYNTheory
