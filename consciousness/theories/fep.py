"""
FEP - Free Energy Principle (Friston, 2010)
=============================================
Consciousness relates to minimizing prediction error (surprise).
A conscious system actively updates its internal model to reduce
the gap between predictions and observations.

Measures: prediction error reduction over time, learning rate (slope),
active inference via mode.
"""
from __future__ import annotations

import math
from collections import deque

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


class FEPTheory(ConsciousnessTheory):
    """Free Energy Principle implementation with sliding window."""

    def __init__(self):
        self._pe_history: deque = deque(maxlen=20)

    @property
    def name(self) -> str:
        return "FEP"

    def compute(self, state: OrganismState) -> TheoryScore:
        components = {}
        diagnostics = {}

        pe_current = state.signals.get("prediction_error", 0.3)
        self._pe_history.append(pe_current)
        diagnostics["pe_current"] = round(pe_current, 4)

        # Not enough history for temporal analysis — return 0.0, not NaN
        if len(self._pe_history) < 2:
            diagnostics["degraded"] = True
            diagnostics["reason"] = "insufficient_history"
            return TheoryScore(
                theory=self.name,
                value=0.0,
                components={"pe_reduction": 0.0, "learning_rate": 0.0, "active_inference": 0.0},
                diagnostics=diagnostics,
            )

        # 1. Error reduction: max(0, pe(t-1) - pe(t))
        pe_prev = self._pe_history[-2]
        error_reduction = max(0.0, pe_prev - pe_current)
        components["pe_reduction"] = round(error_reduction, 4)

        # 2. Learning rate: negative slope of pe_history = system is learning
        learning_rate = 0.0
        if len(self._pe_history) >= 5:
            n = len(self._pe_history)
            xs = list(range(n))
            ys = list(self._pe_history)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs)
            if den > 1e-10:
                slope = num / den
                # Negative slope = learning (PE decreasing)
                learning_rate = max(0.0, -slope * 5.0)
                learning_rate = min(1.0, learning_rate)
                diagnostics["pe_slope"] = round(slope, 6)
        components["learning_rate"] = round(learning_rate, 4)

        # 3. Active inference: system actively changing world to reduce surprise
        mode = state.mode
        if mode in ("Implement", "Consolidate"):
            active_inference = 1.0
        elif mode == "Explore":
            active_inference = 0.5
        elif mode == "Debate":
            active_inference = 0.4
        else:
            active_inference = 0.2
        components["active_inference"] = round(active_inference, 4)

        # Composite
        value = (
            0.40 * error_reduction
            + 0.35 * learning_rate
            + 0.25 * active_inference
        )
        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )


# Backward compatibility
FreeEnergyPrinciple = FEPTheory
