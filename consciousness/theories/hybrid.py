"""
HYBRID - Adaptive weighted combination of 7 other theories
============================================================
Weights learned by correlation with output quality.
Initial weights uniform. Updated every 10 ticks after 20+ history.
Returns NaN before 20 ticks (not enough data for weight learning).
"""
from __future__ import annotations

import math
from typing import Dict, List

from consciousness.theories.base import ConsciousnessTheory, TheoryScore
from organism.organism_state import OrganismState


class HybridTheory(ConsciousnessTheory):
    """Adaptive hybrid: weighted mean of other theories."""

    def __init__(self):
        self._theory_names = ["MDM", "GWT", "HOT", "FEP", "IIT", "DYN", "RPT"]
        n = len(self._theory_names)
        self._weights = {name: 1.0 / n for name in self._theory_names}
        self._lr = 0.02
        self._score_history: List[Dict[str, float]] = []
        self._quality_history: List[float] = []
        self._tick_count = 0

    @property
    def name(self) -> str:
        return "Hybrid"

    def compute(self, state: OrganismState) -> TheoryScore:
        self._tick_count += 1
        components = {}
        diagnostics = {}
        other_scores = state.theory_scores  # May be empty on first call

        # Compute quality from available data
        quality = self._compute_quality(state)

        # Record history
        self._score_history.append(dict(other_scores))
        self._quality_history.append(quality)

        # Keep only last 100
        if len(self._quality_history) > 100:
            self._score_history = self._score_history[-100:]
            self._quality_history = self._quality_history[-100:]

        # Before 20 ticks: use uniform weights (no learned adaptation yet)
        if self._tick_count < 20:
            diagnostics["degraded"] = True
            diagnostics["reason"] = "insufficient_history_uniform_weights"

        # Update weights every 10 ticks
        if self._tick_count % 10 == 0 and len(self._quality_history) >= 20:
            self._update_weights()

        # Compute weighted average, excluding NaN scores
        weighted_sum = 0.0
        total_weight = 0.0
        for theory_name in self._theory_names:
            score = other_scores.get(theory_name, 0.0)
            if isinstance(score, float) and math.isnan(score):
                continue  # Skip NaN theories
            w = self._weights.get(theory_name, 0.0)
            weighted_sum += w * score
            total_weight += w
            components[f"w_{theory_name}"] = round(w, 4)
            components[f"s_{theory_name}"] = round(score, 4)

        if total_weight > 0:
            value = weighted_sum / total_weight
        else:
            diagnostics["degraded"] = True
            diagnostics["reason"] = "no_valid_scores"
            return TheoryScore(
                theory=self.name,
                value=0.0,
                components=components,
                diagnostics=diagnostics,
            )

        value = max(0.0, min(1.0, value))

        return TheoryScore(
            theory=self.name,
            value=round(value, 4),
            components=components,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _compute_quality(state: OrganismState) -> float:
        """Multi-criteria quality proxy from available data."""
        # WM confidence: how confident are the claims
        wm_conf = state.wm_stats.get("avg_confidence", 0.5)

        # Judge confidence (when available)
        judge_conf = 0.5
        if state.judge_verdict:
            judge_conf = state.judge_verdict.confidence

        # Agent participation: more active agents = richer output
        n_active = sum(1 for t in state.agent_turns if t.text.strip())
        participation = min(1.0, n_active / 3.0)

        # Moderate conflict = productive debate (not too much, not too little)
        conflict = state.signals.get("conflict", 0.5)
        conflict_quality = 1.0 - abs(conflict - 0.5) * 2.0

        return (
            0.30 * wm_conf
            + 0.25 * judge_conf
            + 0.25 * participation
            + 0.20 * conflict_quality
        )

    def _update_weights(self):
        """Update weights: theories correlated with quality get more weight."""
        mean_q = sum(self._quality_history) / len(self._quality_history)

        for theory_name in self._theory_names:
            theory_vals = [
                h.get(theory_name, 0.0) for h in self._score_history
            ]
            # Skip if theory was mostly NaN
            valid = [v for v in theory_vals if not (isinstance(v, float) and math.isnan(v))]
            if len(valid) < 10:
                continue

            mean_t = sum(valid) / len(valid)

            # Pearson numerator (on valid entries only)
            pairs = [
                (t, q) for t, q in zip(theory_vals, self._quality_history)
                if not (isinstance(t, float) and math.isnan(t))
            ]
            if len(pairs) < 10:
                continue

            num = sum((t - mean_t) * (q - mean_q) for t, q in pairs)
            den_t = math.sqrt(sum((t - mean_t) ** 2 for t, _ in pairs))
            den_q = math.sqrt(sum((q - mean_q) ** 2 for _, q in pairs))

            if den_t > 0.01 and den_q > 0.01:
                corr = num / (den_t * den_q)
                self._weights[theory_name] *= (1.0 + 0.1 * corr)

        # Normalize
        total = sum(self._weights.values())
        if total > 0:
            for k in self._weights:
                self._weights[k] /= total


# Backward compatibility
AdaptiveHybrid = HybridTheory
