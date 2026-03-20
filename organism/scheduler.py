"""
P2 — Scheduler : modes globaux + status agents
================================================
Transitions non hardcodées via softmax sur scores de mode
+ hystérésis + dwell time minimum.

Le scheduler ne "pense" pas : il mesure, calcule, sélectionne.

Usage:
    sched = Scheduler()
    sched.update_signals(ControlSignals(energy=0.8, novelty=0.7, conflict=0.2))
    mode, changed = sched.tick()
    params = sched.get_agent_params(AgentId.A)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from organism.types import (
    Mode, AgentId, AgentStatus, ControlSignals, AgentParams,
)


# ── Configuration par mode ────────────────────────────────────────

# Agent status par mode : {mode: {agent: status}}
MODE_AGENT_STATUS: Dict[Mode, Dict[AgentId, AgentStatus]] = {
    Mode.IDLE: {
        AgentId.A: AgentStatus.SUPPORT,
        AgentId.B: AgentStatus.SUPPORT,
        AgentId.C: AgentStatus.SUPPORT,
    },
    Mode.EXPLORE: {
        AgentId.A: AgentStatus.LEAD,
        AgentId.B: AgentStatus.SUPPORT,
        AgentId.C: AgentStatus.OPPOSE,
    },
    Mode.DEBATE: {
        AgentId.A: AgentStatus.SUPPORT,
        AgentId.B: AgentStatus.LEAD,
        AgentId.C: AgentStatus.OPPOSE,
    },
    Mode.IMPLEMENT: {
        AgentId.A: AgentStatus.OPPOSE,
        AgentId.B: AgentStatus.SUPPORT,
        AgentId.C: AgentStatus.LEAD,
    },
    Mode.CONSOLIDATE: {
        AgentId.A: AgentStatus.SUPPORT,
        AgentId.B: AgentStatus.SUPPORT,
        AgentId.C: AgentStatus.SUPPORT,
    },
    Mode.RECOVER: {
        AgentId.A: AgentStatus.OPPOSE,
        AgentId.B: AgentStatus.OPPOSE,
        AgentId.C: AgentStatus.OPPOSE,
    },
}

# Paramètres opérationnels par status
STATUS_PARAMS: Dict[AgentStatus, Dict[str, Any]] = {
    AgentStatus.LEAD: {
        "call_weight": 1.0,
        "token_budget": 4096,
        "can_veto": True,
        "must_output_schema": False,
    },
    AgentStatus.SUPPORT: {
        "call_weight": 0.5,
        "token_budget": 2048,
        "can_veto": False,
        "must_output_schema": False,
    },
    AgentStatus.OPPOSE: {
        "call_weight": 0.25,
        "token_budget": 1024,
        "can_veto": True,
        "must_output_schema": True,
    },
}

# Dwell time minimum par mode (en ticks)
MODE_DWELL_MIN: Dict[Mode, int] = {
    Mode.IDLE: 1,
    Mode.EXPLORE: 5,
    Mode.DEBATE: 5,
    Mode.IMPLEMENT: 8,
    Mode.CONSOLIDATE: 4,
    Mode.RECOVER: 3,
}


# ── Scoring : comment chaque signal influence chaque mode ─────────

def _score_idle(s: ControlSignals) -> float:
    """Idle favorisé quand énergie basse, pas de stimulus."""
    return (
        0.5 * (1.0 - s.energy)
        + 0.2 * (1.0 - s.novelty)
        + 0.2 * (1.0 - s.conflict)
        + 0.1 * (1.0 - s.impl_pressure)
    )


def _score_explore(s: ControlSignals) -> float:
    """Explore favorisé quand nouveauté haute, énergie disponible."""
    return (
        0.4 * s.novelty
        + 0.25 * s.energy
        + 0.15 * s.prediction_error
        + 0.1 * (1.0 - s.conflict)
        + 0.1 * (1.0 - s.cost_pressure)
    )


def _score_debate(s: ControlSignals) -> float:
    """Debate favorisé quand conflit haut, cohésion basse."""
    return (
        0.4 * s.conflict
        + 0.3 * (1.0 - s.cohesion)
        + 0.15 * s.energy
        + 0.15 * s.prediction_error
    )


def _score_implement(s: ControlSignals) -> float:
    """Implement favorisé quand pression d'implémentation haute, cohésion bonne."""
    return (
        0.4 * s.impl_pressure
        + 0.25 * s.cohesion
        + 0.2 * s.energy
        + 0.15 * (1.0 - s.conflict)
    )


def _score_consolidate(s: ControlSignals) -> float:
    """Consolidate favorisé après activité intense, besoin de stabiliser."""
    return (
        0.3 * (1.0 - s.cohesion)
        + 0.25 * s.cost_pressure
        + 0.2 * (1.0 - s.novelty)
        + 0.15 * (1.0 - s.energy)
        + 0.1 * s.prediction_error
    )


def _score_recover(s: ControlSignals) -> float:
    """Recover favorisé quand énergie très basse ou erreurs hautes."""
    return (
        0.5 * (1.0 - s.energy)
        + 0.3 * s.prediction_error
        + 0.2 * s.cost_pressure
    )


MODE_SCORERS = {
    Mode.IDLE: _score_idle,
    Mode.EXPLORE: _score_explore,
    Mode.DEBATE: _score_debate,
    Mode.IMPLEMENT: _score_implement,
    Mode.CONSOLIDATE: _score_consolidate,
    Mode.RECOVER: _score_recover,
}


# ── Softmax ───────────────────────────────────────────────────────

def softmax(scores: Dict[Mode, float], temperature: float = 1.0) -> Dict[Mode, float]:
    """Softmax sur les scores de mode. temperature contrôle la netteté."""
    if temperature <= 0:
        temperature = 0.01
    max_score = max(scores.values())
    exps = {m: math.exp((s - max_score) / temperature) for m, s in scores.items()}
    total = sum(exps.values())
    return {m: e / total for m, e in exps.items()}


# ── Scheduler ─────────────────────────────────────────────────────

@dataclass
class ModeTransition:
    """Enregistre une transition de mode."""
    from_mode: Mode
    to_mode: Mode
    tick: int
    scores: Dict[str, float]
    reason: str


class Scheduler:
    """
    Scheduler de l'organisme. Gère les modes globaux et les status agents.

    Pas de hardcoding de séquences : les transitions émergent des signaux
    via softmax + hystérésis + dwell time.
    """

    # Fix E : budget tokens réduit en Idle (output cap)
    # 600 = marge pour thinking (~400) + contenu visible (~200)
    IDLE_TOKEN_BUDGET: int = 600

    def __init__(
        self,
        initial_mode: Mode = Mode.IDLE,
        hysteresis: float = 0.15,
        softmax_temperature: float = 0.5,
        max_idle_dwell: int = 8,
    ):
        self.current_mode: Mode = initial_mode
        self.hysteresis: float = hysteresis
        self.temperature: float = softmax_temperature
        self._max_idle_dwell: int = max_idle_dwell  # Fix D

        self.signals: ControlSignals = ControlSignals(
            energy=1.0,
            novelty=0.5,       # Était 0.0 → le système démarrait en Idle
            prediction_error=0.3,
        )
        self._tick_count: int = 0
        self._dwell_ticks: int = 0  # Ticks passés dans le mode courant
        self._history: List[ModeTransition] = []
        self._veto_boost: float = 0.0  # Decaying boost pour Debate après veto

    def register_veto(self) -> None:
        """Signal qu'un veto a été posé ce tick. Booste Debate sur plusieurs ticks (decay)."""
        self._veto_boost = 0.5  # Initial boost, decays 40% per tick

    def update_signals(self, signals: ControlSignals) -> None:
        """Met à jour les signaux de contrôle."""
        self.signals = signals

    def tick(self) -> Tuple[Mode, bool]:
        """
        Avance d'un tick. Évalue si un changement de mode est nécessaire.

        Returns:
            (mode_courant, mode_a_changé)
        """
        self._tick_count += 1
        self._dwell_ticks += 1

        # Calculer les scores bruts (includes veto boost if active)
        raw_scores = self._compute_raw_scores()
        # Decay veto boost each tick (0.5 → 0.3 → 0.18 → 0.11 → negligible)
        self._veto_boost *= 0.6

        # Appliquer hystérésis : bonus au mode courant
        scores_with_hyst = dict(raw_scores)
        scores_with_hyst[self.current_mode] += self.hysteresis

        # Mode fatigue : pénalité croissante si on dépasse dwell_min × 3
        max_dwell = MODE_DWELL_MIN.get(self.current_mode, 3) * 3
        if self._dwell_ticks > max_dwell:
            excess = self._dwell_ticks - max_dwell
            fatigue = 0.05 * excess
            scores_with_hyst[self.current_mode] -= fatigue
            if fatigue > 0.1:
                import logging as _log
                _log.getLogger("organism.scheduler").info(
                    "Mode fatigue: %s dwell=%d penalty=%.2f",
                    self.current_mode.value, self._dwell_ticks, fatigue)

        # Appliquer softmax
        probs = softmax(scores_with_hyst, self.temperature)

        # Sélection : argmax (déterministe)
        best_mode = max(probs, key=probs.get)  # type: ignore[arg-type]

        # Vérifier dwell time
        dwell_min = MODE_DWELL_MIN.get(self.current_mode, 3)
        if best_mode != self.current_mode and self._dwell_ticks <= dwell_min:
            # Pas encore assez de ticks dans le mode courant
            return self.current_mode, False

        # Transition ?
        if best_mode != self.current_mode:
            transition = ModeTransition(
                from_mode=self.current_mode,
                to_mode=best_mode,
                tick=self._tick_count,
                scores={m.value: round(s, 4) for m, s in raw_scores.items()},
                reason=f"softmax_argmax (dwell={self._dwell_ticks})",
            )
            self._history.append(transition)
            self.current_mode = best_mode
            self._dwell_ticks = 0
            return best_mode, True

        return self.current_mode, False

    def get_agent_params(self, agent: AgentId) -> AgentParams:
        """Retourne les paramètres opérationnels d'un agent selon le mode courant."""
        if agent == AgentId.O:
            # O est toujours Lead, budget max, veto
            return AgentParams(
                agent=AgentId.O,
                status=AgentStatus.LEAD,
                call_weight=1.0,
                token_budget=8192,
                can_veto=True,
                must_output_schema=False,
            )

        status_map = MODE_AGENT_STATUS.get(self.current_mode, {})
        status = status_map.get(agent, AgentStatus.SUPPORT)
        params = STATUS_PARAMS.get(status, STATUS_PARAMS[AgentStatus.SUPPORT])

        result = AgentParams(
            agent=agent,
            status=status,
            call_weight=params["call_weight"],
            token_budget=params["token_budget"],
            can_veto=params["can_veto"],
            must_output_schema=params["must_output_schema"],
        )
        # Fix E : Idle output cap — budget tokens réduit
        if self.current_mode == Mode.IDLE:
            result.token_budget = min(result.token_budget, self.IDLE_TOKEN_BUDGET)
        return result

    def get_all_agent_params(self) -> Dict[AgentId, AgentParams]:
        """Retourne les paramètres de tous les agents."""
        return {a: self.get_agent_params(a) for a in AgentId}

    def get_mode_scores(self) -> Dict[Mode, float]:
        """Retourne les scores bruts actuels de chaque mode."""
        return self._compute_raw_scores()

    def get_mode_probabilities(self) -> Dict[Mode, float]:
        """Retourne les probabilités softmax actuelles (avec hystérésis + fatigue)."""
        raw = self._compute_raw_scores()
        raw[self.current_mode] += self.hysteresis
        max_dwell = MODE_DWELL_MIN.get(self.current_mode, 3) * 3
        if self._dwell_ticks > max_dwell:
            raw[self.current_mode] -= 0.05 * (self._dwell_ticks - max_dwell)
        return softmax(raw, self.temperature)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def dwell_ticks(self) -> int:
        return self._dwell_ticks

    @property
    def history(self) -> List[ModeTransition]:
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du scheduler."""
        return {
            "current_mode": self.current_mode.value,
            "tick_count": self._tick_count,
            "dwell_ticks": self._dwell_ticks,
            "transitions": len(self._history),
            "signals": {
                "energy": self.signals.energy,
                "novelty": self.signals.novelty,
                "conflict": self.signals.conflict,
                "impl_pressure": self.signals.impl_pressure,
                "cohesion": self.signals.cohesion,
                "cost_pressure": self.signals.cost_pressure,
                "prediction_error": self.signals.prediction_error,
            },
        }

    def _compute_raw_scores(self) -> Dict[Mode, float]:
        """Calcule le score brut de chaque mode à partir des signaux."""
        scores = {mode: scorer(self.signals) for mode, scorer in MODE_SCORERS.items()}
        # Fix D : Idle fatigue — pénalise Idle si on y reste trop longtemps
        # Sauf si cost_pressure élevée (on économise volontairement)
        if (self.current_mode == Mode.IDLE
                and self._dwell_ticks > self._max_idle_dwell
                and self.signals.cost_pressure < 0.8):
            excess = self._dwell_ticks - self._max_idle_dwell
            scores[Mode.IDLE] -= 0.05 * excess
            scores[Mode.EXPLORE] += 0.02 * excess
        # Veto boost: a veto is a strong conflict signal → push toward Debate
        # Decaying boost: persists over ~3-4 ticks after each veto
        if self._veto_boost > 0.05:
            scores[Mode.DEBATE] += self._veto_boost
        return scores
