"""
OrganismState - Snapshot d'un tick pour les theories de conscience
=================================================================
Dataclass immuable qui capture l'etat complet d'un tick.
Les theories de conscience operent UNIQUEMENT sur cette structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from organism.types import AgentId, AgentStatus, ControlSignals, Mode


@dataclass(frozen=True)
class AgentTurnSnapshot:
    """Snapshot immuable d'un turn agent pour les theories."""
    agent: str
    status: str
    text: str
    token_out: int
    novelty: float
    conflict: float
    cohesion: float
    impl_pressure: float
    veto: bool = False
    latency_ms: float = 0.0


@dataclass(frozen=True)
class CompetitionPattern:
    """Donnees de competition du judge (pour MDM)."""
    ranking: tuple  # (1er, 2eme, 3eme...) agent ids
    margin_1v2: float  # Marge entre 1er et 2eme
    margin_2v3: float  # Marge entre 2eme et 3eme
    counterfactual: str  # "A aurait gagne si..."


@dataclass(frozen=True)
class JudgeVerdict:
    """Verdict structure du juge (8B local)."""
    winner: Optional[str]  # AgentId value, None if judge_failed
    reason: str
    confidence: float  # 0-1
    signals: Dict[str, float] = field(default_factory=dict)
    claims: tuple = ()  # Tuple of dicts for immutability
    competition: Optional[CompetitionPattern] = None
    raw_json: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class OrganismState:
    """
    Snapshot complet d'un tick - interface pour les theories de conscience.

    Immutable (frozen) : les theories ne peuvent pas modifier l'etat.
    Agent-count agnostic : agent_turns est une liste de taille N.
    """
    tick_id: int
    mode: str
    condition: str  # "full", "no_judge", "single_agent", "random_judge"
    signals: Dict[str, float]  # 7 control signals
    agent_turns: tuple  # Tuple[AgentTurnSnapshot, ...]
    judge_verdict: Optional[JudgeVerdict] = None

    # Memory stats
    wm_stats: Dict[str, Any] = field(default_factory=dict)
    l0r_stats: Dict[str, Any] = field(default_factory=dict)

    # Theory scores (rempli apres calcul)
    theory_scores: Dict[str, float] = field(default_factory=dict)

    # Previous state (pour les theories temporelles comme FEP, DYN)
    prev_signals: Optional[Dict[str, float]] = None
    prev_mode: Optional[str] = None
    mode_changed: bool = False

    # History windows (pour DYN, RPT)
    recent_winners: tuple = ()  # Derniers N winners
    recent_margins: tuple = ()  # Derniers N margin_1v2

    @staticmethod
    def from_tick(
        tick_id,
        mode,
        mode_changed,
        signals,
        agent_turns,
        judge_verdict=None,
        wm_stats=None,
        l0r_stats=None,
        condition="full",
        prev_signals=None,
        prev_mode=None,
        recent_winners=None,
        recent_margins=None,
    ):
        """Factory depuis les objets internes de l'organisme."""
        turn_snapshots = tuple(
            AgentTurnSnapshot(
                agent=t.agent.value if hasattr(t.agent, "value") else str(t.agent),
                status=t.status.value if hasattr(t.status, "value") else str(t.status),
                text=t.text or "",
                token_out=t.token_out,
                novelty=t.novelty,
                conflict=t.conflict,
                cohesion=t.cohesion,
                impl_pressure=t.impl_pressure,
                veto=getattr(t, "veto", False),
                latency_ms=t.latency_ms,
            )
            for t in agent_turns
        )

        sig_dict = {
            "energy": signals.energy,
            "novelty": signals.novelty,
            "conflict": signals.conflict,
            "impl_pressure": signals.impl_pressure,
            "cohesion": signals.cohesion,
            "cost_pressure": signals.cost_pressure,
            "prediction_error": signals.prediction_error,
        }

        prev_sig_dict = None
        if prev_signals:
            prev_sig_dict = {
                "energy": prev_signals.energy,
                "novelty": prev_signals.novelty,
                "conflict": prev_signals.conflict,
                "impl_pressure": prev_signals.impl_pressure,
                "cohesion": prev_signals.cohesion,
                "cost_pressure": prev_signals.cost_pressure,
                "prediction_error": prev_signals.prediction_error,
            }

        return OrganismState(
            tick_id=tick_id,
            mode=mode.value if hasattr(mode, "value") else str(mode),
            condition=condition,
            signals=sig_dict,
            agent_turns=turn_snapshots,
            judge_verdict=judge_verdict,
            wm_stats=wm_stats or {},
            l0r_stats=l0r_stats or {},
            prev_signals=prev_sig_dict,
            prev_mode=prev_mode.value if prev_mode and hasattr(prev_mode, "value") else (str(prev_mode) if prev_mode else None),
            mode_changed=mode_changed,
            recent_winners=tuple(recent_winners or []),
            recent_margins=tuple(recent_margins or []),
        )
