"""
Types partagés pour l'organisme CRISTAL.
Dataclasses et enums utilisés par Mr (P0), L0R (P1), Scheduler (P2) et WorldModel (P3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class EventType(str, Enum):
    """Types d'événements dans Mr."""
    AGENT_MESSAGE = "agent_message"
    TICK_START = "tick_start"
    TICK_END = "tick_end"
    MODE_CHANGE = "mode_change"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    PREDICTION_MADE = "prediction_made"
    PREDICTION_CHECKED = "prediction_checked"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CLAIM_ADDED = "claim_added"
    CLAIM_REVISED = "claim_revised"
    CLAIM_CONTRADICTED = "claim_contradicted"


class AgentId(str, Enum):
    """Identifiants d'agents."""
    A = "A"  # Explorer (ex creative_sampler)
    B = "B"  # Critic (ex safety_controller)
    C = "C"  # Builder (nouveau)
    O = "O"  # Orchestrator


@dataclass
class MrEvent:
    """
    Un événement dans la mémoire de réalité (Mr).
    Chaque ligne du JSONL correspond à un MrEvent.
    Convention: chunk_id = "mr:" + hash[:16].
    """
    event_id: str
    ts: float
    type: EventType
    tick_id: int
    agent: AgentId
    mode: str

    # Métriques LLM
    token_in: int = 0
    token_out: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0

    # Signaux cognitifs (0.0–1.0)
    novelty: float = 0.0
    conflict: float = 0.0
    cohesion: float = 0.0
    impl_pressure: float = 0.0
    prediction_error: float = 0.0

    # Hash chain
    chunk_id: str = ""
    prev_hash: str = ""
    hash: str = ""

    # Payload optionnel
    payload: Optional[Dict[str, Any]] = None

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Dict trié par clés pour hachage canonique déterministe.
        Exclut 'hash' pour que le hash soit calculable."""
        d: Dict[str, Any] = {
            "event_id": self.event_id,
            "ts": self.ts,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "tick_id": self.tick_id,
            "agent": self.agent.value if isinstance(self.agent, AgentId) else self.agent,
            "mode": self.mode,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "novelty": self.novelty,
            "conflict": self.conflict,
            "cohesion": self.cohesion,
            "impl_pressure": self.impl_pressure,
            "prediction_error": self.prediction_error,
            "chunk_id": self.chunk_id,
            "prev_hash": self.prev_hash,
        }
        if self.payload is not None:
            d["payload"] = self.payload
        return dict(sorted(d.items()))

    def to_line_dict(self) -> Dict[str, Any]:
        """Dict complet pour sérialisation JSONL (inclut hash)."""
        d = self.to_canonical_dict()
        d["hash"] = self.hash
        return d


@dataclass
class L0RSlot:
    """
    Un slot dans le ring buffer L0R.
    Pointe vers un chunk_id dans Mr (pas le texte complet).
    """
    chunk_id: str
    salience: float = 0.5
    novelty: float = 0.0
    conflict: float = 0.0
    ttl: int = 10
    reuse_count: int = 0
    inserted_at: float = field(default_factory=time.time)

    def composite_score(self) -> float:
        """Score composite pour le classement dans l'evidence pack."""
        return (
            0.4 * self.salience
            + 0.3 * self.novelty
            + 0.2 * self.conflict
            + 0.1 * min(self.reuse_count / 5.0, 1.0)
        )


# ── P2 : Scheduler types ─────────────────────────────────────────


class Mode(str, Enum):
    """Modes globaux de l'organisme (6 max)."""
    IDLE = "Idle"
    EXPLORE = "Explore"
    DEBATE = "Debate"
    IMPLEMENT = "Implement"
    CONSOLIDATE = "Consolidate"
    RECOVER = "Recover"


class AgentStatus(str, Enum):
    """Rôle opérationnel d'un agent dans le mode courant."""
    LEAD = "Lead"
    SUPPORT = "Support"
    OPPOSE = "Oppose"


@dataclass
class ControlSignals:
    """
    Les 7 signaux de contrôle du scheduler.
    Mis à jour à chaque tick par l'orchestrateur.
    """
    energy: float = 1.0             # 0..1  réserve énergétique
    novelty: float = 0.0            # 0..1  nouveauté détectée
    conflict: float = 0.0           # 0..1  contradictions actives
    impl_pressure: float = 0.0     # 0..1  pression d'implémentation
    cohesion: float = 1.0           # 0..1  cohérence interne (1=aucune contradiction)
    cost_pressure: float = 0.0     # 0..1  pression coût (tokens/min, latence)
    prediction_error: float = 0.0  # 0..1  écart projections vs observations


@dataclass
class AgentParams:
    """
    Paramètres opérationnels d'un agent, déterminés par son status.
    Pas de texte/roleplay — uniquement des valeurs numériques.
    """
    agent: AgentId
    status: AgentStatus
    call_weight: float = 1.0       # Probabilité relative d'appel
    token_budget: int = 2048       # Budget tokens max
    can_veto: bool = False         # Droit de veto
    must_output_schema: bool = False  # Format structuré obligatoire


# ── P3 : World Model types ───────────────────────────────────────


class ClaimStatus(str, Enum):
    """Statut d'une claim dans le world model."""
    HYPOTHESIS = "Hypothesis"    # Pas encore validée
    SUPPORTED = "Supported"      # Renforcée par des évidences
    CONTRADICTED = "Contradicted" # Contredite par des évidences
    RETRACTED = "Retracted"      # Retirée (plus active)


class ClaimRelation(str, Enum):
    """Types de liens entre claims."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


@dataclass
class ClaimLink:
    """Lien orienté entre deux claims."""
    relation: ClaimRelation
    target_claim_id: str
    source_chunk_id: str = ""    # chunk_id Mᵣ qui justifie ce lien
    created_at: float = field(default_factory=time.time)


@dataclass
class Claim:
    """
    Une claim dans le world model.
    Provenance obligatoire : sans chunk_ids valides, ne peut pas être un 'fait'.
    """
    claim_id: str
    content: str
    confidence: float                        # 0..1
    provenance: List[str] = field(default_factory=list)  # chunk_ids Mᵣ
    source_agent: AgentId = AgentId.O
    status: ClaimStatus = ClaimStatus.HYPOTHESIS
    created_at: float = field(default_factory=time.time)
    revised_at: float = 0.0
    links: List[ClaimLink] = field(default_factory=list)

    def has_provenance(self) -> bool:
        """Vérifie si la claim a au moins une provenance valide."""
        return len(self.provenance) > 0

    def is_fact(self, min_confidence: float = 0.7) -> bool:
        """Une claim est un 'fait' ssi provenance + confidence suffisante + pas contredite."""
        return (
            self.has_provenance()
            and self.confidence >= min_confidence
            and self.status not in (ClaimStatus.CONTRADICTED, ClaimStatus.RETRACTED)
        )


@dataclass
class Prediction:
    """
    Une prédiction faite par un agent, liée à une claim.
    Cycle: prediction_made → prediction_checked → confidence update.
    """
    prediction_id: str
    claim_id: str
    predicted_by: AgentId
    prediction: str                  # Ce qui est attendu
    predicted_at: float = field(default_factory=time.time)
    checked: bool = False
    outcome: Optional[bool] = None   # True=correct, False=wrong, None=pas encore vérifié
    checked_at: Optional[float] = None
    confidence_delta: float = 0.0    # Variation de confidence appliquée
