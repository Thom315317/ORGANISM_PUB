"""
CRISTAL Organism
=================
Remplacement progressif du pipeline CRISTAL.
P0: Mémoire de réalité (Mr) — journal d'événements avec chaîne de hachage
P1: L0R Ring Pointer Memory — mémoire de travail avec pack d'évidences
P2: Scheduler — modes globaux + status agents (softmax + hystérésis + dwell)
P3: World Model (Mʷ) — graphe de claims + provenance + prédictions
P4: Orchestrator — boucle interne multi-agents
P5: Agent Wrapper — connexion aux LLMs Ollama
"""

__version__ = "0.6.0"

from organism.types import (
    MrEvent, EventType, AgentId, L0RSlot,
    Mode, AgentStatus, ControlSignals, AgentParams,
    Claim, ClaimStatus, ClaimRelation, ClaimLink, Prediction,
)
from organism.mr import RealityMemory
from organism.l0r import L0RRing, EvidencePack
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.orchestrator import Orchestrator, AgentTurn, TickResult
from organism.agent_wrapper import OllamaAgentFn
from organism.organism_state import (
    OrganismState, JudgeVerdict, CompetitionPattern, AgentTurnSnapshot,
)
from organism.judge import JudgePipeline
from organism.stem import StateEvolutionModel
