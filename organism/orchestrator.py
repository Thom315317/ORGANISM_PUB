"""
P4 — Orchestrator : boucle interne multi-agents
=================================================
Le cerveau de l'organisme. Coordonne les agents A/B/C dans un tick,
collecte les signaux, met à jour le scheduler, et alimente le world model.

L'orchestrateur ne génère PAS de texte lui-même.
Il reçoit des "agent_fn" callables qui appellent les LLMs réels.
Sans agent_fn, il tourne en mode dry (pour les tests).

Cycle d'un tick:
    1. tick_start → log Mr
    2. Scheduler.tick() → mode + agent params
    3. L0R decay + build evidence pack
    4. Pour chaque agent (trié par call_weight desc):
       a. Construire le prompt (evidence pack + world model facts + consigne mode)
       b. Appeler agent_fn(agent_id, prompt, params) → AgentTurn
       c. Logger dans Mr (AGENT_MESSAGE)
       d. Insérer dans L0R
       e. Collecter signaux cognitifs
    5. Agréger les signaux → update scheduler
    6. tick_end → log Mr
    7. Retourner TickResult

Usage:
    orch = Orchestrator(mr=mr, l0r=l0r, scheduler=scheduler, world_model=wm)
    result = orch.run_tick()           # Un tick complet
    results = orch.run_ticks(n=5)      # N ticks
    orch.inject_user_message("Bonjour")  # Signal externe
"""
from __future__ import annotations

import logging
import re
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from organism.types import (
    MrEvent, EventType, AgentId, AgentParams, AgentStatus,
    Mode, ControlSignals,
)
from organism.mr import RealityMemory
from organism.l0r import L0RRing, EvidencePack
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.web_search import web_search, reset_tick_counter
from organism.organism_state import (
    OrganismState, JudgeVerdict, CompetitionPattern,
)

log = logging.getLogger("organism.orchestrator")

# Regex pour détecter les tags <search> dans les réponses agents
_SEARCH_TAG_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)

# Patterns d'assertions substantives en français
_CLAIM_PATTERNS_FR = re.compile(
    r"(?:^|\. )"                               # Début de phrase
    r"("
    r"[A-ZÀ-Ü][^.!?]{15,120}"                 # Phrase commençant par majuscule, 15-120 chars
    r"(?:"
    r"(?:est|sont|a|ont|peut|permet|implique|"  # Verbes assertifs
    r"constitue|représente|signifie|montre|"
    r"prouve|démontre|crée|produit|génère|"
    r"is|are|has|can|means|shows|proves|creates)"
    r"[^.!?]{5,80}"                            # Suite de l'assertion
    r")"
    r")"
    r"[.!]",                                   # Fin de phrase
    re.MULTILINE,
)

# Patterns à exclure (méta-commentaires, instructions)
_CLAIM_EXCLUDE = re.compile(
    r"(?:mon rôle|je dois|il faut|la consigne|le mode|"
    r"agent|draft|résumé|réponse|format|instruction|"
    r"my role|i must|the task|the prompt)",
    re.IGNORECASE,
)


def _extract_claims_from_text(text: str) -> list:
    """
    Extrait les assertions substantives d'un texte agent.
    Retourne les phrases qui affirment quelque chose sur le monde.
    """
    claims = []
    for m in _CLAIM_PATTERNS_FR.finditer(text):
        sentence = m.group(1).strip()
        # Exclure les méta-commentaires
        if _CLAIM_EXCLUDE.search(sentence):
            continue
        # Exclure les phrases trop courtes ou trop longues
        if len(sentence) < 20 or len(sentence) > 150:
            continue
        claims.append(sentence)
    return claims


# ── Dataclasses résultat ──────────────────────────────────────────


@dataclass
class AgentTurn:
    """Résultat d'un appel agent dans un tick."""
    agent: AgentId
    status: AgentStatus
    text: str = ""
    token_in: int = 0
    token_out: int = 0
    latency_ms: float = 0.0
    # Signaux cognitifs auto-évalués par l'agent (ou extraits)
    novelty: float = 0.0
    conflict: float = 0.0
    cohesion: float = 1.0
    impl_pressure: float = 0.0
    # Claims proposées par l'agent (optionnel)
    proposed_claims: List[Dict[str, Any]] = field(default_factory=list)
    # Veto ? (seuls Lead et Oppose peuvent)
    veto: bool = False
    veto_reason: str = ""
    # Retry sur réponse vide (Fix C)
    empty_retry: bool = False
    # Mr chunk_id (rempli par l'orchestrateur après log)
    chunk_id: str = ""
    # Thinking text (séparé du content par think=True)
    thinking_text: str = ""


@dataclass
class TickResult:
    """Résultat complet d'un tick."""
    tick_id: int
    mode: Mode
    mode_changed: bool
    agent_turns: List[AgentTurn]
    evidence_pack: Optional[EvidencePack] = None
    signals: Optional[ControlSignals] = None
    vetoed: bool = False
    veto_agent: Optional[AgentId] = None
    veto_reason: str = ""
    elapsed_ms: float = 0.0
    total_tokens: int = 0
    claims_added: int = 0
    expired_slots: int = 0
    judge_verdict: Optional[JudgeVerdict] = None
    organism_state: Optional[OrganismState] = None


# Type du callable agent
# fn(agent_id, prompt, params) → AgentTurn
AgentFn = Callable[[AgentId, str, AgentParams], AgentTurn]


# ── Strip thinking traces from agent output ──────────────────────
_THINKING_MARKERS = [
    'thinking process', 'my approach', 'let me think',
    'i need to', 'step 1', 'analyze the request',
    'planning:', 'analysis:', '**analyze', '**determine',
    '**formulate', '**evaluate', 'constraint:', 'context:',
    'role:', 'task:',
]


def strip_thinking(text: str) -> str:
    """Remove reasoning traces from agent output.

    Conservative approach:
    1. Remove tagged thinking blocks (<think>...</think>)
    2. Remove plain-text thinking blocks at the START of the response
       (identified by known headers, ending at first non-thinking paragraph)
    3. Safety: if we removed >90% of text, still return stripped version
    """
    if not text:
        return text

    original_len = len(text)

    # Layer 1: Tagged thinking (safest — clean delimiters)
    text = re.sub(
        r'<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>',
        '', text, flags=re.DOTALL,
    )

    # Layer 2: Plain-text thinking at START of response
    paragraphs = text.split('\n\n')
    content_start = 0

    for i, para in enumerate(paragraphs):
        para_lower = para.lower().strip()
        is_thinking = any(marker in para_lower for marker in _THINKING_MARKERS)
        if is_thinking:
            content_start = i + 1
        else:
            break  # First non-thinking paragraph → stop stripping

    if content_start > 0 and content_start < len(paragraphs):
        text = '\n\n'.join(paragraphs[content_start:])
    elif content_start >= len(paragraphs):
        # Everything was thinking — no real content exists
        text = ''

    text = text.strip()

    # Safety: if we removed more than 90%, the draft was genuinely all thinking
    if len(text) < original_len * 0.1 and original_len > 100:
        return text  # Still return stripped

    return text


# ── Veto Budget ──────────────────────────────────────────────────


class VetoBudget:
    """
    Auto-regulated veto cost: Fibonacci cooldown per agent.

    Each veto increments the agent's consecutive-veto counter.
    Cooldown (in ticks) = fib(consecutive_vetos).
    While on cooldown, the agent's veto flag is silently converted
    to a normal disagreement — the draft text is preserved, only the
    blocking power is removed.

    Forgiveness: each tick without a veto decrements the counter by 1,
    so an agent that stops abusing vetos gradually recovers full rights.

    Uses absolute tick timestamps (blocked_until) to avoid off-by-one
    issues with decrement-based cooldowns.

    Sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
    """

    def __init__(self):
        self._consecutive: Dict[str, int] = {}     # agent → consecutive veto count
        self._blocked_until: Dict[str, int] = {}   # agent → tick_id until which blocked
        self._current_tick: int = 0

    def can_veto(self, agent_id: str) -> bool:
        return self._current_tick > self._blocked_until.get(agent_id, 0)

    def register_veto(self, agent_id: str):
        n = self._consecutive.get(agent_id, 0) + 1
        self._consecutive[agent_id] = n
        self._blocked_until[agent_id] = self._current_tick + self._fib(n)

    def register_no_veto(self, agent_id: str):
        n = self._consecutive.get(agent_id, 0)
        if n > 0:
            self._consecutive[agent_id] = n - 1

    def tick(self, tick_id: int):
        """Update current tick. Call at tick start."""
        self._current_tick = tick_id

    def cooldown_remaining(self, agent_id: str) -> int:
        return max(0, self._blocked_until.get(agent_id, 0) - self._current_tick)

    @staticmethod
    def _fib(n: int) -> int:
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a


# ── Orchestrator ──────────────────────────────────────────────────


class Orchestrator:
    """
    Boucle interne multi-agents. Coordonne P0-P3 dans chaque tick.

    Modes de fonctionnement:
    - Avec agent_fn: appelle un vrai LLM par agent
    - Sans agent_fn (dry mode): génère des AgentTurn vides (pour tests)
    """

    def __init__(
        self,
        mr: RealityMemory,
        l0r: L0RRing,
        scheduler: Scheduler,
        world_model: WorldModel,
        agent_fn: Optional[AgentFn] = None,
        max_agents_per_tick: int = 3,
        judge_pipeline=None,
        condition: str = "full",
        theories: Optional[List] = None,
        stem=None,
        bench_mode: bool = False,
    ):
        self._mr = mr
        self._l0r = l0r
        self._scheduler = scheduler
        self._wm = world_model
        self._agent_fn = agent_fn
        self._max_agents_per_tick = max_agents_per_tick
        self._judge = judge_pipeline
        self._condition = condition
        self._theories = theories or []
        self._stem = stem
        self._bench_mode = bench_mode

        self._tick_id: int = 0
        self._history: List[TickResult] = []
        self._user_messages: List[str] = []  # File de messages utilisateur
        self._recent_winners: List[str] = []  # Historique des gagnants pour le judge
        self._recent_margins: List[float] = []  # Historique margin_1v2
        self._discarded_drafts: List[dict] = []  # Drafts forfaits (thinking_only)
        self._prev_signals: Optional[ControlSignals] = None
        self._prev_mode: Optional[Mode] = None
        self._signal_history: List[ControlSignals] = []  # fenêtre pour prediction_error
        self._veto_budget = VetoBudget()

    # ── Public API ────────────────────────────────────────────────

    def inject_user_message(self, message: str) -> MrEvent:
        """
        Injecte un message utilisateur dans l'organisme.
        Le message est loggé dans Mr et ajouté à la file.
        Booste immédiatement novelty pour sortir de Idle.
        """
        self._user_messages.append(message)
        event = self._mr.append(
            event_type=EventType.AGENT_MESSAGE,
            tick_id=self._tick_id,
            agent=AgentId.O,
            mode=self._scheduler.current_mode.value,
            payload={"role": "user", "text": message[:2000]},
        )
        self._l0r.insert(
            chunk_id=event.chunk_id,
            salience=1.0,  # Messages user = salience max
            novelty=1.0,
        )
        # Auto-search web pour enrichir le contexte avec du réel
        # Désactivé en mode bench (non reproductible, variable confondante)
        results_text = None if self._bench_mode else web_search(message, max_results=3)
        if results_text:
            log.info("inject_user_message: auto-search(%r) → %d chars",
                     message[:50], len(results_text))
            ws_event = self._mr.append(
                event_type=EventType.TOOL_RESULT,
                tick_id=self._tick_id,
                agent=AgentId.O,
                mode=self._scheduler.current_mode.value,
                payload={"tool": "web_search", "query": message[:100],
                         "text": results_text[:1000]},
            )
            self._l0r.insert(
                chunk_id=ws_event.chunk_id,
                salience=0.85,
                novelty=0.7,
            )

        # Boost immédiat des signaux pour sortir de Idle
        prev = self._scheduler.signals
        self._scheduler.update_signals(ControlSignals(
            energy=max(prev.energy, 0.8),
            novelty=max(prev.novelty, 0.7),
            conflict=prev.conflict,
            impl_pressure=prev.impl_pressure,
            cohesion=prev.cohesion,
            cost_pressure=prev.cost_pressure,
            prediction_error=max(prev.prediction_error, 0.3),
        ))
        return event

    def run_tick(self) -> TickResult:
        """Exécute un tick complet de l'organisme."""
        t0 = time.time()
        self._tick_id += 1
        self._veto_budget.tick(self._tick_id)
        reset_tick_counter()
        self._wm.tick_id = self._tick_id

        # 1. tick_start
        self._mr.append(
            event_type=EventType.TICK_START,
            tick_id=self._tick_id,
            agent=AgentId.O,
            mode=self._scheduler.current_mode.value,
        )

        # 2. Scheduler tick → mode + transition
        mode, mode_changed = self._scheduler.tick()
        if mode_changed:
            self._mr.append(
                event_type=EventType.MODE_CHANGE,
                tick_id=self._tick_id,
                agent=AgentId.O,
                mode=mode.value,
                payload={
                    "from": self._scheduler.history[-1].from_mode.value
                    if self._scheduler.history else "?",
                    "to": mode.value,
                },
            )

        # 3. L0R decay + evidence pack
        expired = self._l0r.tick_decay()
        pack = self._l0r.build_evidence_pack(budget_tokens=2000)

        # 4. Sélectionner les agents à appeler
        agents_to_call = self._select_agents(mode)

        # 5. Appeler chaque agent
        turns: List[AgentTurn] = []
        vetoed = False
        veto_agent = None
        veto_reason = ""
        claims_added = 0

        for agent_id in agents_to_call:
            params = self._scheduler.get_agent_params(agent_id)
            prompt = self._build_agent_prompt(agent_id, params, pack, mode)

            # Appel agent
            if self._agent_fn:
                turn = self._agent_fn(agent_id, prompt, params)
            else:
                turn = AgentTurn(
                    agent=agent_id,
                    status=params.status,
                )

            # Logger dans Mr
            event = self._mr.append(
                event_type=EventType.AGENT_MESSAGE,
                tick_id=self._tick_id,
                agent=agent_id,
                mode=mode.value,
                token_in=turn.token_in,
                token_out=turn.token_out,
                latency_ms=turn.latency_ms,
                novelty=turn.novelty,
                conflict=turn.conflict,
                cohesion=turn.cohesion,
                impl_pressure=turn.impl_pressure,
                payload={
                    "text": turn.text[:1500],
                    **({"thinking_text": turn.thinking_text[:2000]} if turn.thinking_text else {}),
                } if turn.text else None,
            )
            turn.chunk_id = event.chunk_id

            # Strip thinking traces — events.jsonl already has the raw text
            raw_len = len(turn.text) if turn.text else 0
            if turn.text:
                turn.text = strip_thinking(turn.text)

            # Discard drafts that are pure thinking (no real content)
            if not turn.text or len(turn.text.strip()) < 20:
                aid_str = agent_id.value if hasattr(agent_id, "value") else str(agent_id)
                log.warning("[tick %d] %s draft discarded (thinking_only, raw=%d chars)",
                            self._tick_id, aid_str, raw_len)
                self._mr.append(
                    event_type=EventType.AGENT_MESSAGE,
                    tick_id=self._tick_id,
                    agent=agent_id,
                    mode=mode.value,
                    payload={"type": "draft_discarded", "reason": "thinking_only",
                             "raw_length": raw_len},
                )
                self._discarded_drafts.append({
                    "tick": self._tick_id, "agent": aid_str, "reason": "thinking_only",
                })
                continue  # Skip L0R, claims, judge — agent forfeits this tick

            # Insérer dans L0R
            if turn.text:
                self._l0r.insert(
                    chunk_id=event.chunk_id,
                    salience=0.5 + 0.3 * turn.novelty,
                    novelty=turn.novelty,
                    conflict=turn.conflict,
                )

            # Extraire les <search> tags et exécuter les recherches web
            search_queries = _SEARCH_TAG_RE.findall(turn.text)
            for q in search_queries[:2]:
                q = q.strip()
                if not q:
                    continue
                results_text = web_search(q)
                if results_text:
                    log.info("[tick %d] web_search(%r) → %d chars",
                             self._tick_id, q, len(results_text))
                    ws_event = self._mr.append(
                        event_type=EventType.TOOL_RESULT,
                        tick_id=self._tick_id,
                        agent=AgentId.O,
                        mode=mode.value,
                        payload={"tool": "web_search", "query": q,
                                 "text": results_text[:1000]},
                    )
                    self._l0r.insert(
                        chunk_id=ws_event.chunk_id,
                        salience=0.9,
                        novelty=0.8,
                    )
            # Strip <search> tags du texte visible
            if search_queries:
                turn.text = _SEARCH_TAG_RE.sub("", turn.text).strip()

            # Traiter les claims proposées (explicites)
            for claim_data in turn.proposed_claims:
                self._wm.add_claim(
                    content=claim_data.get("content", ""),
                    confidence=claim_data.get("confidence", 0.5),
                    provenance=[event.chunk_id],
                    source_agent=agent_id,
                )
                claims_added += 1

            # Extraire les claims substantives du texte agent (heuristiques FR)
            if turn.text and len(turn.text) > 50:
                extracted = _extract_claims_from_text(turn.text)
                for claim_text in extracted[:3]:  # Max 3 claims par agent par tick
                    self._wm.add_claim(
                        content=claim_text,
                        confidence=0.45,  # Hypothèse — pas encore validée
                        provenance=[event.chunk_id],
                        source_agent=agent_id,
                    )
                    claims_added += 1

            # Veto ? — regulated by Fibonacci cooldown budget
            aid_str = agent_id.value if hasattr(agent_id, "value") else str(agent_id)
            if turn.veto and params.can_veto:
                if self._veto_budget.can_veto(aid_str):
                    self._veto_budget.register_veto(aid_str)
                    self._scheduler.register_veto()  # Signal → boost Debate
                    vetoed = True
                    veto_agent = agent_id
                    veto_reason = turn.veto_reason
                    turns.append(turn)
                    break  # Le veto arrête le tick
                else:
                    # On cooldown: convert veto to normal disagreement
                    remaining = self._veto_budget.cooldown_remaining(aid_str)
                    log.info("[tick %d] %s wanted veto but on cooldown (%d ticks remaining)",
                             self._tick_id, aid_str, remaining)
                    turn.veto = False  # Draft preserved, blocking power removed
                    self._veto_budget.register_no_veto(aid_str)
            else:
                self._veto_budget.register_no_veto(aid_str)

            turns.append(turn)

        # 6. Judge pipeline (si disponible)
        # Un veto exclut le draft vetoé mais ne supprime PAS le jugement.
        # Le juge a besoin de savoir qu'il y a eu conflit (info pour HOT, MDM).
        judge_verdict = None
        if self._judge and turns:
            judgeable_turns = [t for t in turns if not t.veto]
            if len(judgeable_turns) >= 2:
                try:
                    judge_verdict = self._judge.evaluate(
                        judgeable_turns, self._recent_winners,
                    )
                except Exception:
                    log.exception("Judge pipeline failed at tick %d", self._tick_id)
            elif len(judgeable_turns) == 1:
                # Un seul draft non-vetoé → gagne par défaut, pas de compétition
                sole = judgeable_turns[0]
                aid = sole.agent.value if hasattr(sole.agent, "value") else str(sole.agent)
                # FIX 8: Complete ranking with all agents
                all_aids = []
                for t in turns:
                    a = t.agent.value if hasattr(t.agent, "value") else str(t.agent)
                    if a not in all_aids:
                        all_aids.append(a)
                full_ranking = tuple([aid] + sorted(a for a in all_aids if a != aid))
                judge_verdict = JudgeVerdict(
                    winner=aid,
                    reason="Seul draft non-vetoé",
                    confidence=0.5,
                    signals={"novelty": sole.novelty, "conflict": sole.conflict,
                             "cohesion": sole.cohesion, "impl_pressure": sole.impl_pressure},
                    claims=(),
                    competition=CompetitionPattern(
                        ranking=full_ranking,
                        margin_1v2=1.0,
                        margin_2v3=0.0,
                        counterfactual="Draft vetoé exclu",
                    ),
                )
                log.info("[tick %d] Sole non-vetoed draft: %s wins by default (ranking_generated_from_winner)", self._tick_id, aid)
            else:
                log.info("[tick %d] All drafts vetoed — no judge verdict", self._tick_id)

        # 7. Agréger les signaux → update scheduler
        # Si le judge a produit des signals, les utiliser en priorité
        # Si judge_failed (signals=None), on tombe dans le else → _aggregate_signals
        if judge_verdict and judge_verdict.signals:
            js = judge_verdict.signals
            prev_energy = self._scheduler.signals.energy
            total_tokens_so_far = sum(t.token_in + t.token_out for t in turns)
            energy_cost = min(0.3, total_tokens_so_far / 20000.0)
            signals = ControlSignals(
                energy=max(0.0, prev_energy - energy_cost),
                novelty=js.get("novelty", 0.0),
                conflict=js.get("conflict", 0.0),
                impl_pressure=js.get("impl_pressure", 0.0),
                cohesion=js.get("cohesion", 0.5),
                cost_pressure=min(1.0, total_tokens_so_far / 10000.0),
                prediction_error=self._scheduler.signals.prediction_error,
            )
        else:
            signals = self._aggregate_signals(turns)
        self._scheduler.update_signals(signals)

        # 7b. Compute prediction_error dynamically
        self._signal_history.append(signals)
        if len(self._signal_history) > 5:
            self._signal_history = self._signal_history[-5:]

        if len(self._signal_history) >= 2:
            sig_names = ["novelty", "conflict", "cohesion", "impl_pressure"]
            history_before = self._signal_history[:-1]
            expected = {
                s: sum(getattr(h, s) for h in history_before) / len(history_before)
                for s in sig_names
            }
            actual = {s: getattr(signals, s) for s in sig_names}
            pe_signals = sum(abs(expected[s] - actual[s]) for s in sig_names) / len(sig_names)
            pe_mode = 1.0 if mode_changed else 0.0
            pe = 0.4 * pe_mode + 0.6 * pe_signals
        else:
            pe = 0.3  # prior non-informatif

        signals = ControlSignals(
            energy=signals.energy,
            novelty=signals.novelty,
            conflict=signals.conflict,
            impl_pressure=signals.impl_pressure,
            cohesion=signals.cohesion,
            cost_pressure=signals.cost_pressure,
            prediction_error=pe,
        )
        self._scheduler.update_signals(signals)

        # 8. Post-judge: winning draft -> Mr+L0R, claims -> WorldModel
        if judge_verdict:
            # Log winning draft with high salience
            winner_turn = None
            if judge_verdict.winner is not None:
                for t in turns:
                    aid = t.agent.value if hasattr(t.agent, "value") else str(t.agent)
                    if aid == judge_verdict.winner:
                        winner_turn = t
                        break
            if winner_turn and winner_turn.text:
                _sig = judge_verdict.signals or {}
                self._l0r.insert(
                    chunk_id=winner_turn.chunk_id,
                    salience=1.0,
                    novelty=_sig.get("novelty", 0.0),
                    conflict=_sig.get("conflict", 0.0),
                )

            # Judge claims -> WorldModel
            judge_claims_added = 0
            for claim in judge_verdict.claims:
                if isinstance(claim, dict) and claim.get("text"):
                    status_map = {
                        "hypothesis": 0.4,
                        "supported": 0.7,
                        "contradicted": 0.2,
                    }
                    claim_status = claim.get("status", "hypothesis")
                    conf = status_map.get(claim_status, 0.4)
                    source_aid = claim.get("source", "")
                    prov = []
                    for t in turns:
                        aid = t.agent.value if hasattr(t.agent, "value") else str(t.agent)
                        if aid == source_aid and t.chunk_id:
                            prov.append(t.chunk_id)
                    self._wm.add_claim(
                        content=claim["text"],
                        confidence=conf,
                        provenance=prov,
                        source_agent=AgentId.O,
                        status_hint=claim_status,
                    )
                    claims_added += 1
                    judge_claims_added += 1

            # Fallback: if judge produced no claims, extract from winning draft
            if judge_claims_added == 0 and winner_turn and winner_turn.text:
                extracted = _extract_claims_from_text(winner_turn.text)
                winner_aid = judge_verdict.winner or ""
                prov = [winner_turn.chunk_id] if winner_turn.chunk_id else []
                for claim_text in extracted[:3]:
                    self._wm.add_claim(
                        content=claim_text,
                        confidence=0.5,
                        provenance=prov,
                        source_agent=AgentId.O,
                        status_hint="hypothesis",
                    )
                    claims_added += 1

            # Track winners & margins (skip judge_failed ticks)
            if judge_verdict.winner is not None:
                self._recent_winners.append(judge_verdict.winner)
                if len(self._recent_winners) > 50:
                    self._recent_winners = self._recent_winners[-50:]
            if judge_verdict.competition:
                self._recent_margins.append(judge_verdict.competition.margin_1v2)
                if len(self._recent_margins) > 50:
                    self._recent_margins = self._recent_margins[-50:]

            # Adaptive judge temperature based on margin variance
            if hasattr(self._judge, 'adapt_temperature') and len(self._recent_margins) >= 5:
                self._judge.adapt_temperature(self._recent_margins)

        # 9. Build OrganismState snapshot
        organism_state = OrganismState.from_tick(
            tick_id=self._tick_id,
            mode=mode,
            mode_changed=mode_changed,
            signals=signals,
            agent_turns=turns,
            judge_verdict=judge_verdict,
            wm_stats={**self._wm.get_stats(), "claims_added_this_tick": claims_added},
            l0r_stats={"active_slots": len(self._l0r)},
            condition=self._condition,
            prev_signals=self._prev_signals,
            prev_mode=self._prev_mode,
            recent_winners=self._recent_winners[-10:],
            recent_margins=self._recent_margins[-10:],
        )

        # 9b. Compute consciousness theory scores
        # Hybrid must run AFTER the 7 others (it reads their scores)
        theory_scores = {}
        hybrid_theory = None
        for theory in self._theories:
            if getattr(theory, 'name', '') == "Hybrid":
                hybrid_theory = theory
                continue
            try:
                score = theory.compute(organism_state)
                theory_scores[score.theory] = score.value
            except Exception:
                log.exception("Theory %s failed at tick %d",
                              getattr(theory, 'name', '?'), self._tick_id)
        # Store non-Hybrid scores so Hybrid can read them
        if theory_scores:
            organism_state.theory_scores.update(theory_scores)
        # Now compute Hybrid with all other scores available
        if hybrid_theory:
            try:
                score = hybrid_theory.compute(organism_state)
                organism_state.theory_scores[score.theory] = score.value
            except Exception:
                log.exception("Theory Hybrid failed at tick %d", self._tick_id)

        # 9c. STEM: accumulate state vector
        if self._stem:
            try:
                self._stem.on_tick(organism_state)
            except Exception:
                log.exception("STEM failed at tick %d", self._tick_id)

        # Save for next tick
        self._prev_signals = signals
        self._prev_mode = mode

        # 10. tick_end
        elapsed_ms = (time.time() - t0) * 1000
        total_tokens = sum(t.token_in + t.token_out for t in turns)

        self._mr.append(
            event_type=EventType.TICK_END,
            tick_id=self._tick_id,
            agent=AgentId.O,
            mode=mode.value,
            token_in=sum(t.token_in for t in turns),
            token_out=sum(t.token_out for t in turns),
            latency_ms=elapsed_ms,
            novelty=signals.novelty,
            conflict=signals.conflict,
            cohesion=signals.cohesion,
            impl_pressure=signals.impl_pressure,
            payload={
                "agents_called": len(turns),
                "vetoed": vetoed,
                "claims_added": claims_added,
                "expired_slots": expired,
                "judge_winner": judge_verdict.winner if judge_verdict else None,
                "judge_confidence": judge_verdict.confidence if judge_verdict else None,
                "_anon_map": judge_verdict.raw_json.get("_anon_map") if judge_verdict and judge_verdict.raw_json else None,
                "_anon_reverse": judge_verdict.raw_json.get("_anon_reverse") if judge_verdict and judge_verdict.raw_json else None,
            },
        )

        # Drainer la file user
        self._user_messages.clear()

        result = TickResult(
            tick_id=self._tick_id,
            mode=mode,
            mode_changed=mode_changed,
            agent_turns=turns,
            evidence_pack=pack,
            signals=signals,
            vetoed=vetoed,
            veto_agent=veto_agent,
            veto_reason=veto_reason,
            elapsed_ms=elapsed_ms,
            total_tokens=total_tokens,
            claims_added=claims_added,
            expired_slots=expired,
            judge_verdict=judge_verdict,
            organism_state=organism_state,
        )
        self._history.append(result)
        return result

    def run_ticks(self, n: int = 1) -> List[TickResult]:
        """Exécute N ticks consécutifs."""
        return [self.run_tick() for _ in range(n)]

    # ── Queries ───────────────────────────────────────────────────

    @property
    def tick_id(self) -> int:
        return self._tick_id

    @property
    def history(self) -> List[TickResult]:
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques globales de l'orchestrateur."""
        total_turns = sum(len(r.agent_turns) for r in self._history)
        total_tokens = sum(r.total_tokens for r in self._history)
        total_vetoes = sum(1 for r in self._history if r.vetoed)
        total_claims = sum(r.claims_added for r in self._history)

        return {
            "tick_count": self._tick_id,
            "total_agent_turns": total_turns,
            "total_tokens": total_tokens,
            "total_vetoes": total_vetoes,
            "total_claims_added": total_claims,
            "current_mode": self._scheduler.current_mode.value,
            "mr_events": self._mr.event_count,
            "l0r_active_slots": len(self._l0r),
            "wm_stats": self._wm.get_stats(),
            "scheduler_stats": self._scheduler.get_stats(),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _select_agents(self, mode: Mode) -> List[AgentId]:
        """
        Sélectionne les agents à appeler ce tick, triés par call_weight desc.
        Exclut O (l'orchestrateur ne s'appelle pas lui-même).
        """
        agents = [AgentId.A, AgentId.B, AgentId.C]
        params = {a: self._scheduler.get_agent_params(a) for a in agents}

        # Trier par call_weight décroissant
        agents.sort(key=lambda a: params[a].call_weight, reverse=True)

        # Limiter au max
        return agents[:self._max_agents_per_tick]

    def _build_agent_prompt(
        self,
        agent: AgentId,
        params: AgentParams,
        pack: EvidencePack,
        mode: Mode,
    ) -> str:
        """
        Construit le prompt pour un agent.
        Minimal : mode (1 phrase) + status (1 ligne) + contexte + facts.
        Pas de [ROLE], pas de [STATUS] technique, pas de [BOOTSTRAP] technique.
        """
        sections = []

        _en = getattr(self, '_language', 'fr') == 'en'

        # 1. Messages utilisateur — PRIORITÉ ABSOLUE
        if self._user_messages:
            sections.append("Answer this question:" if _en else "Réponds à cette question :")
            for msg in self._user_messages:
                sections.append(f">>> {msg}")
            sections.append("")  # ligne vide

        # 2. Mode — une phrase, pas un paragraphe
        if _en:
            mode_instruction = {
                Mode.IDLE: "Think briefly about a topic that interests you.",
                Mode.EXPLORE: "Explore a new angle. Propose a hypothesis.",
                Mode.DEBATE: "Challenge or defend an idea. Take a position.",
                Mode.IMPLEMENT: "Propose something concrete: a plan, a synthesis, a formulation.",
                Mode.CONSOLIDATE: "Summarize what has been established. Identify points of agreement.",
                Mode.RECOVER: "Pause. 1 sentence max.",
            }
        else:
            mode_instruction = {
                Mode.IDLE: "Réfléchis brièvement sur un sujet qui t'intéresse.",
                Mode.EXPLORE: "Explore une nouvelle piste. Propose une hypothèse.",
                Mode.DEBATE: "Conteste ou défends une idée. Prends position.",
                Mode.IMPLEMENT: "Propose quelque chose de concret : un plan, une synthèse, une formulation.",
                Mode.CONSOLIDATE: "Résume ce qui a été établi. Identifie les points d'accord.",
                Mode.RECOVER: "Pause. 1 phrase max.",
            }
        sections.append(mode_instruction.get(mode, "Think." if _en else "Pense."))

        # 3. Status — une ligne
        if params.status == AgentStatus.LEAD:
            sections.append("You lead the discussion." if _en else "Tu mènes la réflexion.")
        elif params.status == AgentStatus.OPPOSE:
            sections.append("Challenge what has been said. If it is dangerous or false, say VETO." if _en else "Conteste ce qui a été dit. Si c'est dangereux ou faux, dis VETO.")
        # Support = pas d'instruction spéciale

        # 4. Contexte — les derniers échanges (dédupliqués)
        if pack and pack.slots:
            deduped = self._dedupe_evidence(pack)
            if deduped:
                sections.append("")
                sections.append("Recent context:" if _en else "Contexte récent :")
                sections.append("\n".join(deduped))

        # 5. Facts du World Model (si non vide)
        facts = self._wm.get_facts()
        if facts:
            facts_text = "\n".join(f"- {f.content}" for f in facts[:8])
            sections.append(("\nEstablished facts:\n" if _en else "\nFaits établis :\n") + facts_text)

        # 6. Bootstrap si premier tick sans contexte ni user
        if not pack or not pack.slots:
            if not self._user_messages:
                sections.append(
                    "\nThis is the beginning. Pick a topic that interests you and "
                    "formulate a first thought." if _en else
                    "\nC'est le début. Choisis un sujet qui t'intéresse et "
                    "formule une première réflexion."
                )

        return "\n".join(sections)

    @staticmethod
    def _dedupe_evidence(pack: EvidencePack) -> List[str]:
        """
        Déduplique les entrées de l'evidence pack.
        Garde les user messages + max 1 occurrence de chaque texte agent.
        """
        seen_texts: set = set()
        lines: List[str] = []
        for slot, event in zip(pack.slots, pack.events):
            if not event.payload:
                continue
            text = event.payload.get("text", "")
            role = event.payload.get("role", "")
            if not text:
                continue
            # Toujours garder les user messages et web results
            tool = event.payload.get("tool", "")
            if role == "user":
                lines.append(f"[{slot.composite_score():.2f}] USER: {text[:200]}")
                continue
            if tool == "web_search":
                lines.append(f"[{slot.composite_score():.2f}] WEB: {text[:300]}")
                continue
            # Pour les messages agent, dédupliquer par contenu (premiers 80 chars)
            key = text[:80].lower().strip()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            agent = event.agent.value if hasattr(event.agent, 'value') else event.agent
            lines.append(f"[{slot.composite_score():.2f}] {agent}: {text[:400]}")
        return lines

    def _aggregate_signals(self, turns: List[AgentTurn]) -> ControlSignals:
        """
        Agrège les signaux cognitifs des turns en ControlSignals pour le scheduler.
        """
        if not turns:
            return self._scheduler.signals  # Garder les signaux précédents

        n = len(turns)
        total_tokens = sum(t.token_in + t.token_out for t in turns)

        # Moyennes pondérées par status (Lead pèse plus)
        weights = {
            AgentStatus.LEAD: 1.0,
            AgentStatus.SUPPORT: 0.6,
            AgentStatus.OPPOSE: 0.8,
        }
        total_weight = sum(weights.get(t.status, 0.5) for t in turns)
        if total_weight == 0:
            total_weight = 1.0

        def weighted_avg(attr: str) -> float:
            return sum(
                getattr(t, attr, 0.0) * weights.get(t.status, 0.5)
                for t in turns
            ) / total_weight

        # Énergie : diminue avec le nombre de tokens consommés
        prev_energy = self._scheduler.signals.energy
        energy_cost = min(0.3, total_tokens / 20000.0)
        energy = max(0.0, prev_energy - energy_cost)

        # Cost pressure : proportionnelle aux tokens
        cost_pressure = min(1.0, total_tokens / 10000.0)

        return ControlSignals(
            energy=energy,
            novelty=weighted_avg("novelty"),
            conflict=weighted_avg("conflict"),
            impl_pressure=weighted_avg("impl_pressure"),
            cohesion=weighted_avg("cohesion"),
            cost_pressure=cost_pressure,
            prediction_error=0.0,  # sera recalculé dans run_tick() étape 7b
        )
