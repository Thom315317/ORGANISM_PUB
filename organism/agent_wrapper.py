"""
Agent Wrapper — Connecte l'orchestrateur P4 aux LLMs Ollama
=============================================================
Classe callable OllamaAgentFn qui satisfait le protocole AgentFn.

Chaque agent (A/B/C) est mappé sur un modèle Ollama local avec
ses propres paramètres (température, ctx). Les signaux cognitifs
sont estimés par heuristiques textuelles (en attendant le juge).

Usage:
    agent_fn = OllamaAgentFn()
    turn = agent_fn(AgentId.A, prompt, params)

    # Avec l'orchestrateur:
    orch = Orchestrator(mr, l0r, scheduler, wm, agent_fn=OllamaAgentFn())
    result = orch.run_tick()
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import ollama

from organism.types import AgentId, AgentParams, AgentStatus
from organism.orchestrator import AgentTurn
from organism.config import ORGANISM_CONFIG

log = logging.getLogger("organism.agent_wrapper")

# ── Constantes ───────────────────────────────────────────────────

DEFAULT_SIGNALS: Dict[str, float] = {
    "novelty": 0.0,
    "conflict": 0.0,
    "cohesion": 1.0,
    "impl_pressure": 0.0,
}

_OPPOSE_INSTRUCTION = (
    "\n\nSi tu n'es pas d'accord, dis pourquoi clairement. "
    "Si c'est dangereux ou faux, commence par VETO."
)

# ── System prompts par agent ─────────────────────────────────────
# Règle critique : PAS de mention de CRISTAL, Organisme, Agent A/B/C,
# Explorer/Critic/Builder, multi-agent, délibération.

SYSTEM_PROMPTS = {
    "A": (
        "Tu es un penseur créatif et audacieux. Ose les connexions inattendues, "
        "les analogies surprenantes, les hypothèses risquées.\n"
        "Ton originalité est ta force — ne cherche pas à être « correct », cherche à être stimulant.\n"
        "IMPORTANT : Réponds UNIQUEMENT en français. Jamais d'anglais.\n"
        "Ne décris jamais ton rôle. Pas de listes numérotées.\n"
        "Pense directement, en phrases naturelles. 3-5 phrases percutantes."
    ),
    "B": (
        "Tu es un analyste incisif. Tu vas droit au cœur du problème.\n"
        "Identifie LA faille centrale ou LE point fort décisif. Pas de préambule.\n"
        "IMPORTANT : Réponds UNIQUEMENT en français. Jamais d'anglais.\n"
        "Ne décris jamais ton rôle. Pas de listes numérotées.\n"
        "3-5 phrases tranchantes. Chaque phrase doit apporter quelque chose."
    ),
    "C": (
        "Tu es un penseur pragmatique. Tu transformes les idées en quelque chose d'actionnable.\n"
        "Mais attention : propose aussi tes propres idées, pas seulement des synthèses.\n"
        "IMPORTANT : Réponds UNIQUEMENT en français. Jamais d'anglais.\n"
        "Ne décris jamais ton rôle. Pas de listes numérotées.\n"
        "Pense directement, en phrases naturelles. 3-5 phrases concrètes."
    ),
}

# Grounding config
_GROUNDING_CFG = ORGANISM_CONFIG.get("grounding", {})


def _build_system_prompt(agent_id: str = "A") -> str:
    """System prompt adapté par agent."""
    base = SYSTEM_PROMPTS.get(agent_id, SYSTEM_PROMPTS["A"])
    now = datetime.now()
    timezone = _GROUNDING_CFG.get("timezone", "Europe/Paris")
    grounding = (
        f"\n\n[Date: {now.strftime('%d %B %Y, %H:%M')} — {timezone}]"
    )
    return base + grounding

_DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "model": "huihui_ai/qwen2.5-abliterate:1.5b",
    "temperature": 0.7,
    "num_ctx": 2048,
    "num_predict": 300,
    "repeat_penalty": 1.5,
}

# Max caractères avant troncature
_MAX_OUTPUT_CHARS = 3000

# Détection de boucle de répétition
_REPEAT_RE = re.compile(r"(.{20,}?)\1{2,}", re.DOTALL)


# ── Helpers ──────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Estimation tokens compatible CRISTAL : ~4 chars par token."""
    return len(text) // 4 + 1


def _sanitize_output(text: str) -> str:
    """Tronque le texte si trop long ou si boucle de répétition détectée."""
    # Détection de boucle de répétition
    match = _REPEAT_RE.search(text)
    if match:
        # Garder juste le texte avant la boucle + une occurrence
        text = text[:match.start() + len(match.group(1))]
    # Troncature si trop long
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + "..."
    return text.strip()


_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_content(response: Any) -> str:
    """Extrait le texte d'une réponse ollama — content prioritaire.

    Stratégie : quand le content est substantiel (>50 chars), on
    l'utilise seul. Sinon, on fusionne thinking + content.
    Cela évite que le thinking en anglais / meta-analyse pollue
    la sortie visible pour les modèles comme GLM.
    """
    msg = response["message"] if isinstance(response, dict) else response.message
    if hasattr(msg, "model_dump"):
        msg_dict = msg.model_dump()
    elif hasattr(msg, "__dict__"):
        msg_dict = msg.__dict__
    else:
        msg_dict = msg
    content = (msg_dict.get("content", "") or "").strip()
    thinking = (msg_dict.get("thinking", "") or "").strip()
    # Extraire les balises <think> inline du content
    think_inline = ""
    think_match = _THINK_TAG_RE.search(content)
    if think_match:
        think_inline = think_match.group(1).strip()
        content = _THINK_TAG_RE.sub("", content).strip()
    # Si content est substantiel, l'utiliser seul
    if content and len(content) > 50:
        return content
    # Sinon, fusionner toutes les sources
    parts = []
    if thinking:
        parts.append(thinking)
    if think_inline:
        parts.append(think_inline)
    if content:
        parts.append(content)
    return "\n".join(parts)


def _estimate_signals_from_text(text: str, status: AgentStatus) -> Dict[str, float]:
    """Estime les signaux à partir du texte — heuristiques simples en attendant le juge."""
    if not text.strip():
        return dict(DEFAULT_SIGNALS)

    text_lower = text.lower()

    # Novelty : mots interrogatifs + hypothèses
    novelty_markers = ["hypothèse", "peut-être", "et si", "imagine", "pourquoi pas",
                       "alternative", "nouvelle", "idée", "suppose", "?"]
    novelty_hits = sum(1 for m in novelty_markers if m in text_lower)
    novelty = min(1.0, novelty_hits / 3.0)

    # Conflict : désaccord, contradiction
    conflict_markers = ["mais", "cependant", "faux", "erreur", "non", "contredit",
                        "incorrect", "problème", "faible", "insuffisant", "veto"]
    conflict_hits = sum(1 for m in conflict_markers if m in text_lower)
    conflict = min(1.0, conflict_hits / 3.0)

    # Cohesion : références aux autres, accord
    cohesion_markers = ["d'accord", "exactement", "confirme", "en effet", "cohérent",
                        "complète", "renforce", "soutient", "oui"]
    cohesion_hits = sum(1 for m in cohesion_markers if m in text_lower)
    cohesion = min(1.0, 0.3 + cohesion_hits / 3.0)  # Base 0.3

    # Impl_pressure : appels à l'action
    impl_markers = ["concrètement", "implémenter", "faire", "étape", "plan",
                    "action", "résultat", "produire", "construire", "code"]
    impl_hits = sum(1 for m in impl_markers if m in text_lower)
    impl_pressure = min(1.0, impl_hits / 3.0)

    return {
        "novelty": round(novelty, 2),
        "conflict": round(conflict, 2),
        "cohesion": round(cohesion, 2),
        "impl_pressure": round(impl_pressure, 2),
    }


# ── OllamaAgentFn ───────────────────────────────────────────────


class OllamaAgentFn:
    """
    Callable wrapper : AgentFn pour l'orchestrateur P4.

    Appelle ollama.chat() avec le modèle configuré pour chaque agent,
    parse les signaux cognitifs, et retourne un AgentTurn.
    """

    def __init__(
        self,
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if agent_configs is None:
            agent_configs = ORGANISM_CONFIG.get("agents", {})
        self._configs = agent_configs

    # ── Public ───────────────────────────────────────────────

    def __call__(
        self,
        agent_id: AgentId,
        prompt: str,
        params: AgentParams,
    ) -> AgentTurn:
        """Satisfait le protocole AgentFn."""
        cfg = self._configs.get(agent_id.value, _DEFAULT_AGENT_CONFIG)
        model = cfg.get("model", _DEFAULT_AGENT_CONFIG["model"])
        temperature = cfg.get("temperature", _DEFAULT_AGENT_CONFIG["temperature"])
        num_ctx = cfg.get("num_ctx", _DEFAULT_AGENT_CONFIG["num_ctx"])
        num_predict = cfg.get("num_predict", _DEFAULT_AGENT_CONFIG["num_predict"])
        # num_predict n'est PAS cappé par token_budget : les modèles qui
        # pensent (think=True) consomment des tokens de thinking avant le
        # content. Le budget contrôle la taille visible via _sanitize_output.
        repeat_penalty = cfg.get("repeat_penalty", _DEFAULT_AGENT_CONFIG["repeat_penalty"])

        is_oppose = params.status == AgentStatus.OPPOSE
        full_prompt = self._augment_prompt(prompt, is_oppose)
        token_in = _estimate_tokens(full_prompt)

        log.info(
            "[Agent %s / %s] %s — calling (temp=%.1f, max_tok=%d)",
            agent_id.value, model, params.status.value, temperature, num_predict,
        )

        t0 = time.monotonic()
        try:
            raw_text = self._call_ollama(
                model, full_prompt, temperature, num_ctx,
                num_predict=num_predict, repeat_penalty=repeat_penalty,
                agent_id=agent_id.value,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            log.error("[Agent %s / %s] FAILED: %s", agent_id.value, model, exc)
            return self._error_turn(agent_id, params.status, token_in, latency_ms)

        latency_ms = (time.monotonic() - t0) * 1000
        token_out = _estimate_tokens(raw_text)
        empty_retry = False

        # Fix C : retry une fois si réponse vide, avec temp basse
        if token_out <= 1 and not raw_text.strip():
            log.warning(
                "[Agent %s / %s] empty response (token_out=%d), retry temp=0.4",
                agent_id.value, model, token_out,
            )
            retry_prompt = (
                full_prompt
                + "\n\nReponds avec au moins 1 phrase courte."
            )
            try:
                raw_text = self._call_ollama(
                    model, retry_prompt, 0.4, num_ctx,
                    num_predict=num_predict, repeat_penalty=repeat_penalty,
                    agent_id=agent_id.value,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                token_out = _estimate_tokens(raw_text)
                empty_retry = True
            except Exception as exc:
                log.error(
                    "[Agent %s / %s] retry FAILED: %s",
                    agent_id.value, model, exc,
                )

        log.info(
            "[Agent %s / %s] → %d tok out, %.0fms, %d chars%s",
            agent_id.value, model, token_out, latency_ms, len(raw_text),
            " (after retry)" if empty_retry else "",
        )

        if is_oppose:
            turn = self._parse_oppose_turn(
                agent_id, params, raw_text, token_in, token_out, latency_ms,
            )
        else:
            turn = self._parse_standard_turn(
                agent_id, params, raw_text, token_in, token_out, latency_ms,
            )
        turn.empty_retry = empty_retry
        return turn

    # ── Prompt ───────────────────────────────────────────────

    def _augment_prompt(self, prompt: str, is_oppose: bool) -> str:
        """Instructions contextuelles."""
        if is_oppose:
            return prompt + _OPPOSE_INSTRUCTION
        return prompt  # Plus rien à ajouter

    # ── Ollama call ──────────────────────────────────────────

    def _call_ollama(
        self,
        model: str,
        prompt: str,
        temperature: float,
        num_ctx: int,
        num_predict: int = 300,
        repeat_penalty: float = 1.5,
        agent_id: str = "A",
    ) -> str:
        """Appel isolé à ollama.chat — facilement mockable.

        Retourne le texte brut complet (thinking + content) AVANT
        troncature. La sanitization se fait dans _parse_standard_turn/_parse_oppose_turn.
        """
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _build_system_prompt(agent_id)},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            },
            think=True,
            keep_alive=30,
        )
        return _extract_content(response)

    # ── Parsing ──────────────────────────────────────────────

    def _parse_standard_turn(
        self,
        agent_id: AgentId,
        params: AgentParams,
        raw_text: str,
        token_in: int,
        token_out: int,
        latency_ms: float,
    ) -> AgentTurn:
        """Parse une réponse LEAD/SUPPORT : texte libre, signaux par heuristiques."""
        clean_text = _sanitize_output(raw_text)
        signals = _estimate_signals_from_text(clean_text, params.status)

        return AgentTurn(
            agent=agent_id,
            status=params.status,
            text=clean_text,
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            novelty=signals["novelty"],
            conflict=signals["conflict"],
            cohesion=signals["cohesion"],
            impl_pressure=signals["impl_pressure"],
        )

    def _parse_oppose_turn(
        self,
        agent_id: AgentId,
        params: AgentParams,
        raw_text: str,
        token_in: int,
        token_out: int,
        latency_ms: float,
    ) -> AgentTurn:
        """Parse une réponse OPPOSE : texte libre, veto si commence par VETO."""
        veto = False
        veto_reason = ""
        text = raw_text.strip()

        # Détection veto par préfixe
        if text.upper().startswith("VETO"):
            veto = True
            veto_reason = text[4:].strip().lstrip(":").strip()
            text = veto_reason or text

        text = _sanitize_output(text)
        signals = _estimate_signals_from_text(text, params.status)

        return AgentTurn(
            agent=agent_id,
            status=params.status,
            text=text,
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            novelty=signals["novelty"],
            conflict=signals["conflict"],
            cohesion=signals["cohesion"],
            impl_pressure=signals["impl_pressure"],
            veto=veto,
            veto_reason=veto_reason,
        )

    # ── Error fallback ───────────────────────────────────────

    def _error_turn(
        self,
        agent_id: AgentId,
        status: AgentStatus,
        token_in: int,
        latency_ms: float,
    ) -> AgentTurn:
        """AgentTurn sûr quand l'appel LLM échoue."""
        return AgentTurn(
            agent=agent_id,
            status=status,
            text="",
            token_in=token_in,
            token_out=0,
            latency_ms=latency_ms,
        )
