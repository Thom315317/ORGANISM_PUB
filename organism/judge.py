"""
Judge Pipeline - Evaluation des drafts agents
===============================================
Pipeline en deux etapes :
1. Summarizer (120B cloud) : resume N drafts + safety check
2. Judge (8B local) : choisit le meilleur, produit competition data

Le juge est PETIT expres (8B). Un petit juge hesite plus,
produisant des donnees de competition plus riches (margin_1v2).

Usage:
    pipeline = JudgePipeline()
    verdict = pipeline.evaluate(agent_turns, recent_winners)
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

import ollama

from organism.types import AgentId
from organism.config import ORGANISM_CONFIG
from organism.organism_state import JudgeVerdict, CompetitionPattern

log = logging.getLogger("organism.judge")

# -- Config ---------------------------------------------------------------

_JUDGE_CFG = ORGANISM_CONFIG.get("judge", {})
_SUMMARIZER_MODEL = _JUDGE_CFG.get("summarizer_model", "gpt-oss:120b-cloud")
_JUDGE_MODEL = _JUDGE_CFG.get("judge_model", "deepseek-r1:8b")
_SUMMARIZER_TEMP = _JUDGE_CFG.get("summarizer_temperature", 0.3)
_JUDGE_TEMP = _JUDGE_CFG.get("judge_temperature", 0.5)
_SUMMARIZER_CTX = _JUDGE_CFG.get("summarizer_num_ctx", 4096)
_JUDGE_CTX = _JUDGE_CFG.get("judge_num_ctx", 4096)
_SUMMARIZER_PREDICT = _JUDGE_CFG.get("summarizer_num_predict", 4000)
_JUDGE_PREDICT = _JUDGE_CFG.get("judge_num_predict", 4000)
# -- Auto-detect : cache par modèle (session-level) -------------------------
# model -> (use_think: bool, use_format_json: bool)
_MODEL_CAPS: Dict[str, tuple] = {}

# -- JSON extraction -------------------------------------------------------

_THINK_RE = re.compile(r"<(?:think|thinking)>.*?</(?:think|thinking)>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extrait le premier objet JSON valide d'un texte (robuste R1/reasoning)."""
    text = _THINK_RE.sub("", text)

    # 1. Try markdown ```json ... ``` blocks first
    for m in _CODE_BLOCK_RE.finditer(text):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue

    # 2. Find ALL { positions, extract balanced blocks, return the LARGEST valid one
    brace_positions = [i for i, c in enumerate(text) if c == '{']
    best = None
    for start in brace_positions:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        parsed = json.loads(candidate)
                        if best is None or len(candidate) > len(best[1]):
                            best = (parsed, candidate)
                    except json.JSONDecodeError:
                        pass
                    break
    if best is not None:
        return best[0]

    # 3. Repair truncated JSON: close unclosed braces
    brace_starts = [i for i, c in enumerate(text) if c == '{']
    if brace_starts:
        candidate = text[brace_starts[0]:]
        open_braces = candidate.count('{') - candidate.count('}')
        if open_braces > 0:
            repaired = candidate + '}' * open_braces
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # 4. Fallback: try the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _extract_fields(response: Any) -> tuple:
    """Extrait (content, thinking) d'une reponse ollama.

    Gere les variantes :
    - champ 'thinking' (deepseek-r1, gpt-oss)
    - champ 'think' (certains modeles Qwen)
    - balises <think>/<thinking> dans le content (modeles sans champ separe)
    """
    msg = response["message"] if isinstance(response, dict) else response.message
    if hasattr(msg, "model_dump"):
        msg_dict = msg.model_dump()
    elif hasattr(msg, "__dict__"):
        msg_dict = msg.__dict__
    else:
        msg_dict = msg if isinstance(msg, dict) else {"content": str(msg)}

    content = (msg_dict.get("content", "") or "").strip()

    # Chercher le thinking dans tous les champs possibles
    thinking = ""
    for field in ("thinking", "think", "reasoning"):
        val = (msg_dict.get(field, "") or "").strip()
        if val:
            thinking = val
            break

    # Si pas de champ thinking, extraire depuis les balises dans content
    if not thinking:
        for tag in ("think", "thinking", "reasoning"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                break

    # Nettoyer le content (retirer les balises think/thinking/reasoning)
    content = re.sub(
        r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
        "", content, flags=re.DOTALL
    ).strip()

    return content, thinking


def _extract_content(response: Any) -> str:
    """Extrait le texte utile d'une reponse ollama (compat legacy)."""
    content, thinking = _extract_fields(response)
    if content and len(content) > 30:
        return content
    parts = []
    if thinking:
        parts.append(thinking)
    if content:
        parts.append(content)
    return "\n".join(parts)


# -- Auto-detect : appel ollama avec détection automatique -----------------

def _has_valid_json(content: str, thinking: str) -> bool:
    """Vérifie si content ou thinking contient du JSON parseable."""
    if content:
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            if _extract_json(content):
                return True
    if thinking and _extract_json(thinking):
        return True
    return False


def _ollama_chat_smart(
    model: str,
    messages: list,
    options: dict,
    keep_alive: int = 30,
) -> Any:
    """Appelle ollama.chat avec auto-detection de think + format=json.

    Premier appel par modèle : teste 3 combinaisons dans l'ordre :
    1. think=True + format=json  (meilleur cas)
    2. format=json seul          (si think pose problème)
    3. texte libre               (si format=json n'est pas supporté)

    Mémorise la combinaison qui marche. Les appels suivants = 0 retry.
    """
    caps = _MODEL_CAPS.get(model)

    if caps is not None:
        use_think, use_format = caps
        kwargs = dict(
            model=model, messages=messages, options=options,
            keep_alive=keep_alive,
        )
        if use_format:
            kwargs["format"] = "json"
        if use_think:
            kwargs["think"] = True
        return ollama.chat(**kwargs)

    # Phase de découverte : tester les combinaisons
    combos = [
        (True, True, "think+json"),
        (False, True, "json only"),
        (False, False, "free text"),
    ]

    last_response = None
    for use_think, use_format, label in combos:
        try:
            kwargs = dict(
                model=model, messages=messages, options=options,
                keep_alive=keep_alive,
            )
            if use_format:
                kwargs["format"] = "json"
            if use_think:
                kwargs["think"] = True

            response = ollama.chat(**kwargs)
            last_response = response
            content, thinking = _extract_fields(response)

            if _has_valid_json(content, thinking):
                _MODEL_CAPS[model] = (use_think, use_format)
                log.info("auto-detect: %s -> %s (cache)", model, label)
                return response

            log.warning("auto-detect: %s %s -> JSON invalide, next...", model, label)

        except Exception as exc:
            log.warning("auto-detect: %s %s -> erreur: %s", model, label, exc)

    # Toutes les combinaisons ont échoué — cacher free text
    _MODEL_CAPS[model] = (False, False)
    log.error("auto-detect: %s -> aucune combinaison ne produit du JSON", model)
    return last_response


# -- Prompts ---------------------------------------------------------------

def _build_summarizer_prompt(drafts: Dict[str, str]) -> str:
    """Prompt pour le summarizer. Les drafts sont déjà anonymisés (clés = '1','2','3')."""
    n = len(drafts)
    draft_text = "\n\n".join(
        f"--- Draft {label} ---\n{text[:4000]}"
        for label, text in drafts.items()
    )
    return (
        f"Tu recois {n} drafts de penseurs differents. Pour chaque draft :\n"
        "1. Resume en 60 mots max en PRESERVANT le style de raisonnement "
        "(creatif, analytique, pragmatique — c'est important pour le juge)\n"
        "2. Qualite principale : CREATIF, PROFOND, CONCRET, ou AUTRE\n"
        "3. Safety : GO si acceptable, VETO + raison sinon\n"
        "4. 1-2 assertions SUBSTANTIVES du contenu "
        "(ce que le draft affirme, PAS sur le draft lui-meme)\n"
        "Identifie la tension principale entre les drafts (1 phrase).\n"
        "\nReponds en JSON strict :\n"
        '{"summaries": {"<id>": {"summary": "...", '
        '"quality": "CREATIF|PROFOND|CONCRET|AUTRE", '
        '"safety": "GO|VETO", '
        '"assertions": ["affirmation sur le sujet..."]}}, '
        '"main_tension": "..."}\n\n'
        f"{draft_text}"
    )


def _build_judge_prompt(
    summaries: Dict[str, Dict[str, str]],
    main_tension: str,
    recent_winners: List[str],
) -> str:
    """Prompt pour le juge. Les résumés sont déjà anonymisés (clés = '1','2','3').

    L'historique des gagnants est présenté de façon anonyme :
    pas de noms d'agents, juste une note anti-stagnation si un
    contributeur domine.
    """
    n = len(summaries)
    summaries_text = "\n".join(
        f"  Draft {label}: {s.get('summary', '?')}"
        for label, s in summaries.items()
    )

    # Anti-stagnation note (anonyme — pas de noms d'agents)
    dominance_note = ""
    if len(recent_winners) >= 5:
        from collections import Counter
        counts = Counter(recent_winners[-10:])
        _, top_count = counts.most_common(1)[0]
        if top_count >= 6:
            dominance_note = (
                f"\nATTENTION : un meme contributeur a gagne {top_count}/10 derniers ticks. "
                "Cherche ACTIVEMENT les qualites des autres drafts. "
                "Un draft original ou profond peut valoir plus qu'un draft simplement clair.\n"
            )

    return (
        f"Tu juges {n} resumes de pensees. Tu es un evaluateur EXIGEANT et EQUITABLE.\n\n"
        "Les drafts sont presentes dans un ORDRE ALEATOIRE. "
        "NE donne PAS d'avantage au premier ou au dernier draft.\n\n"
        "Ton role : comme un prof qui encourage la diversite de pensee.\n"
        "- Un draft CREATIF (angle original, analogie surprenante) a de la valeur "
        "meme s'il est moins structure.\n"
        "- Un draft PROFOND (analyse incisive, faille identifiee) a de la valeur "
        "meme s'il est moins concret.\n"
        "- Un draft CONCRET (plan, synthese) a de la valeur "
        "mais ne devrait PAS gagner par defaut.\n\n"
        "Evalue uniquement la qualite du contenu, la pertinence et "
        "la profondeur du raisonnement. La longueur de la reponse "
        "ne doit PAS influencer ton jugement. Une reponse courte et "
        "precise est superieure a une reponse longue et diluee.\n\n"
        f"Resumes :\n{summaries_text}\n\n"
        f"Tension principale : {main_tension}\n"
        f"{dominance_note}\n"
        "CLAIMS : extrais les affirmations SUBSTANTIVES sur le sujet "
        "(ex: 'la conscience est une illusion narrative'). "
        "PAS d'observations sur les drafts eux-memes.\n"
        "Si un claim contredit un fait anterieur, mets status=contradicted.\n\n"
        "SIGNAUX — calibre avec precision sur TOUTE l'echelle 0-100 :\n"
        "- novelty: 0-20 = redite pure, 20-40 = variation mineure, "
        "40-60 = angle different, 60-80 = idee nouvelle, "
        "80-100 = concept jamais vu dans cette conversation\n"
        "- conflict: 0-20 = consensus total, 20-40 = nuances de style, "
        "40-60 = approches differentes sur un point, "
        "60-80 = desaccord substantiel, "
        "80-100 = contradiction directe entre les drafts\n"
        "- cohesion: 0-20 = aucun lien, 20-40 = theme commun vague, "
        "40-60 = construisent sur les memes idees, "
        "60-80 = synthese coherente, 80-100 = convergence totale\n"
        "- impl_pressure: 0-20 = purement theorique, "
        "20-40 = pistes vagues, 40-60 = plan esquisse, "
        "60-80 = plan actionnable, 80-100 = pret a implementer\n\n"
        "IMPORTANT : repartis tes signaux sur toute l'echelle. "
        "Si les 3 drafts disent des choses similaires avec des "
        "styles differents, conflict doit etre SOUS 40. "
        "Reserve 80+ pour les cas exceptionnels.\n\n"
        "Reponds en JSON strict :\n"
        '{"winner": "<id>", "reason": "pourquoi CE draft apporte le plus", '
        '"confidence": 0-100, '
        '"signals": {"novelty": 0-100, "conflict": 0-100, "cohesion": 0-100, "impl_pressure": 0-100}, '
        '"claims": [{"text": "affirmation substantive...", '
        '"status": "hypothesis|supported|contradicted", "source": "<id>"}], '
        '"competition": {"ranking": ["1er", "2eme", "3eme"], '
        '"margin_1v2": 0-100, "margin_2v3": 0-100, '
        '"counterfactual": "... aurait gagne si ..."}}\n'
        "IMPORTANT : utilise des valeurs ENTIERES precises de 0 a 100 "
        "(ex: 63, 27, 81). Evite les nombres ronds (50, 60, 70).\n"
        "margin_1v2 FAIBLE (10-35) = les drafts sont proches, hesitation. "
        "margin_1v2 HAUTE (70-95) = un draft domine clairement."
    )


# -- JudgePipeline ---------------------------------------------------------


def normalize_judge_verdict(
    data: Dict[str, Any],
    valid_agents: List[str],
) -> tuple:
    """
    Normalize raw judge JSON into clean verdict data.

    Rules:
      R1: winner == competition.ranking[0] (invariant)
      R2: if no valid winner/ranking → winner = None (judge_failed)
      R3: complete ranking to len(valid_agents)
      R4: canonicalize agent IDs (strip 'Agent ', case, etc.)
      R5: normalize 0-100 → 0.0-1.0

    Returns:
        (winner, ranking, confidence, signals, claims, comp_data, audit)
        where audit = {"fixes": [...], "judge_failed": bool}
    """
    audit: Dict[str, Any] = {"fixes": [], "judge_failed": False}
    norm = JudgePipeline._norm100
    normalize_id = JudgePipeline._normalize_agent_id

    # --- R4+R2: Winner ---
    raw_winner = str(data.get("winner", ""))
    winner = normalize_id(raw_winner, valid_agents)
    if not winner:
        # Try ranking[0] as fallback
        raw_ranking = data.get("competition", {}).get("ranking", [])
        if isinstance(raw_ranking, list):
            for r in raw_ranking:
                w = normalize_id(str(r), valid_agents)
                if w:
                    winner = w
                    audit["fixes"].append(f"winner_from_ranking: '{raw_winner}' → {winner}")
                    break
    if not winner:
        # R2: winner = None, count as judge_failed
        winner = None
        audit["judge_failed"] = True
        audit["fixes"].append(f"winner_none: raw='{raw_winner}'")

    # --- R5: Confidence ---
    confidence = norm(data.get("confidence", 50), 0.5)

    # --- R5: Signals ---
    raw_signals = data.get("signals", {})
    if not isinstance(raw_signals, dict):
        raw_signals = {}
    signals = {
        "novelty": norm(raw_signals.get("novelty", 0), 0.0),
        "conflict": norm(raw_signals.get("conflict", 0), 0.0),
        "cohesion": norm(raw_signals.get("cohesion", 50), 0.5),
        "impl_pressure": norm(raw_signals.get("impl_pressure", 0), 0.0),
    }

    # --- Claims ---
    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
    claims = tuple(
        {"text": str(c.get("text", "")),
         "status": str(c.get("status", "hypothesis")),
         "source": normalize_id(str(c.get("source", "")), valid_agents) or str(c.get("source", ""))}
        for c in raw_claims
        if isinstance(c, dict) and c.get("text")
    )

    # --- R4+R3+R1: Ranking ---
    raw_comp = data.get("competition", {})
    if not isinstance(raw_comp, dict):
        raw_comp = {}
    raw_ranking = raw_comp.get("ranking", [])
    if not isinstance(raw_ranking, list):
        raw_ranking = []

    ranking = []
    for r in raw_ranking:
        normalized = normalize_id(str(r), valid_agents)
        if normalized and normalized not in ranking:
            ranking.append(normalized)

    # FIX 2: Audit — record original ranking state before completion
    audit["ranking_original"] = list(ranking)
    audit["ranking_original_len"] = len(ranking)

    # R1: ensure winner == ranking[0]
    if winner is not None:
        if ranking and ranking[0] != winner:
            audit["fixes"].append(f"ranking_reorder: ranking[0]={ranking[0]} → {winner}")
            if winner in ranking:
                ranking.remove(winner)
            ranking.insert(0, winner)
        elif not ranking:
            ranking = [winner]

    # R3: complete ranking to all valid agents
    was_corrected = False
    for agent in valid_agents:
        if agent not in ranking:
            ranking.append(agent)
            was_corrected = True
    if not ranking and winner is None:
        ranking = list(valid_agents)
        was_corrected = True
    audit["was_corrected"] = was_corrected

    ranking_tuple = tuple(ranking)

    # Competition data
    comp_data = {
        "margin_1v2": norm(raw_comp.get("margin_1v2", 50), 0.5),
        "margin_2v3": norm(raw_comp.get("margin_2v3", 0), 0.0),
        "counterfactual": str(raw_comp.get("counterfactual", "")),
    }

    return winner, ranking_tuple, confidence, signals, claims, comp_data, audit


class JudgePipeline:
    """
    Pipeline Summarizer (120B) -> Judge (8B).

    Température adaptative : le juge ajuste sa température en fonction
    de la variance des margins récentes. Trop peu de variance → on monte
    la température (plus d'exploration). Trop → on descend (plus stable).
    """

    # Bornes de température adaptative
    _TEMP_MIN = 0.3
    _TEMP_MAX = 0.8
    _TARGET_VAR_LOW = 0.02   # en-dessous → trop uniforme, monter temp
    _TARGET_VAR_HIGH = 0.15  # au-dessus → trop chaotique, baisser temp
    _ADAPT_WINDOW = 20       # fenêtre glissante pour la variance

    def __init__(
        self,
        summarizer_model: Optional[str] = None,
        judge_model: Optional[str] = None,
    ):
        self._summarizer_model = summarizer_model or _SUMMARIZER_MODEL
        self._judge_model = judge_model or _JUDGE_MODEL
        self._judge_temp = _JUDGE_TEMP
        self._call_count = 0
        self._valid_json_count = 0
        self._temp_history: List[float] = []

    def evaluate(
        self,
        agent_turns: list,
        recent_winners: Optional[List[str]] = None,
    ) -> Optional[JudgeVerdict]:
        """
        Evalue les drafts agents et retourne un verdict.

        Les drafts sont anonymisés (shuffled, relabeled "1","2","3")
        avant d'être envoyés au summarizer et au juge. Le verdict
        est dé-anonymisé avant retour.
        """
        if not agent_turns:
            return None

        drafts = {}
        for turn in agent_turns:
            aid = turn.agent.value if hasattr(turn.agent, "value") else str(turn.agent)
            if turn.text and turn.text.strip():
                drafts[aid] = turn.text.strip()

        # All agent IDs (including those with empty drafts) for ranking completion
        all_agent_ids = []
        for turn in agent_turns:
            aid = turn.agent.value if hasattr(turn.agent, "value") else str(turn.agent)
            if aid not in all_agent_ids:
                all_agent_ids.append(aid)

        if len(drafts) < 2:
            if drafts:
                winner_id = next(iter(drafts))
                # FIX 8: Complete ranking with all agents, not just drafters
                full_ranking = [winner_id] + sorted(a for a in all_agent_ids if a != winner_id)
                log.info("ranking_generated_from_winner: winner=%s, ranking=%s", winner_id, full_ranking)
                # FIX 11: Use agent heuristics for novelty/impl_pressure if available
                sole_turn = agent_turns[0] if agent_turns else None
                if sole_turn and hasattr(sole_turn, 'novelty'):
                    signals = {
                        "novelty": getattr(sole_turn, 'novelty', 0.0),
                        "conflict": 0.0,  # No conflict with single draft
                        "cohesion": 1.0,
                        "impl_pressure": getattr(sole_turn, 'impl_pressure', 0.0),
                    }
                else:
                    signals = {"novelty": 0.0, "conflict": 0.0, "cohesion": 1.0, "impl_pressure": 0.0}
                return JudgeVerdict(
                    winner=winner_id,
                    reason="Draft unique",
                    confidence=1.0,
                    signals=signals,
                    competition=CompetitionPattern(
                        ranking=tuple(full_ranking),
                        margin_1v2=1.0,
                        margin_2v3=0.0,
                        counterfactual="",
                    ),
                    raw_json={"_audit": {"fixes": ["ranking_generated_from_winner"]}},
                )
            return None

        self._call_count += 1
        recent = recent_winners or []

        # --- Anonymisation : shuffle + relabel "1","2","3" ---
        real_ids = list(drafts.keys())
        shuffled_ids = real_ids[:]
        random.shuffle(shuffled_ids)
        # Mapping: anonymous label → real agent ID
        anon_to_real = {str(i + 1): aid for i, aid in enumerate(shuffled_ids)}
        real_to_anon = {aid: str(i + 1) for i, aid in enumerate(shuffled_ids)}
        log.debug("judge anonymisation: %s", real_to_anon)

        # Anonymised drafts
        anon_drafts = {real_to_anon[aid]: text for aid, text in drafts.items()}

        # Step 1: Summarizer (receives anonymous drafts)
        summaries, main_tension = self._call_summarizer(anon_drafts)
        if not summaries:
            log.warning("Summarizer failed, using raw drafts as summaries")
            summaries = {
                label: {"summary": text[:100], "safety": "GO"}
                for label, text in anon_drafts.items()
            }
            main_tension = ""

        for label, s in summaries.items():
            if s.get("safety", "GO").upper().startswith("VETO"):
                real_aid = anon_to_real.get(label, label)
                log.warning("Safety VETO on draft %s (anon=%s): %s", real_aid, label, s.get("safety"))

        # Step 2: Judge (receives anonymous summaries)
        anon_agents = [str(i + 1) for i in range(len(shuffled_ids))]
        verdict = self._call_judge(summaries, main_tension, recent, anon_agents)

        # --- Dé-anonymisation du verdict ---
        if verdict:
            verdict = self._deanonymize_verdict(verdict, anon_to_real, real_ids)
            # Store anon mapping in raw_json for audit (FIX 5E)
            if verdict.raw_json and isinstance(verdict.raw_json, dict):
                verdict.raw_json["_anon_map"] = real_to_anon
                verdict.raw_json["_anon_reverse"] = anon_to_real

        return verdict

    @staticmethod
    def _deanonymize_verdict(
        verdict: JudgeVerdict,
        anon_to_real: Dict[str, str],
        real_ids: List[str],
    ) -> JudgeVerdict:
        """Remplace les labels anonymes (1,2,3) par les vrais IDs (A,B,C)."""

        def _remap(label: Optional[str]) -> Optional[str]:
            if label is None:
                return None
            return anon_to_real.get(str(label).strip(), label)

        new_winner = _remap(verdict.winner)

        new_ranking = None
        new_competition = None
        if verdict.competition:
            new_ranking = tuple(_remap(r) for r in verdict.competition.ranking)
            # Compléter le ranking avec les agents manquants
            for aid in real_ids:
                if aid not in new_ranking:
                    new_ranking = new_ranking + (aid,)
            new_competition = CompetitionPattern(
                ranking=new_ranking,
                margin_1v2=verdict.competition.margin_1v2,
                margin_2v3=verdict.competition.margin_2v3,
                counterfactual=verdict.competition.counterfactual,
            )

        new_claims = tuple(
            {**c, "source": _remap(c.get("source", ""))}
            for c in verdict.claims
        ) if verdict.claims else ()

        return JudgeVerdict(
            winner=new_winner,
            reason=verdict.reason,
            confidence=verdict.confidence,
            signals=verdict.signals,
            claims=new_claims,
            competition=new_competition,
            raw_json=verdict.raw_json,
        )

    def adapt_temperature(self, recent_margins: List[float]) -> float:
        """
        Auto-régulation de la température du juge.

        Calcule la variance des margins récentes et ajuste :
        - Variance basse (< 0.02) → le juge est trop uniforme → monter temp
        - Variance haute (> 0.15) → trop chaotique → baisser temp
        - Entre les deux → maintenir

        Retourne la nouvelle température.
        """
        if len(recent_margins) < 5:
            return self._judge_temp  # Pas assez de données

        window = recent_margins[-self._ADAPT_WINDOW:]
        n = len(window)
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / n

        old_temp = self._judge_temp

        if var < self._TARGET_VAR_LOW:
            # Trop uniforme → explorer plus
            self._judge_temp = min(self._TEMP_MAX, self._judge_temp + 0.05)
        elif var > self._TARGET_VAR_HIGH:
            # Trop chaotique → stabiliser
            self._judge_temp = max(self._TEMP_MIN, self._judge_temp - 0.05)
        # Sinon : zone optimale, ne rien changer

        if self._judge_temp != old_temp:
            log.info("Judge temp adapted: %.2f → %.2f (var=%.4f, window=%d)",
                     old_temp, self._judge_temp, var, n)

        self._temp_history.append(self._judge_temp)
        return self._judge_temp

    def _call_summarizer(self, drafts: Dict[str, str]) -> tuple:
        """Appelle le summarizer."""
        prompt = _build_summarizer_prompt(drafts)
        t0 = time.monotonic()

        try:
            response = _ollama_chat_smart(
                model=self._summarizer_model,
                messages=[
                    {"role": "system", "content": "Tu es un evaluateur objectif. Reponds uniquement en JSON."},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": _SUMMARIZER_TEMP,
                    "num_ctx": _SUMMARIZER_CTX,
                    "num_predict": _SUMMARIZER_PREDICT,
                },
            )
            content, thinking = _extract_fields(response)
            latency = (time.monotonic() - t0) * 1000
            log.info("Summarizer [%s] -> content=%d chars, thinking=%d chars, %.0fms",
                     self._summarizer_model, len(content), len(thinking), latency)

            data = None
            # 1. Try content directly (format="json" gives clean JSON)
            if content:
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    data = _extract_json(content)

            # 2. Fallback: extract JSON from thinking field
            if not data and thinking:
                log.info("Summarizer: content empty, extracting from thinking (%d chars)", len(thinking))
                data = _extract_json(thinking)

            # 3. Last resort: combine everything
            if not data:
                raw = _extract_content(response)
                data = _extract_json(raw)

            if data and "summaries" in data:
                return data["summaries"], data.get("main_tension", "")
            log.warning("Summarizer JSON invalid: content=%s thinking=%s",
                        content[:100], thinking[:100])
            return None, ""

        except Exception as exc:
            log.error("Summarizer failed: %s", exc)
            return None, ""

    def _call_judge(
        self,
        summaries: Dict[str, Dict[str, str]],
        main_tension: str,
        recent_winners: List[str],
        valid_agents: List[str],
    ) -> Optional[JudgeVerdict]:
        """Appelle le juge 8B local."""
        prompt = _build_judge_prompt(summaries, main_tension, recent_winners)
        t0 = time.monotonic()

        try:
            response = _ollama_chat_smart(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": "Tu es un juge impartial. Reponds uniquement en JSON."},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": self._judge_temp,
                    "num_ctx": _JUDGE_CTX,
                    "num_predict": _JUDGE_PREDICT,
                },
            )
            content, thinking = _extract_fields(response)
            latency = (time.monotonic() - t0) * 1000
            log.info("Judge [%s] -> content=%d chars, thinking=%d chars, %.0fms",
                     self._judge_model, len(content), len(thinking), latency)

            data = None
            # 1. Try content directly
            if content:
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    data = _extract_json(content)

            # 2. Fallback: extract JSON from thinking field
            if not data and thinking:
                log.info("Judge: content empty, extracting from thinking (%d chars)", len(thinking))
                data = _extract_json(thinking)

            # 3. Last resort
            if not data:
                raw = _extract_content(response)
                data = _extract_json(raw)

            if not data:
                log.warning("Judge JSON invalid: %s", content[:300])
                return self._fallback_verdict(valid_agents)

            self._valid_json_count += 1
            return self._parse_verdict(data, valid_agents)

        except Exception as exc:
            log.error("Judge failed: %s", exc)
            return self._fallback_verdict(valid_agents)

    @staticmethod
    def _norm100(val: float, default: float = 0.0) -> float:
        """Normalise une valeur 0-100 vers 0.0-1.0 (retro-compatible 0-1)."""
        v = float(val) if val is not None else default
        if v > 1.0:
            v = v / 100.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _normalize_agent_id(raw: str, valid_agents: List[str]) -> Optional[str]:
        """Normalize winner variants: 'Agent B', 'Draft 1', 'agent_b', ' B ', 'b', '1' → match."""
        cleaned = (raw.strip().upper()
                   .replace("AGENT", "").replace("DRAFT", "")
                   .replace("_", "").replace(" ", ""))
        if cleaned in [a.upper() for a in valid_agents]:
            for a in valid_agents:
                if a.upper() == cleaned:
                    return a
        return None

    def _parse_verdict(
        self, data: Dict[str, Any], valid_agents: List[str],
    ) -> JudgeVerdict:
        """Parse le JSON du juge en JudgeVerdict (robust + normalized)."""
        winner, ranking, confidence, signals, claims, comp_data, audit = \
            normalize_judge_verdict(data, valid_agents)

        competition = None
        if ranking:
            competition = CompetitionPattern(
                ranking=ranking,
                margin_1v2=comp_data["margin_1v2"],
                margin_2v3=comp_data["margin_2v3"],
                counterfactual=comp_data["counterfactual"],
            )

        # Store audit in raw_json
        raw = dict(data)
        raw["_audit"] = audit

        if audit["judge_failed"]:
            log.warning("judge: judge_failed — no valid winner (raw='%s')",
                        data.get("winner", ""))
        elif audit["fixes"]:
            log.info("judge: normalized with fixes: %s", audit["fixes"])

        return JudgeVerdict(
            winner=winner,
            reason=str(data.get("reason", "")),
            confidence=confidence,
            signals=signals,
            claims=claims,
            competition=competition,
            raw_json=raw,
        )

    def _fallback_verdict(self, valid_agents: List[str]) -> JudgeVerdict:
        """Verdict de fallback quand le juge echoue.
        winner=None — tick comptabilise comme judge_failed.
        signals=None → l'orchestrateur utilisera _aggregate_signals(turns)
        au lieu de constantes artificielles [0, 0, 0.5, 0]."""
        return JudgeVerdict(
            winner=None,
            reason="fallback - judge parse error",
            confidence=0.0,
            signals=None,
            competition=CompetitionPattern(
                ranking=tuple(valid_agents),
                margin_1v2=0.5,
                margin_2v3=0.0,
                counterfactual="",
            ),
            raw_json={"_audit": {"fixes": ["fallback_verdict"], "judge_failed": True}},
        )

    @property
    def valid_json_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._valid_json_count / self._call_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self._call_count,
            "valid_json_count": self._valid_json_count,
            "valid_json_rate": round(self.valid_json_rate, 3),
            "summarizer_model": self._summarizer_model,
            "judge_model": self._judge_model,
            "judge_temperature": round(self._judge_temp, 3),
            "temp_history": [round(t, 3) for t in self._temp_history[-20:]],
        }
