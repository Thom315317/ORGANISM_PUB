"""
P3 — World Model (Mʷ) + Projections
=====================================
Graphe de claims avec provenance obligatoire, liens SUPPORTS/CONTRADICTS/SUPERSEDES,
et système de prédictions avec mise à jour de confidence.

Le world model ne fabrique rien : il enregistre, lie, et expose.
Toute mutation est tracée dans Mᵣ (CLAIM_ADDED, CLAIM_REVISED, CLAIM_CONTRADICTED).

Invariants :
  - Une claim sans provenance (chunk_ids Mᵣ) ne sort JAMAIS comme "fait"
  - Une contradiction ↓ confidence + lien CONTRADICTED_BY
  - prediction_made → prediction_checked → confidence update automatique

Usage:
    wm = WorldModel(mr=mr)
    claim = wm.add_claim("L'utilisateur préfère le français", 0.8,
                         provenance=[event.chunk_id], source_agent=AgentId.A)
    wm.contradict_claim(claim.claim_id, contradicting_chunk_id=e2.chunk_id,
                        source_agent=AgentId.B)
    facts = wm.get_facts(min_confidence=0.7)
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set

from organism.types import (
    AgentId, EventType,
    Claim, ClaimStatus, ClaimRelation, ClaimLink, Prediction,
)
from organism.mr import RealityMemory

log = logging.getLogger("organism.world_model")

# Mots de négation français pour détection de contradiction
_NEGATION_WORDS = frozenset({
    "ne", "pas", "plus", "jamais", "rien", "aucun", "aucune",
    "non", "ni", "sans", "impossible", "faux", "inexact",
    "not", "no", "never", "cannot", "impossible", "false",
})

# Mots-outils à ignorer pour le calcul de similarité
_STOP_WORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "car", "que", "qui", "quoi", "dont",
    "est", "sont", "a", "ont", "fait", "peut", "dans", "sur", "par",
    "pour", "avec", "en", "ce", "cette", "ces", "il", "elle", "ils",
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "and", "or", "but", "for", "with", "on", "at", "from", "by",
})


def _claim_id() -> str:
    """Génère un identifiant unique pour une claim."""
    return f"claim:{uuid.uuid4().hex[:16]}"


def _prediction_id() -> str:
    """Génère un identifiant unique pour une prédiction."""
    return f"pred:{uuid.uuid4().hex[:16]}"


def _tokenize(text: str) -> Set[str]:
    """Tokenise un texte en mots significatifs (lowercase, sans stop words)."""
    words = re.findall(r"[a-zàâäéèêëïîôùûüçœæ]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _has_negation(text: str) -> bool:
    """Détecte la présence de négation dans un texte."""
    words = set(re.findall(r"[a-zàâäéèêëïîôùûüçœæ]+", text.lower()))
    return bool(words & _NEGATION_WORDS)


def _semantic_overlap(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Jaccard similarity entre deux ensembles de tokens."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class WorldModel:
    """
    World Model (Mʷ) : graphe de claims avec provenance.
    Toutes les mutations sont tracées dans Mᵣ.

    Auto-contradiction : quand une nouvelle claim partage du vocabulaire
    avec une existante mais porte une négation, l'ancienne est contredite.
    """

    def __init__(self, mr: RealityMemory, tick_id: int = 0):
        self._mr = mr
        self._tick_id = tick_id
        self._claims: Dict[str, Claim] = {}
        self._predictions: Dict[str, Prediction] = {}
        self._claim_tokens: Dict[str, Set[str]] = {}  # Cache de tokens par claim

    @property
    def tick_id(self) -> int:
        return self._tick_id

    @tick_id.setter
    def tick_id(self, value: int) -> None:
        self._tick_id = value

    # ── Claims ────────────────────────────────────────────────────

    def add_claim(
        self,
        content: str,
        confidence: float,
        provenance: Optional[List[str]] = None,
        source_agent: AgentId = AgentId.O,
        status_hint: Optional[str] = None,
    ) -> Claim:
        """
        Ajoute une claim au world model.
        Provenance = liste de chunk_ids Mᵣ.
        Sans provenance, la claim reste HYPOTHESIS (jamais "fait").

        Auto-contradiction : si la nouvelle claim partage du vocabulaire
        avec une existante mais porte une négation opposée, l'ancienne
        est automatiquement contredite.
        """
        confidence = max(0.0, min(1.0, confidence))
        prov = provenance or []

        # Filtrer les claims méta-évaluatives (sur les drafts, pas sur le monde)
        content_lower = content.lower()
        meta_patterns = [
            "le resume", "le draft", "la reponse", "summary",
            "doit etre en", "format", "should be", "response must",
        ]
        if any(p in content_lower for p in meta_patterns):
            log.debug("Claim filtrée (méta-évaluative) : %s", content[:80])
            # Créer quand même mais avec confidence basse pour ne pas polluer get_facts
            confidence = min(confidence, 0.15)

        claim = Claim(
            claim_id=_claim_id(),
            content=content,
            confidence=confidence,
            provenance=list(prov),
            source_agent=source_agent,
            status=ClaimStatus.HYPOTHESIS,
            created_at=time.time(),
        )

        # Si le juge a dit "contradicted", marquer directement
        if status_hint == "contradicted":
            claim.status = ClaimStatus.CONTRADICTED
            claim.confidence = min(claim.confidence, 0.25)
        elif claim.has_provenance():
            claim.status = ClaimStatus.SUPPORTED

        self._claims[claim.claim_id] = claim
        new_tokens = _tokenize(content)
        self._claim_tokens[claim.claim_id] = new_tokens

        # Auto-contradiction : chercher les claims existantes contradites
        if new_tokens:
            self._auto_detect_contradictions(claim, new_tokens, prov)

        # Tracer dans Mᵣ
        self._mr.append(
            event_type=EventType.CLAIM_ADDED,
            tick_id=self._tick_id,
            agent=source_agent,
            payload={
                "claim_id": claim.claim_id,
                "content": content,
                "confidence": confidence,
                "provenance": prov,
                "status": claim.status.value,
            },
        )

        return claim

    def _auto_detect_contradictions(
        self,
        new_claim: Claim,
        new_tokens: Set[str],
        provenance: List[str],
    ) -> None:
        """
        Détecte automatiquement les contradictions entre la nouvelle claim
        et les claims existantes.

        Logique : si deux claims partagent ≥40% de vocabulaire mais une
        porte une négation et l'autre non → contradiction.
        """
        new_has_neg = _has_negation(new_claim.content)

        for cid, existing in list(self._claims.items()):
            if cid == new_claim.claim_id:
                continue
            if existing.status in (ClaimStatus.RETRACTED, ClaimStatus.CONTRADICTED):
                continue

            existing_tokens = self._claim_tokens.get(cid)
            if existing_tokens is None:
                existing_tokens = _tokenize(existing.content)
                self._claim_tokens[cid] = existing_tokens

            overlap = _semantic_overlap(new_tokens, existing_tokens)
            if overlap < 0.35:
                continue

            existing_has_neg = _has_negation(existing.content)

            # Contradiction si négation asymétrique sur contenu similaire
            if new_has_neg != existing_has_neg:
                chunk_id = provenance[0] if provenance else ""
                log.info(
                    "Auto-contradiction : '%s' vs '%s' (overlap=%.2f)",
                    new_claim.content[:60], existing.content[:60], overlap,
                )
                self.contradict_claim(
                    cid,
                    contradicting_chunk_id=chunk_id,
                    source_agent=new_claim.source_agent,
                    confidence_penalty=0.25,
                )

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retourne une claim par son ID."""
        return self._claims.get(claim_id)

    def revise_claim(
        self,
        claim_id: str,
        new_confidence: float,
        reason_chunk_id: str = "",
        source_agent: AgentId = AgentId.O,
    ) -> Optional[Claim]:
        """
        Révise la confidence d'une claim.
        Ajoute la raison à la provenance si fournie.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return None

        old_confidence = claim.confidence
        claim.confidence = max(0.0, min(1.0, new_confidence))
        claim.revised_at = time.time()

        if reason_chunk_id:
            claim.provenance.append(reason_chunk_id)

        # Mettre à jour le status si nécessaire
        if claim.confidence < 0.3 and claim.status == ClaimStatus.SUPPORTED:
            claim.status = ClaimStatus.CONTRADICTED

        # Tracer dans Mᵣ
        self._mr.append(
            event_type=EventType.CLAIM_REVISED,
            tick_id=self._tick_id,
            agent=source_agent,
            payload={
                "claim_id": claim_id,
                "old_confidence": round(old_confidence, 4),
                "new_confidence": round(claim.confidence, 4),
                "reason_chunk_id": reason_chunk_id,
            },
        )

        return claim

    def contradict_claim(
        self,
        claim_id: str,
        contradicting_chunk_id: str,
        source_agent: AgentId = AgentId.B,
        confidence_penalty: float = 0.3,
    ) -> Optional[Claim]:
        """
        Contredit une claim :
        1. Baisse la confidence (penalty)
        2. Ajoute un lien CONTRADICTS
        3. Passe en status CONTRADICTED si confidence < 0.3
        4. Trace dans Mᵣ (CLAIM_CONTRADICTED)
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return None

        old_confidence = claim.confidence
        claim.confidence = max(0.0, claim.confidence - confidence_penalty)
        claim.revised_at = time.time()

        # Lien CONTRADICTED_BY
        claim.links.append(ClaimLink(
            relation=ClaimRelation.CONTRADICTS,
            target_claim_id="",  # Pas de claim cible — c'est une contradiction externe
            source_chunk_id=contradicting_chunk_id,
        ))

        # Status update
        if claim.confidence < 0.3:
            claim.status = ClaimStatus.CONTRADICTED

        # Tracer dans Mᵣ
        self._mr.append(
            event_type=EventType.CLAIM_CONTRADICTED,
            tick_id=self._tick_id,
            agent=source_agent,
            conflict=min(1.0, confidence_penalty),
            payload={
                "claim_id": claim_id,
                "old_confidence": round(old_confidence, 4),
                "new_confidence": round(claim.confidence, 4),
                "contradicting_chunk_id": contradicting_chunk_id,
                "penalty": confidence_penalty,
            },
        )

        return claim

    def support_claim(
        self,
        claim_id: str,
        supporting_chunk_id: str,
        source_agent: AgentId = AgentId.A,
        confidence_boost: float = 0.1,
    ) -> Optional[Claim]:
        """
        Renforce une claim :
        1. Augmente la confidence
        2. Ajoute le chunk_id à la provenance
        3. Ajoute un lien SUPPORTS
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return None

        claim.confidence = min(1.0, claim.confidence + confidence_boost)
        claim.revised_at = time.time()
        claim.provenance.append(supporting_chunk_id)

        claim.links.append(ClaimLink(
            relation=ClaimRelation.SUPPORTS,
            target_claim_id="",
            source_chunk_id=supporting_chunk_id,
        ))

        # Si la claim était contredite mais regagne en confidence
        if claim.confidence >= 0.5 and claim.status == ClaimStatus.CONTRADICTED:
            claim.status = ClaimStatus.SUPPORTED

        # Si la claim était une hypothèse sans provenance mais en a maintenant
        if claim.status == ClaimStatus.HYPOTHESIS and claim.has_provenance():
            claim.status = ClaimStatus.SUPPORTED

        self._mr.append(
            event_type=EventType.CLAIM_REVISED,
            tick_id=self._tick_id,
            agent=source_agent,
            payload={
                "claim_id": claim_id,
                "action": "support",
                "confidence": round(claim.confidence, 4),
                "supporting_chunk_id": supporting_chunk_id,
            },
        )

        return claim

    def retract_claim(
        self,
        claim_id: str,
        source_agent: AgentId = AgentId.O,
    ) -> Optional[Claim]:
        """Retire une claim (la marque RETRACTED, ne la supprime pas)."""
        claim = self._claims.get(claim_id)
        if claim is None:
            return None

        claim.status = ClaimStatus.RETRACTED
        claim.confidence = 0.0
        claim.revised_at = time.time()

        self._mr.append(
            event_type=EventType.CLAIM_REVISED,
            tick_id=self._tick_id,
            agent=source_agent,
            payload={
                "claim_id": claim_id,
                "action": "retract",
            },
        )

        return claim

    def link_claims(
        self,
        from_claim_id: str,
        to_claim_id: str,
        relation: ClaimRelation,
        source_chunk_id: str = "",
    ) -> bool:
        """Crée un lien orienté entre deux claims."""
        from_claim = self._claims.get(from_claim_id)
        to_claim = self._claims.get(to_claim_id)
        if from_claim is None or to_claim is None:
            return False

        from_claim.links.append(ClaimLink(
            relation=relation,
            target_claim_id=to_claim_id,
            source_chunk_id=source_chunk_id,
        ))

        return True

    # ── Queries ───────────────────────────────────────────────────

    def get_facts(self, min_confidence: float = 0.7) -> List[Claim]:
        """
        Retourne les claims qui sont des "faits" :
        - Provenance valide (au moins 1 chunk_id)
        - Confidence >= min_confidence
        - Pas contredite ni retirée
        """
        return [
            c for c in self._claims.values()
            if c.is_fact(min_confidence)
        ]

    def get_active_claims(self) -> List[Claim]:
        """Retourne toutes les claims non retirées."""
        return [
            c for c in self._claims.values()
            if c.status != ClaimStatus.RETRACTED
        ]

    def get_claims_by_status(self, status: ClaimStatus) -> List[Claim]:
        """Retourne les claims avec un status donné."""
        return [c for c in self._claims.values() if c.status == status]

    def get_contradictions(self) -> List[Claim]:
        """Retourne les claims contredites."""
        return self.get_claims_by_status(ClaimStatus.CONTRADICTED)

    def get_claim_graph(self, claim_id: str) -> Dict[str, Any]:
        """
        Retourne le sous-graphe d'une claim : ses liens et les claims liées.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return {}

        linked = []
        for link in claim.links:
            target = self._claims.get(link.target_claim_id)
            linked.append({
                "relation": link.relation.value,
                "target_claim_id": link.target_claim_id,
                "target_content": target.content if target else None,
                "source_chunk_id": link.source_chunk_id,
            })

        return {
            "claim_id": claim.claim_id,
            "content": claim.content,
            "confidence": claim.confidence,
            "status": claim.status.value,
            "provenance_count": len(claim.provenance),
            "links": linked,
        }

    # ── Predictions ───────────────────────────────────────────────

    def make_prediction(
        self,
        claim_id: str,
        prediction: str,
        agent: AgentId = AgentId.A,
    ) -> Optional[Prediction]:
        """
        Enregistre une prédiction liée à une claim.
        Trace PREDICTION_MADE dans Mᵣ.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return None

        pred = Prediction(
            prediction_id=_prediction_id(),
            claim_id=claim_id,
            predicted_by=agent,
            prediction=prediction,
        )

        self._predictions[pred.prediction_id] = pred

        # Tracer dans Mᵣ
        self._mr.append(
            event_type=EventType.PREDICTION_MADE,
            tick_id=self._tick_id,
            agent=agent,
            payload={
                "prediction_id": pred.prediction_id,
                "claim_id": claim_id,
                "prediction": prediction,
            },
        )

        return pred

    def check_prediction(
        self,
        prediction_id: str,
        outcome: bool,
        confidence_delta: float = 0.1,
    ) -> Optional[Prediction]:
        """
        Vérifie une prédiction :
        - outcome=True  → confidence de la claim augmente de +delta
        - outcome=False → confidence de la claim diminue de -delta
        Trace PREDICTION_CHECKED dans Mᵣ.
        """
        pred = self._predictions.get(prediction_id)
        if pred is None or pred.checked:
            return None

        pred.checked = True
        pred.outcome = outcome
        pred.checked_at = time.time()

        # Appliquer le delta sur la claim
        claim = self._claims.get(pred.claim_id)
        if claim is not None:
            delta = confidence_delta if outcome else -confidence_delta
            old_conf = claim.confidence
            claim.confidence = max(0.0, min(1.0, claim.confidence + delta))
            claim.revised_at = time.time()
            pred.confidence_delta = delta

            # Update status si nécessaire
            if not outcome and claim.confidence < 0.3:
                claim.status = ClaimStatus.CONTRADICTED

        # Tracer dans Mᵣ
        self._mr.append(
            event_type=EventType.PREDICTION_CHECKED,
            tick_id=self._tick_id,
            agent=pred.predicted_by,
            prediction_error=0.0 if outcome else 1.0,
            payload={
                "prediction_id": prediction_id,
                "claim_id": pred.claim_id,
                "outcome": outcome,
                "confidence_delta": pred.confidence_delta,
            },
        )

        return pred

    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Retourne une prédiction par son ID."""
        return self._predictions.get(prediction_id)

    def get_unchecked_predictions(self) -> List[Prediction]:
        """Retourne les prédictions non encore vérifiées."""
        return [p for p in self._predictions.values() if not p.checked]

    def get_predictions_for_claim(self, claim_id: str) -> List[Prediction]:
        """Retourne toutes les prédictions liées à une claim."""
        return [p for p in self._predictions.values() if p.claim_id == claim_id]

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du world model."""
        claims = list(self._claims.values())
        active = [c for c in claims if c.status != ClaimStatus.RETRACTED]
        facts = self.get_facts()

        return {
            "total_claims": len(claims),
            "active_claims": len(active),
            "facts": len(facts),
            "hypotheses": len([c for c in claims if c.status == ClaimStatus.HYPOTHESIS]),
            "supported": len([c for c in claims if c.status == ClaimStatus.SUPPORTED]),
            "contradicted": len([c for c in claims if c.status == ClaimStatus.CONTRADICTED]),
            "retracted": len([c for c in claims if c.status == ClaimStatus.RETRACTED]),
            "total_predictions": len(self._predictions),
            "unchecked_predictions": len(self.get_unchecked_predictions()),
            "avg_confidence": (
                sum(c.confidence for c in active) / len(active)
                if active else 0.0
            ),
        }
