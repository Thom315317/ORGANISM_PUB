"""
P1 — L0R Ring Pointer Memory
==============================
Buffer circulaire de pointeurs chunk_id vers Mr.

Chaque slot contient: chunk_id, salience, novelty, conflict, ttl, reuse_count.
Le ring ne stocke PAS le texte — il pointe vers Mr via chunk_id.

L'evidence pack builder sélectionne les top-K slots, résout les chunk_ids
dans Mr, et assemble le texte dans un budget de tokens.

Usage:
    l0r = L0RRing(mr=reality_memory)
    l0r.insert(chunk_id="mr:abc123", salience=0.8, novelty=0.6)
    l0r.tick_decay()
    pack = l0r.build_evidence_pack(budget_tokens=2000)
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from organism.types import L0RSlot, MrEvent
from organism.config import ORGANISM_CONFIG


@dataclass
class EvidencePack:
    """
    Pack d'évidences résolu: chunk_ids -> texte réel depuis Mr.
    Prêt à être injecté dans un prompt LLM.
    """
    slots: List[L0RSlot]
    events: List[MrEvent]
    total_tokens: int
    budget_tokens: int
    dropped_count: int = 0

    def to_prompt_text(self) -> str:
        """Formate le pack pour injection dans un prompt."""
        lines = []
        for slot, event in zip(self.slots, self.events):
            score = f"{slot.composite_score():.2f}"
            agent = event.agent.value if hasattr(event.agent, 'value') else event.agent
            etype = event.type.value if hasattr(event.type, 'value') else event.type
            payload_str = ""
            if event.payload:
                text = event.payload.get("text", "")
                if text:
                    payload_str = f" | {text[:200]}"
                else:
                    payload_str = f" | {json.dumps(event.payload, ensure_ascii=False)[:200]}"
            lines.append(f"[{score}] {agent}:{etype}{payload_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_ids": [s.chunk_id for s in self.slots],
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "dropped_count": self.dropped_count,
            "slot_count": len(self.slots),
        }


class L0RRing:
    """
    Ring buffer de pointeurs vers Mr avec scoring et TTL.
    Pas de persistence propre — reconstructible depuis Mr via replay().
    """

    def __init__(self, mr: Any = None, ring_size: Optional[int] = None):
        cfg = ORGANISM_CONFIG.get("l0r", {})
        self._ring_size = ring_size or cfg.get("ring_size", 64)
        self._budget_tokens = cfg.get("evidence_pack_budget_tokens", 2000)
        self._token_ratio = cfg.get("token_estimate_ratio", 3.5)
        self._default_ttl = cfg.get("default_ttl", 10)

        self._ring: deque[L0RSlot] = deque(maxlen=self._ring_size)
        self._mr = mr

        # Index rapide chunk_id -> slot
        self._index: Dict[str, L0RSlot] = {}

    def insert(
        self,
        chunk_id: str,
        salience: float = 0.5,
        novelty: float = 0.0,
        conflict: float = 0.0,
        ttl: Optional[int] = None,
    ) -> L0RSlot:
        """
        Insère un nouveau slot. Si chunk_id existe déjà, le promote.
        """
        if chunk_id in self._index:
            return self.promote(chunk_id)

        slot = L0RSlot(
            chunk_id=chunk_id,
            salience=salience,
            novelty=novelty,
            conflict=conflict,
            ttl=ttl if ttl is not None else self._default_ttl,
        )

        # Si le ring est plein, le deque évince automatiquement le plus ancien
        if len(self._ring) >= self._ring_size:
            evicted = self._ring[0]
            self._index.pop(evicted.chunk_id, None)

        self._ring.append(slot)
        self._index[chunk_id] = slot
        return slot

    def promote(self, chunk_id: str) -> Optional[L0RSlot]:
        """Reset TTL et incrémente reuse_count."""
        slot = self._index.get(chunk_id)
        if slot is None:
            return None
        slot.ttl = self._default_ttl
        slot.reuse_count += 1
        return slot

    def tick_decay(self) -> int:
        """
        Décrémente le TTL de tous les slots. Supprime ceux à TTL <= 0.
        Appeler une fois par tick.
        Returns: nombre de slots expirés.
        """
        expired_count = 0
        surviving: List[L0RSlot] = []

        for slot in self._ring:
            slot.ttl -= 1
            if slot.ttl <= 0:
                self._index.pop(slot.chunk_id, None)
                expired_count += 1
            else:
                surviving.append(slot)

        self._ring = deque(surviving, maxlen=self._ring_size)
        return expired_count

    def build_evidence_pack(
        self,
        budget_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> EvidencePack:
        """
        Construit un evidence pack: top-K slots par score,
        résout les chunk_ids dans Mr, respecte le budget tokens.
        """
        budget = budget_tokens or self._budget_tokens
        k = top_k or self._ring_size

        sorted_slots = sorted(
            self._ring,
            key=lambda s: s.composite_score(),
            reverse=True,
        )[:k]

        selected_slots: List[L0RSlot] = []
        selected_events: List[MrEvent] = []
        total_tokens = 0
        dropped = 0

        if self._mr is not None:
            chunk_ids = [s.chunk_id for s in sorted_slots]
            events_map = self._mr.get_events_by_chunk_ids(chunk_ids)

            for slot in sorted_slots:
                event = events_map.get(slot.chunk_id)
                if event is None:
                    dropped += 1
                    continue

                event_text_len = self._estimate_event_text_length(event)
                event_tokens = max(1, int(event_text_len / self._token_ratio))

                if total_tokens + event_tokens > budget:
                    dropped += 1
                    continue

                selected_slots.append(slot)
                selected_events.append(event)
                total_tokens += event_tokens
                slot.reuse_count += 1
        else:
            for slot in sorted_slots:
                selected_slots.append(slot)

        return EvidencePack(
            slots=selected_slots,
            events=selected_events,
            total_tokens=total_tokens,
            budget_tokens=budget,
            dropped_count=dropped,
        )

    def get_active_slots(self) -> List[L0RSlot]:
        """Retourne tous les slots actifs (TTL > 0)."""
        return list(self._ring)

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du ring."""
        slots = list(self._ring)
        if not slots:
            return {
                "size": 0,
                "capacity": self._ring_size,
                "avg_ttl": 0,
                "avg_salience": 0,
                "avg_reuse": 0,
            }
        return {
            "size": len(slots),
            "capacity": self._ring_size,
            "avg_ttl": sum(s.ttl for s in slots) / len(slots),
            "avg_salience": sum(s.salience for s in slots) / len(slots),
            "avg_reuse": sum(s.reuse_count for s in slots) / len(slots),
            "max_composite": max(s.composite_score() for s in slots),
            "min_ttl": min(s.ttl for s in slots),
        }

    def clear(self) -> None:
        """Vide complètement le ring."""
        self._ring.clear()
        self._index.clear()

    def __len__(self) -> int:
        return len(self._ring)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._index

    @staticmethod
    def _estimate_event_text_length(event: MrEvent) -> int:
        """Estime la longueur en caractères d'un événement sérialisé."""
        base = 80
        if event.payload:
            text = event.payload.get("text", "")
            if text:
                return base + len(text)
            return base + len(json.dumps(event.payload, ensure_ascii=False))
        return base
