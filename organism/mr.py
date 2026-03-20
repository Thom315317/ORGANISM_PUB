"""
P0 — Mémoire de Réalité (Mr)
=============================
Journal d'événements append-only avec chaîne de hachage SHA-256.

Chaque événement contient prev_hash (hash de l'événement précédent) et
hash (SHA-256 du JSON canonique + prev_hash). Toute modification d'une
ligne casse la chaîne → détection de falsification.

Usage:
    mr = RealityMemory()
    event = mr.append(EventType.AGENT_MESSAGE, tick_id=1, agent=AgentId.O, ...)
    events = mr.replay()
    ok, broken_at = mr.verify_chain()
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator

from organism.types import MrEvent, EventType, AgentId
from organism.config import ORGANISM_CONFIG, BASE_DIR


GENESIS_HASH = "0" * 64


def _ulid() -> str:
    """ULID approximatif : timestamp hex + random. Orderable par temps."""
    ts = int(time.time() * 1000)
    return f"{ts:012x}-{uuid.uuid4().hex[:12]}"


def _canonical_json(d: Dict[str, Any]) -> str:
    """JSON canonique: clés triées, pas d'espaces, ensure_ascii=False."""
    return json.dumps(d, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def _sha256(data: str) -> str:
    """SHA-256 hex digest complet (64 chars)."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class RealityMemory:
    """
    Mémoire de réalité append-only avec chaîne de hachage.
    Thread-safe via un verrou interne.
    """

    def __init__(self, path: Optional[str] = None):
        cfg = ORGANISM_CONFIG.get("mr", {})
        if path is None:
            path = str(BASE_DIR / cfg.get("path", "data/organism/mr_events.jsonl"))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._prev_hash: str = GENESIS_HASH
        self._event_count: int = 0

        self._restore_tail()

    def _restore_tail(self) -> None:
        """Lit la dernière ligne du fichier pour restaurer prev_hash."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        last_line = ""
        count = 0
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
                    count += 1
        if last_line:
            try:
                last_event = json.loads(last_line)
                self._prev_hash = last_event.get("hash", GENESIS_HASH)
                self._event_count = count
            except json.JSONDecodeError:
                pass

    def append(
        self,
        event_type: EventType,
        tick_id: int,
        agent: AgentId,
        mode: str = "Idle",
        token_in: int = 0,
        token_out: int = 0,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        novelty: float = 0.0,
        conflict: float = 0.0,
        cohesion: float = 0.0,
        impl_pressure: float = 0.0,
        prediction_error: float = 0.0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> MrEvent:
        """
        Crée et persiste un événement. Thread-safe.
        Retourne l'objet MrEvent avec chunk_id et hash remplis.
        """
        with self._lock:
            event_id = _ulid()
            ts = time.time()

            event = MrEvent(
                event_id=event_id,
                ts=ts,
                type=event_type,
                tick_id=tick_id,
                agent=agent,
                mode=mode,
                token_in=token_in,
                token_out=token_out,
                latency_ms=latency_ms,
                cost=cost,
                novelty=novelty,
                conflict=conflict,
                cohesion=cohesion,
                impl_pressure=impl_pressure,
                prediction_error=prediction_error,
                prev_hash=self._prev_hash,
                payload=payload,
            )

            # chunk_id provisoire (sera dans le canonical dict)
            event.chunk_id = ""

            # Calcul du hash
            canonical = _canonical_json(event.to_canonical_dict())
            event.hash = _sha256(canonical)
            event.chunk_id = f"mr:{event.hash[:16]}"

            # Recalculer le hash avec le chunk_id final
            canonical = _canonical_json(event.to_canonical_dict())
            event.hash = _sha256(canonical)

            # Écriture append-only
            line = json.dumps(event.to_line_dict(), ensure_ascii=False,
                              separators=(',', ':'))
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

            self._prev_hash = event.hash
            self._event_count += 1

            return event

    def replay(
        self,
        event_types: Optional[List[EventType]] = None,
        agent: Optional[AgentId] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        limit: int = 0,
    ) -> List[MrEvent]:
        """
        Relit le journal et retourne les événements filtrés.

        Args:
            event_types: filtrer par types (None = tous)
            agent: filtrer par agent (None = tous)
            tick_range: (min_tick, max_tick) inclusif
            limit: max nombre de résultats (0 = illimité)
        """
        results: List[MrEvent] = []
        type_set = set(t.value for t in event_types) if event_types else None

        for raw in self._iter_lines():
            if type_set and raw.get("type") not in type_set:
                continue
            if agent and raw.get("agent") != agent.value:
                continue
            if tick_range:
                tid = raw.get("tick_id", 0)
                if tid < tick_range[0] or tid > tick_range[1]:
                    continue

            event = self._dict_to_event(raw)
            results.append(event)

            if limit and len(results) >= limit:
                break

        return results

    def get_event_by_chunk_id(self, chunk_id: str) -> Optional[MrEvent]:
        """Retrouve un événement par son chunk_id. Scan linéaire."""
        for raw in self._iter_lines():
            if raw.get("chunk_id") == chunk_id:
                return self._dict_to_event(raw)
        return None

    def get_events_by_chunk_ids(self, chunk_ids: List[str]) -> Dict[str, MrEvent]:
        """Retrouve plusieurs événements par chunk_id en un seul scan."""
        wanted = set(chunk_ids)
        found: Dict[str, MrEvent] = {}
        for raw in self._iter_lines():
            cid = raw.get("chunk_id", "")
            if cid in wanted:
                found[cid] = self._dict_to_event(raw)
                wanted.discard(cid)
                if not wanted:
                    break
        return found

    def verify_chain(self) -> Tuple[bool, int]:
        """
        Vérifie l'intégrité de la chaîne de hachage.

        Returns:
            (is_valid, broken_at_line) — broken_at_line = -1 si valide
        """
        expected_prev = GENESIS_HASH
        line_num = 0

        for raw in self._iter_lines():
            line_num += 1
            if raw.get("prev_hash") != expected_prev:
                return False, line_num

            stored_hash = raw.pop("hash", "")
            canonical = _canonical_json(dict(sorted(raw.items())))
            computed = _sha256(canonical)

            if computed != stored_hash:
                return False, line_num

            expected_prev = stored_hash

        return True, -1

    def tail(self, n: int = 10) -> List[MrEvent]:
        """Retourne les n derniers événements."""
        all_lines: List[Dict[str, Any]] = []
        for raw in self._iter_lines():
            all_lines.append(raw)
        return [self._dict_to_event(r) for r in all_lines[-n:]]

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def last_hash(self) -> str:
        return self._prev_hash

    def _iter_lines(self) -> Iterator[Dict[str, Any]]:
        """Itère sur les lignes JSONL du fichier."""
        if not self.path.exists():
            return
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def _dict_to_event(d: Dict[str, Any]) -> MrEvent:
        """Reconstruit un MrEvent depuis un dict JSONL."""
        etype = d.get("type", "agent_message")
        agent = d.get("agent", "O")
        return MrEvent(
            event_id=d.get("event_id", ""),
            ts=d.get("ts", 0.0),
            type=EventType(etype) if etype in EventType._value2member_map_ else EventType.AGENT_MESSAGE,
            tick_id=d.get("tick_id", 0),
            agent=AgentId(agent) if agent in AgentId._value2member_map_ else AgentId.O,
            mode=d.get("mode", "Idle"),
            token_in=d.get("token_in", 0),
            token_out=d.get("token_out", 0),
            latency_ms=d.get("latency_ms", 0.0),
            cost=d.get("cost", 0.0),
            novelty=d.get("novelty", 0.0),
            conflict=d.get("conflict", 0.0),
            cohesion=d.get("cohesion", 0.0),
            impl_pressure=d.get("impl_pressure", 0.0),
            prediction_error=d.get("prediction_error", 0.0),
            chunk_id=d.get("chunk_id", ""),
            prev_hash=d.get("prev_hash", ""),
            hash=d.get("hash", ""),
            payload=d.get("payload"),
        )
