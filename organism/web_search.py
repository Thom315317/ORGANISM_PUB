"""
Recherche web légère pour l'Organisme CRISTAL.
Wrapper standalone autour de DuckDuckGo (duckduckgo-search).
Pas de dépendance à Executor/ToolRunner/Guardian.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from organism.config import ORGANISM_CONFIG

log = logging.getLogger("organism.web_search")

_grounding_cfg = ORGANISM_CONFIG.get("grounding", {})
_MAX_SEARCHES_PER_TICK = _grounding_cfg.get("max_searches_per_tick", 2)
_SEARCH_TIMEOUT = _grounding_cfg.get("search_timeout", 10)

# Rate-limit simple : timestamp du dernier appel
_last_search_ts: float = 0.0
_COOLDOWN_SECS = 3.0

# Compteur par tick (reset par l'orchestrateur)
_searches_this_tick: int = 0


def reset_tick_counter() -> None:
    """Appelé par l'orchestrateur au début de chaque tick."""
    global _searches_this_tick
    _searches_this_tick = 0


def web_search(query: str, max_results: int = 3) -> str:
    """Recherche DuckDuckGo, retourne du texte formaté pour injection L0R.

    Retourne une chaîne vide si la recherche échoue ou si le rate-limit
    est atteint. Pas d'exception propagée.
    """
    global _last_search_ts, _searches_this_tick

    if not _grounding_cfg.get("enabled", True):
        return ""

    if _searches_this_tick >= _MAX_SEARCHES_PER_TICK:
        log.debug("web_search: max searches/tick reached (%d)", _MAX_SEARCHES_PER_TICK)
        return ""

    # Cooldown
    now = time.monotonic()
    if now - _last_search_ts < _COOLDOWN_SECS:
        log.debug("web_search: cooldown (%.1fs)", _COOLDOWN_SECS - (now - _last_search_ts))
        return ""

    query = (query or "").strip()
    if not query or len(query) < 3:
        return ""

    _last_search_ts = time.monotonic()
    _searches_this_tick += 1

    # Import DDG (ddgs ou duckduckgo_search)
    DDGS = _get_ddgs_class()
    if DDGS is None:
        log.warning("web_search: DuckDuckGo package not available")
        return ""

    try:
        results = _do_search(DDGS, query, max_results)
    except Exception as exc:
        log.error("web_search failed for %r: %s", query, exc)
        return ""

    if not results:
        return ""

    lines = [f"[WEB] Recherche: {query}"]
    for r in results:
        title = r.get("title", "").strip()
        snippet = r.get("body", r.get("snippet", "")).strip()
        url = r.get("href", r.get("url", "")).strip()
        if title or snippet:
            line = f"- {title}"
            if snippet:
                line += f" — {snippet[:200]}"
            if url:
                line += f" ({url})"
            lines.append(line)

    return "\n".join(lines)


# ── Internals ──────────────────────────────────────────────────


_ddgs_class = None
_ddgs_checked = False


def _get_ddgs_class():
    """Import paresseux de DDGS avec fallback."""
    global _ddgs_class, _ddgs_checked
    if _ddgs_checked:
        return _ddgs_class
    _ddgs_checked = True
    try:
        from ddgs import DDGS  # type: ignore
        _ddgs_class = DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore
        _ddgs_class = DDGS
        return DDGS
    except ImportError:
        pass
    return None


def _do_search(DDGS, query: str, max_results: int) -> List[dict]:
    """Exécute la recherche DDG avec timeout."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r)
    return results
