"""
perturbation.py — Transform operators for Organism V2
=====================================================
Perturbation = LLM-generated text transformation injected into
shared agent context at a specific tick.

Source: selected (winning) draft from previous tick.
Output: transformed text injected via orchestrator.inject_user_message().

All perturbation calls use temperature=0.0 for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import ollama

log = logging.getLogger("organism_v2.perturbation")

# Model used for all perturbation transforms
_PERTURBATION_MODEL = "gpt-oss:120b-cloud"

# File-based perturbation cache
_cache: Optional[dict] = None
_cache_path: Optional[Path] = None


def set_cache_path(path: Path) -> None:
    """Set the cache file location. Called by bench_v2 at startup."""
    global _cache, _cache_path
    _cache_path = path
    if _cache_path.exists():
        try:
            _cache = json.loads(_cache_path.read_text())
            log.info("Perturbation cache loaded: %d entries", len(_cache))
        except Exception:
            _cache = {}
    else:
        _cache = {}


def _cache_key(condition: str, tick: int, input_text: str) -> str:
    raw = f"{condition}|{tick}|{input_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _save_cache() -> None:
    if _cache is not None and _cache_path is not None:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        _cache_path.write_text(json.dumps(_cache, ensure_ascii=False))


def _call_llm(instruction: str, draft_text: str, condition: str = "", tick: int = -1) -> str:
    """Call Ollama with temp=0 to transform a draft. Uses file cache."""
    # Check cache
    if _cache is not None and condition and tick >= 0:
        key = _cache_key(condition, tick, draft_text)
        if key in _cache:
            log.info("Perturbation cache HIT: %s (tick %d)", key[:8], tick)
            return _cache[key]

    prompt = f"{instruction}\n\n{draft_text[:3000]}"
    try:
        resp = ollama.chat(
            model=_PERTURBATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 3000, "num_ctx": 65536},  # explicit — not read from cristal.json
            think=False,
        )
        content = resp.get("message", {}).get("content", "")
        content = re.sub(
            r"<(?:think|thinking)>.*?</(?:think|thinking)>",
            "", content, flags=re.DOTALL,
        ).strip()
        log.info("Perturbation LLM: %d chars → %d chars", len(draft_text), len(content))

        # Store in cache
        if _cache is not None and condition and tick >= 0 and content:
            key = _cache_key(condition, tick, draft_text)
            _cache[key] = content
            _save_cache()

        return content
    except Exception as exc:
        log.error("Perturbation LLM failed: %s", exc)
        return ""


def neutral(draft_text: str, condition: str = "", tick: int = -1) -> str:
    """Neutral perturbation: reformulate without changing content."""
    return _call_llm(
        "Reformulate the following text using different words and sentence structure. "
        "Do not add, remove, or modify any factual claim, argument, or conclusion. "
        "Your output MUST be approximately the same length as the input. "
        "Do NOT summarize or shorten.",
        draft_text, condition=condition, tick=tick,
    )


def compression(draft_text: str, condition: str = "", tick: int = -1) -> str:
    """Strong perturbation: compress to one sentence."""
    return _call_llm(
        "Summarize the following in exactly one sentence.",
        draft_text, condition=condition, tick=tick,
    )


def inversion(draft_text: str, condition: str = "", tick: int = -1) -> str:
    """Strong perturbation: argue the opposite."""
    return _call_llm(
        "Argue the opposite position of the following.",
        draft_text, condition=condition, tick=tick,
    )


# Perturbation schedule per condition
PERTURBATION_SCHEDULE = {
    "A": {},  # No perturbation
    "B": {
        15: neutral,
        35: neutral,
    },
    "C": {
        15: compression,
        35: inversion,
    },
    "E": {
        15: compression,
        35: inversion,
    },
    "E_B": {
        15: compression,
        35: inversion,
    },
    "E_C": {
        15: compression,
        35: inversion,
    },
    "R": {
        15: compression,
        35: inversion,
    },
    "F": {
        10: compression,
        20: inversion,
        30: compression,
        40: inversion,
    },
}


def get_perturbation(condition: str, tick: int):
    """Return (operator_fn, operator_name) or (None, None) for this tick."""
    schedule = PERTURBATION_SCHEDULE.get(condition, {})
    fn = schedule.get(tick)
    if fn is None:
        return None, None
    name = fn.__name__
    return fn, name
