"""
Configuration pour l'organisme CRISTAL.
Charge la section 'organism' depuis config/cristal.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def load_organism_config() -> Dict[str, Any]:
    """Charge la section 'organism' depuis cristal.json.
    Retourne un dict avec des defaults si la section n'existe pas."""
    config_path = BASE_DIR / "config" / "cristal.json"
    defaults: Dict[str, Any] = {
        "mr": {
            "path": "data/organism/mr_events.jsonl",
            "max_file_size_mb": 100,
            "flush_interval_events": 1,
        },
        "l0r": {
            "ring_size": 64,
            "default_ttl": 10,
            "evidence_pack_budget_tokens": 2000,
            "token_estimate_ratio": 3.5,
        },
        "agents": {
            "A": {"model": "glm-4.6:cloud", "temperature": 0.9, "num_ctx": 4096, "num_predict": 1500, "repeat_penalty": 1.2},
            "B": {"model": "deepseek-v3.1:671b-cloud", "temperature": 0.3, "num_ctx": 4096, "num_predict": 1500, "repeat_penalty": 1.2},
            "C": {"model": "qwen3-coder:480b-cloud", "temperature": 0.5, "num_ctx": 8192, "num_predict": 1500, "repeat_penalty": 1.2},
        },
        "grounding": {
            "enabled": True,
            "location": "France",
            "timezone": "Europe/Paris",
            "max_searches_per_tick": 2,
            "search_timeout": 10,
        },
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        organism_cfg = full_config.get("organism", {})
        for key in defaults:
            if key not in organism_cfg:
                organism_cfg[key] = defaults[key]
            elif isinstance(defaults[key], dict):
                merged = {**defaults[key], **organism_cfg[key]}
                organism_cfg[key] = merged
        return organism_cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults


ORGANISM_CONFIG = load_organism_config()
