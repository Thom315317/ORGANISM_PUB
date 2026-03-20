"""
metrics_v2.py — Embedding-based time series for Organism V2
============================================================
Computes per-tick raw metrics from agent drafts using
sentence-transformers embeddings.

Primary model: all-mpnet-base-v2 (768-dim)
Fallback: all-MiniLM-L6-v2 (384-dim)

All metrics are raw scalars or vectors — no derived/windowed
computations here. Post-processing is separate.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Utilise le cache local, évite les requêtes HuggingFace au chargement
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Supprime les warnings verbeux de sentence-transformers / transformers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger("organism_v2.metrics_v2")

_embed_model = None
_embed_dim = 0


def load_embedding_model() -> int:
    """Load the embedding model. Returns embedding dimension."""
    global _embed_model, _embed_dim
    if _embed_model is not None:
        return _embed_dim

    from sentence_transformers import SentenceTransformer

    for model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
        try:
            _embed_model = SentenceTransformer(model_name)
            _embed_dim = _embed_model.get_sentence_embedding_dimension()
            log.info("Loaded embedding model: %s (dim=%d)", model_name, _embed_dim)
            return _embed_dim
        except Exception as exc:
            log.warning("Failed to load %s: %s", model_name, exc)

    raise RuntimeError("No embedding model available")


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, dim) array."""
    if _embed_model is None:
        load_embedding_model()
    # Replace empty strings with a placeholder to avoid empty encoding
    safe = [t if t.strip() else "." for t in texts]
    return _embed_model.encode(safe, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    return 1.0 - cosine_similarity(a, b)


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance between N embeddings.
    Returns 0.0 if fewer than 2 non-trivial embeddings."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(cosine_distance(embeddings[i], embeddings[j]))
    return float(np.mean(distances)) if distances else 0.0


def judge_score_dispersion_from_verdict(verdict) -> float:
    """Compute std of inferred quality scores from judge verdict.

    Quality scores derived from ranking + margins:
      rank 1: 1.0
      rank 2: 1.0 - margin_1v2
      rank 3: (1.0 - margin_1v2) - margin_2v3
    """
    if not verdict or not verdict.competition:
        return 0.0
    m12 = verdict.competition.margin_1v2
    m23 = verdict.competition.margin_2v3
    scores = [1.0, 1.0 - m12, max(0.0, 1.0 - m12 - m23)]
    return float(np.std(scores))


class TickMetrics:
    """Collects per-tick metrics during a run."""

    def __init__(self):
        self.state_vectors_mean: List[np.ndarray] = []
        self.state_vectors_selected: List[np.ndarray] = []
        self.claim_cosine_variance: List[float] = []
        self.judge_score_dispersion: List[float] = []
        self.draft_velocity: List[float] = []
        self.post_selection_variance: List[float] = []
        self.quality_per_tick: List[float] = []
        self._prev_selected: Optional[np.ndarray] = None

    def record_tick(
        self,
        agent_drafts: Dict[str, str],
        winner_id: Optional[str],
        verdict,
    ) -> None:
        """Record all metrics for one tick.

        Args:
            agent_drafts: {agent_id: draft_text} for all agents with text
            winner_id: ID of the winning agent (from judge verdict)
            verdict: JudgeVerdict object
        """
        # Embed all non-empty drafts
        agent_ids = list(agent_drafts.keys())
        texts = [agent_drafts[aid] for aid in agent_ids]

        if not texts:
            dim = _embed_dim or 768
            zero = np.zeros(dim)
            self.state_vectors_mean.append(zero)
            self.state_vectors_selected.append(zero)
            self.claim_cosine_variance.append(0.0)
            self.judge_score_dispersion.append(0.0)
            self.draft_velocity.append(float("nan"))
            self.post_selection_variance.append(0.0)
            self.quality_per_tick.append(0.0)
            self._prev_selected = zero
            return

        embeddings = embed_texts(texts)  # (N, dim)

        # state_vector_mean = mean of all agent embeddings
        sv_mean = embeddings.mean(axis=0)
        self.state_vectors_mean.append(sv_mean)

        # state_vector_selected = embedding of winning draft
        if winner_id and winner_id in agent_ids:
            idx = agent_ids.index(winner_id)
            sv_selected = embeddings[idx]
        else:
            # Fallback: use first agent
            sv_selected = embeddings[0]
        self.state_vectors_selected.append(sv_selected)

        # claim_cosine_variance = mean pairwise cosine distance
        self.claim_cosine_variance.append(
            mean_pairwise_cosine_distance(embeddings)
        )

        # ranking_disagreement = std of quality scores
        self.judge_score_dispersion.append(
            judge_score_dispersion_from_verdict(verdict)
        )

        # draft_velocity = cosine_distance(selected[t], selected[t-1])
        if self._prev_selected is not None:
            self.draft_velocity.append(
                cosine_distance(sv_selected, self._prev_selected)
            )
        else:
            self.draft_velocity.append(float("nan"))
        self._prev_selected = sv_selected.copy()

        # post_selection_variance = cosine_distance(selected, mean)
        self.post_selection_variance.append(
            cosine_distance(sv_selected, sv_mean)
        )

        # quality_per_tick = judge confidence
        if verdict:
            self.quality_per_tick.append(float(verdict.confidence))
        else:
            self.quality_per_tick.append(0.0)

    def compute_sim_curves(
        self,
        perturbation_ticks: List[int],
        k_max: int = 15,
    ) -> Dict[str, Dict[str, List[Optional[float]]]]:
        """Compute similarity curves for each perturbation event.

        For each t_p in perturbation_ticks:
          R_pre_mean     = mean(sv_mean[t_p-3], sv_mean[t_p-2], sv_mean[t_p-1])
          R_pre_selected = mean(sv_selected[t_p-3], ..., sv_selected[t_p-1])
          sim_mean[k]     = cosine_similarity(R_pre_mean, sv_mean[t_p+k])
          sim_selected[k] = cosine_similarity(R_pre_selected, sv_selected[t_p+k])
          for k in 1..k_max
        """
        result = {}
        n = len(self.state_vectors_mean)

        for t_p in perturbation_ticks:
            # Compute R_pre (average of 3 ticks before perturbation)
            pre_indices = [t_p - 3, t_p - 2, t_p - 1]
            valid_pre_mean = [
                self.state_vectors_mean[i]
                for i in pre_indices
                if 0 <= i < n
            ]
            valid_pre_sel = [
                self.state_vectors_selected[i]
                for i in pre_indices
                if 0 <= i < n
            ]

            if not valid_pre_mean:
                result[f"tick_{t_p}"] = {"mean": [], "selected": []}
                continue

            r_pre_mean = np.mean(valid_pre_mean, axis=0)
            r_pre_selected = np.mean(valid_pre_sel, axis=0)

            sim_mean_curve = []
            sim_sel_curve = []

            for k in range(1, k_max + 1):
                idx = t_p + k
                if idx < n:
                    sim_mean_curve.append(
                        cosine_similarity(r_pre_mean, self.state_vectors_mean[idx])
                    )
                    sim_sel_curve.append(
                        cosine_similarity(r_pre_selected, self.state_vectors_selected[idx])
                    )
                else:
                    sim_mean_curve.append(None)
                    sim_sel_curve.append(None)

            result[f"tick_{t_p}"] = {
                "mean": sim_mean_curve,
                "selected": sim_sel_curve,
            }

        return result

    def to_dict(self) -> Dict:
        """Export all time series as JSON-serializable dict."""
        def _to_list(arr_list):
            return [
                a.tolist() if isinstance(a, np.ndarray) else a
                for a in arr_list
            ]

        def _float_list(lst):
            return [
                None if (isinstance(v, float) and math.isnan(v)) else round(v, 6)
                for v in lst
            ]

        return {
            "state_vector_mean": _to_list(self.state_vectors_mean),
            "state_vector_selected": _to_list(self.state_vectors_selected),
            "claim_cosine_variance": _float_list(self.claim_cosine_variance),
            "judge_score_dispersion": _float_list(self.judge_score_dispersion),
            "draft_velocity": _float_list(self.draft_velocity),
            "post_selection_variance": _float_list(self.post_selection_variance),
            "quality_per_tick": _float_list(self.quality_per_tick),
        }
