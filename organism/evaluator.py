"""
Evaluator — Metriques d'evaluation pour l'Organism CRISTAL
===========================================================
Observateur passif : recoit les TickResult apres chaque tick,
calcule des metriques, ecrit en JSONL (1 ligne/tick) + summary.json.

Ne modifie aucun module existant. Lecture seule sur Scheduler/WorldModel.

Usage:
    evaluator = Evaluator(run_id="org_123", output_dir="runs")
    evaluator.on_tick_end(result, scheduler, world_model)
    evaluator.on_user_injection("Bonjour", tick_id=5)
    evaluator.finalize()  # ecrit summary.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False


# ── Constantes ────────────────────────────────────────────────────

HASH_DIM = 256  # Dimension du vecteur de hashing trick
WINDOW_SIZE = 20  # Fenetre glissante pour mode_entropy et signal_stability


# ── Helpers ───────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Tokenise un texte : lowercase, supprime ponctuation, split."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()


def _text_hash(text: str) -> str:
    """Hash court (8 chars hex) d'un texte."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]


def _percentile(values: List[float], p: float) -> float:
    """Calcule le percentile p (0-100) d'une liste de valeurs."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100.0) * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def _linear_slope(values: List[float]) -> float:
    """Pente de regression lineaire sur une liste de valeurs."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


# ── Evaluator ─────────────────────────────────────────────────────


class Evaluator:
    """
    Observateur passif. Calcule des metriques par tick et les ecrit en JSONL.

    Metriques :
      - repetition_3gram : proportion de 3-grams repetes
      - dedup_ratio : tokens uniques / tokens totaux
      - hashvec_novelty : 1 - cosine(hashvec_t, hashvec_{t-1})
      - agent_balance_entropy : Shannon entropy des longueurs de texte par agent
      - vocab_richness : mots uniques / mots totaux
      - mode_entropy_w20 : Shannon entropy des modes (fenetre 20)
      - wm_supported_ratio, wm_contradicted_ratio, wm_churn
      - user_pending, ticks_since_user_injection
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str = "runs",
        condition: str = "organism",
    ):
        self._run_id = run_id
        self._condition = condition

        # Dossier de sortie
        self._run_dir = Path(output_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Fichier JSONL metrics — mode 'w' : chaque run repart à zéro
        self._metrics_path = self._run_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, 'w', encoding='utf-8')

        # Garde-fou anti-doublons
        self._written_tick_ids: set = set()

        # Etat interne
        self._tick_count = 0
        self._prev_hashvec: Optional[List[float]] = None
        self._mode_history: List[str] = []
        self._prev_wm_stats: Optional[Dict[str, Any]] = None

        # Tracking injections user
        self._pending_injections: List[Dict[str, Any]] = []
        self._resolved_latencies: List[int] = []

        # Accumulation pour summary
        self._all_metrics: List[Dict[str, Any]] = []

        # Embedding model (lazy init, None if unavailable)
        self._embed_model: Any = None
        self._embed_loaded = False
        self._prev_embedding: Any = None

    # ── Public API ────────────────────────────────────────────────

    def on_tick_end(self, result: Any, scheduler: Any, wm: Any) -> None:
        """
        Appele apres chaque tick. Calcule les metriques et ecrit 1 ligne JSONL.

        Args:
            result: TickResult de l'orchestrateur
            scheduler: Scheduler (pour get_mode_probabilities)
            wm: WorldModel (pour get_stats)
        """
        self._tick_count += 1
        ts = time.time()

        # Extraire les textes des agents
        agent_texts = []
        agents_data = []
        for turn in result.agent_turns:
            text = (turn.text or "").strip()
            agent_texts.append(text)
            agents_data.append({
                "agent": turn.agent.value if hasattr(turn.agent, 'value') else str(turn.agent),
                "status": turn.status.value if hasattr(turn.status, 'value') else str(turn.status),
                "text_len": len(text),
                "text_hash": _text_hash(text) if text else "",
                "token_in": turn.token_in,
                "token_out": turn.token_out,
                "latency_ms": round(turn.latency_ms, 1),
                "novelty": round(turn.novelty, 3),
                "conflict": round(turn.conflict, 3),
                "cohesion": round(turn.cohesion, 3),
                "impl_pressure": round(turn.impl_pressure, 3),
                "veto": turn.veto,
            })

        # Mode
        mode_str = result.mode.value if hasattr(result.mode, 'value') else str(result.mode)
        self._mode_history.append(mode_str)

        # Mode probs (depuis scheduler)
        mode_probs = {}
        try:
            raw_probs = scheduler.get_mode_probabilities()
            mode_probs = {
                (k.value if hasattr(k, 'value') else str(k)): round(v, 4)
                for k, v in raw_probs.items()
            }
        except Exception:
            pass

        # Signals
        signals = {}
        if result.signals:
            s = result.signals
            signals = {
                "energy": round(s.energy, 4),
                "novelty": round(s.novelty, 4),
                "conflict": round(s.conflict, 4),
                "impl_pressure": round(s.impl_pressure, 4),
                "cohesion": round(s.cohesion, 4),
                "cost_pressure": round(s.cost_pressure, 4),
                "prediction_error": round(s.prediction_error, 4),
            }

        # WM stats
        wm_stats = {}
        try:
            wm_stats = wm.get_stats()
        except Exception:
            pass

        # WM churn
        wm_churn = self._compute_wm_churn(wm_stats, result.claims_added)

        # Calculer les metriques
        hashvec_nov = round(self._hashvec_novelty(agent_texts), 4)
        metrics = {
            "repetition_3gram": round(self._repetition_3gram(agent_texts), 4),
            "dedup_ratio": round(self._dedup_ratio(agent_texts), 4),
            "hashvec_novelty": hashvec_nov,
            "agent_balance_entropy": round(self._agent_balance_entropy(result.agent_turns), 4),
            "vocab_richness": round(self._vocab_richness(agent_texts), 4),
            "mode_entropy_w20": round(self._mode_entropy(), 4),
            "wm_supported_ratio": round(self._wm_ratio(wm_stats, "supported"), 4),
            "wm_contradicted_ratio": round(self._wm_ratio(wm_stats, "contradicted"), 4),
            "wm_churn": wm_churn,
            "user_pending": len(self._pending_injections) > 0,
            "ticks_since_user_injection": self._ticks_since_injection(result.tick_id),
            "cohesion": round(signals.get("cohesion", 0.0), 4),
            "signal_mode_corr": round(self._signal_mode_corr(), 4),
            "novelty_spike": self._novelty_spike(hashvec_nov),
        }

        # Optional embedding metrics
        sem_sim = self._semantic_similarity(agent_texts)
        if sem_sim is not None:
            metrics["semantic_similarity"] = sem_sim
        topic_d = self._topic_drift(agent_texts)
        if topic_d is not None:
            metrics["topic_drift"] = topic_d

        # Verifier reactivite user
        self._check_user_response(agent_texts, result.tick_id)

        # Veto
        veto_agent = None
        if result.veto_agent:
            veto_agent = (
                result.veto_agent.value
                if hasattr(result.veto_agent, 'value')
                else str(result.veto_agent)
            )

        # Judge verdict (Phase 1)
        judge_data = None
        if hasattr(result, 'judge_verdict') and result.judge_verdict:
            v = result.judge_verdict
            judge_data = {
                "winner": v.winner,
                "reason": v.reason[:200] if v.reason else "",
                "confidence": round(v.confidence, 4),
                "signals": v.signals,
                "judge_failed": v.winner is None,
            }
            if v.competition:
                judge_data["competition"] = {
                    "ranking": list(v.competition.ranking),
                    "margin_1v2": round(v.competition.margin_1v2, 4),
                    "margin_2v3": round(v.competition.margin_2v3, 4),
                    "counterfactual": v.competition.counterfactual[:100] if v.competition.counterfactual else "",
                }
            # Expose audit info from normalization
            if v.raw_json and isinstance(v.raw_json, dict):
                if "_audit" in v.raw_json:
                    judge_data["_audit"] = v.raw_json["_audit"]
                if "_anon_map" in v.raw_json:
                    judge_data["_anon_map"] = v.raw_json["_anon_map"]
                if "_anon_reverse" in v.raw_json:
                    judge_data["_anon_reverse"] = v.raw_json["_anon_reverse"]

        # Theory scores (Phase 2)
        theory_scores = None
        if hasattr(result, 'organism_state') and result.organism_state:
            ts_dict = result.organism_state.theory_scores
            if ts_dict:
                theory_scores = {k: round(v, 4) for k, v in ts_dict.items()}

        # Construire la ligne JSONL
        row = {
            "run_id": self._run_id,
            "condition": self._condition,
            "tick_id": result.tick_id,
            "ts": round(ts, 3),
            "mode": mode_str,
            "mode_changed": result.mode_changed,
            "mode_probs": mode_probs,
            "agents": agents_data,
            "signals": signals,
            "wm_stats": {
                "total_claims": wm_stats.get("total_claims", 0),
                "supported": wm_stats.get("supported", 0),
                "contradicted": wm_stats.get("contradicted", 0),
                "hypotheses": wm_stats.get("hypotheses", 0),
                "retracted": wm_stats.get("retracted", 0),
                "avg_confidence": round(wm_stats.get("avg_confidence", 0.0), 4),
            },
            "veto_flag": result.vetoed,
            "veto_agent": veto_agent,
            "total_tokens": result.total_tokens,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "claims_added": result.claims_added,
            "judge_verdict": judge_data,
            "theory_scores": theory_scores,
            "metrics": metrics,
        }

        # Garde-fou anti-doublons
        if result.tick_id in self._written_tick_ids:
            import logging as _log
            _log.getLogger("organism.evaluator").warning(
                "SKIP duplicate tick_id=%d in metrics.jsonl", result.tick_id)
            return

        # Ecrire en JSONL
        line = json.dumps(row, ensure_ascii=False, separators=(',', ':'))
        self._metrics_file.write(line + '\n')
        self._metrics_file.flush()
        self._written_tick_ids.add(result.tick_id)

        # Accumuler pour summary
        self._all_metrics.append(row)

    def on_user_injection(self, text: str, tick_id: int) -> None:
        """
        Appele quand un message user est injecte.
        Stocke les mots-cles pour tracker la reactivite.
        """
        words = set(w for w in _tokenize(text) if len(w) > 3)
        # Baseline novelty at injection time
        baseline = 0.0
        if self._all_metrics:
            baseline = self._all_metrics[-1].get("metrics", {}).get(
                "hashvec_novelty", 0.0
            )
        self._pending_injections.append({
            "tick_id": tick_id,
            "words": words,
            "resolved": False,
            "novelty_baseline": baseline,
        })

    def finalize(self) -> None:
        """
        Ecrit summary.json, metrics.csv, verifie l'integrite et ferme les fichiers.
        """
        summary = self._compute_summary()
        summary_path = self._run_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Export CSV
        self._export_csv()

        self._metrics_file.close()

        # Verification post-run : pas de doublons, bon nombre de ticks
        self._verify_integrity()

    def _verify_integrity(self) -> None:
        """Verifie metrics.jsonl et events.jsonl apres la fin du run."""
        import logging as _log
        logger = _log.getLogger("organism.evaluator")
        expected = self._tick_count

        # metrics.jsonl
        try:
            with open(self._metrics_path, 'r', encoding='utf-8') as f:
                tick_ids = [json.loads(line)['tick_id'] for line in f if line.strip()]
            n_dup = len(tick_ids) - len(set(tick_ids))
            if n_dup > 0:
                logger.error("INTEGRITY: %d ticks dupliques dans metrics.jsonl", n_dup)
            if len(tick_ids) != expected:
                logger.warning("INTEGRITY: metrics.jsonl a %d ticks (attendu %d)",
                               len(tick_ids), expected)
            else:
                logger.info("metrics.jsonl OK: %d ticks, 0 doublons", len(tick_ids))
        except Exception as exc:
            logger.error("INTEGRITY check failed on metrics.jsonl: %s", exc)

        # events.jsonl
        events_path = self._run_dir / "events.jsonl"
        if events_path.exists():
            try:
                with open(events_path, 'r', encoding='utf-8') as f:
                    all_events = [json.loads(line) for line in f if line.strip()]
                te_ids = [e['tick_id'] for e in all_events
                          if e.get('type') == 'tick_end']
                n_dup_te = len(te_ids) - len(set(te_ids))
                if n_dup_te > 0:
                    logger.error("INTEGRITY: %d tick_end dupliques dans events.jsonl", n_dup_te)
                else:
                    logger.info("events.jsonl OK: %d tick_end, 0 doublons", len(te_ids))
            except Exception as exc:
                logger.error("INTEGRITY check failed on events.jsonl: %s", exc)

    def _export_csv(self) -> None:
        """Flatten all tick metrics into a CSV file."""
        if not self._all_metrics:
            return
        # Collect all metric keys across ticks
        all_keys: set = set()
        for row in self._all_metrics:
            all_keys.update(row.get("metrics", {}).keys())
        metric_keys = sorted(all_keys)

        columns = [
            "tick_id", "mode", "mode_changed",
            "total_tokens", "elapsed_ms", "claims_added", "veto_flag",
        ] + metric_keys

        csv_path = self._run_dir / "metrics.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self._all_metrics:
                flat: Dict[str, Any] = {
                    "tick_id": row["tick_id"],
                    "mode": row["mode"],
                    "mode_changed": row.get("mode_changed", False),
                    "total_tokens": row["total_tokens"],
                    "elapsed_ms": row["elapsed_ms"],
                    "claims_added": row.get("claims_added", 0),
                    "veto_flag": row.get("veto_flag", False),
                }
                metrics = row.get("metrics", {})
                for k in metric_keys:
                    flat[k] = metrics.get(k, "")
                writer.writerow(flat)

    # ── Metriques : Rumination ────────────────────────────────────

    def _repetition_3gram(self, texts: List[str]) -> float:
        """Proportion de 3-grams repetes dans les textes concatenes."""
        words: List[str] = []
        for t in texts:
            words.extend(_tokenize(t))
        if len(words) < 3:
            return 0.0
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        counts = Counter(trigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(trigrams) if trigrams else 0.0

    def _dedup_ratio(self, texts: List[str]) -> float:
        """Ratio tokens uniques / tokens totaux. 1.0 = tout unique."""
        tokens: List[str] = []
        for t in texts:
            tokens.extend(_tokenize(t))
        return len(set(tokens)) / len(tokens) if tokens else 1.0

    # ── Metriques : Diversite thematique ──────────────────────────

    def _hashvec(self, texts: List[str]) -> List[float]:
        """Hashing trick : bag-of-words → vecteur de dimension HASH_DIM."""
        vec = [0.0] * HASH_DIM
        for t in texts:
            for w in _tokenize(t):
                idx = hash(w) % HASH_DIM
                vec[idx] += 1.0
        return vec

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Similarite cosinus entre deux vecteurs."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def _hashvec_novelty(self, texts: List[str]) -> float:
        """1 - cosine(hashvec_t, hashvec_{t-1}). Premier tick = 1.0."""
        curr = self._hashvec(texts)
        if self._prev_hashvec is None:
            self._prev_hashvec = curr
            return 1.0
        cos = self._cosine(self._prev_hashvec, curr)
        self._prev_hashvec = curr
        return 1.0 - cos

    @staticmethod
    def _agent_balance_entropy(agent_turns: list) -> float:
        """Shannon entropy normalisee des longueurs de texte par agent."""
        lengths: Dict[str, int] = {}
        for turn in agent_turns:
            key = turn.agent.value if hasattr(turn.agent, 'value') else str(turn.agent)
            lengths[key] = lengths.get(key, 0) + len(turn.text or "")
        total = sum(lengths.values())
        if total == 0 or len(lengths) < 2:
            return 0.0
        entropy = 0.0
        for length in lengths.values():
            p = length / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(lengths))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def _vocab_richness(texts: List[str]) -> float:
        """Mots uniques / mots totaux."""
        all_words: List[str] = []
        for t in texts:
            all_words.extend(_tokenize(t))
        if not all_words:
            return 0.0
        return len(set(all_words)) / len(all_words)

    # ── Metriques : Transitions de mode ───────────────────────────

    def _mode_entropy(self, window: int = WINDOW_SIZE) -> float:
        """Shannon entropy des modes sur les N derniers ticks."""
        recent = self._mode_history[-window:]
        if not recent:
            return 0.0
        counts = Counter(recent)
        total = len(recent)
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values() if c > 0
        )
        max_e = math.log2(min(len(counts), 6))
        return entropy / max_e if max_e > 0 else 0.0

    # ── Metriques : Correlation signal/mode ────────────────────────

    @staticmethod
    def _spearman_rank_corr(x: List[float], y: List[float]) -> float:
        """Spearman rank correlation (pure Python, no scipy)."""
        n = len(x)
        if n < 3:
            return 0.0

        def _ranks(vals: List[float]) -> List[float]:
            indexed = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and vals[indexed[j]] == vals[indexed[j + 1]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k]] = avg_rank
                i = j + 1
            return ranks

        rx, ry = _ranks(x), _ranks(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        return 1.0 - (6 * d_sq) / (n * (n * n - 1))

    def _signal_mode_corr(self) -> float:
        """Spearman corr between mean signal magnitude and mode_changed (window=WINDOW_SIZE)."""
        recent = self._all_metrics[-WINDOW_SIZE:]
        if len(recent) < 3:
            return 0.0
        signal_mags: List[float] = []
        mode_flags: List[float] = []
        for row in recent:
            sigs = row.get("signals", {})
            mag = _mean([abs(v) for v in sigs.values() if isinstance(v, (int, float))])
            signal_mags.append(mag)
            mode_flags.append(1.0 if row.get("mode_changed") else 0.0)
        return self._spearman_rank_corr(signal_mags, mode_flags)

    # ── Metriques : World Model ───────────────────────────────────

    @staticmethod
    def _wm_ratio(wm_stats: Dict[str, Any], key: str) -> float:
        """Ratio d'un type de claim sur le total."""
        total = wm_stats.get("total_claims", 0)
        if total == 0:
            return 0.0
        return wm_stats.get(key, 0) / total

    def _compute_wm_churn(
        self, wm_stats: Dict[str, Any], claims_added: int
    ) -> int:
        """Claims ajoutees + changements de status ce tick."""
        churn = claims_added
        if self._prev_wm_stats is not None:
            for key in ("supported", "contradicted", "retracted"):
                prev = self._prev_wm_stats.get(key, 0)
                curr = wm_stats.get(key, 0)
                churn += abs(curr - prev)
        self._prev_wm_stats = dict(wm_stats) if wm_stats else None
        return churn

    # ── Metriques : Reactivite user ───────────────────────────────

    def _check_user_response(
        self, agent_texts: List[str], current_tick_id: int
    ) -> None:
        """Verifie si les agents mentionnent le sujet d'une injection pending."""
        all_words = set()
        for t in agent_texts:
            all_words.update(w for w in _tokenize(t) if len(w) > 3)

        for inj in self._pending_injections:
            if inj["resolved"]:
                continue
            overlap = len(all_words & inj["words"])
            if overlap >= 2:
                latency = current_tick_id - inj["tick_id"]
                inj["resolved"] = True
                self._resolved_latencies.append(latency)

    def _novelty_spike(self, current_novelty: float) -> Optional[float]:
        """Max novelty spike relative to baseline across unresolved injections."""
        spikes: List[float] = []
        for inj in self._pending_injections:
            if not inj["resolved"]:
                spike = current_novelty - inj.get("novelty_baseline", 0.0)
                spikes.append(spike)
        return round(max(spikes), 4) if spikes else None

    def _ticks_since_injection(self, current_tick_id: int) -> Optional[int]:
        """Retourne le nb de ticks depuis la derniere injection non-resolue."""
        for inj in reversed(self._pending_injections):
            if not inj["resolved"]:
                return current_tick_id - inj["tick_id"]
        return None

    # ── Metriques : Embedding (optional) ──────────────────────────

    def _get_embed_model(self) -> Any:
        """Lazy-load the embedding model (or return None)."""
        if not self._embed_loaded:
            self._embed_loaded = True
            if _HAS_EMBEDDINGS:
                try:
                    self._embed_model = SentenceTransformer(
                        "all-MiniLM-L6-v2"
                    )
                except Exception:
                    self._embed_model = None
        return self._embed_model

    def _semantic_similarity(self, texts: List[str]) -> Optional[float]:
        """Mean pairwise cosine similarity between agent texts (embedding-based)."""
        model = self._get_embed_model()
        if model is None or len(texts) < 2:
            return None
        non_empty = [t for t in texts if t.strip()]
        if len(non_empty) < 2:
            return None
        embeddings = model.encode(non_empty)
        import numpy as np
        n = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                dot = float(np.dot(embeddings[i], embeddings[j]))
                ni = float(np.linalg.norm(embeddings[i]))
                nj = float(np.linalg.norm(embeddings[j]))
                sims.append(dot / (ni * nj) if ni > 0 and nj > 0 else 0.0)
        return round(sum(sims) / len(sims), 4) if sims else None

    def _topic_drift(self, texts: List[str]) -> Optional[float]:
        """1 - cosine(embedding_t, embedding_{t-1}). None if no embeddings."""
        model = self._get_embed_model()
        if model is None:
            return None
        combined = " ".join(t for t in texts if t.strip())
        if not combined.strip():
            return None
        curr = model.encode([combined])[0]
        if self._prev_embedding is None:
            self._prev_embedding = curr
            return 0.0
        import numpy as np
        dot = float(np.dot(self._prev_embedding, curr))
        n1 = float(np.linalg.norm(self._prev_embedding))
        n2 = float(np.linalg.norm(curr))
        cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
        self._prev_embedding = curr
        return round(1.0 - cos, 4)

    # ── Summary ───────────────────────────────────────────────────

    def _compute_summary(self) -> Dict[str, Any]:
        """Calcule le summary a partir de tous les ticks enregistres."""
        if not self._all_metrics:
            return {"run_id": self._run_id, "condition": self._condition,
                    "total_ticks": 0}

        total_ticks = len(self._all_metrics)

        # Extraire les valeurs de metriques
        def metric_values(key: str) -> List[float]:
            return [
                row["metrics"].get(key, 0.0)
                for row in self._all_metrics
                if isinstance(row["metrics"].get(key), (int, float))
            ]

        # Mode distribution
        mode_counts = Counter(row["mode"] for row in self._all_metrics)
        mode_dist = {m: round(c / total_ticks, 4) for m, c in mode_counts.items()}

        # Transitions count
        transitions = sum(1 for row in self._all_metrics if row.get("mode_changed"))

        # Veto rate
        veto_count = sum(1 for row in self._all_metrics if row.get("veto_flag"))

        # Latencies
        latencies = [row["elapsed_ms"] for row in self._all_metrics]

        # Tokens
        tokens_list = [row["total_tokens"] for row in self._all_metrics]

        # Meaningful turns (texte > 50 chars)
        meaningful_per_tick = []
        for row in self._all_metrics:
            meaningful = sum(1 for a in row["agents"] if a["text_len"] > 50)
            meaningful_per_tick.append(meaningful)

        # Cohesion values for trend
        cohesion_values = metric_values("cohesion")

        # Claim confidence values
        confidence_values = [
            row["wm_stats"].get("avg_confidence", 0.0)
            for row in self._all_metrics
        ]

        # Signal stability (variance sur derniere fenetre)
        signal_stability = {}
        signal_names = [
            "energy", "novelty", "conflict", "impl_pressure",
            "cohesion", "cost_pressure", "prediction_error",
        ]
        for name in signal_names:
            values = [
                row["signals"].get(name, 0.0)
                for row in self._all_metrics[-WINDOW_SIZE:]
            ]
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                signal_stability[name] = round(variance, 6)
            else:
                signal_stability[name] = 0.0

        # User response latencies
        inj_count = len(self._pending_injections)

        # Judge audit counters
        judge_total = 0
        judge_failed = 0
        judge_fixes = 0
        winner_mismatch = 0
        ranking_completed = 0
        for row in self._all_metrics:
            jd = row.get("judge_verdict")
            if jd:
                judge_total += 1
                if jd.get("judge_failed"):
                    judge_failed += 1
                audit = jd.get("_audit", {})
                fixes = audit.get("fixes", [])
                if fixes:
                    judge_fixes += 1
                if any("ranking_reorder" in f for f in fixes):
                    winner_mismatch += 1
                # FIX 2: Count rankings that were completed (< 3 agents)
                comp = jd.get("competition", {})
                if comp:
                    ranking = comp.get("ranking", [])
                    orig_len = audit.get("ranking_original_len")
                    if orig_len is not None and orig_len < 3:
                        ranking_completed += 1
                    elif len(ranking) == 3 and any("complete" in f.lower() for f in fixes):
                        ranking_completed += 1

        return {
            "run_id": self._run_id,
            "condition": self._condition,
            "total_ticks": total_ticks,
            "total_tokens": sum(tokens_list),
            "total_duration_ms": round(sum(latencies), 1),
            "diversity": {
                "mean_repetition_3gram": round(_mean(metric_values("repetition_3gram")), 4),
                "mean_dedup_ratio": round(_mean(metric_values("dedup_ratio")), 4),
                "mean_hashvec_novelty": round(_mean(metric_values("hashvec_novelty")), 4),
                "mean_agent_balance_entropy": round(_mean(metric_values("agent_balance_entropy")), 4),
                "mean_vocab_richness": round(_mean(metric_values("vocab_richness")), 4),
            },
            "modes": {
                "distribution": mode_dist,
                "mean_mode_entropy_w20": round(_mean(metric_values("mode_entropy_w20")), 4),
                "transitions_count": transitions,
            },
            "deliberation": {
                "veto_rate": round(veto_count / total_ticks, 4) if total_ticks else 0.0,
                "mean_wm_supported_ratio": round(_mean(metric_values("wm_supported_ratio")), 4),
                "mean_wm_contradicted_ratio": round(_mean(metric_values("wm_contradicted_ratio")), 4),
                "mean_wm_churn": round(_mean([
                    row["metrics"].get("wm_churn", 0)
                    for row in self._all_metrics
                ]), 2),
            },
            "responsiveness": {
                "injections_count": inj_count,
                "mean_user_response_latency_ticks": round(
                    _mean([float(x) for x in self._resolved_latencies]), 2
                ) if self._resolved_latencies else None,
                "p50_user_response_latency": _percentile(
                    [float(x) for x in self._resolved_latencies], 50
                ) if self._resolved_latencies else None,
                "p95_user_response_latency": _percentile(
                    [float(x) for x in self._resolved_latencies], 95
                ) if self._resolved_latencies else None,
            },
            "efficiency": {
                "mean_tokens_per_tick": round(_mean([float(x) for x in tokens_list]), 1),
                "mean_latency_ms": round(_mean(latencies), 1),
                "p50_latency_ms": round(_percentile(latencies, 50), 1),
                "p95_latency_ms": round(_percentile(latencies, 95), 1),
                "mean_meaningful_turns_per_tick": round(
                    _mean([float(x) for x in meaningful_per_tick]), 2
                ),
            },
            "convergence": {
                "cohesion_trend_slope": round(_linear_slope(cohesion_values), 6),
                "signal_stability_w20": signal_stability,
                "claim_confidence_trend_slope": round(
                    _linear_slope(confidence_values), 6
                ),
            },
            "judge_audit": {
                "total_verdicts": judge_total,
                "judge_failed": judge_failed,
                "judge_failed_rate": round(judge_failed / judge_total, 4) if judge_total else 0.0,
                "normalized_fixes": judge_fixes,
                "winner_ranking_mismatch": winner_mismatch,
                "ranking_completed": ranking_completed,
                "ranking_completed_rate": round(ranking_completed / judge_total, 4) if judge_total else 0.0,
            },
        }


def _mean(values: List[float]) -> float:
    """Moyenne d'une liste, 0.0 si vide."""
    return sum(values) / len(values) if values else 0.0
