"""
stem_metrics.py — STEM Metrics Pack
====================================
Standalone module that computes a comprehensive metrics pack from
events.jsonl + metrics.jsonl data. No Flask or organism imports.

Usage:
    from tools.stem_metrics import compute_stem_metrics
    m = compute_stem_metrics(events_path, metrics_path)
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Signal keys used for PCA ─────────────────────────────────
_SIGNAL_KEYS = ["novelty", "conflict", "cohesion", "impl_pressure",
                "prediction_error", "cost"]

_ALL_MODES = ["Idle", "Explore", "Debate", "Implement", "Consolidate", "Recover"]


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load_data(
    events_path: Path,
    metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read events.jsonl + metrics.jsonl and return unified tick data."""
    # ── Read tick_end events ──
    tick_data: List[Dict] = []
    with open(events_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "tick_end":
                tick_data.append(ev)

    n = len(tick_data)
    modes = [t.get("mode", "Idle") for t in tick_data]
    tick_ids = [t.get("tick_id", i + 1) for i, t in enumerate(tick_data)]

    # Signal matrix (n x 6)
    vectors = []
    for t in tick_data:
        vec = [float(t.get(k, 0.0)) for k in _SIGNAL_KEYS]
        vectors.append(vec)
    X = np.array(vectors, dtype=np.float64) if n > 0 else np.empty((0, len(_SIGNAL_KEYS)))

    # ── Read metrics.jsonl (richer data) ──
    metrics_rows: Dict[int, Dict] = {}
    if metrics_path and metrics_path.exists():
        try:
            with open(metrics_path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        tid = row.get("tick_id", row.get("tick", 0))
                        metrics_rows[tid] = row
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    # Per-tick enriched data
    per_tick: List[Dict] = []
    for i, td in enumerate(tick_data):
        tid = tick_ids[i]
        mr = metrics_rows.get(tid, {})
        jv = mr.get("judge_verdict") or {}
        comp = jv.get("competition") or {}
        wm = mr.get("wm_stats") or {}
        sigs = mr.get("signals") or {}

        per_tick.append({
            "tick_id": tid,
            "mode": modes[i],
            "signals": {k: float(sigs.get(k, td.get(k, 0.0))) for k in _SIGNAL_KEYS},
            "margin_1v2": comp.get("margin_1v2"),
            "margin_2v3": comp.get("margin_2v3"),
            "judge_winner": jv.get("winner"),
            "judge_confidence": jv.get("confidence"),
            "judge_failed": jv.get("judge_failed", False),
            "claims_added": mr.get("claims_added", 0),
            "wm_total_claims": wm.get("total_claims", 0),
            "wm_supported": wm.get("supported", 0),
            "wm_contradicted": wm.get("contradicted", 0),
            "wm_avg_confidence": wm.get("avg_confidence", 0.0),
            "theory_scores": mr.get("theory_scores") or {},
        })

    return {
        "n": n,
        "tick_ids": tick_ids,
        "modes": modes,
        "X": X,
        "per_tick": per_tick,
    }


# ═══════════════════════════════════════════════════════════════
# PCA computation
# ═══════════════════════════════════════════════════════════════

def _compute_pca(X: np.ndarray) -> Dict[str, Any]:
    """Compute PCA and return eigvals, eigvecs, projection, cov matrix."""
    n = X.shape[0]
    if n < 3 or X.shape[1] < 2:
        return {
            "proj": np.zeros((n, 3)),
            "eigvals": np.zeros(3),
            "eigvecs": np.eye(X.shape[1] if X.shape[1] > 0 else 3, 3),
            "cov": np.eye(3),
            "mean": np.zeros(X.shape[1] if X.shape[1] > 0 else 3),
        }

    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n_comp = min(3, len(eigvals))
    proj = Xc @ eigvecs[:, :n_comp]

    # Pad to 3 columns if needed
    if proj.shape[1] < 3:
        pad = np.zeros((n, 3 - proj.shape[1]))
        proj = np.hstack([proj, pad])

    return {
        "proj": proj,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "cov": cov,
        "mean": mean,
    }


def _compute_velocities(proj: np.ndarray) -> np.ndarray:
    """Euclidean distance between consecutive PCA points."""
    n = proj.shape[0]
    vel = np.zeros(n)
    for i in range(1, n):
        vel[i] = np.linalg.norm(proj[i] - proj[i - 1])
    return vel


# ═══════════════════════════════════════════════════════════════
# Attractor detection
# ═══════════════════════════════════════════════════════════════

def _detect_attractors(
    modes: List[str],
    proj: np.ndarray,
    velocities: np.ndarray,
    per_tick: List[Dict],
    min_dwell: int = 3,
) -> List[Dict[str, Any]]:
    """Detect attractors as contiguous same-mode regions with dwell >= min_dwell."""
    n = len(modes)
    attractors = []
    if n < 5:
        return attractors

    i = 0
    att_idx = 0
    while i < n:
        j = i
        while j < n and modes[j] == modes[i]:
            j += 1
        dwell = j - i
        if dwell >= min_dwell:
            region = proj[i:j]
            centroid = region.mean(axis=0)
            dists = np.linalg.norm(region - centroid, axis=1)
            radius = float(max(dists.max(), 0.3))

            # Mean speed inside attractor
            mean_speed = float(velocities[i:j].mean()) if j > i else 0.0

            # Mean signals inside
            mean_sigs = {}
            for sk in _SIGNAL_KEYS:
                vals = [per_tick[k]["signals"].get(sk, 0.0) for k in range(i, j)]
                mean_sigs[sk] = round(float(np.mean(vals)), 4) if vals else 0.0

            # Mean margins
            m1v2 = [per_tick[k]["margin_1v2"] for k in range(i, j)
                     if per_tick[k]["margin_1v2"] is not None]
            m2v3 = [per_tick[k]["margin_2v3"] for k in range(i, j)
                     if per_tick[k]["margin_2v3"] is not None]

            # Purity: % ticks matching dominant mode (trivially 1.0 for contiguous)
            purity = 1.0  # By construction, all ticks in the region have same mode

            attractors.append({
                "id": f"A#{att_idx}",
                "start": i,
                "end": j,
                "dominant_mode": modes[i],
                "dwell_ticks": dwell,
                "centroid": [round(float(c), 4) for c in centroid],
                "radius": round(radius, 4),
                "mean_speed": round(mean_speed, 4),
                "mean_signals": mean_sigs,
                "mean_margin_1v2": round(float(np.mean(m1v2)), 4) if m1v2 else None,
                "mean_margin_2v3": round(float(np.mean(m2v3)), 4) if m2v3 else None,
                "purity": round(purity, 4),
                "entry_count": 0,
                "exit_count": 0,
            })
            att_idx += 1
        i = j

    # Compute entry/exit counts
    for ai, att in enumerate(attractors):
        if ai > 0:
            att["entry_count"] = 1
        if ai < len(attractors) - 1:
            att["exit_count"] = 1

    # Silhouette proxy: min inter-centroid distance / own radius
    centroids_arr = np.array([a["centroid"] for a in attractors])
    for ai, att in enumerate(attractors):
        if len(attractors) > 1:
            dists_to_others = [
                np.linalg.norm(centroids_arr[ai] - centroids_arr[oj])
                for oj in range(len(attractors)) if oj != ai
            ]
            min_inter = min(dists_to_others)
            att["silhouette_proxy"] = round(
                min_inter / max(att["radius"], 0.01), 4
            )
        else:
            att["silhouette_proxy"] = None

    return attractors


def _assign_ticks_to_attractors(
    n: int,
    attractors: List[Dict],
) -> List[Optional[int]]:
    """Map each tick to its attractor index (or None if not in an attractor)."""
    att_ids: List[Optional[int]] = [None] * n
    for ai, att in enumerate(attractors):
        for t in range(att["start"], att["end"]):
            att_ids[t] = ai
    return att_ids


# ═══════════════════════════════════════════════════════════════
# Helper: statistics
# ═══════════════════════════════════════════════════════════════

def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson correlation, return None if insufficient data."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def _gini(values: List[float]) -> float:
    """Gini coefficient of a list of values."""
    if not values:
        return 0.0
    arr = np.sort(np.array(values, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def _autocorrelation(x: np.ndarray, max_lag: int = 10) -> List[Optional[float]]:
    """Compute autocorrelation for lags 1..max_lag."""
    n = len(x)
    if n < 3:
        return [None] * max_lag
    x = x - x.mean()
    var = (x ** 2).sum()
    if var < 1e-12:
        return [0.0] * max_lag
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            result.append(None)
        else:
            r = float((x[:n - lag] * x[lag:]).sum() / var)
            result.append(round(r, 4))
    return result


# ═══════════════════════════════════════════════════════════════
# B1: Geometry / Dimensionality
# ═══════════════════════════════════════════════════════════════

def compute_geometry_metrics(
    proj: np.ndarray,
    eigvals: np.ndarray,
    cov: np.ndarray,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute geometry & dimensionality metrics."""
    n = proj.shape[0]
    if n < 3:
        return {"error": "insufficient_data", "n_ticks": n}

    # Apply window
    if window and window < n:
        proj = proj[-window:]

    total_var = max(eigvals.sum(), 1e-12)
    ev = eigvals / total_var

    # Explained variance
    n_ev = min(3, len(ev))
    explained_variance = [round(float(ev[i]), 4) for i in range(n_ev)]

    # Effective dim (Renyi entropy)
    normed = eigvals[eigvals > 1e-12] / total_var
    eff_dim = float(np.exp(-np.sum(normed * np.log(normed + 1e-30))))

    # Participation ratio
    pr = float((eigvals.sum()) ** 2 / max((eigvals ** 2).sum(), 1e-30))

    # Covariance metrics
    k = min(3, cov.shape[0])
    cov_sub = cov[:k, :k]
    det = np.linalg.det(cov_sub)
    log_det_cov = float(np.log(max(det, 1e-30)))
    trace_cov = float(np.trace(cov))

    # Anisotropy
    if len(eigvals) >= 3 and (eigvals[1] + eigvals[2]) > 1e-12:
        anisotropy = float(eigvals[0] / (eigvals[1] + eigvals[2]))
    else:
        anisotropy = None

    # Radius metrics
    mean_pos = proj.mean(axis=0)
    radii = np.linalg.norm(proj - mean_pos, axis=1)
    mean_radius = float(radii.mean())
    p95_radius = float(np.percentile(radii, 95))

    # Remote analysis
    remote_mask = radii > p95_radius
    remote_rate = float(remote_mask.mean())

    # Max consecutive remote ticks
    max_run = 0
    current_run = 0
    for r in remote_mask:
        if r:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Remote return count
    remote_return = 0
    for i in range(1, len(remote_mask)):
        if remote_mask[i] and not remote_mask[i - 1]:
            remote_return += 1

    # Centroid drift (first half vs second half)
    half = len(proj) // 2
    if half > 0:
        drift = float(np.linalg.norm(proj[:half].mean(axis=0) - proj[half:].mean(axis=0)))
    else:
        drift = 0.0

    return {
        "explained_variance": explained_variance,
        "effective_dim": round(eff_dim, 4),
        "participation_ratio": round(pr, 4),
        "log_det_cov": round(log_det_cov, 4),
        "trace_cov": round(trace_cov, 4),
        "anisotropy": round(anisotropy, 4) if anisotropy is not None else None,
        "centroid_drift": round(drift, 4),
        "mean_radius": round(mean_radius, 4),
        "p95_radius": round(p95_radius, 4),
        "remote_rate": round(remote_rate, 4),
        "remote_max_run": max_run,
        "remote_return_count": remote_return,
    }


# ═══════════════════════════════════════════════════════════════
# B2: Dynamics
# ═══════════════════════════════════════════════════════════════

def compute_dynamics_metrics(
    modes: List[str],
    velocities: np.ndarray,
    per_tick: List[Dict],
    attractors: List[Dict],
    att_ids: List[Optional[int]],
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute dynamics metrics."""
    n = len(modes)
    if n < 3:
        return {"error": "insufficient_data", "n_ticks": n}

    # Apply window
    if window and window < n:
        start = n - window
        modes = modes[start:]
        velocities = velocities[start:]
        per_tick = per_tick[start:]
        att_ids = att_ids[start:]
        n = len(modes)

    vel = velocities

    # Speed distribution
    speed_stats = {
        "mean": round(float(vel.mean()), 4),
        "median": round(float(np.median(vel)), 4),
        "p95": round(float(np.percentile(vel, 95)), 4),
        "p99": round(float(np.percentile(vel, 99)), 4),
    }

    # Spike rate
    p95_speed = np.percentile(vel, 95)
    spike_mask = vel > p95_speed
    spike_rate = float(spike_mask.mean())

    # Spike → switch attractor
    spike_to_switch = None
    if att_ids and spike_mask.sum() > 0:
        switches = 0
        spikes = 0
        for i in range(len(spike_mask) - 1):
            if spike_mask[i]:
                spikes += 1
                if att_ids[i] is not None and att_ids[i + 1] is not None and att_ids[i] != att_ids[i + 1]:
                    switches += 1
        if spikes > 0:
            spike_to_switch = round(switches / spikes, 4)

    # Dwell distribution
    dwell_dist = [a["dwell_ticks"] for a in attractors]

    # Recurrence (returns to same attractor mode within W=20)
    recurrence = 0
    for i in range(len(attractors)):
        for j in range(i + 1, len(attractors)):
            if attractors[j]["start"] - attractors[i]["end"] <= 20:
                if attractors[j]["dominant_mode"] == attractors[i]["dominant_mode"]:
                    recurrence += 1

    # Mode transition matrix
    mode_trans = Counter()
    for i in range(n - 1):
        mode_trans[(modes[i], modes[i + 1])] += 1

    mode_trans_matrix: Dict[str, Dict[str, float]] = {}
    for m in _ALL_MODES:
        total = sum(mode_trans.get((m, m2), 0) for m2 in _ALL_MODES)
        if total > 0:
            mode_trans_matrix[m] = {
                m2: round(mode_trans.get((m, m2), 0) / total, 4)
                for m2 in _ALL_MODES
            }

    # Attractor transition matrix
    att_trans = Counter()
    prev_att = None
    for aid in att_ids:
        if aid is not None:
            if prev_att is not None and aid != prev_att:
                att_trans[(prev_att, aid)] += 1
            prev_att = aid

    # Mode entropy
    mode_counts = Counter(modes)
    mode_entropy = _shannon_entropy(mode_counts)

    # Attractor entropy
    att_occ = Counter(a for a in att_ids if a is not None)
    att_entropy = _shannon_entropy(att_occ)

    # Autocorrelations
    speed_autocorr = _autocorrelation(vel, 10)

    # Impl_pressure autocorrelation
    impl_p = np.array([pt["signals"].get("impl_pressure", 0.0) for pt in per_tick])
    impl_autocorr = _autocorrelation(impl_p, 5)

    # Mode autocorrelation (binary encoding per mode)
    mode_autocorr: Dict[str, List[Optional[float]]] = {}
    for m in _ALL_MODES:
        binary = np.array([1.0 if md == m else 0.0 for md in modes])
        if binary.sum() > 0:
            mode_autocorr[m] = _autocorrelation(binary, 10)

    # Markovity check: |P(m_{t+1}|m_t, m_{t-1}) - P(m_{t+1}|m_t)|
    markovity_delta = _compute_markovity(modes)

    return {
        "speed": speed_stats,
        "spike_rate": round(spike_rate, 4),
        "spike_to_switch": spike_to_switch,
        "dwell_distribution": dwell_dist,
        "recurrence": recurrence,
        "mode_transition_matrix": mode_trans_matrix,
        "attractor_transition_count": dict(
            (f"{k[0]}->{k[1]}", v) for k, v in att_trans.items()
        ),
        "mode_entropy": round(mode_entropy, 4),
        "attractor_entropy": round(att_entropy, 4),
        "speed_autocorr": speed_autocorr,
        "impl_pressure_autocorr": impl_autocorr,
        "mode_autocorr": mode_autocorr,
        "markovity_delta": markovity_delta,
    }


def _compute_markovity(modes: List[str]) -> Optional[float]:
    """Compare P(m_{t+1}|m_t) vs P(m_{t+1}|m_t,m_{t-1})."""
    n = len(modes)
    if n < 10:
        return None

    # First-order: P(m_{t+1}|m_t)
    first_order = Counter()
    first_denom = Counter()
    for i in range(n - 1):
        first_order[(modes[i], modes[i + 1])] += 1
        first_denom[modes[i]] += 1

    # Second-order: P(m_{t+1}|m_t, m_{t-1})
    second_order = Counter()
    second_denom = Counter()
    for i in range(1, n - 1):
        second_order[(modes[i - 1], modes[i], modes[i + 1])] += 1
        second_denom[(modes[i - 1], modes[i])] += 1

    # Max |P2 - P1|
    max_delta = 0.0
    for (m_prev, m_cur, m_next), count in second_order.items():
        p2 = count / max(second_denom[(m_prev, m_cur)], 1)
        p1 = first_order.get((m_cur, m_next), 0) / max(first_denom.get(m_cur, 0), 1)
        max_delta = max(max_delta, abs(p2 - p1))

    return round(max_delta, 4)


# ═══════════════════════════════════════════════════════════════
# B3: Attractor metrics (global)
# ═══════════════════════════════════════════════════════════════

def compute_attractor_global_metrics(
    attractors: List[Dict],
    n_ticks: int,
) -> Dict[str, Any]:
    """Global attractor metrics."""
    if not attractors:
        return {
            "n_attractors": 0,
            "dwell_max": 0,
            "gini_dwell": 0.0,
            "total_dwell": 0,
            "coverage": 0.0,
        }

    dwells = [a["dwell_ticks"] for a in attractors]
    total_dwell = sum(dwells)

    return {
        "n_attractors": len(attractors),
        "dwell_max": max(dwells),
        "gini_dwell": round(_gini(dwells), 4),
        "total_dwell": total_dwell,
        "coverage": round(total_dwell / max(n_ticks, 1), 4),
    }


# ═══════════════════════════════════════════════════════════════
# B4: Judge / competition metrics
# ═══════════════════════════════════════════════════════════════

def compute_judge_metrics(
    per_tick: List[Dict],
    velocities: np.ndarray,
    proj: np.ndarray,
) -> Dict[str, Any]:
    """Metrics from judge verdicts and competition data."""
    m1v2 = [pt["margin_1v2"] for pt in per_tick if pt["margin_1v2"] is not None]
    m2v3 = [pt["margin_2v3"] for pt in per_tick if pt["margin_2v3"] is not None]

    if not m1v2:
        return {"available": False}

    m1v2_arr = np.array(m1v2)
    m2v3_arr = np.array(m2v3) if m2v3 else np.array([])

    result: Dict[str, Any] = {"available": True}

    # Margin stats
    result["margin_1v2"] = {
        "mean": round(float(m1v2_arr.mean()), 4),
        "std": round(float(m1v2_arr.std()), 4),
        "min": round(float(m1v2_arr.min()), 4),
        "max": round(float(m1v2_arr.max()), 4),
    }
    if len(m2v3_arr) > 0:
        result["margin_2v3"] = {
            "mean": round(float(m2v3_arr.mean()), 4),
            "std": round(float(m2v3_arr.std()), 4),
            "min": round(float(m2v3_arr.min()), 4),
            "max": round(float(m2v3_arr.max()), 4),
        }

    # Correlations: need aligned arrays (only ticks where margin exists)
    n = len(per_tick)
    vel_aligned = []
    m1_aligned = []
    impl_aligned = []
    for i in range(n):
        if per_tick[i]["margin_1v2"] is not None:
            vel_aligned.append(velocities[i])
            m1_aligned.append(per_tick[i]["margin_1v2"])
            impl_aligned.append(per_tick[i]["signals"].get("impl_pressure", 0.0))

    vel_a = np.array(vel_aligned)
    m1_a = np.array(m1_aligned)
    impl_a = np.array(impl_aligned)

    result["corr_speed_margin1v2"] = _pearson(vel_a, m1_a)
    result["corr_impl_speed"] = _pearson(impl_a, vel_a)

    # Remote indicator vs margin
    n = len(per_tick)
    p95_r = np.percentile(np.linalg.norm(proj - proj.mean(axis=0), axis=1), 95)
    remote_all = np.linalg.norm(proj - proj.mean(axis=0), axis=1) > p95_r
    remote_aligned = []
    m1_full = []
    for i in range(len(per_tick)):
        if per_tick[i]["margin_1v2"] is not None:
            remote_aligned.append(float(remote_all[i]))
            m1_full.append(per_tick[i]["margin_1v2"])

    if len(remote_aligned) >= 3:
        result["corr_remote_margin1v2"] = _pearson(
            np.array(remote_aligned), np.array(m1_full)
        )
    else:
        result["corr_remote_margin1v2"] = None

    # Signal-PC correlations
    signal_pc_corr: Dict[str, Dict[str, Optional[float]]] = {}
    for si, sk in enumerate(_SIGNAL_KEYS):
        sig_vals = np.array([pt["signals"].get(sk, 0.0) for pt in per_tick])
        signal_pc_corr[sk] = {}
        for pc in range(min(3, proj.shape[1])):
            signal_pc_corr[sk][f"PC{pc + 1}"] = _pearson(sig_vals, proj[:, pc])
    result["corr_signals_pc"] = signal_pc_corr

    return result


# ═══════════════════════════════════════════════════════════════
# B5: World model / claims
# ═══════════════════════════════════════════════════════════════

def compute_claims_metrics(
    per_tick: List[Dict],
    velocities: np.ndarray,
) -> Dict[str, Any]:
    """Claims and world-model metrics."""
    total_claims_last = 0
    supported_last = 0
    contradicted_last = 0
    churn = []

    for pt in per_tick:
        total_claims_last = pt["wm_total_claims"]
        supported_last = pt["wm_supported"]
        contradicted_last = pt["wm_contradicted"]
        churn.append(pt["claims_added"])

    if total_claims_last == 0 and not any(c > 0 for c in churn):
        return {"available": False}

    churn_arr = np.array(churn, dtype=np.float64)

    result: Dict[str, Any] = {"available": True}

    result["supported_ratio"] = round(
        supported_last / max(total_claims_last, 1), 4
    )
    result["contradicted_ratio"] = round(
        contradicted_last / max(total_claims_last, 1), 4
    )
    result["churn_mean"] = round(float(churn_arr.mean()), 4)
    result["churn_std"] = round(float(churn_arr.std()), 4)
    result["churn_max"] = int(churn_arr.max())

    # Correlations
    result["corr_churn_speed"] = _pearson(churn_arr, velocities)

    return result


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

def compute_stem_metrics(
    events_path: Path,
    metrics_path: Optional[Path] = None,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the full STEM metrics pack.

    Returns a dict with keys:
        n_ticks, geometry, dynamics, attractors, attractors_global,
        judge, claims
    """
    data = _load_data(events_path, metrics_path)
    n = data["n"]
    if n == 0:
        return {"n_ticks": 0, "error": "no_data"}

    pca = _compute_pca(data["X"])
    proj = pca["proj"]
    vel = _compute_velocities(proj)

    attractors = _detect_attractors(
        data["modes"], proj, vel, data["per_tick"]
    )
    att_ids = _assign_ticks_to_attractors(n, attractors)

    geometry = compute_geometry_metrics(proj, pca["eigvals"], pca["cov"], window)
    dynamics = compute_dynamics_metrics(
        data["modes"], vel, data["per_tick"], attractors, att_ids, window
    )
    att_global = compute_attractor_global_metrics(attractors, n)
    judge = compute_judge_metrics(data["per_tick"], vel, proj)
    claims = compute_claims_metrics(data["per_tick"], vel)

    # Sanitize attractors for export (remove internal start/end)
    att_export = []
    for a in attractors:
        ae = dict(a)
        ae.pop("start", None)
        ae.pop("end", None)
        att_export.append(ae)

    return {
        "n_ticks": n,
        "geometry": geometry,
        "dynamics": dynamics,
        "attractors": att_export,
        "attractors_global": att_global,
        "judge": judge,
        "claims": claims,
    }
