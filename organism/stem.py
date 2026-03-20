"""
STEM - State Evolution Model
==============================
Accumulates state vectors from each tick, produces trajectory analyses.

Dimensions (25-33D depending on N agents + K theories):
  7 signals + mode(encoded) + N agent_active + balance_entropy
  + 2 judge (confidence, margin_1v2) + 3 WM (total, supported, contradicted)
  + 2 memory (l0r_slots, energy) + K theory scores

Analyses every 20 ticks:
  - PCA 2D/3D
  - Velocity (L2 norm between consecutive states)
  - Phase transitions (velocity > mu + 2*sigma)
  - Attractors (KMeans, min dwell time)
  - Effective dimensionality (participation ratio)

API:
  stem.on_tick(state)     -> accumulate
  stem.analyze()          -> run PCA, detect attractors, phase transitions
  stem.snapshot()         -> dict for WebSocket emission
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from organism.organism_state import OrganismState

log = logging.getLogger("organism.stem")

# Mode encoding
_MODE_MAP = {
    "Idle": 0.0,
    "Explore": 0.2,
    "Debate": 0.4,
    "Implement": 0.6,
    "Consolidate": 0.8,
    "Recover": 1.0,
}

_ANALYSIS_INTERVAL = 20  # Ticks between full analyses


@dataclass
class PCAResult:
    """PCA projection results."""
    components_2d: List[Tuple[float, float]] = field(default_factory=list)
    components_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    explained_variance: List[float] = field(default_factory=list)  # top-3 / sum(all)
    total_variance_explained: float = 0.0  # sum(top-3) / sum(all)
    all_eigenvalues: List[float] = field(default_factory=list)  # ALL eigenvalues (raw)


@dataclass
class Attractor:
    """Detected attractor in state space."""
    center_2d: Tuple[float, float] = (0.0, 0.0)
    center_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dwell_ticks: int = 0
    dominant_mode: str = ""
    radius: float = 0.0


@dataclass
class PhaseTransition:
    """Detected phase transition."""
    tick_id: int = 0
    velocity: float = 0.0
    from_mode: str = ""
    to_mode: str = ""


@dataclass
class STEMAnalysis:
    """Complete STEM analysis result."""
    pca: Optional[PCAResult] = None
    velocities: List[float] = field(default_factory=list)
    phase_transitions: List[PhaseTransition] = field(default_factory=list)
    attractors: List[Attractor] = field(default_factory=list)
    effective_dimensionality: float = 0.0
    n_ticks: int = 0


class StateEvolutionModel:
    """
    Accumulates state vectors and produces trajectory analyses.
    Agent-count agnostic: vector dimensions adapt to N agents + K theories.
    """

    def __init__(self, analysis_interval: int = _ANALYSIS_INTERVAL):
        self._vectors: List[List[float]] = []
        self._tick_ids: List[int] = []
        self._modes: List[str] = []
        self._analysis_interval = analysis_interval
        self._last_analysis: Optional[STEMAnalysis] = None
        self._dim_names: List[str] = []

    def on_tick(self, state: OrganismState) -> None:
        """Convert OrganismState to vector and accumulate."""
        vec, names = self._state_to_vector(state)
        self._vectors.append(vec)
        self._tick_ids.append(state.tick_id)
        self._modes.append(state.mode)
        self._dim_names = names

        # Auto-analyze at interval
        if len(self._vectors) % self._analysis_interval == 0:
            self.analyze()

    def analyze(self) -> STEMAnalysis:
        """Run full analysis on accumulated vectors."""
        if len(self._vectors) < 5:
            self._last_analysis = STEMAnalysis(n_ticks=len(self._vectors))
            return self._last_analysis

        analysis = STEMAnalysis(n_ticks=len(self._vectors))

        # 1. PCA
        analysis.pca = self._compute_pca()

        # 2. Velocities
        analysis.velocities = self._compute_velocities()

        # 3. Phase transitions
        analysis.phase_transitions = self._detect_phase_transitions(
            analysis.velocities
        )

        # 4. Attractors (on PCA 2D)
        if analysis.pca and analysis.pca.components_2d:
            analysis.attractors = self._detect_attractors(
                analysis.pca.components_2d
            )

        # 5. Effective dimensionality
        analysis.effective_dimensionality = self._effective_dimensionality()

        self._last_analysis = analysis
        return analysis

    def snapshot(self) -> Dict[str, Any]:
        """Dict for WebSocket emission."""
        if not self._last_analysis:
            if len(self._vectors) >= 5:
                self.analyze()
            else:
                return {"n_ticks": len(self._vectors), "ready": False}

        a = self._last_analysis
        result: Dict[str, Any] = {
            "n_ticks": a.n_ticks,
            "ready": True,
            "effective_dim": round(a.effective_dimensionality, 2),
            "effective_dim_3pc": round(self._effective_dimensionality_3pc(), 2),
        }

        if a.pca:
            result["pca_2d"] = [
                {"x": round(p[0], 4), "y": round(p[1], 4)}
                for p in a.pca.components_2d
            ]
            result["pca_3d"] = [
                {"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}
                for p in a.pca.components_3d
            ]
            result["explained_variance"] = [
                round(v, 4) for v in a.pca.explained_variance
            ]
            result["total_variance_3d"] = round(a.pca.total_variance_explained, 4)
            result["all_eigenvalues"] = [
                round(v, 6) for v in a.pca.all_eigenvalues
            ]

        result["modes"] = self._modes[-a.n_ticks:]
        result["tick_ids"] = self._tick_ids[-a.n_ticks:]

        result["velocities"] = [round(v, 4) for v in a.velocities]

        result["phase_transitions"] = [
            {
                "tick_id": pt.tick_id,
                "velocity": round(pt.velocity, 4),
                "from_mode": pt.from_mode,
                "to_mode": pt.to_mode,
            }
            for pt in a.phase_transitions
        ]

        result["attractors"] = [
            {
                "center_2d": {"x": round(att.center_2d[0], 4),
                              "y": round(att.center_2d[1], 4)},
                "dwell_ticks": att.dwell_ticks,
                "dominant_mode": att.dominant_mode,
                "radius": round(att.radius, 4),
            }
            for att in a.attractors
        ]

        return result

    # -- Vectorization -------------------------------------------------------

    def _state_to_vector(self, state: OrganismState) -> Tuple[List[float], List[str]]:
        """Convert OrganismState to fixed-dimension vector."""
        vec = []
        names = []

        # 7 signals
        for key in ["energy", "novelty", "conflict", "impl_pressure",
                     "cohesion", "cost_pressure", "prediction_error"]:
            vec.append(state.signals.get(key, 0.0))
            names.append(f"sig_{key}")

        # Mode encoded
        vec.append(_MODE_MAP.get(state.mode, 0.0))
        names.append("mode")

        # N agent active flags
        for t in state.agent_turns:
            active = 1.0 if t.text.strip() else 0.0
            vec.append(active)
            names.append(f"agent_{t.agent}_active")

        # Balance entropy (how balanced are agent outputs)
        lengths = [len(t.text) for t in state.agent_turns if t.text.strip()]
        if lengths:
            total = sum(lengths)
            if total > 0:
                probs = [l / total for l in lengths]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                max_ent = math.log2(len(lengths)) if len(lengths) > 1 else 1.0
                vec.append(entropy / max_ent if max_ent > 0 else 0.0)
            else:
                vec.append(0.0)
        else:
            vec.append(0.0)
        names.append("balance_entropy")

        # 2 judge dimensions
        if state.judge_verdict:
            vec.append(state.judge_verdict.confidence)
            m12 = 0.5
            if state.judge_verdict.competition:
                m12 = state.judge_verdict.competition.margin_1v2
            vec.append(m12)
        else:
            vec.extend([0.0, 0.0])
        names.extend(["judge_confidence", "judge_margin_1v2"])

        # 3 WM dimensions
        wm = state.wm_stats
        vec.append(min(1.0, wm.get("total_claims", 0) / 30.0))
        vec.append(min(1.0, wm.get("supported", 0) / 15.0))
        vec.append(min(1.0, wm.get("contradicted", 0) / 10.0))
        names.extend(["wm_total", "wm_supported", "wm_contradicted"])

        # 2 memory dimensions
        vec.append(min(1.0, state.l0r_stats.get("active_slots", 0) / 64.0))
        vec.append(state.signals.get("energy", 0.5))
        names.extend(["l0r_fill", "mem_energy"])

        # K theory scores (NaN → 0.0 to prevent propagation in PCA)
        for theory_name in sorted(state.theory_scores.keys()):
            val = state.theory_scores[theory_name]
            vec.append(val if math.isfinite(val) else 0.0)
            names.append(f"theory_{theory_name}")

        return vec, names

    # -- PCA (pure Python, no numpy dependency) ------------------------------

    def _compute_pca(self) -> PCAResult:
        """Simple PCA via power iteration (no numpy needed)."""
        if len(self._vectors) < 3:
            return PCAResult()

        # Ensure all vectors same length (pad shorter ones), sanitize NaN/inf
        max_dim = max(len(v) for v in self._vectors)
        data = [
            [x if math.isfinite(x) else 0.0 for x in v]
            + [0.0] * (max_dim - len(v))
            for v in self._vectors
        ]
        n = len(data)
        d = max_dim

        # Center data
        means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
        centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]

        # Drop zero-variance dimensions (constant columns degenerate the cov matrix)
        variances = [sum(centered[k][j] ** 2 for k in range(n)) / max(1, n - 1) for j in range(d)]
        active_dims = [j for j in range(d) if variances[j] > 1e-12]
        if len(active_dims) < 2:
            return PCAResult()
        d_active = len(active_dims)
        centered_active = [[centered[k][j] for j in active_dims] for k in range(n)]

        # Covariance matrix (d_active x d_active)
        cov = [[0.0] * d_active for _ in range(d_active)]
        for i in range(d_active):
            for j in range(i, d_active):
                val = sum(centered_active[k][i] * centered_active[k][j] for k in range(n)) / max(1, n - 1)
                cov[i][j] = val
                cov[j][i] = val

        # Full eigendecomposition via power iteration + deflation (ALL eigenvalues)
        all_eigenvalues = []
        all_eigenvectors = []
        residual_cov = [row[:] for row in cov]

        for _ in range(d_active):
            ev, eigval = self._power_iteration(residual_cov, d_active)
            if eigval < 1e-10:
                break
            all_eigenvalues.append(eigval)
            all_eigenvectors.append(ev)
            # Deflate
            for i in range(d_active):
                for j in range(d_active):
                    residual_cov[i][j] -= eigval * ev[i] * ev[j]

        # Top 3 eigenvectors for projection
        top3_eigvecs = all_eigenvectors[:3]

        # Project data (using active dims of centered data)
        components_2d = []
        components_3d = []
        for row in centered_active:
            p2 = [0.0, 0.0]
            p3 = [0.0, 0.0, 0.0]
            for k in range(min(len(top3_eigvecs), 3)):
                proj = sum(row[j] * top3_eigvecs[k][j] for j in range(d_active))
                if k < 2:
                    p2[k] = proj
                p3[k] = proj
            components_2d.append(tuple(p2))
            components_3d.append(tuple(p3))

        # explained_variance: top-3 eigenvalues normalized by sum of ALL eigenvalues
        total_var_all = sum(all_eigenvalues)
        explained = [ev / total_var_all if total_var_all > 0 else 0.0
                     for ev in all_eigenvalues[:3]]

        return PCAResult(
            components_2d=components_2d,
            components_3d=components_3d,
            explained_variance=explained,
            total_variance_explained=sum(explained),
            all_eigenvalues=all_eigenvalues,
        )

    @staticmethod
    def _power_iteration(matrix, d, max_iter=100):
        """Find dominant eigenvector via power iteration."""
        # Initial vector
        vec = [1.0 / math.sqrt(d)] * d
        eigenvalue = 0.0

        for _ in range(max_iter):
            # Matrix-vector multiply
            new_vec = [sum(matrix[i][j] * vec[j] for j in range(d)) for i in range(d)]
            # Norm
            norm = math.sqrt(sum(v * v for v in new_vec))
            if norm < 1e-12:
                return vec, 0.0
            new_vec = [v / norm for v in new_vec]
            # Eigenvalue estimate (Rayleigh quotient)
            eigenvalue = sum(
                new_vec[i] * sum(matrix[i][j] * new_vec[j] for j in range(d))
                for i in range(d)
            )
            # Convergence check
            diff = sum((new_vec[i] - vec[i]) ** 2 for i in range(d))
            vec = new_vec
            if diff < 1e-10:
                break

        return vec, eigenvalue

    # -- Velocities -----------------------------------------------------------

    def _compute_velocities(self) -> List[float]:
        """L2 norm between consecutive state vectors."""
        velocities = [0.0]
        max_dim = max(len(v) for v in self._vectors)
        for i in range(1, len(self._vectors)):
            v1 = self._vectors[i - 1] + [0.0] * (max_dim - len(self._vectors[i - 1]))
            v2 = self._vectors[i] + [0.0] * (max_dim - len(self._vectors[i]))
            dist = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(v1, v2)
                if math.isfinite(a) and math.isfinite(b)
            ))
            velocities.append(dist)
        return velocities

    # -- Phase transitions ----------------------------------------------------

    def _detect_phase_transitions(
        self, velocities: List[float],
    ) -> List[PhaseTransition]:
        """Detect phase transitions: velocity > mu + 2*sigma."""
        if len(velocities) < 5:
            return []

        mean_v = sum(velocities) / len(velocities)
        var_v = sum((v - mean_v) ** 2 for v in velocities) / len(velocities)
        std_v = math.sqrt(var_v)
        threshold = mean_v + 2.0 * std_v

        transitions = []
        modes = self._modes
        for i, vel in enumerate(velocities):
            if vel > threshold:
                from_mode = modes[i - 1] if i > 0 else ""
                to_mode = modes[i] if i < len(modes) else ""
                transitions.append(PhaseTransition(
                    tick_id=self._tick_ids[i] if i < len(self._tick_ids) else 0,
                    velocity=vel,
                    from_mode=from_mode,
                    to_mode=to_mode,
                ))
        return transitions

    # -- Attractors -----------------------------------------------------------

    def _detect_attractors(
        self,
        points_2d: List[Tuple[float, float]],
        min_dwell: int = 5,
        n_clusters: int = 5,
    ) -> List[Attractor]:
        """Simple KMeans-like clustering on 2D PCA points."""
        if len(points_2d) < min_dwell:
            return []

        # Mini KMeans (Lloyd's algorithm)
        k = min(n_clusters, len(points_2d) // min_dwell)
        if k < 1:
            return []

        # Initialize centers evenly spaced
        step = max(1, len(points_2d) // k)
        centers = [points_2d[i * step] for i in range(k)]

        for _ in range(20):  # Max iterations
            # Assign
            assignments = []
            for p in points_2d:
                dists = [
                    (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                    for c in centers
                ]
                assignments.append(dists.index(min(dists)))

            # Update centers
            new_centers = []
            for ci in range(k):
                members = [points_2d[j] for j in range(len(points_2d))
                           if assignments[j] == ci]
                if members:
                    cx = sum(m[0] for m in members) / len(members)
                    cy = sum(m[1] for m in members) / len(members)
                    new_centers.append((cx, cy))
                else:
                    new_centers.append(centers[ci])

            if new_centers == centers:
                break
            centers = new_centers

        # Build attractors from clusters
        attractors = []
        for ci in range(k):
            member_indices = [j for j in range(len(points_2d))
                              if assignments[j] == ci]
            if len(member_indices) < min_dwell:
                continue

            members = [points_2d[j] for j in member_indices]
            cx = sum(m[0] for m in members) / len(members)
            cy = sum(m[1] for m in members) / len(members)
            radius = max(
                math.sqrt((m[0] - cx) ** 2 + (m[1] - cy) ** 2)
                for m in members
            ) if members else 0.0

            # Dominant mode in this cluster
            cluster_modes = [self._modes[j] for j in member_indices
                             if j < len(self._modes)]
            mode_counts = Counter(cluster_modes)
            dominant_mode = mode_counts.most_common(1)[0][0] if mode_counts else ""

            attractors.append(Attractor(
                center_2d=(cx, cy),
                dwell_ticks=len(member_indices),
                dominant_mode=dominant_mode,
                radius=radius,
            ))

        return attractors

    # -- Effective dimensionality ---------------------------------------------

    def _effective_dimensionality(self) -> float:
        """Participation ratio on ALL eigenvalues: (sum(λ))² / sum(λ²)."""
        if len(self._vectors) < 3:
            return 0.0

        # Use ALL eigenvalues from PCA (not just top 3)
        if self._last_analysis and self._last_analysis.pca:
            evs = self._last_analysis.pca.all_eigenvalues
        else:
            pca = self._compute_pca()
            evs = pca.all_eigenvalues

        if not evs or sum(evs) == 0:
            return 0.0

        sum_lambda = sum(evs)
        sum_lambda2 = sum(e ** 2 for e in evs)
        if sum_lambda2 == 0:
            return 0.0

        return (sum_lambda ** 2) / sum_lambda2

    def _effective_dimensionality_3pc(self) -> float:
        """Legacy: participation ratio on top-3 eigenvalues only (for comparison)."""
        if self._last_analysis and self._last_analysis.pca:
            evs = self._last_analysis.pca.explained_variance
        else:
            return 0.0
        if not evs or sum(evs) == 0:
            return 0.0
        sum_lambda = sum(evs)
        sum_lambda2 = sum(e ** 2 for e in evs)
        return (sum_lambda ** 2) / sum_lambda2 if sum_lambda2 > 0 else 0.0
