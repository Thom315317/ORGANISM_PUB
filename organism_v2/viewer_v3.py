#!/usr/bin/env python3
"""
viewer_v3.py — V3 Semantic Trajectory Viewer
==============================================
Upgraded from viewer_v2.py. New features:
  - UMAP 3D projection (toggle PCA/UMAP)
  - Connectome edges (cosine sim > threshold in raw embedding space)
  - Metric coloring (CCV, velocity, PSV, quality, ranking disagreement)
  - Trail fading (last 12 ticks visible during playback)
  - Distance matrix heatmap

Usage:
    python organism_v2/viewer_v3.py --runs-dir runs/bench_v2/ --port 8767
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from flask import Flask, jsonify, Response, send_from_directory, request

log = logging.getLogger("viewer_v3")
app = Flask(__name__)

RUNS_DIR: Path = Path(".")
_V2_DIR = Path(__file__).resolve().parent

# Optional UMAP
try:
    warnings.filterwarnings("ignore", message="n_jobs value.*random_state", category=UserWarning)
    from umap import UMAP as _UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    log.warning("umap-learn not installed — UMAP projection unavailable")

# Cache: (run_name, n_ticks) → computed heavy fields (UMAP, connectome, dist_matrix)
_heavy_cache: Dict[tuple, dict] = {}


@app.route("/three.min.js")
def serve_threejs():
    return send_from_directory(str(_V2_DIR), "three.min.js", mimetype="application/javascript")


@app.route("/api/runs")
def api_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and (d / "results.json").exists():
            runs.append(d.name)
    return jsonify({"runs": runs})


def _load_results(run_name: str):
    p = RUNS_DIR / run_name / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _valid_mask(vectors: List) -> np.ndarray:
    """Return boolean mask: True for rows that are finite and non-zero-norm."""
    arr = np.array(vectors, dtype=np.float64)
    finite = np.all(np.isfinite(arr), axis=1)
    nonzero = np.linalg.norm(arr, axis=1) > 1e-12
    return finite & nonzero


def _pca_3d(vectors):
    arr = np.array(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return ([[0, 0, 0]] * arr.shape[0] if arr.ndim == 2 else [], [0, 0, 0])
    centered = arr - arr.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:3]
    basis = eigvecs[:, idx]
    projected = centered @ basis
    mx = np.abs(projected).max()
    if mx > 0:
        projected = projected / mx
    total = eigvals.sum()
    explained = [round(float(eigvals[i] / total), 4) if total > 0 else 0 for i in idx]
    return projected.tolist(), explained


def _umap_3d(vectors):
    """UMAP 3D projection. Returns (points, n_neighbors_used) or (None, 0)."""
    if not _HAS_UMAP:
        return None, 0
    try:
        arr = np.array(vectors, dtype=np.float64)
        mask = _valid_mask(vectors)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 4:
            return None, 0
        valid = arr[valid_idx]
        k = min(10, len(valid) - 1)
        reducer = _UMAP(n_components=3, n_neighbors=k,
                        min_dist=0.1, metric='cosine', random_state=42)
        projected = reducer.fit_transform(valid)
        mx = np.abs(projected).max()
        if mx > 0:
            projected = projected / mx
        # Map back to full array (invalid rows get [0,0,0])
        full = np.zeros((arr.shape[0], 3))
        for out_i, orig_i in enumerate(valid_idx):
            full[orig_i] = projected[out_i]
        return full.tolist(), k
    except Exception as exc:
        log.warning("UMAP failed: %s", exc)
        return None, 0


def _connectome(vectors, threshold=0.80, max_edges=500):
    """Compute connectome edges from raw embeddings."""
    arr = np.array(vectors, dtype=np.float64)
    mask = _valid_mask(vectors)
    n = arr.shape[0]
    # Precompute norms
    norms = np.linalg.norm(arr, axis=1)
    edges = []
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(i + 2, n):  # |i-j| > 1
            if not mask[j]:
                continue
            sim = float(np.dot(arr[i], arr[j]) / (norms[i] * norms[j]))
            if sim > threshold:
                edges.append({"i": i, "j": j, "sim": round(sim, 4)})
    # Cap at max_edges (keep highest sim)
    if len(edges) > max_edges:
        edges.sort(key=lambda e: -e["sim"])
        edges = edges[:max_edges]
    return edges


def _dist_matrix(vectors):
    """Cosine distance matrix."""
    arr = np.array(vectors, dtype=np.float64)
    mask = _valid_mask(vectors)
    n = arr.shape[0]
    norms = np.linalg.norm(arr, axis=1)
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            if not mask[i] or not mask[j]:
                row.append(None)
            else:
                sim = float(np.dot(arr[i], arr[j]) / (norms[i] * norms[j]))
                row.append(round(1.0 - sim, 4))
        mat.append(row)
    return mat


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/data")
def api_data():
    run_name = request.args.get("run", "")
    if not run_name:
        return jsonify({"error": "No run selected", "ticks": 0})

    results = _load_results(run_name)
    if not results:
        return jsonify({"error": f"No results.json in {run_name}", "ticks": 0})

    sv_mean = results.get("state_vector_mean", [])
    sv_selected = results.get("state_vector_selected", [])
    n = len(sv_mean)

    if n < 2:
        return jsonify({"error": f"Only {n} ticks, need >= 2", "ticks": n})

    pts_mean, explained_mean = _pca_3d(sv_mean)
    pts_selected, explained_selected = _pca_3d(sv_selected)

    # Heavy computations cached by (run, n_ticks) — recomputed only when new ticks arrive
    cache_key = (run_name, n)
    if cache_key not in _heavy_cache:
        umap_mean, umap_k = _umap_3d(sv_mean)
        umap_selected, _ = _umap_3d(sv_selected)
        conn = _connectome(sv_mean)
        dm = _dist_matrix(sv_mean)
        _heavy_cache[cache_key] = {
            "umap_mean": umap_mean,
            "umap_selected": umap_selected,
            "umap_k": umap_k,
            "connectome": conn,
            "dist_matrix": dm,
        }
    cached = _heavy_cache[cache_key]
    umap_mean = cached["umap_mean"]
    umap_selected = cached["umap_selected"]
    umap_k = cached["umap_k"]
    conn = cached["connectome"]
    dm = cached["dist_matrix"]

    # Mean-Selected gap: mean cosine distance across ticks
    ms_gap = None
    if len(sv_mean) == len(sv_selected) and len(sv_mean) > 0:
        dists = []
        for i in range(len(sv_mean)):
            vm = np.array(sv_mean[i])
            vs = np.array(sv_selected[i])
            nm = np.linalg.norm(vm)
            ns = np.linalg.norm(vs)
            if nm > 0 and ns > 0:
                sim = float(np.dot(vm, vs) / (nm * ns))
                dists.append(1.0 - sim)
        if dists:
            ms_gap = round(float(np.mean(dists)), 4)

    pert_ticks = [p["tick"] for p in results.get("perturbation_log", [])]
    pert_types = {str(p["tick"]): p["type"] for p in results.get("perturbation_log", [])}

    return jsonify({
        "condition": results.get("condition", "?"),
        "seed": results.get("seed", 0),
        "ticks": n,
        "warmup": results.get("warmup_ticks", 5),
        "points_mean": pts_mean,
        "points_selected": pts_selected,
        "explained_mean": explained_mean,
        "explained_selected": explained_selected,
        "umap_mean": umap_mean,
        "umap_selected": umap_selected,
        "n_neighbors_used": umap_k,
        "connectome": conn,
        "dist_matrix": dm,
        "claim_cosine_variance": results.get("claim_cosine_variance", []),
        "judge_score_dispersion": results.get("judge_score_dispersion", results.get("ranking_disagreement", [])),
        "draft_velocity": results.get("draft_velocity", []),
        "post_selection_variance": results.get("post_selection_variance", []),
        "quality_per_tick": results.get("quality_per_tick", []),
        "perturbation_ticks": pert_ticks,
        "perturbation_types": pert_types,
        "sim_curves": results.get("sim_curves"),
        "ms_gap": ms_gap,
    })


# ── HTML ──────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Organism V3 — Semantic Trajectory</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0e14; color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    overflow: hidden;
}
#header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(10,14,20,0.95); border-bottom: 1px solid #1e2530;
    padding: 6px 12px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
#header h1 { font-size: 13px; font-weight: 600; color: #7ee787; white-space: nowrap; }
.stat { font-size: 10px; white-space: nowrap; }
.stat .label { color: #6e7681; }
.stat .value { color: #f0f6fc; font-weight: 600; }
select, .btn, .view-toggle {
    background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: 2px 7px; border-radius: 4px; cursor: pointer; font-size: 10px;
    font-family: inherit;
}
select { min-width: 120px; }
.btn:hover { background: #30363d; }
.view-toggle.active { background: #388bfd; border-color: #388bfd; color: #fff; }
.view-toggle:disabled { opacity: 0.3; cursor: not-allowed; }
.view-toggle.on { background: #238636; border-color: #238636; color: #fff; }
#transport { display: flex; align-items: center; gap: 4px; }
#transport .btn { font-size: 12px; padding: 1px 6px; }
#timeline {
    position: fixed; bottom: 20px; left: 0; right: 320px; z-index: 10;
    padding: 0 12px; display: flex; align-items: center; gap: 8px;
}
#timeline input[type=range] { flex: 1; accent-color: #7ee787; }
#timeline .tick-label { font-size: 10px; color: #8b949e; min-width: 60px; }
#three-container { position: fixed; top: 40px; left: 0; right: 320px; bottom: 40px; }
#sidebar {
    position: fixed; top: 40px; right: 0; bottom: 0; width: 320px;
    background: rgba(10,14,20,0.95); border-left: 1px solid #1e2530;
    padding: 8px; overflow-y: auto; font-size: 10px;
}
.panel { margin-bottom: 10px; }
.panel-title { color: #7ee787; font-size: 10px; font-weight: 600; margin-bottom: 3px; }
canvas.chart {
    width: 100%; height: 50px; background: #161b22;
    border-radius: 4px; margin-bottom: 2px; display: block;
}
.legend-row { display: flex; align-items: center; gap: 5px; margin: 1px 0; font-size: 9px; }
.legend-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
#status { position: fixed; bottom: 2px; left: 8px; font-size: 9px; color: #484f58; z-index: 10; }
#tick-info { position: fixed; bottom: 40px; left: 12px; z-index: 10; font-size: 10px; color: #8b949e; }
.slider-row { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
.slider-row input[type=range] { flex: 1; accent-color: #7ee787; }
.slider-row label { min-width: 100px; font-size: 10px; color: #8b949e; }
.note { font-size: 9px; color: #484f58; margin-top: 2px; }
</style>
</head>
<body>
<div id="header">
    <h1>V3</h1>
    <select id="run-select" onchange="onRunChange()"><option value="">— run —</option></select>
    <span class="stat"><span class="label">C </span><span class="value" id="s-cond">—</span></span>
    <span class="stat"><span class="label">S </span><span class="value" id="s-seed">—</span></span>
    <span class="stat"><span class="label">T </span><span class="value" id="s-ticks">—</span></span>
    <span class="stat"><span class="value" id="s-proj">—</span></span>
    <button class="view-toggle" onclick="setView('mean')">Mean</button>
    <button class="view-toggle active" onclick="setView('selected')">Selected</button>
    <span class="stat" id="ms-gap-container" style="display:none"><span class="label">M-S gap </span><span class="value" id="ms-gap">—</span></span>
    <span style="color:#30363d">|</span>
    <button class="view-toggle active" id="btn-pca" onclick="setProj('pca')">PCA</button>
    <button class="view-toggle" id="btn-umap" onclick="setProj('umap')">UMAP</button>
    <button class="view-toggle" id="btn-dist3d" onclick="setProj('dist3d')">DIST-3D</button>
    <span style="color:#30363d">|</span>
    <button class="view-toggle" id="btn-conn" onclick="toggleConn()">Connectome</button>
    <button class="view-toggle" id="btn-trail" onclick="toggleTrail()">Trail</button>
    <button class="view-toggle" id="btn-octants" onclick="toggleOctants()">Octants</button>
    <div id="transport">
        <button class="btn" onclick="stepBack()">◀</button>
        <button class="btn" id="btn-play" onclick="togglePlay()">▶</button>
        <button class="btn" onclick="stepForward()">▶▶</button>
        <select id="speed-select" onchange="setSpeed(this.value)">
            <option value="2000">0.5×</option>
            <option value="1000" selected>1×</option>
            <option value="500">2×</option>
            <option value="200">5×</option>
        </select>
    </div>
</div>
<div id="three-container"></div>
<div id="timeline">
    <span class="tick-label" id="tl-label">Tick —</span>
    <input type="range" id="tl-slider" min="1" max="50" value="50" oninput="onSlider(this.value)">
    <span class="tick-label" id="tl-max">/ —</span>
</div>
<div id="tick-info"></div>
<div id="sidebar">
    <div class="panel">
        <div class="panel-title">Sphere Color</div>
        <select id="color-select" onchange="setColorMode(this.value)" style="width:100%">
            <option value="temporal">Temporal (default)</option>
            <option value="claim_cosine_variance">CCV</option>
            <option value="draft_velocity">Velocity</option>
            <option value="post_selection_variance">PSV</option>
            <option value="quality_per_tick">Quality</option>
            <option value="judge_score_dispersion">Judge Score Dispersion</option>
        </select>
    </div>
    <div class="panel" id="conn-panel" style="display:none">
        <div class="panel-title">Connectome</div>
        <div class="slider-row">
            <label id="conn-label">Threshold: 0.80</label>
            <input type="range" id="conn-slider" min="70" max="99" value="80" step="1"
                   oninput="onConnThreshold(this.value)">
        </div>
        <div class="note" id="conn-note">Edges: cosine sim in embedding space | positions: PCA</div>
        <div class="note" id="conn-count">0 edges</div>
        <div class="note" style="color:#8b949e;font-size:10px;margin-top:4px">Edges reflect temporal semantic cohesion of the collective mean vector — not directly comparable across conditions with different agent counts.</div>
    </div>
    <div class="panel">
        <div class="panel-title">Claim Cosine Variance</div>
        <canvas class="chart" id="chart-ccv"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Draft Velocity</div>
        <canvas class="chart" id="chart-dv"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Post-Selection Variance</div>
        <canvas class="chart" id="chart-psv"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Quality per Tick</div>
        <canvas class="chart" id="chart-q"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Ranking Disagreement</div>
        <canvas class="chart" id="chart-rd"></canvas>
    </div>
    <div class="panel" id="sim-panel" style="display:none">
        <div class="panel-title">Similarity Curves</div>
        <canvas class="chart" id="chart-sim" style="height:80px"></canvas>
        <div class="note">solid=t15 dashed=t35 | green=mean blue=selected</div>
    </div>
    <div class="panel">
        <div class="panel-title">Distance Matrix</div>
        <canvas class="chart" id="chart-dm" style="height:150px"></canvas>
    </div>
    <div class="panel">
        <div class="panel-title">Legend</div>
        <div class="legend-row"><div class="legend-dot" style="background:#388bfd"></div> Warmup</div>
        <div class="legend-row"><div class="legend-dot" style="background:#7ee787"></div> Normal</div>
        <div class="legend-row"><div class="legend-dot" style="background:#f85149"></div> Perturbation</div>
        <div class="legend-row"><div class="legend-dot" style="background:#ffa657"></div> Post-pert</div>
        <div class="legend-row"><div class="legend-dot" style="background:#f0f6fc"></div> Cursor</div>
    </div>
</div>
<div id="status">Loading...</div>

<script src="/three.min.js"></script>
<script>
// ── State ──
let scene, camera, renderer;
let data = null, allMeshes = [], lineMesh = null, cursorMesh = null;
let connMesh = null, connVisible = false, connThreshold = 0.80;
let dm3Mesh = null; // dist-3d surface mesh
let currentView = 'selected', currentProj = 'pca', colorMode = 'temporal';
let trailOn = false;
let octantsOn = false, octantMeshes = [];
let rotAngle = 0;
let camTheta = Math.PI/4, camPhi = 0.5, camRadius = 2.5;
let isDragging = false, lastMX = 0, lastMY = 0, userControl = false;
let displayTick = -1;
let _lastRun = '';
let playing = false, playInterval = null, playSpeed = 1000;
let refreshInterval = null;

// ── Helpers ──
function lerpColor(c1, c2, t) {
    const r = ((c1>>16)&0xff) + t*(((c2>>16)&0xff)-((c1>>16)&0xff));
    const g = ((c1>>8)&0xff) + t*(((c2>>8)&0xff)-((c1>>8)&0xff));
    const b = (c1&0xff) + t*((c2&0xff)-(c1&0xff));
    return (Math.round(r)<<16)|(Math.round(g)<<8)|Math.round(b);
}

// ── Init ──
function init3D() {
    const c = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, c.clientWidth/c.clientHeight, 0.1, 100);
    camera.position.set(2, 1.5, 2);
    camera.lookAt(0, 0, 0);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(c.clientWidth, c.clientHeight);
    renderer.setClearColor(0x0a0e14);
    c.appendChild(renderer.domElement);
    // Axes + labels
    const axNames = ['PC1','PC2','PC3'];
    const axColors = [0x58a6ff, 0x7ee787, 0xffa657];
    for (let i = 0; i < 3; i++) {
        const g = new THREE.BufferGeometry();
        const p = [new THREE.Vector3(0,0,0)];
        const e = new THREE.Vector3(0,0,0); e.setComponent(i, 1.2); p.push(e);
        g.setFromPoints(p);
        const l = new THREE.Line(g, new THREE.LineBasicMaterial({color: axColors[i]}));
        l.userData.axis = true; scene.add(l);
        const cv = document.createElement('canvas'); cv.width=128; cv.height=64;
        const cx = cv.getContext('2d');
        cx.fillStyle = '#'+axColors[i].toString(16).padStart(6,'0');
        cx.font = 'bold 40px monospace'; cx.textAlign = 'center'; cx.fillText(axNames[i],64,44);
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),transparent:true}));
        sp.scale.set(0.3,0.15,1);
        const lp = new THREE.Vector3(0,0,0); lp.setComponent(i,1.35); sp.position.copy(lp);
        sp.userData.axis = true; scene.add(sp);
    }
    // Cursor
    cursorMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.06,16,16),
        new THREE.MeshBasicMaterial({color:0xf0f6fc})
    );
    cursorMesh.visible = false; cursorMesh.userData.cursor = true;
    scene.add(cursorMesh);

    window.addEventListener('resize', () => {
        camera.aspect = c.clientWidth/c.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(c.clientWidth, c.clientHeight);
    });

    // ── Contrôles souris ──
    c.addEventListener('mousedown', e => {
        isDragging = true; lastMX = e.clientX; lastMY = e.clientY;
        userControl = true;
    });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const dx = e.clientX - lastMX, dy = e.clientY - lastMY;
        lastMX = e.clientX; lastMY = e.clientY;
        camTheta -= dx * 0.005;
        camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, camPhi + dy * 0.005));
    });
    c.addEventListener('wheel', e => {
        e.preventDefault();
        camRadius = Math.max(0.5, Math.min(8, camRadius + e.deltaY * 0.002));
        userControl = true;
    }, { passive: false });
}

// ── Points access ──
function getPoints() {
    if (!data) return [];
    if (currentProj === 'dist3d') return data.points_mean; // fallback, not used in dist3d rendering
    if (currentProj === 'umap') {
        return currentView === 'mean' ? (data.umap_mean||data.points_mean) : (data.umap_selected||data.points_selected);
    }
    return currentView === 'mean' ? data.points_mean : data.points_selected;
}

// ── Color ──
function getTemporalColor(i) {
    const tick = i+1, warmup = data.warmup||5;
    const pertSet = new Set(data.perturbation_ticks||[]);
    if (pertSet.has(tick)) return 0xf85149;
    for (const pt of pertSet) { if (tick>pt && tick<=pt+5) return 0xffa657; }
    if (tick <= warmup) return 0x388bfd;
    const t = (i-warmup)/Math.max(1,data.ticks-warmup);
    return (Math.floor(30+t*96)<<16)|(Math.floor(180+t*51)<<8)|Math.floor(50+t*87);
}

function getMetricColor(i) {
    if (colorMode === 'temporal') return getTemporalColor(i);
    const vals = data[colorMode];
    if (!vals || i >= vals.length) return 0x333333;
    const v = vals[i];
    if (v === null || v === undefined || isNaN(v)) return 0x333333;
    // Normalize
    const valid = vals.filter(x => x!==null && x!==undefined && !isNaN(x));
    if (valid.length === 0) return 0x333333;
    const mn = Math.min(...valid), mx = Math.max(...valid);
    const range = mx-mn || 1;
    const t = (v-mn)/range;
    return lerpColor(0x0000ff, 0xff0000, t);
}

function applyColors() {
    if (!data) return;
    allMeshes.forEach((m, i) => {
        m.material.color.setHex(getMetricColor(i));
    });
}

// ── Build ──
function buildTrajectory() {
    const keep = scene.children.filter(c => c.userData.axis || c.userData.cursor || c === connMesh);
    scene.children = keep;
    allMeshes = []; lineMesh = null;
    if (!data || !data.ticks) return;

    const pts = getPoints(); const n = pts.length;
    // Line
    const pos = [];
    for (let i=0;i<n;i++) pos.push(pts[i][0],pts[i][1],pts[i][2]);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
    lineMesh = new THREE.Line(lg, new THREE.LineBasicMaterial({color:0x30363d,transparent:true,opacity:0.4}));
    scene.add(lineMesh);
    // Spheres
    for (let i=0;i<n;i++) {
        const pertSet = new Set(data.perturbation_ticks||[]);
        const sz = pertSet.has(i+1) ? 0.035 : 0.018;
        const g = new THREE.SphereGeometry(sz,8,8);
        const m = new THREE.MeshBasicMaterial({color:getMetricColor(i),transparent:true,opacity:1.0});
        const mesh = new THREE.Mesh(g,m);
        mesh.position.set(pts[i][0],pts[i][1],pts[i][2]);
        scene.add(mesh);
        allMeshes.push(mesh);
    }
    updateConnectome();
    updateVisibility();
}

// ── DIST-3D Surface ──
function clearDistSurface() {
    if (dm3Mesh) { scene.remove(dm3Mesh); dm3Mesh = null; }
    // Remove pert lines
    scene.children.filter(c => c.userData.dm3pert).forEach(c => scene.remove(c));
}

function buildDistSurface() {
    // Clear trajectory objects
    const keep = scene.children.filter(c => c.userData.axis || c.userData.cursor);
    scene.children = keep;
    allMeshes = []; lineMesh = null;
    if (dm3Mesh) { scene.remove(dm3Mesh); dm3Mesh = null; }
    scene.children.filter(c => c.userData.dm3pert).forEach(c => scene.remove(c));

    if (!data || !data.dist_matrix) return;
    const dm = data.dist_matrix;
    const n = dm.length;
    if (n < 2) return;

    const maxT = displayTick < 0 ? n : Math.min(displayTick, n);
    const scale = 2.0 / n; // fit in [-1,1] range
    const zScale = 1.5;

    // PlaneGeometry: width segments = n-1, height segments = n-1
    const geom = new THREE.PlaneGeometry(2, 2, n - 1, n - 1);
    const colors = new Float32Array(n * n * 3);
    const posAttr = geom.attributes.position;

    for (let iy = 0; iy < n; iy++) {
        for (let ix = 0; ix < n; ix++) {
            const idx = iy * n + ix;
            const v = (ix < maxT && iy < maxT && dm[iy][ix] !== null) ? dm[iy][ix] : 0;
            const revealed = (ix < maxT && iy < maxT);

            // Position: X = tick i, Y = tick j, Z = distance
            posAttr.setXYZ(idx,
                -1 + ix * scale + scale * 0.5,
                1 - iy * scale - scale * 0.5,
                revealed ? v * zScale : 0
            );

            // Color
            let r, g, b;
            if (!revealed || dm[iy][ix] === null) {
                r = 13/255; g = 17/255; b = 23/255;
            } else {
                // Cursor highlight: row/col = maxT-1
                if (ix === maxT - 1 || iy === maxT - 1) {
                    r = 1; g = 1; b = 1;
                } else {
                    const t = Math.max(0, Math.min(1, v));
                    r = (13 + t * (248 - 13)) / 255;
                    g = (17 + t * (81 - 17)) / 255;
                    b = (23 + t * (73 - 23)) / 255;
                }
            }
            colors[idx * 3] = r;
            colors[idx * 3 + 1] = g;
            colors[idx * 3 + 2] = b;
        }
    }

    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    posAttr.needsUpdate = true;
    geom.computeVertexNormals();

    const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    dm3Mesh = new THREE.Mesh(geom, mat);
    scene.add(dm3Mesh);

    // Perturbation lines on surface
    const pt = data.perturbation_ticks || [];
    for (const tick of pt) {
        const idx = tick - 1;
        if (idx < 0 || idx >= n) continue;
        const x = -1 + idx * scale + scale * 0.5;
        // Vertical line (column)
        const vpts = [];
        for (let iy = 0; iy < n; iy++) {
            const v = (idx < maxT && iy < maxT && dm[iy][idx] !== null) ? dm[iy][idx] : 0;
            vpts.push(x, 1 - iy * scale - scale * 0.5, v * zScale + 0.01);
        }
        const vg = new THREE.BufferGeometry();
        vg.setAttribute('position', new THREE.Float32BufferAttribute(vpts, 3));
        const vl = new THREE.Line(vg, new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.5}));
        vl.userData.dm3pert = true;
        scene.add(vl);
        // Horizontal line (row)
        const hpts = [];
        const y = 1 - idx * scale - scale * 0.5;
        for (let ix = 0; ix < n; ix++) {
            const v = (ix < maxT && idx < maxT && dm[idx][ix] !== null) ? dm[idx][ix] : 0;
            hpts.push(-1 + ix * scale + scale * 0.5, y, v * zScale + 0.01);
        }
        const hg = new THREE.BufferGeometry();
        hg.setAttribute('position', new THREE.Float32BufferAttribute(hpts, 3));
        const hl = new THREE.Line(hg, new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.5}));
        hl.userData.dm3pert = true;
        scene.add(hl);
    }
}

function updateDistSurface() {
    if (currentProj !== 'dist3d' || !data || !data.dist_matrix || !dm3Mesh) return;
    const dm = data.dist_matrix;
    const n = dm.length;
    const maxT = displayTick < 0 ? n : Math.min(displayTick, n);
    const scale = 2.0 / n;
    const zScale = 1.5;

    const posAttr = dm3Mesh.geometry.attributes.position;
    const colAttr = dm3Mesh.geometry.attributes.color;

    for (let iy = 0; iy < n; iy++) {
        for (let ix = 0; ix < n; ix++) {
            const idx = iy * n + ix;
            const v = (ix < maxT && iy < maxT && dm[iy][ix] !== null) ? dm[iy][ix] : 0;
            const revealed = (ix < maxT && iy < maxT);

            posAttr.setZ(idx, revealed ? v * zScale : 0);

            let r, g, b;
            if (!revealed || dm[iy][ix] === null) {
                r = 13/255; g = 17/255; b = 23/255;
            } else if (ix === maxT - 1 || iy === maxT - 1) {
                r = 1; g = 1; b = 1;
            } else {
                const t = Math.max(0, Math.min(1, v));
                r = (13 + t * (248 - 13)) / 255;
                g = (17 + t * (81 - 17)) / 255;
                b = (23 + t * (73 - 23)) / 255;
            }
            colAttr.setXYZ(idx, r, g, b);
        }
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
}

// ── Connectome ──
function updateConnectome() {
    if (connMesh) { scene.remove(connMesh); connMesh = null; }
    if (!data || !data.connectome || !connVisible) return;

    const pts = getPoints();
    const edges = data.connectome.filter(e => e.sim >= connThreshold);
    const positions = []; const colors = [];

    edges.forEach(e => {
        if (e.i >= pts.length || e.j >= pts.length) return;
        const p1 = pts[e.i], p2 = pts[e.j];
        positions.push(p1[0],p1[1],p1[2], p2[0],p2[1],p2[2]);
        // Force encodée dans la luminosité : faible=bleu sombre, fort=cyan→blanc/or
        const t = Math.max(0, Math.min(1, (e.sim - connThreshold) / (1.0 - connThreshold)));
        let c;
        if (t < 0.5) {
            c = lerpColor(0x0a1a2e, 0x00bcd4, t * 2);       // sombre → cyan
        } else {
            c = lerpColor(0x00bcd4, 0xfff176, (t - 0.5) * 2); // cyan → or clair
        }
        const r=((c>>16)&0xff)/255, g=((c>>8)&0xff)/255, b=(c&0xff)/255;
        colors.push(r,g,b, r,g,b);
    });

    if (positions.length === 0) { document.getElementById('conn-count').textContent = '0 edges'; return; }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors,3));
    // Blending additif : les connexions denses/fortes s'accumulent et "brillent"
    const mat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 1.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    connMesh = new THREE.LineSegments(geo, mat);
    connMesh.renderOrder = -1;
    scene.add(connMesh);
    document.getElementById('conn-count').textContent = edges.length + ' edges';
}

function onConnThreshold(val) {
    connThreshold = parseInt(val) / 100;
    document.getElementById('conn-label').textContent = 'Threshold: ' + connThreshold.toFixed(2);
    updateConnectome();
}

function toggleConn() {
    connVisible = !connVisible;
    document.getElementById('btn-conn').classList.toggle('on', connVisible);
    document.getElementById('conn-panel').style.display = connVisible ? 'block' : 'none';
    updateConnectome();
}

// ── Trail ──
function toggleTrail() {
    trailOn = !trailOn;
    document.getElementById('btn-trail').classList.toggle('on', trailOn);
    updateVisibility();
}

// ── Octants ──
// 8 demi-espaces (signe de x,y,z) chacun d'une couleur distincte, très transparent
const OCTANT_COLORS = [
    0x4fc3f7, // +++ cyan clair
    0xf48fb1, // ++- rose
    0xa5d6a7, // +-+ vert menthe
    0xffe082, // +-- ambre
    0xce93d8, // -++ violet
    0xef9a9a, // -+- rouge pastel
    0x80deea, // --+ turquoise
    0xffcc80, // --- orange pastel
];

function buildOctants() {
    octantMeshes.forEach(m => scene.remove(m));
    octantMeshes = [];
    const signs = [1, -1];
    let idx = 0;
    for (const sx of signs) for (const sy of signs) for (const sz of signs) {
        const geo = new THREE.BoxGeometry(1.0, 1.0, 1.0);
        const mat = new THREE.MeshBasicMaterial({
            color: OCTANT_COLORS[idx],
            transparent: true,
            opacity: 0.07,
            depthWrite: false,
            side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(sx * 0.5, sy * 0.5, sz * 0.5);
        mesh.renderOrder = -2;
        octantMeshes.push(mesh);
        idx++;
    }
}

function toggleOctants() {
    octantsOn = !octantsOn;
    document.getElementById('btn-octants').classList.toggle('on', octantsOn);
    if (octantsOn) {
        if (octantMeshes.length === 0) buildOctants();
        octantMeshes.forEach(m => scene.add(m));
    } else {
        octantMeshes.forEach(m => scene.remove(m));
    }
}

// ── Visibility ──
function updateVisibility() {
    if (!data) return;
    const pts = getPoints();
    const maxTick = displayTick < 0 ? data.ticks : displayTick;
    const trailWindow = 12;

    allMeshes.forEach((m, i) => {
        const tick = i + 1;
        if (tick > maxTick) { m.visible = false; m.material.opacity = 1.0; return; }
        if (trailOn) {
            const start = Math.max(1, maxTick - trailWindow + 1);
            if (tick < start) { m.visible = false; return; }
            m.visible = true;
            m.material.opacity = (tick - start + 1) / trailWindow;
        } else {
            m.visible = true;
            m.material.opacity = 1.0;
        }
    });

    // Ligne : complète en mode normal, fenêtre glissante en mode trail
    if (lineMesh) {
        if (trailOn && maxTick > 0) {
            const start = Math.max(0, maxTick - trailWindow);
            const end = Math.min(pts.length, maxTick);
            const pos = [];
            for (let i = start; i < end; i++) pos.push(pts[i][0], pts[i][1], pts[i][2]);
            lineMesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        } else {
            const pos = [];
            for (let i = 0; i < Math.min(maxTick, pts.length); i++) pos.push(pts[i][0],pts[i][1],pts[i][2]);
            lineMesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        }
        lineMesh.geometry.attributes.position.needsUpdate = true;
    }

    // Cursor
    if (maxTick > 0 && maxTick <= pts.length) {
        const p = pts[maxTick-1];
        cursorMesh.position.set(p[0],p[1],p[2]);
        cursorMesh.visible = true;
    } else { cursorMesh.visible = false; }

    // Tick info
    const tick = maxTick;
    const info = [];
    if (tick > 0 && tick <= data.ticks) {
        const get = (k) => { const v = data[k]; return v && v[tick-1]; };
        const ccv=get('claim_cosine_variance'), dv=get('draft_velocity');
        const psv=get('post_selection_variance'), q=get('quality_per_tick');
        if (ccv!=null) info.push('CCV='+ccv.toFixed(3));
        if (dv!=null&&!isNaN(dv)) info.push('Vel='+dv.toFixed(3));
        if (psv!=null) info.push('PSV='+psv.toFixed(3));
        if (q!=null) info.push('Q='+q.toFixed(2));
        const pt = data.perturbation_types||{};
        if (pt[String(tick)]) info.push('PERT='+pt[String(tick)]);
    }
    document.getElementById('tick-info').textContent = info.join('  ');

    // Update dist-3d surface if active
    updateDistSurface();
}

// ── Charts ──
function drawChart(cid, values, pertTicks, color, ht) {
    const cv = document.getElementById(cid); if (!cv) return;
    const ctx = cv.getContext('2d');
    const w = cv.width = cv.clientWidth*2, h = cv.height = cv.clientHeight*2;
    ctx.clearRect(0,0,w,h);
    const vals = values.filter(v => v!==null && !isNaN(v));
    if (!vals.length) return;
    const mn=Math.min(...vals), mx=Math.max(...vals), rng=mx-mn||1;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    let started=false;
    for (let i=0;i<values.length;i++) {
        const v=values[i]; if(v===null||isNaN(v)) continue;
        const x=(i/Math.max(1,values.length-1))*w;
        const y=h-((v-mn)/rng)*(h-8)-4;
        if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
    }
    ctx.stroke();
    if(pertTicks){ctx.fillStyle='rgba(248,81,73,0.3)';
        for(const pt of pertTicks){const idx=pt-1;if(idx>=0&&idx<values.length){
            const x=(idx/Math.max(1,values.length-1))*w;ctx.fillRect(x-2,0,4,h);}}
    }
    if(ht>0&&ht<=values.length){const idx=ht-1;const v=values[idx];
        if(v!==null&&!isNaN(v)){const x=(idx/Math.max(1,values.length-1))*w;
        const y=h-((v-mn)/rng)*(h-8)-4;
        ctx.fillStyle='#f0f6fc';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();}}
}

function drawSimCurves() {
    const panel=document.getElementById('sim-panel');
    if(!data||!data.sim_curves){panel.style.display='none';return;}
    panel.style.display='block';
    const cv=document.getElementById('chart-sim');
    const ctx=cv.getContext('2d');
    const w=cv.width=cv.clientWidth*2, h=cv.height=cv.clientHeight*2;
    ctx.clearRect(0,0,w,h);
    const colors={mean:'#7ee787',selected:'#388bfd'};
    let allV=[];
    for(const cd of Object.values(data.sim_curves))
        for(const vals of Object.values(cd)) vals.forEach(v=>{if(v!==null)allV.push(v);});
    const mn=allV.length?Math.min(...allV):0, mx=allV.length?Math.max(...allV):1, rng=mx-mn||1;
    for(const[tk,cd]of Object.entries(data.sim_curves)){
        for(const[type,vals]of Object.entries(cd)){
            ctx.strokeStyle=colors[type]||'#888';ctx.lineWidth=2;
            ctx.setLineDash(tk.includes('35')?[8,5]:[]);ctx.beginPath();let s=false;
            for(let k=0;k<vals.length;k++){if(vals[k]===null)continue;
                const x=(k/Math.max(1,vals.length-1))*w;
                const y=h-((vals[k]-mn)/rng)*(h-8)-4;
                if(!s){ctx.moveTo(x,y);s=true;}else ctx.lineTo(x,y);}
            ctx.stroke();}}
    ctx.setLineDash([]);
}

function drawDistMatrix() {
    const cv = document.getElementById('chart-dm');
    if (!cv || !data || !data.dist_matrix) return;
    const ctx = cv.getContext('2d');
    const w = cv.width = cv.clientWidth * 2, h = cv.height = cv.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const dm = data.dist_matrix, n = dm.length;
    if (n === 0) return;
    const maxT = displayTick < 0 ? n : Math.min(displayTick, n);
    const cw = w / n, ch = h / n;
    // Background: dark for unrevealed cells
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < maxT; i++) {
        for (let j = 0; j < maxT; j++) {
            const v = dm[i][j];
            if (v === null) { ctx.fillStyle = '#21262d'; }
            else {
                const t = Math.max(0, Math.min(1, v));
                const r = Math.round(13 + t * (248 - 13));
                const g = Math.round(17 + t * (81 - 17));
                const b = Math.round(23 + t * (73 - 23));
                ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
            }
            ctx.fillRect(j * cw, i * ch, Math.ceil(cw), Math.ceil(ch));
        }
    }
    // Perturbation lines
    const pt = data.perturbation_ticks || [];
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
    for (const tick of pt) {
        const idx = tick - 1;
        if (idx >= 0 && idx < maxT) {
            ctx.beginPath(); ctx.moveTo(idx * cw, 0); ctx.lineTo(idx * cw, maxT * ch); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, idx * ch); ctx.lineTo(maxT * cw, idx * ch); ctx.stroke();
        }
    }
}

function drawAllCharts() {
    if (!data) return;
    const pt = data.perturbation_ticks;
    const ht = displayTick < 0 ? data.ticks : displayTick;
    drawChart('chart-ccv', data.claim_cosine_variance, pt, '#7ee787', ht);
    drawChart('chart-dv', data.draft_velocity, pt, '#388bfd', ht);
    drawChart('chart-psv', data.post_selection_variance, pt, '#ffa657', ht);
    drawChart('chart-q', data.quality_per_tick, pt, '#f0f6fc', ht);
    drawChart('chart-rd', data.judge_score_dispersion, pt, '#d2a8ff', ht);
    drawSimCurves();
    drawDistMatrix();
}

// ── View/Proj/Color toggles ──
function setView(v) {
    currentView = v;
    document.querySelectorAll('.view-toggle').forEach(b => {
        if (b.textContent === 'Mean' || b.textContent === 'Selected')
            b.classList.toggle('active', b.textContent.toLowerCase() === v);
    });
    buildTrajectory(); updateProjLabel(); drawAllCharts();
}

function setProj(p) {
    if (p === 'umap' && (!data || !data.umap_mean)) return;
    if (p === 'dist3d' && (!data || !data.dist_matrix)) return;
    currentProj = p;
    document.getElementById('btn-pca').classList.toggle('active', p==='pca');
    document.getElementById('btn-umap').classList.toggle('active', p==='umap');
    document.getElementById('btn-dist3d').classList.toggle('active', p==='dist3d');
    if (p === 'dist3d') {
        buildDistSurface();
    } else {
        clearDistSurface();
        buildTrajectory();
    }
    updateProjLabel(); drawAllCharts();
    document.getElementById('conn-note').textContent =
        'Edges: cosine sim in embedding space | positions: ' + p.toUpperCase();
}

function updateProjLabel() {
    if (!data) return;
    if (currentProj === 'dist3d') {
        document.getElementById('s-proj').textContent = 'DIST-3D (cosine)';
    } else if (currentProj === 'umap') {
        document.getElementById('s-proj').textContent =
            'UMAP (cosine, k=' + (data.n_neighbors_used||'?') + ')';
    } else {
        const exp = currentView === 'mean' ? data.explained_mean : data.explained_selected;
        document.getElementById('s-proj').textContent = exp ?
            'PCA ' + exp.map(v=>(v*100).toFixed(1)+'%').join('/') : 'PCA';
    }
}

function setColorMode(mode) {
    colorMode = mode;
    applyColors();
}

// ── Playback ──
function setDisplayTick(t) {
    if (!data) return;
    displayTick = Math.max(1, Math.min(data.ticks, t));
    document.getElementById('tl-slider').value = displayTick;
    document.getElementById('tl-label').textContent = 'Tick ' + displayTick;
    updateVisibility(); drawAllCharts();
}
function onSlider(val) { stopPlay(); setDisplayTick(parseInt(val)); }
function stepForward() { if(!data)return; setDisplayTick((displayTick<0?data.ticks:displayTick)+1); }
function stepBack() { if(!data)return; setDisplayTick((displayTick<0?1:displayTick)-1); }
function togglePlay() {
    if (playing) { stopPlay(); return; }
    playing = true; document.getElementById('btn-play').textContent = '⏸';
    if (displayTick < 0 || displayTick >= data.ticks) setDisplayTick(1);
    playInterval = setInterval(() => {
        if (!data || displayTick >= data.ticks) { stopPlay(); return; }
        setDisplayTick(displayTick + 1);
    }, playSpeed);
}
function stopPlay() {
    playing = false; document.getElementById('btn-play').textContent = '▶';
    if (playInterval) { clearInterval(playInterval); playInterval = null; }
}
function setSpeed(ms) { playSpeed = parseInt(ms); if (playing) { stopPlay(); togglePlay(); } }

// ── Run selector ──
async function loadRunList() {
    const r = await fetch('/api/runs'); const d = await r.json();
    const sel = document.getElementById('run-select');
    sel.innerHTML = '<option value="">— run —</option>';
    for (const name of d.runs) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name; sel.appendChild(opt);
    }
    if (d.runs.length > 0) { sel.value = d.runs[0]; onRunChange(); }
}

async function onRunChange() {
    const run = document.getElementById('run-select').value;
    if (!run) return; stopPlay();
    await fetchData(run);
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => fetchData(run), 5000);
}

async function fetchData(run) {
    try {
        const r = await fetch('/api/data?run=' + encodeURIComponent(run));
        data = await r.json();
        if (data.error) { document.getElementById('status').textContent = data.error; return; }
        document.getElementById('s-cond').textContent = data.condition;
        document.getElementById('s-seed').textContent = data.seed;
        document.getElementById('s-ticks').textContent = data.ticks;
        document.getElementById('tl-slider').max = data.ticks;
        document.getElementById('tl-max').textContent = '/ ' + data.ticks;
        // UMAP button
        const ub = document.getElementById('btn-umap');
        if (!data.umap_mean) { ub.disabled = true; ub.title = '(unavailable)';
            if (currentProj === 'umap') setProj('pca');
        } else { ub.disabled = false; ub.title = ''; }
        // DIST-3D button
        const db = document.getElementById('btn-dist3d');
        if (!data.dist_matrix) { db.disabled = true; db.title = '(unavailable)';
            if (currentProj === 'dist3d') setProj('pca');
        } else { db.disabled = false; db.title = ''; }
        updateProjLabel();
        // Mean-Selected gap indicator
        const gapEl = document.getElementById('ms-gap');
        const gapC = document.getElementById('ms-gap-container');
        if (data.ms_gap !== null && data.ms_gap !== undefined) {
            gapC.style.display = '';
            gapEl.textContent = data.ms_gap.toFixed(3);
            if (data.ms_gap < 0.05) gapEl.style.color = '#7ee787';
            else if (data.ms_gap <= 0.15) gapEl.style.color = '#d29922';
            else gapEl.style.color = '#f85149';
        } else { gapC.style.display = 'none'; }
        if (currentProj === 'dist3d') { buildDistSurface(); } else { buildTrajectory(); }
        // On run change or first load: show all ticks
        // On refresh of same run: keep current position unless ticks grew
        if (displayTick < 0 || displayTick > data.ticks || run !== _lastRun) {
            displayTick = data.ticks;
            document.getElementById('tl-slider').value = displayTick;
            document.getElementById('tl-label').textContent = 'Tick ' + displayTick;
        }
        _lastRun = run;
        updateVisibility(); drawAllCharts();
        document.getElementById('status').textContent = 'Updated: ' + new Date().toLocaleTimeString();
    } catch (e) { document.getElementById('status').textContent = 'Error: ' + e.message; }
}

// ── Animation ──
function animate() {
    requestAnimationFrame(animate);
    if (!userControl) {
        rotAngle += 0.003;
        camTheta = rotAngle;
    }
    camera.position.x = camRadius * Math.sin(camPhi) * Math.cos(camTheta);
    camera.position.y = camRadius * Math.cos(camPhi);
    camera.position.z = camRadius * Math.sin(camPhi) * Math.sin(camTheta);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
}

// ── Keyboard ──
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') stepForward();
    else if (e.key === 'ArrowLeft') stepBack();
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'c') toggleConn();
    else if (e.key === 't') toggleTrail();
});

// ── Boot ──
init3D();
animate();
loadRunList();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="V3 Semantic Trajectory Viewer")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory containing V2 run folders (default: runs/bench_v2/)")
    parser.add_argument("--run", type=str, default=None,
                        help="Single run directory (legacy compat)")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    global RUNS_DIR
    if args.run:
        RUNS_DIR = Path(args.run).parent
    elif args.runs_dir:
        RUNS_DIR = Path(args.runs_dir)
    else:
        RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "bench_v2"

    if not RUNS_DIR.exists():
        print(f"[ERROR] Directory not found: {RUNS_DIR}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    print(f"\n  V3 Viewer: http://localhost:{args.port}")
    print(f"  Runs dir: {RUNS_DIR}")
    print(f"  UMAP: {'available' if _HAS_UMAP else 'unavailable'}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
