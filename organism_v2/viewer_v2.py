#!/usr/bin/env python3
"""
viewer_v2.py — V2 Semantic Trajectory Viewer
==============================================
3D PCA visualization of 768-dim embedding trajectories from bench_v2.

Features:
  - Run selector dropdown (scans all runs in --runs-dir)
  - Playback: play/pause, prev/next tick, speed slider, timeline scrubber
  - 3D rotating trajectory (Mean / Selected toggle)
  - Perturbation ticks highlighted
  - Sidebar charts: cosine variance, draft velocity, post-selection var, quality
  - Similarity curves panel (conditions B/C/E)
  - Auto-refresh for live runs

Usage:
    python organism_v2/viewer_v2.py --runs-dir runs/bench_v2/
    # Open http://localhost:8766
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, Response, send_from_directory, request

log = logging.getLogger("viewer_v2")
app = Flask(__name__)

RUNS_DIR: Path = Path(".")
_V2_DIR = Path(__file__).resolve().parent


@app.route("/three.min.js")
def serve_threejs():
    return send_from_directory(str(_V2_DIR), "three.min.js", mimetype="application/javascript")


@app.route("/api/runs")
def api_runs():
    """List all available V2 runs."""
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


def _pca_3d(vectors):
    arr = np.array(vectors)
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

    if n < 3:
        return jsonify({"error": f"Only {n} ticks, need >= 3", "ticks": n})

    pts_mean, explained_mean = _pca_3d(sv_mean)
    pts_selected, explained_selected = _pca_3d(sv_selected)

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
        "claim_cosine_variance": results.get("claim_cosine_variance", []),
        "ranking_disagreement": results.get("ranking_disagreement", []),
        "draft_velocity": results.get("draft_velocity", []),
        "post_selection_variance": results.get("post_selection_variance", []),
        "quality_per_tick": results.get("quality_per_tick", []),
        "perturbation_ticks": pert_ticks,
        "perturbation_types": pert_types,
        "sim_curves": results.get("sim_curves"),
    })


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Organism V2 — Semantic Trajectory</title>
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
    padding: 8px 16px; display: flex; align-items: center; gap: 16px;
    flex-wrap: wrap;
}
#header h1 { font-size: 14px; font-weight: 600; color: #7ee787; white-space: nowrap; }
.stat { font-size: 11px; white-space: nowrap; }
.stat .label { color: #6e7681; }
.stat .value { color: #f0f6fc; font-weight: 600; }
select, .btn, .view-toggle {
    background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
    padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;
    font-family: inherit;
}
select { min-width: 140px; }
.btn:hover { background: #30363d; }
.view-toggle.active { background: #388bfd; border-color: #388bfd; color: #fff; }
#transport { display: flex; align-items: center; gap: 6px; }
#transport .btn { font-size: 13px; padding: 2px 8px; }
#timeline {
    position: fixed; bottom: 24px; left: 0; right: 320px; z-index: 10;
    padding: 0 16px; display: flex; align-items: center; gap: 10px;
}
#timeline input[type=range] { flex: 1; accent-color: #7ee787; }
#timeline .tick-label { font-size: 11px; color: #8b949e; min-width: 70px; }
#three-container { position: fixed; top: 44px; left: 0; right: 320px; bottom: 48px; }
#sidebar {
    position: fixed; top: 44px; right: 0; bottom: 0; width: 320px;
    background: rgba(10,14,20,0.95); border-left: 1px solid #1e2530;
    padding: 10px; overflow-y: auto; font-size: 11px;
}
.panel { margin-bottom: 12px; }
.panel-title { color: #7ee787; font-size: 11px; font-weight: 600; margin-bottom: 4px; }
canvas.chart {
    width: 100%; height: 60px; background: #161b22;
    border-radius: 4px; margin-bottom: 2px; display: block;
}
.legend-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: 10px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
#status { position: fixed; bottom: 4px; left: 8px; font-size: 10px; color: #484f58; z-index: 10; }
#tick-info {
    position: fixed; bottom: 48px; left: 16px; z-index: 10;
    font-size: 11px; color: #8b949e;
}
</style>
</head>
<body>
<div id="header">
    <h1>V2 Trajectory</h1>
    <select id="run-select" onchange="onRunChange()"><option value="">— select run —</option></select>
    <span class="stat"><span class="label">Cond </span><span class="value" id="s-cond">—</span></span>
    <span class="stat"><span class="label">Seed </span><span class="value" id="s-seed">—</span></span>
    <span class="stat"><span class="label">Ticks </span><span class="value" id="s-ticks">—</span></span>
    <span class="stat"><span class="label">PCA </span><span class="value" id="s-pca">—</span></span>
    <button class="view-toggle active" onclick="setView('mean')">Mean</button>
    <button class="view-toggle" onclick="setView('selected')">Selected</button>
    <div id="transport">
        <button class="btn" onclick="stepBack()" title="Previous tick">◀</button>
        <button class="btn" id="btn-play" onclick="togglePlay()" title="Play/Pause">▶</button>
        <button class="btn" onclick="stepForward()" title="Next tick">▶▶</button>
        <select id="speed-select" onchange="setSpeed(this.value)" title="Playback speed">
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
        <div class="panel-title">Similarity Curves (post-perturbation)</div>
        <canvas class="chart" id="chart-sim" style="height:100px"></canvas>
        <div style="font-size:10px;color:#6e7681">solid=tick15 dashed=tick35 | green=mean blue=selected</div>
    </div>
    <div class="panel">
        <div class="panel-title">Legend</div>
        <div class="legend-row"><div class="legend-dot" style="background:#388bfd"></div> Warmup (1-5)</div>
        <div class="legend-row"><div class="legend-dot" style="background:#7ee787"></div> Normal</div>
        <div class="legend-row"><div class="legend-dot" style="background:#f85149"></div> Perturbation</div>
        <div class="legend-row"><div class="legend-dot" style="background:#ffa657"></div> Post-pert (k≤5)</div>
        <div class="legend-row"><div class="legend-dot" style="background:#f0f6fc"></div> Current tick</div>
    </div>
</div>
<div id="status">Loading runs...</div>

<script src="/three.min.js"></script>
<script>
// ── State ──
let scene, camera, renderer;
let data = null, allMeshes = [], lineMesh = null, cursorMesh = null;
let currentView = 'mean';
let rotAngle = 0;
let displayTick = -1; // -1 = show all
let playing = false, playInterval = null, playSpeed = 1000;
let refreshInterval = null;

// ── Init ──
function init3D() {
    const c = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, c.clientWidth / c.clientHeight, 0.1, 100);
    camera.position.set(2, 1.5, 2);
    camera.lookAt(0, 0, 0);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(c.clientWidth, c.clientHeight);
    renderer.setClearColor(0x0a0e14);
    c.appendChild(renderer.domElement);
    // Axes + labels
    const axMat = new THREE.LineBasicMaterial({ color: 0x30363d });
    const axNames = ['PC1', 'PC2', 'PC3'];
    const axColors = [0x58a6ff, 0x7ee787, 0xffa657];
    for (let i = 0; i < 3; i++) {
        const g = new THREE.BufferGeometry();
        const p = [new THREE.Vector3(0,0,0)];
        const e = new THREE.Vector3(0,0,0); e.setComponent(i, 1.2); p.push(e);
        g.setFromPoints(p);
        const lm = new THREE.LineBasicMaterial({ color: axColors[i] });
        const l = new THREE.Line(g, lm); l.userData.axis = true;
        scene.add(l);
        // Label sprite
        const canvas2 = document.createElement('canvas');
        canvas2.width = 128; canvas2.height = 64;
        const ctx2 = canvas2.getContext('2d');
        ctx2.fillStyle = '#' + axColors[i].toString(16).padStart(6, '0');
        ctx2.font = 'bold 40px monospace';
        ctx2.textAlign = 'center';
        ctx2.fillText(axNames[i], 64, 44);
        const tex = new THREE.CanvasTexture(canvas2);
        const spMat = new THREE.SpriteMaterial({ map: tex, transparent: true });
        const sp = new THREE.Sprite(spMat);
        sp.scale.set(0.3, 0.15, 1);
        const lp = new THREE.Vector3(0,0,0); lp.setComponent(i, 1.35);
        sp.position.copy(lp);
        sp.userData.axis = true;
        scene.add(sp);
    }
    // Cursor sphere
    const cg = new THREE.SphereGeometry(0.06, 16, 16);
    const cm = new THREE.MeshBasicMaterial({ color: 0xf0f6fc });
    cursorMesh = new THREE.Mesh(cg, cm);
    cursorMesh.visible = false;
    scene.add(cursorMesh);

    window.addEventListener('resize', () => {
        camera.aspect = c.clientWidth / c.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(c.clientWidth, c.clientHeight);
    });
}

function getPoints() {
    if (!data) return [];
    return currentView === 'mean' ? data.points_mean : data.points_selected;
}

function getColor(i) {
    const tick = i + 1;
    const warmup = data.warmup || 5;
    const pertSet = new Set(data.perturbation_ticks || []);
    if (pertSet.has(tick)) return 0xf85149;
    for (const pt of pertSet) { if (tick > pt && tick <= pt + 5) return 0xffa657; }
    if (tick <= warmup) return 0x388bfd;
    const t = (i - warmup) / Math.max(1, data.ticks - warmup);
    return (Math.floor(30+t*96) << 16) | (Math.floor(180+t*51) << 8) | Math.floor(50+t*87);
}

function buildTrajectory() {
    // Clear non-axis, non-cursor
    const keep = scene.children.filter(c => c.userData.axis || c === cursorMesh);
    scene.children = keep;
    allMeshes = [];
    if (!data || !data.ticks) return;

    const pts = getPoints();
    const n = pts.length;

    // Line
    const positions = [];
    for (let i = 0; i < n; i++) positions.push(pts[i][0], pts[i][1], pts[i][2]);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    lineMesh = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.4 }));
    scene.add(lineMesh);

    // Spheres
    for (let i = 0; i < n; i++) {
        const pertSet = new Set(data.perturbation_ticks || []);
        const sz = pertSet.has(i+1) ? 0.035 : 0.018;
        const g = new THREE.SphereGeometry(sz, 8, 8);
        const m = new THREE.MeshBasicMaterial({ color: getColor(i) });
        const mesh = new THREE.Mesh(g, m);
        mesh.position.set(pts[i][0], pts[i][1], pts[i][2]);
        scene.add(mesh);
        allMeshes.push(mesh);
    }
    updateVisibility();
}

function updateVisibility() {
    if (!data) return;
    const pts = getPoints();
    const maxTick = displayTick < 0 ? data.ticks : displayTick;
    allMeshes.forEach((m, i) => { m.visible = (i < maxTick); });
    if (lineMesh) lineMesh.visible = true;
    // Cursor
    if (maxTick > 0 && maxTick <= pts.length) {
        const p = pts[maxTick - 1];
        cursorMesh.position.set(p[0], p[1], p[2]);
        cursorMesh.visible = true;
    } else {
        cursorMesh.visible = false;
    }
    // Tick info
    const tick = maxTick;
    const info = [];
    if (tick > 0 && tick <= data.ticks) {
        const ccv = data.claim_cosine_variance[tick-1];
        const dv = data.draft_velocity[tick-1];
        const psv = data.post_selection_variance[tick-1];
        const q = data.quality_per_tick[tick-1];
        if (ccv != null) info.push('CCV=' + ccv.toFixed(3));
        if (dv != null && !isNaN(dv)) info.push('Vel=' + dv.toFixed(3));
        if (psv != null) info.push('PSV=' + psv.toFixed(3));
        if (q != null) info.push('Q=' + q.toFixed(3));
        const pt = data.perturbation_types || {};
        if (pt[String(tick)]) info.push('PERT=' + pt[String(tick)]);
    }
    document.getElementById('tick-info').textContent = info.join('  ');
}

// ── Charts ──
function drawChart(canvasId, values, pertTicks, color, highlightTick) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * 2;
    const h = canvas.height = canvas.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const vals = values.filter(v => v !== null && !isNaN(v));
    if (vals.length === 0) return;
    const mn = Math.min(...vals); const mx = Math.max(...vals);
    const range = mx - mn || 1;
    // Line
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    for (let i = 0; i < values.length; i++) {
        const v = values[i]; if (v === null || isNaN(v)) continue;
        const x = (i / Math.max(1, values.length - 1)) * w;
        const y = h - ((v - mn) / range) * (h - 10) - 5;
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Pert markers
    if (pertTicks) {
        ctx.fillStyle = 'rgba(248,81,73,0.3)';
        for (const pt of pertTicks) {
            const idx = pt - 1;
            if (idx >= 0 && idx < values.length) {
                const x = (idx / Math.max(1, values.length - 1)) * w;
                ctx.fillRect(x - 3, 0, 6, h);
            }
        }
    }
    // Highlight current tick
    if (highlightTick > 0 && highlightTick <= values.length) {
        const idx = highlightTick - 1;
        const v = values[idx];
        if (v !== null && !isNaN(v)) {
            const x = (idx / Math.max(1, values.length - 1)) * w;
            const y = h - ((v - mn) / range) * (h - 10) - 5;
            ctx.fillStyle = '#f0f6fc';
            ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
        }
    }
}

function drawSimCurves() {
    const panel = document.getElementById('sim-panel');
    if (!data || !data.sim_curves) { panel.style.display = 'none'; return; }
    panel.style.display = 'block';
    const canvas = document.getElementById('chart-sim');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * 2;
    const h = canvas.height = canvas.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const colors = { mean: '#7ee787', selected: '#388bfd' };
    // Collect all values for global scale
    let allVals = [];
    for (const cd of Object.values(data.sim_curves)) {
        for (const vals of Object.values(cd)) {
            vals.forEach(v => { if (v !== null) allVals.push(v); });
        }
    }
    const mn = allVals.length ? Math.min(...allVals) : 0;
    const mx = allVals.length ? Math.max(...allVals) : 1;
    const range = mx - mn || 1;

    for (const [tickKey, curveData] of Object.entries(data.sim_curves)) {
        for (const [type, vals] of Object.entries(curveData)) {
            ctx.strokeStyle = colors[type] || '#888';
            ctx.lineWidth = 2;
            ctx.setLineDash(tickKey.includes('35') ? [8, 5] : []);
            ctx.beginPath();
            let started = false;
            for (let k = 0; k < vals.length; k++) {
                if (vals[k] === null) continue;
                const x = (k / Math.max(1, vals.length - 1)) * w;
                const y = h - ((vals[k] - mn) / range) * (h - 10) - 5;
                if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    }
    ctx.setLineDash([]);
}

function drawAllCharts() {
    if (!data) return;
    const pt = data.perturbation_ticks;
    const ht = displayTick < 0 ? data.ticks : displayTick;
    drawChart('chart-ccv', data.claim_cosine_variance, pt, '#7ee787', ht);
    drawChart('chart-dv', data.draft_velocity, pt, '#388bfd', ht);
    drawChart('chart-psv', data.post_selection_variance, pt, '#ffa657', ht);
    drawChart('chart-q', data.quality_per_tick, pt, '#f0f6fc', ht);
    drawChart('chart-rd', data.ranking_disagreement, pt, '#d2a8ff', ht);
    drawSimCurves();
}

// ── View toggle ──
function setView(v) {
    currentView = v;
    document.querySelectorAll('.view-toggle').forEach(b => {
        b.classList.toggle('active', b.textContent.toLowerCase() === v);
    });
    buildTrajectory();
    updatePCA();
    drawAllCharts();
}

function updatePCA() {
    if (!data) return;
    const exp = currentView === 'mean' ? data.explained_mean : data.explained_selected;
    if (exp) {
        document.getElementById('s-pca').textContent =
            exp.map(v => (v * 100).toFixed(1) + '%').join(' / ');
    }
}

// ── Playback ──
function setDisplayTick(t) {
    if (!data) return;
    displayTick = Math.max(1, Math.min(data.ticks, t));
    document.getElementById('tl-slider').value = displayTick;
    document.getElementById('tl-label').textContent = 'Tick ' + displayTick;
    updateVisibility();
    drawAllCharts();
}

function onSlider(val) {
    stopPlay();
    setDisplayTick(parseInt(val));
}

function stepForward() {
    if (!data) return;
    const cur = displayTick < 0 ? data.ticks : displayTick;
    setDisplayTick(cur + 1);
}

function stepBack() {
    if (!data) return;
    const cur = displayTick < 0 ? 1 : displayTick;
    setDisplayTick(cur - 1);
}

function togglePlay() {
    if (playing) { stopPlay(); return; }
    playing = true;
    document.getElementById('btn-play').textContent = '⏸';
    if (displayTick < 0 || displayTick >= data.ticks) displayTick = 0;
    playInterval = setInterval(() => {
        if (!data || displayTick >= data.ticks) { stopPlay(); return; }
        setDisplayTick(displayTick + 1);
    }, playSpeed);
}

function stopPlay() {
    playing = false;
    document.getElementById('btn-play').textContent = '▶';
    if (playInterval) { clearInterval(playInterval); playInterval = null; }
}

function setSpeed(ms) {
    playSpeed = parseInt(ms);
    if (playing) { stopPlay(); togglePlay(); }
}

// ── Run selector ──
async function loadRunList() {
    const r = await fetch('/api/runs');
    const d = await r.json();
    const sel = document.getElementById('run-select');
    sel.innerHTML = '<option value="">— select run —</option>';
    for (const name of d.runs) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
    }
    // Auto-select first
    if (d.runs.length > 0) {
        sel.value = d.runs[0];
        onRunChange();
    }
}

async function onRunChange() {
    const run = document.getElementById('run-select').value;
    if (!run) return;
    stopPlay();
    await fetchData(run);
    // Start auto-refresh for this run
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => fetchData(run), 5000);
}

async function fetchData(run) {
    try {
        const r = await fetch('/api/data?run=' + encodeURIComponent(run));
        data = await r.json();
        if (data.error) {
            document.getElementById('status').textContent = data.error;
            return;
        }
        document.getElementById('s-cond').textContent = data.condition;
        document.getElementById('s-seed').textContent = data.seed;
        document.getElementById('s-ticks').textContent = data.ticks;
        document.getElementById('tl-slider').max = data.ticks;
        document.getElementById('tl-max').textContent = '/ ' + data.ticks;
        updatePCA();
        buildTrajectory();
        if (displayTick < 0 || displayTick > data.ticks) {
            displayTick = data.ticks;
            document.getElementById('tl-slider').value = displayTick;
            document.getElementById('tl-label').textContent = 'Tick ' + displayTick;
        }
        updateVisibility();
        drawAllCharts();
        document.getElementById('status').textContent =
            'Updated: ' + new Date().toLocaleTimeString();
    } catch (e) {
        document.getElementById('status').textContent = 'Error: ' + e.message;
    }
}

// ── Animation ──
function animate() {
    requestAnimationFrame(animate);
    rotAngle += 0.003;
    camera.position.x = 2.5 * Math.cos(rotAngle);
    camera.position.z = 2.5 * Math.sin(rotAngle);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
}

// ── Keyboard ──
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') stepForward();
    else if (e.key === 'ArrowLeft') stepBack();
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
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
    parser = argparse.ArgumentParser(description="V2 Semantic Trajectory Viewer")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory containing V2 run folders (default: runs/bench_v2/)")
    parser.add_argument("--run", type=str, default=None,
                        help="Single run directory (legacy compat)")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    global RUNS_DIR
    if args.run:
        # Legacy: --run points to a single run dir, parent is runs_dir
        RUNS_DIR = Path(args.run).parent
    elif args.runs_dir:
        RUNS_DIR = Path(args.runs_dir)
    else:
        RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "bench_v2"

    if not RUNS_DIR.exists():
        print(f"[ERROR] Directory not found: {RUNS_DIR}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    print(f"\n  V2 Viewer: http://localhost:{args.port}")
    print(f"  Runs dir: {RUNS_DIR}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
