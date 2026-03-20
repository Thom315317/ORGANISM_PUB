#!/usr/bin/env python3
"""
stem_live.py — Live STEM trajectory visualizer
================================================
Mini web server that displays the STEM state-space trajectory
in real-time during benchmarks.

Usage:
    python tools/stem_live.py --data-dir runs/20260228_204414_bench/organism/
    # Open http://localhost:8765

Reads events.jsonl from --data-dir every 10 seconds.
Standalone — no imports from organism/ or consciousness/.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, jsonify, Response, request

# Ensure project root is on sys.path when running as `python tools/stem_live.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.stem_metrics import compute_stem_metrics
from tools.stem_export import export_per_tick_csv, export_summary_markdown

app = Flask(__name__)
log = logging.getLogger("stem_live")

DATA_DIR: Path = Path(".")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>STEM Live — 3D State-Space Trajectory</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    overflow: hidden;
}
#header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(13,17,23,0.92); border-bottom: 1px solid #30363d;
    padding: 12px 24px; display: flex; align-items: center; gap: 32px;
}
#header h1 { font-size: 16px; font-weight: 600; color: #58a6ff; }
.stat { font-size: 13px; }
.stat .label { color: #8b949e; }
.stat .value { color: #f0f6fc; font-weight: 600; }
.mode-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
}
#three-container {
    position: fixed; top: 52px; left: 0; right: 0; bottom: 32px;
}
#legend {
    position: fixed; bottom: 48px; right: 16px; z-index: 10;
    background: rgba(13,17,23,0.85); border: 1px solid #30363d;
    border-radius: 8px; padding: 10px 14px; font-size: 12px;
}
.legend-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
#status {
    position: fixed; bottom: 36px; left: 16px; z-index: 10;
    font-size: 11px; color: #484f58;
}
#tooltip {
    position: fixed; z-index: 20; pointer-events: none;
    background: rgba(22,27,34,0.95); border: 1px solid #30363d;
    border-radius: 6px; padding: 8px 12px; font-size: 12px;
    display: none; max-width: 320px;
}
#tooltip .tt-tick { color: #58a6ff; font-weight: 600; }
#tooltip .tt-mode { font-weight: 600; }
#tooltip .tt-vel { color: #8b949e; }
#tooltip .tt-theory { color: #8b949e; font-size: 11px; margin-top: 4px; }
#timeline {
    position: fixed; bottom: 0; left: 0; right: 0; height: 28px;
    background: rgba(13,17,23,0.95); border-top: 1px solid #30363d;
    z-index: 10; display: flex; align-items: center; padding: 0 8px;
}
#timeline-bar {
    flex: 1; height: 12px; display: flex; position: relative;
    border-radius: 3px; overflow: hidden; background: #161b22;
}
#timeline-bar .tl-tick { flex: 1; min-width: 1px; }
#timeline-cursor {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: #ffffff; border-radius: 1px; z-index: 1;
}
#controls {
    position: fixed; top: 58px; left: 12px; z-index: 16;
    display: flex; flex-direction: column; gap: 6px;
}
#controls button {
    width: 36px; height: 36px; border-radius: 8px;
    border: 1px solid #30363d; background: rgba(13,17,23,0.85);
    color: #c9d1d9; font-size: 16px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
}
#controls button:hover { background: rgba(48,54,61,0.85); }
#controls button.active { background: rgba(88,166,255,0.25); border-color: #58a6ff; }
#layer-toggles {
    position: fixed; top: 178px; left: 12px; z-index: 16;
    display: flex; flex-direction: column; gap: 4px;
    background: rgba(13,17,23,0.85); border: 1px solid #30363d;
    border-radius: 6px; padding: 8px 10px; font-size: 11px;
}
#layer-toggles label { display: flex; align-items: center; gap: 6px; cursor: pointer; color: #c9d1d9; }
#layer-toggles input[type=checkbox] { accent-color: #58a6ff; cursor: pointer; }
#layer-toggles .slider-row { display: flex; align-items: center; gap: 6px; margin-top: 4px; border-top: 1px solid #30363d; padding-top: 6px; }
#layer-toggles .slider-row input[type=range] { width: 80px; height: 4px; accent-color: #58a6ff; cursor: pointer; }
#layer-toggles .slider-val { color: #58a6ff; font-weight: 600; min-width: 28px; text-align: right; }
#debug-panel {
    position: fixed; bottom: 48px; left: 16px; z-index: 10;
    background: rgba(13,17,23,0.85); border: 1px solid #30363d;
    border-radius: 8px; padding: 8px 12px; font-size: 11px;
}
#debug-toggle { cursor: pointer; color: #8b949e; font-weight: 600; }
#debug-toggle:hover { color: #c9d1d9; }
#debug-content { margin-top: 6px; color: #c9d1d9; }
#debug-content div { margin: 2px 0; }
#mini-panel-left {
    position: fixed; top: 58px; left: 56px; z-index: 15;
    width: 250px; height: 250px;
    background: #000000; border: 1px solid #30363d;
    border-radius: 6px; overflow: hidden;
}
#mini-panel-left canvas { display: block; }
#mini-panel-left .mp-header {
    position: absolute; top: 4px; left: 8px; z-index: 1;
    font-size: 10px; color: #8b949e; font-weight: 600;
    letter-spacing: 0.5px; pointer-events: none;
}
#mini-panel-left .mp-footer {
    position: absolute; bottom: 4px; left: 8px; right: 8px; z-index: 1;
    font-size: 9px; color: #484f58; pointer-events: none;
}
#review-bar {
    position: fixed; bottom: 28px; left: 0; right: 0; height: 32px;
    background: rgba(13,17,23,0.95); border-top: 1px solid #58a6ff;
    z-index: 12; display: none; align-items: center; padding: 0 12px; gap: 8px;
}
#review-bar button {
    width: 28px; height: 24px; border-radius: 4px;
    border: 1px solid #30363d; background: rgba(13,17,23,0.85);
    color: #c9d1d9; font-size: 12px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
#review-bar button:hover { background: rgba(48,54,61,0.85); }
#review-slider {
    flex: 1; height: 4px; -webkit-appearance: none; appearance: none;
    background: #30363d; border-radius: 2px; outline: none; cursor: pointer;
}
#review-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 12px; height: 12px;
    border-radius: 50%; background: #58a6ff; cursor: pointer;
}
#review-label {
    font-size: 11px; color: #58a6ff; font-weight: 600; min-width: 100px; text-align: right;
}
#mini-panel {
    position: fixed; top: 58px; right: 16px; z-index: 15;
    width: 250px; height: 250px;
    background: #000000; border: 1px solid #30363d;
    border-radius: 6px; overflow: hidden;
}
#mini-panel canvas { display: block; }
#mini-panel-header {
    position: absolute; top: 4px; left: 8px; z-index: 1;
    font-size: 10px; color: #8b949e; font-weight: 600;
    letter-spacing: 0.5px; pointer-events: none;
}
#mini-panel-footer {
    position: absolute; bottom: 4px; left: 8px; right: 8px; z-index: 1;
    font-size: 9px; color: #484f58; pointer-events: none;
}
</style>
</head>
<body>

<div id="header">
    <h1>STEM Live 3D</h1>
    <div class="stat"><span class="label">Tick </span><span class="value" id="h-tick">—</span></div>
    <div class="stat"><span class="label">Mode </span><span class="mode-badge" id="h-mode">—</span></div>
    <div class="stat"><span class="label">Dim </span><span class="value" id="h-dim">—</span></div>
    <div class="stat"><span class="label">Attractors </span><span class="value" id="h-attr">—</span></div>
    <div class="stat"><span class="label">Transitions </span><span class="value" id="h-trans">—</span></div>
    <div class="stat"><span class="label">Variance </span><span class="value" id="h-var">—</span></div>
    <div class="stat" style="margin-left:auto">
        <select id="h-run-select" style="
            background:#161b22; color:#c9d1d9; border:1px solid #30363d;
            border-radius:6px; padding:3px 8px; font-size:12px;
            font-family:inherit; cursor:pointer; max-width:340px;
        ">
            <option value="">— loading runs —</option>
        </select>
    </div>
</div>

<div id="controls">
    <button id="btn-play" class="active" title="Auto-rotate">&#9654;</button>
    <button id="btn-pause" title="Pause rotation">&#9646;&#9646;</button>
    <button id="btn-review" title="Review mode">&#8634;</button>
</div>

<div id="layer-toggles">
    <label><input type="checkbox" id="chk-trajectory" checked> Trajectory</label>
    <label><input type="checkbox" id="chk-attractors" checked> Attractors</label>
    <label><input type="checkbox" id="chk-connectors"> Connectors</label>
    <div class="slider-row">
        <span>Last</span>
        <input type="range" id="tick-window" min="50" max="500" value="500">
        <span class="slider-val" id="tick-window-val">all</span>
    </div>
</div>

<div id="mini-panel-left">
    <div class="mp-header">Theory History</div>
    <div class="mp-footer" id="hist-footer">—</div>
</div>

<div id="three-container"></div>

<div id="review-bar">
    <button id="rev-start" title="Start">&#9664;&#9664;</button>
    <button id="rev-play" title="Play">&#9654;</button>
    <button id="rev-pause" title="Pause">&#9646;&#9646;</button>
    <button id="rev-end" title="End">&#9654;&#9654;</button>
    <input type="range" id="review-slider" min="1" max="100" value="1">
    <span id="review-label">—</span>
</div>

<div id="timeline">
    <div id="timeline-bar"></div>
</div>

<div id="legend">
    <div style="font-weight:600;margin-bottom:4px;color:#8b949e">Modes</div>
    <div class="legend-row"><div class="legend-dot" style="background:#888888"></div> <span id="leg-Idle">Idle</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#58a6ff"></div> <span id="leg-Explore">Explore</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#f85149"></div> <span id="leg-Debate">Debate</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#3fb950"></div> <span id="leg-Implement">Implement</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#d2a8ff"></div> <span id="leg-Consolidate">Consolidate</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#f0883e"></div> <span id="leg-Recover">Recover</span></div>
    <div style="border-top:1px solid #30363d;margin-top:6px;padding-top:6px;font-weight:600;color:#8b949e">Axes (PCA)</div>
    <div class="legend-row" style="color:#ff4444" id="leg-pc1">PC1</div>
    <div class="legend-row" style="color:#44ff44" id="leg-pc2">PC2</div>
    <div class="legend-row" style="color:#4488ff" id="leg-pc3">PC3</div>
    <div style="border-top:1px solid #30363d;margin-top:6px;padding-top:6px;color:#8b949e;font-size:11px">
        Wireframe = attractor region<br>
        &#9679; = attractor centroid<br>
        --- = label connector<br>
        (Nt) = dwell ticks
    </div>
    <div style="border-top:1px solid #30363d;margin-top:6px;padding-top:4px;color:#f0883e;font-size:10px;font-style:italic">
        Wireframes/connectors are<br>NOT temporal transitions.
    </div>
</div>

<div id="tooltip">
    <div class="tt-tick"></div>
    <div class="tt-mode"></div>
    <div class="tt-vel"></div>
    <div class="tt-theory"></div>
</div>

<div id="mini-panel">
    <div id="mini-panel-header">Theory Space</div>
    <div id="mini-panel-footer">—</div>
</div>

<div id="debug-panel">
    <div id="debug-toggle">Debug &#9656;</div>
    <div id="debug-content" style="display:none">
        <div id="dbg-dim"></div>
        <div id="dbg-var"></div>
        <div id="dbg-attr"></div>
        <div id="dbg-trans"></div>
        <div id="dbg-idle"></div>
        <div style="border-top:1px solid #30363d;margin-top:6px;padding-top:6px;display:flex;gap:6px;flex-wrap:wrap">
            <button id="btn-export-json" style="font-size:10px;padding:2px 8px;border:1px solid #30363d;background:#161b22;color:#58a6ff;border-radius:4px;cursor:pointer">Export JSON</button>
            <button id="btn-export-csv" style="font-size:10px;padding:2px 8px;border:1px solid #30363d;background:#161b22;color:#58a6ff;border-radius:4px;cursor:pointer">Export CSV</button>
            <button id="btn-screenshot" style="font-size:10px;padding:2px 8px;border:1px solid #30363d;background:#161b22;color:#58a6ff;border-radius:4px;cursor:pointer">Screenshot</button>
            <button id="btn-copy-md" style="font-size:10px;padding:2px 8px;border:1px solid #30363d;background:#161b22;color:#58a6ff;border-radius:4px;cursor:pointer">Copy MD</button>
        </div>
    </div>
</div>

<div id="status">Connecting...</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Constants ──────────────────────────────────────────────
const MODE_COLORS = {
    'Idle':        0x888888,
    'Explore':     0x58a6ff,
    'Debate':      0xf85149,
    'Implement':   0x3fb950,
    'Consolidate': 0xd2a8ff,
    'Recover':     0xf0883e,
};
const MODE_CSS = {
    'Idle':        '#888888',
    'Explore':     '#58a6ff',
    'Debate':      '#f85149',
    'Implement':   '#3fb950',
    'Consolidate': '#d2a8ff',
    'Recover':     '#f0883e',
};
const MODE_BG = {
    'Idle':        'rgba(136,136,136,0.2)',
    'Explore':     'rgba(88,166,255,0.2)',
    'Debate':      'rgba(248,81,73,0.2)',
    'Implement':   'rgba(63,185,80,0.2)',
    'Consolidate': 'rgba(210,168,255,0.2)',
    'Recover':     'rgba(240,136,62,0.2)',
};

// ── Legend label mapping (loaded from /api/legend) ─────────
let LEGEND_MAP = {};
function labelMode(mode) {
    return (LEGEND_MAP && LEGEND_MAP[mode]) ? LEGEND_MAP[mode] : mode;
}
// Fetch legend mapping at startup (async, non-blocking)
fetch('/api/legend').then(r => r.json()).then(m => {
    LEGEND_MAP = m || {};
    // Update legend labels in the sidebar
    for (const [mode, label] of Object.entries(LEGEND_MAP)) {
        const el = document.getElementById('leg-' + mode);
        if (el) el.textContent = label;
    }
}).catch(() => {});

// ── Three.js Setup ─────────────────────────────────────────
const container = document.getElementById('three-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.FogExp2(0x0d1117, 0.004);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
camera.position.set(25, 20, 25);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// ── Lights ─────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x404060, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 15);
scene.add(dirLight);
const pointLight = new THREE.PointLight(0x58a6ff, 0.4, 60);
pointLight.position.set(-5, 10, -5);
scene.add(pointLight);

// ── OrbitControls ──────────────────────────────────────────
let orbitTheta = Math.PI / 4;
let orbitPhi = Math.PI / 4;
let orbitRadius = 28;
let orbitTarget = new THREE.Vector3(0, 0, 0);
let isPlaying = true;
let isDragging = false;
let lastMouse = { x: 0, y: 0 };

function updateCamera() {
    const sinP = Math.sin(orbitPhi);
    const cosP = Math.cos(orbitPhi);
    const sinT = Math.sin(orbitTheta);
    const cosT = Math.cos(orbitTheta);
    camera.position.set(
        orbitTarget.x + orbitRadius * sinP * cosT,
        orbitTarget.y + orbitRadius * cosP,
        orbitTarget.z + orbitRadius * sinP * sinT
    );
    camera.lookAt(orbitTarget);
}

container.addEventListener('mousedown', (e) => {
    isDragging = true;
    lastMouse = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    orbitTheta -= dx * 0.005;
    orbitPhi = Math.max(0.1, Math.min(Math.PI - 0.1, orbitPhi - dy * 0.005));
    lastMouse = { x: e.clientX, y: e.clientY };
    updateCamera();
});
container.addEventListener('wheel', (e) => {
    e.preventDefault();
    orbitRadius = Math.max(5, Math.min(100, orbitRadius + e.deltaY * 0.05));
    updateCamera();
}, { passive: false });

// Play/pause buttons
document.getElementById('btn-play').addEventListener('click', () => {
    isPlaying = true;
    document.getElementById('btn-play').classList.add('active');
    document.getElementById('btn-pause').classList.remove('active');
});
document.getElementById('btn-pause').addEventListener('click', () => {
    isPlaying = false;
    document.getElementById('btn-pause').classList.add('active');
    document.getElementById('btn-play').classList.remove('active');
});

updateCamera();

// ── Layer toggle event listeners ──────────────────────────
document.getElementById('chk-trajectory').addEventListener('change', (e) => {
    trajectoryGroup.visible = e.target.checked;
    if (e.target.checked) applyTickWindow();
});
document.getElementById('chk-attractors').addEventListener('change', (e) => {
    attractorGroup.visible = e.target.checked;
});
document.getElementById('chk-connectors').addEventListener('change', (e) => {
    connectorGroup.visible = e.target.checked;
});
document.getElementById('tick-window').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    const total = tickSpheres.length || parseInt(e.target.max) || 500;
    tickWindow = val >= total ? Infinity : val;
    document.getElementById('tick-window-val').textContent = tickWindow === Infinity ? 'all' : val;
    applyTickWindow();
});

// ── XY Grid ────────────────────────────────────────────────
const gridHelper = new THREE.GridHelper(40, 20, 0x1a1e24, 0x1a1e24);
gridHelper.position.y = -10;
scene.add(gridHelper);

// ── Axes ───────────────────────────────────────────────────
const axesGroup = new THREE.Group();
scene.add(axesGroup);

function makeAxisLine(from, to, color) {
    const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...from),
        new THREE.Vector3(...to),
    ]);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 });
    return new THREE.Line(geo, mat);
}

function makeAxisLabel(text, position, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
    ctx.font = '28px monospace';
    ctx.fillText(text, 4, 40);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.7 });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(...position);
    sprite.scale.set(6, 1.5, 1);
    return sprite;
}

let axisLabels = [];

function buildAxes(variance) {
    while (axesGroup.children.length > 0) axesGroup.remove(axesGroup.children[0]);
    axisLabels = [];

    const len = 12;
    axesGroup.add(makeAxisLine([0,0,0], [len,0,0], 0xff4444));
    const v1 = variance ? (variance[0]*100).toFixed(1) + '%' : '';
    const lx = makeAxisLabel('PC1 ' + v1, [len+2, 0, 0], 0xff4444);
    axesGroup.add(lx); axisLabels.push(lx);

    axesGroup.add(makeAxisLine([0,0,0], [0,len,0], 0x44ff44));
    const v2 = variance ? (variance[1]*100).toFixed(1) + '%' : '';
    const ly = makeAxisLabel('PC2 ' + v2, [0, len+2, 0], 0x44ff44);
    axesGroup.add(ly); axisLabels.push(ly);

    axesGroup.add(makeAxisLine([0,0,0], [0,0,len], 0x4488ff));
    const v3 = variance ? (variance[2]*100).toFixed(1) + '%' : '';
    const lz = makeAxisLabel('PC3 ' + v3, [0, 0, len+2], 0x4488ff);
    axesGroup.add(lz); axisLabels.push(lz);
}
buildAxes(null);

// ── Scene objects ──────────────────────────────────────────
let trajectoryGroup = new THREE.Group();
scene.add(trajectoryGroup);
let attractorGroup = new THREE.Group();
scene.add(attractorGroup);
let connectorGroup = new THREE.Group();
scene.add(connectorGroup);
let currentSphere = null;
let glowSphere = null;
let attractorMeshes = [];

let pointPositions = [];
let pointMeta = [];
let centroidPositions = [];
let centroidMeta = [];
let tickSpheres = [];      // per-tick sphere refs for review visibility
let tickTubes = [];         // {mesh, startTick, endTick} for review visibility
let tickTransitions = [];   // {mesh, tick} for review visibility
let tickHalos = [];         // {light, tick} for review visibility

// ── Data state ─────────────────────────────────────────────
let data = null;
let prevCount = 0;

// ── Scale PCA coords to scene units ────────────────────────
let scaleParams = { cx: 0, cy: 0, cz: 0, scale: 1 };

function computeScaleParams(pts3d) {
    if (!pts3d || pts3d.length === 0) return { cx: 0, cy: 0, cz: 0, scale: 1 };
    let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity, minZ=Infinity, maxZ=-Infinity;
    for (const p of pts3d) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
        if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    }
    const cx = (minX+maxX)/2, cy = (minY+maxY)/2, cz = (minZ+maxZ)/2;
    const range = Math.max(maxX-minX, maxY-minY, maxZ-minZ) || 1;
    const scale = 20 / range;
    return { cx, cy, cz, scale };
}

function scaleCoords(pts3d) {
    if (!pts3d || pts3d.length === 0) return [];
    scaleParams = computeScaleParams(pts3d);
    return applyScale(pts3d, scaleParams);
}

function applyScale(pts3d, params) {
    const { cx, cy, cz, scale } = params;
    return pts3d.map(p => ({
        x: (p.x - cx) * scale,
        y: (p.y - cy) * scale,
        z: (p.z - cz) * scale,
    }));
}

function applyScaleOne(pt, params) {
    const { cx, cy, cz, scale } = params;
    return {
        x: (pt.x - cx) * scale,
        y: (pt.y - cy) * scale,
        z: (pt.z - cz) * scale,
    };
}

// ── Build 3D scene from data ───────────────────────────────
function buildScene() {
    if (!data) return;

    let rawPts = data.pca_3d || null;
    if (!rawPts || rawPts.length === 0) {
        if (data.pca_2d && data.pca_2d.length > 0) {
            rawPts = data.pca_2d.map(p => ({ x: p.x, y: p.y, z: 0 }));
        } else {
            return;
        }
    }

    const pts = scaleCoords(rawPts);
    const modes = data.modes || [];
    const tickIds = data.tick_ids || [];
    const velocities = data.velocities || [];
    const theoryScores = data.theory_scores || [];

    // Clear previous
    scene.remove(trajectoryGroup);
    trajectoryGroup = new THREE.Group();
    scene.add(trajectoryGroup);

    scene.remove(attractorGroup);
    attractorGroup = new THREE.Group();
    scene.add(attractorGroup);

    scene.remove(connectorGroup);
    connectorGroup = new THREE.Group();
    scene.add(connectorGroup);

    if (currentSphere) { scene.remove(currentSphere); currentSphere = null; }
    if (glowSphere) { scene.remove(glowSphere); glowSphere = null; }

    pointPositions = [];
    pointMeta = [];
    centroidPositions = [];
    centroidMeta = [];
    tickSpheres = [];
    tickTubes = [];
    tickTransitions = [];
    tickHalos = [];

    if (pts.length === 0) return;

    // Transition tick set
    const transitionTicks = new Set();
    if (data.phase_transitions) {
        for (const pt of data.phase_transitions) {
            transitionTicks.add(pt.tick_id);
        }
    }

    // ── Tube trajectory (THIN radius=0.06) ──────────────────
    let segStart = 0;
    for (let i = 0; i <= pts.length; i++) {
        const modeNow = i < pts.length ? (modes[i] || 'Idle') : null;
        const modePrev = i > 0 ? (modes[segStart] || 'Idle') : null;
        if (i === pts.length || (i > segStart && modeNow !== modePrev)) {
            if (i - segStart >= 2) {
                const segPts = [];
                for (let j = segStart; j < i; j++) {
                    segPts.push(new THREE.Vector3(pts[j].x, pts[j].y, pts[j].z));
                }
                if (i < pts.length) {
                    segPts.push(new THREE.Vector3(pts[i].x, pts[i].y, pts[i].z));
                }
                const curve = new THREE.CatmullRomCurve3(segPts, false);
                const tubeGeo = new THREE.TubeGeometry(curve, Math.max(4, segPts.length * 3), 0.06, 6, false);
                const color = MODE_COLORS[modePrev] || 0x888888;
                const tubeMat = new THREE.MeshPhongMaterial({
                    color: color, transparent: true, opacity: 0.7, shininess: 40,
                });
                const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
                trajectoryGroup.add(tubeMesh);
                tickTubes.push({ mesh: tubeMesh, endTick: i });
            }
            segStart = i;
        }
    }

    // ── Phase transitions (WHITE EMISSIVE, radius=0.08) ─────
    for (let i = 1; i < pts.length; i++) {
        const tid = tickIds[i] || i;
        if (!transitionTicks.has(tid)) continue;
        const segPts = [
            new THREE.Vector3(pts[i-1].x, pts[i-1].y, pts[i-1].z),
            new THREE.Vector3(pts[i].x, pts[i].y, pts[i].z),
        ];
        const curve = new THREE.CatmullRomCurve3(segPts, false);
        const tubeGeo = new THREE.TubeGeometry(curve, 4, 0.08, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({
            color: 0xffffff, transparent: true, opacity: 0.9,
        });
        const tMesh = new THREE.Mesh(tubeGeo, tubeMat);
        trajectoryGroup.add(tMesh);
        tickTransitions.push({ mesh: tMesh, tick: i });
    }

    // ── Point spheres (one per tick) ────────────────────────
    const sphereGeo = new THREE.SphereGeometry(0.22, 8, 8);
    for (let i = 0; i < pts.length; i++) {
        const mode = modes[i] || 'Idle';
        const color = MODE_COLORS[mode] || 0x888888;
        const mat = new THREE.MeshPhongMaterial({
            color: color, emissive: color, emissiveIntensity: 0.6,
            transparent: true, opacity: 0.8,
        });
        const sphere = new THREE.Mesh(sphereGeo, mat);
        sphere.position.set(pts[i].x, pts[i].y, pts[i].z);
        trajectoryGroup.add(sphere);
        tickSpheres.push(sphere);

        // Point light halo on last 10 ticks
        if (i >= pts.length - 10) {
            const halo = new THREE.PointLight(color, 0.1, 2);
            halo.position.set(pts[i].x, pts[i].y, pts[i].z);
            trajectoryGroup.add(halo);
            tickHalos.push({ light: halo, tick: i });
        }

        pointPositions.push(new THREE.Vector3(pts[i].x, pts[i].y, pts[i].z));
        pointMeta.push({
            tick: tickIds[i] || (i+1),
            mode: mode,
            velocity: velocities[i] || 0,
            theories: theoryScores[i] || {},
        });
    }

    // ── Current point (PULSE animation) ─────────────────────
    const lastIdx = pts.length - 1;
    const lastMode = modes[lastIdx] || 'Idle';
    const lastColor = MODE_COLORS[lastMode] || 0x888888;

    const curGeo = new THREE.SphereGeometry(0.6, 16, 16);
    const curMat = new THREE.MeshPhongMaterial({
        color: lastColor, emissive: lastColor, emissiveIntensity: 1.0,
    });
    currentSphere = new THREE.Mesh(curGeo, curMat);
    currentSphere.position.set(pts[lastIdx].x, pts[lastIdx].y, pts[lastIdx].z);
    scene.add(currentSphere);

    const glowGeo = new THREE.SphereGeometry(1.5, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: lastColor, transparent: true, opacity: 0.25,
    });
    glowSphere = new THREE.Mesh(glowGeo, glowMat);
    glowSphere.position.copy(currentSphere.position);
    scene.add(glowSphere);

    // ── Attractors (wireframe spheres + breathing animation) ──
    attractorMeshes = [];
    if (data.attractors && data.attractors.length > 0) {
        const labelInfos = [];

        for (let ai = 0; ai < data.attractors.length; ai++) {
            const att = data.attractors[ai];
            const c3d = att.center_3d || att.center_2d;
            if (!c3d) continue;

            const rawC = { x: c3d.x, y: c3d.y, z: c3d.z || 0 };
            const sc = applyScaleOne(rawC, scaleParams);

            const color = MODE_COLORS[att.dominant_mode] || 0x484f58;
            let radius = Math.max(1.0, (att.radius || 1.0) * 3);
            if (att.dominant_mode === 'Consolidate') {
                radius *= 0.9;
            }

            // Wireframe sphere with breathing (animated in animate())
            const attGeo = new THREE.SphereGeometry(radius, 12, 8);
            const attMat = new THREE.MeshBasicMaterial({
                color: color, wireframe: true, transparent: true, opacity: 0.03,
            });
            const attMesh = new THREE.Mesh(attGeo, attMat);
            attMesh.position.set(sc.x, sc.y, sc.z);
            attractorGroup.add(attMesh);
            attractorMeshes.push({ mesh: attMesh, index: ai });

            // Centroid sphere (solid, small) for hover detection
            const centGeo = new THREE.SphereGeometry(0.5, 12, 12);
            const centMat = new THREE.MeshPhongMaterial({
                color: color, emissive: color, emissiveIntensity: 0.4,
                transparent: true, opacity: 0.6,
            });
            const centMesh = new THREE.Mesh(centGeo, centMat);
            centMesh.position.set(sc.x, sc.y, sc.z);
            attractorGroup.add(centMesh);
            centroidPositions.push(new THREE.Vector3(sc.x, sc.y, sc.z));
            centroidMeta.push({
                index: ai,
                dominant_mode: att.dominant_mode,
                dwell_ticks: att.dwell_ticks || 0,
                radius: radius,
                mean_speed: att.mean_speed || 0,
                entry_count: att.entry_count || 0,
                exit_count: att.exit_count || 0,
                purity: att.purity || 0,
            });

            labelInfos.push({
                text: labelMode(att.dominant_mode) + ' (' + att.dwell_ticks + 't)',
                cx: sc.x, cy: sc.y, cz: sc.z,
                radius: radius, color: color,
            });
        }

        // Compute label positions with anti-overlap
        for (let i = 0; i < labelInfos.length; i++) {
            const li = labelInfos[i];
            li.labelX = li.cx + li.radius + 1.5;
            li.labelY = li.cy + li.radius * 0.5;
            li.labelZ = li.cz;
            li.needsConnector = false;
        }

        // Push overlapping labels apart vertically
        for (let i = 0; i < labelInfos.length; i++) {
            for (let j = i + 1; j < labelInfos.length; j++) {
                const a = labelInfos[i], b = labelInfos[j];
                const dx = a.labelX - b.labelX;
                const dy = a.labelY - b.labelY;
                const dz = a.labelZ - b.labelZ;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < 4.0) {
                    b.labelY += (4.0 - dist) + 1.5;
                    b.needsConnector = true;
                }
            }
        }

        // Create labels (opacity 0.4) + connector lines (opacity 0.08)
        for (const li of labelInfos) {
            const label = makeAxisLabel(li.text,
                [li.labelX, li.labelY, li.labelZ], li.color);
            label.material.opacity = 0.4;
            attractorGroup.add(label);
            axisLabels.push(label);

            if (li.needsConnector) {
                const lineGeo = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(li.cx, li.cy, li.cz),
                    new THREE.Vector3(li.labelX, li.labelY, li.labelZ),
                ]);
                const lineMat = new THREE.LineDashedMaterial({
                    color: li.color, transparent: true, opacity: 0.15,
                    dashSize: 0.3, gapSize: 0.15,
                });
                const line = new THREE.Line(lineGeo, lineMat);
                line.computeLineDistances();
                connectorGroup.add(line);
            }
        }
    }

    // Axes with variance
    buildAxes(data.explained_variance || null);

    // Grid position
    gridHelper.position.y = -12;

    // Timeline
    buildTimeline();

    // Update tick-window slider max + apply visibility
    const twSlider = document.getElementById('tick-window');
    if (twSlider && pts.length > 0) {
        twSlider.max = pts.length;
        if (parseInt(twSlider.value) > pts.length) twSlider.value = pts.length;
    }

    // Apply layer toggle visibility
    trajectoryGroup.visible = document.getElementById('chk-trajectory').checked;
    attractorGroup.visible = document.getElementById('chk-attractors').checked;
    connectorGroup.visible = document.getElementById('chk-connectors').checked;

    // Apply tick window
    applyTickWindow();
}

// ── Tick window ───────────────────────────────────────────
let tickWindow = Infinity;

function applyTickWindow() {
    if (reviewMode) return;
    if (!document.getElementById('chk-trajectory').checked) return;
    const total = tickSpheres.length;
    const startIdx = tickWindow === Infinity ? 0 : Math.max(0, total - tickWindow);
    for (let i = 0; i < total; i++) {
        tickSpheres[i].visible = (i >= startIdx);
    }
    for (const t of tickTubes) {
        t.mesh.visible = (t.endTick > startIdx);
    }
    for (const t of tickTransitions) {
        t.mesh.visible = (t.tick > startIdx);
    }
    for (const h of tickHalos) {
        h.light.visible = (h.tick >= startIdx);
    }
}

// ── Timeline bar ───────────────────────────────────────────
function buildTimeline() {
    const bar = document.getElementById('timeline-bar');
    bar.innerHTML = '';
    if (!data || !data.modes) return;
    const modes = data.modes;
    const n = modes.length;
    if (n === 0) return;

    for (let i = 0; i < n; i++) {
        const div = document.createElement('div');
        div.className = 'tl-tick';
        div.style.background = MODE_CSS[modes[i]] || '#888888';
        bar.appendChild(div);
    }

    const cursor = document.createElement('div');
    cursor.id = 'timeline-cursor';
    cursor.style.left = ((n - 0.5) / n * 100) + '%';
    bar.appendChild(cursor);
}

// ── Tooltip / Hover ────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.5 };
const mouse = new THREE.Vector2();
const tooltipEl = document.getElementById('tooltip');

container.addEventListener('mousemove', (e) => {
    if (isDragging) { tooltipEl.style.display = 'none'; return; }

    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Check attractor centroids FIRST (priority over tick points)
    let closestCentroid = null;
    let closestCentDist = 2.0;
    for (let i = 0; i < centroidPositions.length; i++) {
        const d = raycaster.ray.distanceToPoint(centroidPositions[i]);
        if (d < closestCentDist) {
            closestCentDist = d;
            closestCentroid = i;
        }
    }

    if (closestCentroid !== null) {
        const cm = centroidMeta[closestCentroid];
        tooltipEl.querySelector('.tt-tick').textContent = 'Attractor A#' + cm.index;
        const modeEl = tooltipEl.querySelector('.tt-mode');
        modeEl.textContent = labelMode(cm.dominant_mode);
        modeEl.style.color = MODE_CSS[cm.dominant_mode] || '#c9d1d9';
        tooltipEl.querySelector('.tt-vel').textContent =
            'dwell=' + cm.dwell_ticks + 't  r=' + cm.radius.toFixed(2) +
            '  spd=' + cm.mean_speed.toFixed(3) +
            '  in/out=' + cm.entry_count + '/' + cm.exit_count;
        tooltipEl.querySelector('.tt-theory').style.display = 'none';
        tooltipEl.style.display = 'block';
        tooltipEl.style.left = (e.clientX + 16) + 'px';
        tooltipEl.style.top = (e.clientY - 10) + 'px';
        return;
    }

    // Then check tick points
    let closest = null;
    let closestDist = 2.0;
    for (let i = 0; i < pointPositions.length; i++) {
        const d = raycaster.ray.distanceToPoint(pointPositions[i]);
        if (d < closestDist) {
            closestDist = d;
            closest = i;
        }
    }

    if (closest !== null) {
        const pm = pointMeta[closest];
        tooltipEl.querySelector('.tt-tick').textContent = 'Tick ' + pm.tick;
        const modeEl = tooltipEl.querySelector('.tt-mode');
        modeEl.textContent = labelMode(pm.mode);
        modeEl.style.color = MODE_CSS[pm.mode] || '#c9d1d9';
        tooltipEl.querySelector('.tt-vel').textContent = 'vel=' + pm.velocity.toFixed(3);

        // Theory scores
        const theoryEl = tooltipEl.querySelector('.tt-theory');
        if (pm.theories && Object.keys(pm.theories).length > 0) {
            const parts = [];
            for (const [name, score] of Object.entries(pm.theories)) {
                if (score !== null && !isNaN(score)) {
                    parts.push(name + '=' + Number(score).toFixed(2));
                }
            }
            theoryEl.textContent = parts.join(' | ');
            theoryEl.style.display = parts.length > 0 ? 'block' : 'none';
        } else {
            theoryEl.style.display = 'none';
        }

        tooltipEl.style.display = 'block';
        tooltipEl.style.left = (e.clientX + 16) + 'px';
        tooltipEl.style.top = (e.clientY - 10) + 'px';
    } else {
        tooltipEl.style.display = 'none';
    }
});

container.addEventListener('mouseleave', () => {
    tooltipEl.style.display = 'none';
});

// ── Resize ─────────────────────────────────────────────────
function resize() {
    const rect = container.getBoundingClientRect();
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height);
}
window.addEventListener('resize', resize);
resize();

// ── Update header ──────────────────────────────────────────
function updateHeader() {
    if (!data) return;
    const n = data.n_ticks || 0;
    const mode = (data.modes && data.modes.length > 0) ? data.modes[data.modes.length - 1] : '—';
    const dim = data.effective_dim != null ? data.effective_dim.toFixed(2) : '—';
    const nAttr = data.attractors ? data.attractors.length : 0;
    const nTrans = data.phase_transitions ? data.phase_transitions.length : 0;
    const variance = data.explained_variance;
    const varStr = variance ? variance.map(v => (v*100).toFixed(1) + '%').join(' / ') : '—';

    document.getElementById('h-tick').textContent = n;
    const modeEl = document.getElementById('h-mode');
    modeEl.textContent = labelMode(mode);
    modeEl.style.background = MODE_BG[mode] || 'transparent';
    modeEl.style.color = MODE_CSS[mode] || '#c9d1d9';
    document.getElementById('h-dim').textContent = dim;
    document.getElementById('h-attr').textContent = nAttr;
    document.getElementById('h-trans').textContent = nTrans;
    document.getElementById('h-var').textContent = varStr;
    // run_label displayed via select dropdown, not text

    const pcaLabels = data.pca_labels || ['','',''];
    const ev = data.explained_variance || [0,0,0];
    document.getElementById('leg-pc1').textContent = 'PC1 (' + (ev[0]*100).toFixed(0) + '%) ' + pcaLabels[0];
    document.getElementById('leg-pc2').textContent = 'PC2 (' + (ev[1]*100).toFixed(0) + '%) ' + pcaLabels[1];
    document.getElementById('leg-pc3').textContent = 'PC3 (' + (ev[2]*100).toFixed(0) + '%) ' + pcaLabels[2];
}

// ── Debug panel ───────────────────────────────────────────
function updateDebugPanel() {
    if (!data) return;
    const dim = data.effective_dim != null ? data.effective_dim.toFixed(3) : '\u2014';
    const ev = data.explained_variance || [0,0,0];
    const nAttr = data.attractors ? data.attractors.length : 0;
    const nTrans = data.phase_transitions ? data.phase_transitions.length : 0;
    const modes = data.modes || [];
    const idleCount = modes.filter(m => m === 'Idle').length;
    const idleRate = modes.length > 0 ? (idleCount / modes.length * 100).toFixed(1) : '\u2014';

    document.getElementById('dbg-dim').textContent = 'eff_dim: ' + dim;
    document.getElementById('dbg-var').textContent = 'variance: ' + ev.map(v => (v*100).toFixed(1) + '%').join(' / ');
    document.getElementById('dbg-attr').textContent = '#attractors: ' + nAttr;
    document.getElementById('dbg-trans').textContent = '#transitions: ' + nTrans;
    document.getElementById('dbg-idle').textContent = 'idle_rate: ' + idleRate + '%';
}

document.getElementById('debug-toggle').addEventListener('click', () => {
    const content = document.getElementById('debug-content');
    const toggle = document.getElementById('debug-toggle');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.innerHTML = 'Debug \u25BE';
    } else {
        content.style.display = 'none';
        toggle.innerHTML = 'Debug \u25B8';
    }
});

// ── Export buttons ────────────────────────────────────────
function downloadBlob(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
}

document.getElementById('btn-export-json').addEventListener('click', async () => {
    const resp = await fetch('/api/metrics');
    const json = await resp.text();
    downloadBlob(json, 'stem_metrics.json', 'application/json');
});

document.getElementById('btn-export-csv').addEventListener('click', async () => {
    const resp = await fetch('/api/export/csv');
    const csv = await resp.text();
    downloadBlob(csv, 'stem_ticks.csv', 'text/csv');
});

document.getElementById('btn-screenshot').addEventListener('click', () => {
    renderer.render(scene, camera);
    const dataUrl = renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl; a.download = 'stem_screenshot.png';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
});

document.getElementById('btn-copy-md').addEventListener('click', async () => {
    const resp = await fetch('/api/export/markdown');
    const md = await resp.text();
    await navigator.clipboard.writeText(md);
    const btn = document.getElementById('btn-copy-md');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy MD'; }, 2000);
});

// ── Fetch loop ─────────────────────────────────────────────
async function fetchData() {
    try {
        const resp = await fetch('/api/stem');
        if (!resp.ok) {
            document.getElementById('status').textContent = 'No data yet (' + resp.status + ')';
            return;
        }
        const newData = await resp.json();
        const pts = newData.pca_3d || newData.pca_2d || [];
        const newCount = pts.length;
        const oldCount = prevCount;

        data = newData;
        prevCount = newCount;

        if (!reviewMode && (newCount !== oldCount || newCount === 0)) {
            try {
                buildScene();
            } catch(err) {
                console.error('buildScene CRASH:', err);
                document.getElementById('status').textContent = 'buildScene error: ' + err.message;
            }
        }
        if (reviewMode) {
            // Update slider max
            reviewSlider.max = data.n_ticks || 1;
        }
        updateHeader();
        updateDebugPanel();
        document.getElementById('status').textContent =
            'Last update: ' + new Date().toLocaleTimeString() + ' — ' + newCount + ' points';
    } catch (e) {
        document.getElementById('status').textContent = 'Error: ' + e.message;
    }
}

// ── Mini-panel: Theory Space polyhedron ────────────────────
const THEORY_COLORS = {
    MDM: 0xf85149, GWT: 0x58a6ff, HOT: 0xd2a8ff, FEP: 0x3fb950,
    IIT: 0xf0883e, DYN: 0x888888, RPT: 0xffffff,
};

const miniPanel = document.getElementById('mini-panel');
const miniScene = new THREE.Scene();
miniScene.background = new THREE.Color(0x000000);

const miniCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
miniCamera.position.set(6, 4.5, 6);
miniCamera.lookAt(0, 0, 0);

const miniRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
miniRenderer.setPixelRatio(window.devicePixelRatio);
miniRenderer.setSize(250, 250);
miniPanel.appendChild(miniRenderer.domElement);

miniScene.add(new THREE.AmbientLight(0xffffff, 0.8));
const miniDirLight = new THREE.DirectionalLight(0xffffff, 0.4);
miniDirLight.position.set(3, 5, 3);
miniScene.add(miniDirLight);

let miniGroup = new THREE.Group();
miniScene.add(miniGroup);
let miniTheta = 0;
let polyData = null;
let polyPrevN = 0;
let miniTheorySpheres = [];
let miniEdgeLines = [];

function makeMiniLabel(text, position, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(text, 32, 22);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.9 });
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(position.x, position.y, position.z);
    sprite.scale.set(1.2, 0.6, 1);
    return sprite;
}

function buildMiniScene() {
    if (!polyData || !polyData.vertices || polyData.vertices.length === 0) return;

    miniScene.remove(miniGroup);
    miniGroup = new THREE.Group();
    miniScene.add(miniGroup);

    const verts = polyData.vertices;
    const edges = polyData.edges || [];
    const SCALE = 4.0;  // scale up positions for visibility

    // Positions scaled
    const positions = verts.map(v => new THREE.Vector3(v.x * SCALE, v.y * SCALE, v.z * SCALE));

    // Draw 21 edges with color/opacity by correlation (animated in animate())
    miniEdgeLines = [];
    for (const e of edges) {
        const absR = Math.abs(e.r);
        const geo = new THREE.BufferGeometry().setFromPoints([positions[e.i], positions[e.j]]);
        let edgeColor, baseOpacity;
        if (absR > 0.5) {
            edgeColor = 0xffffff;
            baseOpacity = 0.3 + absR * 0.4;
        } else if (absR < 0.2) {
            edgeColor = 0xff4444;
            baseOpacity = 0.08 + absR * 0.3;
        } else {
            const t = (absR - 0.2) / 0.3;
            const r = Math.round(255 * (1 - t) + 255 * t);
            const g = Math.round(68 * (1 - t) + 255 * t);
            const b = Math.round(68 * (1 - t) + 255 * t);
            edgeColor = (r << 16) | (g << 8) | b;
            baseOpacity = 0.12 + absR * 0.4;
        }
        const mat = new THREE.LineBasicMaterial({
            color: edgeColor, transparent: true, opacity: baseOpacity,
        });
        const line = new THREE.Line(geo, mat);
        miniGroup.add(line);
        miniEdgeLines.push({ line: line, baseOpacity: baseOpacity });
    }

    // Draw 7 vertex spheres + labels (spheres animated in animate())
    miniTheorySpheres = [];
    for (let i = 0; i < verts.length; i++) {
        const v = verts[i];
        const score = v.score || 0;
        const color = THEORY_COLORS[v.name] || 0xffffff;
        const radius = Math.max(0.08, score * 0.35);
        const sphereGeo = new THREE.SphereGeometry(radius, 10, 10);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: color, emissive: color, emissiveIntensity: 0.6,
            transparent: true, opacity: 0.8,
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.copy(positions[i]);
        miniGroup.add(sphere);
        miniTheorySpheres.push({ mesh: sphere, index: i });

        // Label offset slightly above
        const labelPos = positions[i].clone();
        labelPos.y += radius + 0.4;
        miniGroup.add(makeMiniLabel(v.name, labelPos, color));
    }

    // Stability halos: wireframe sphere around stable vertices
    const stability = polyData.stability || {};
    for (let i = 0; i < verts.length; i++) {
        const v = verts[i];
        const stab = stability[v.name] || 0;
        if (stab > 0.1) {
            const haloOpacity = 0.02 + 0.06 * stab;
            const haloGeo = new THREE.SphereGeometry(0.6, 8, 6);
            const haloMat = new THREE.MeshBasicMaterial({
                color: 0xffffff, wireframe: true, transparent: true, opacity: haloOpacity,
            });
            const halo = new THREE.Mesh(haloGeo, haloMat);
            halo.position.copy(positions[i]);
            miniGroup.add(halo);
        }
    }

    // Update footer
    const scores = polyData.scores || {};
    const topKey = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    const topStr = topKey ? topKey[0] + '=' + topKey[1].toFixed(2) : '—';
    const nWeak = polyData.n_weak || 0;
    const nEdges = polyData.n_edges || 0;
    document.getElementById('mini-panel-footer').textContent =
        topStr + ' | corr<0.5: ' + nWeak + '/' + nEdges;

    polyPrevN = polyData.n_ticks || 0;
}

// Build polyhedron from local data.theory_scores at a specific tick (for review mode)
const POLY_BASE_POS = {
    MDM: [0, 1, 0], GWT: [0.87, 0.5, 0], HOT: [0.87, -0.5, 0],
    IIT: [0, -1, 0], DYN: [-0.87, -0.5, 0], RPT: [-0.87, 0.5, 0], FEP: [0, 0, 1],
};
function buildMiniSceneForTick(tickIdx) {
    if (!data || !data.theory_scores || tickIdx < 1) return;
    const ts = data.theory_scores[Math.min(tickIdx - 1, data.theory_scores.length - 1)] || {};

    miniScene.remove(miniGroup);
    miniGroup = new THREE.Group();
    miniScene.add(miniGroup);
    miniTheorySpheres = [];
    miniEdgeLines = [];

    const SCALE = 4.0;
    const keys = ['MDM','GWT','HOT','IIT','DYN','RPT','FEP'];
    const positions = [];
    const scores = [];

    for (const k of keys) {
        let s = ts[k];
        if (s === null || s === undefined || isNaN(s)) s = 0;
        scores.push(s);
        const bp = POLY_BASE_POS[k];
        positions.push(new THREE.Vector3(bp[0] * s * SCALE, bp[1] * s * SCALE, bp[2] * s * SCALE));
    }

    // Edges (all pairs, simple white with opacity based on score product)
    for (let i = 0; i < keys.length; i++) {
        for (let j = i + 1; j < keys.length; j++) {
            const geo = new THREE.BufferGeometry().setFromPoints([positions[i], positions[j]]);
            const avgS = (scores[i] + scores[j]) / 2;
            const baseOpacity = 0.05 + avgS * 0.2;
            const mat = new THREE.LineBasicMaterial({
                color: 0x666666, transparent: true, opacity: baseOpacity,
            });
            const line = new THREE.Line(geo, mat);
            miniGroup.add(line);
            miniEdgeLines.push({ line: line, baseOpacity: baseOpacity });
        }
    }

    // Vertex spheres + labels
    for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        const s = scores[i];
        const color = THEORY_COLORS[k] || 0xffffff;
        const radius = Math.max(0.08, s * 0.35);
        const sphereGeo = new THREE.SphereGeometry(radius, 10, 10);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: color, emissive: color, emissiveIntensity: 0.6,
            transparent: true, opacity: 0.8,
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.copy(positions[i]);
        miniGroup.add(sphere);
        miniTheorySpheres.push({ mesh: sphere, index: i });

        const labelPos = positions[i].clone();
        labelPos.y += radius + 0.4;
        miniGroup.add(makeMiniLabel(k, labelPos, color));
    }

    // Footer
    const topIdx = scores.indexOf(Math.max(...scores));
    const topStr = keys[topIdx] + '=' + scores[topIdx].toFixed(2);
    document.getElementById('mini-panel-footer').textContent = topStr + ' | tick ' + tickIdx;
}

async function fetchPolyhedron() {
    try {
        const resp = await fetch('/api/polyhedron');
        if (!resp.ok) return;
        const newData = await resp.json();
        polyData = newData;
        if (!reviewMode) buildMiniScene();
    } catch(e) {}
}

// ── Mini-panel LEFT: Theory History ─────────────────────────
const THEORY_ANGLES = { MDM: 0, GWT: 51, HOT: 103, IIT: 154, DYN: 206, RPT: 257, FEP: 309 };
const THEORY_ORDER = ['MDM', 'GWT', 'HOT', 'IIT', 'DYN', 'RPT', 'FEP'];

const histPanel = document.getElementById('mini-panel-left');
const histScene = new THREE.Scene();
histScene.background = new THREE.Color(0x000000);

const histCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
histCamera.position.set(8, 2, 8);
histCamera.lookAt(0, 0, 0);

const histRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
histRenderer.setPixelRatio(window.devicePixelRatio);
histRenderer.setSize(250, 250);
histPanel.appendChild(histRenderer.domElement);

histScene.add(new THREE.AmbientLight(0xffffff, 0.8));
const histDirLight = new THREE.DirectionalLight(0xffffff, 0.3);
histDirLight.position.set(3, 5, 3);
histScene.add(histDirLight);

let histGroup = new THREE.Group();
histScene.add(histGroup);
let histTheta = 0;
let histData = null;

function buildHistoryScene(maxTick) {
    if (!histData || !histData.ticks || histData.ticks.length === 0) return;

    histScene.remove(histGroup);
    histGroup = new THREE.Group();
    histScene.add(histGroup);

    let allTicks = histData.ticks;
    if (maxTick !== undefined && maxTick < allTicks.length) {
        allTicks = allTicks.slice(0, maxTick);
    }
    const totalN = allTicks.length;
    if (totalN === 0) return;

    // Subsample: max 60 points per theory to avoid "stick" effect
    const MAX_SHOW = 60;
    const step = totalN <= MAX_SHOW ? 1 : Math.ceil(totalN / MAX_SHOW);
    const indices = [];
    for (let i = 0; i < totalN; i += step) indices.push(i);
    // Always include last tick
    if (indices[indices.length - 1] !== totalN - 1) indices.push(totalN - 1);
    const nShow = indices.length;

    const RAD = 3.0;
    const HEIGHT = 8.0;

    for (let ti = 0; ti < THEORY_ORDER.length; ti++) {
        const key = THEORY_ORDER[ti];
        const angleDeg = THEORY_ANGLES[key];
        const angleRad = angleDeg * Math.PI / 180;
        const baseX = Math.cos(angleRad) * RAD;
        const baseZ = Math.sin(angleRad) * RAD;
        const color = THEORY_COLORS[key] || 0xffffff;

        const linePositions = [];
        for (let si = 0; si < nShow; si++) {
            const idx = indices[si];
            const score = allTicks[idx].scores[key] || 0;
            const y = (idx / Math.max(totalN - 1, 1)) * HEIGHT - HEIGHT / 2;
            // Sphere displaced radially by score (high score = further out)
            const r = RAD + (score - 0.5) * 1.2;
            const px = Math.cos(angleRad) * r;
            const pz = Math.sin(angleRad) * r;
            const radius = Math.max(0.06, score * 0.25 + 0.04);

            // Opacity: last 5 samples bright, older fade
            const age = nShow - 1 - si;
            const opacity = age < 5 ? 0.9 : Math.max(0.25, 0.9 - (age - 4) * 0.02);

            const sGeo = new THREE.SphereGeometry(radius, 6, 6);
            const sMat = new THREE.MeshPhongMaterial({
                color: color, emissive: color, emissiveIntensity: 0.5,
                transparent: true, opacity: opacity,
            });
            const sphere = new THREE.Mesh(sGeo, sMat);
            sphere.position.set(px, y, pz);
            histGroup.add(sphere);

            linePositions.push(new THREE.Vector3(px, y, pz));
        }

        // Connect subsampled points with segments (opacity varies with avg score)
        for (let si = 0; si < linePositions.length - 1; si++) {
            const idxA = indices[si], idxB = indices[si + 1];
            const sA = allTicks[idxA].scores[key] || 0;
            const sB = allTicks[idxB].scores[key] || 0;
            const avgScore = (sA + sB) / 2;
            const segOpacity = 0.06 + avgScore * 0.35;  // low score = faint, high = visible
            const segGeo = new THREE.BufferGeometry().setFromPoints([
                linePositions[si], linePositions[si + 1],
            ]);
            const segMat = new THREE.LineBasicMaterial({
                color: color, transparent: true, opacity: segOpacity,
            });
            histGroup.add(new THREE.Line(segGeo, segMat));
        }

        // Theory name label at top of column
        const topY = HEIGHT / 2 + 0.6;
        histGroup.add(makeMiniLabel(key,
            { x: baseX, y: topY, z: baseZ }, color));
    }

    // Footer
    const spread = histData.spread || 0;
    document.getElementById('hist-footer').textContent =
        'ticks: ' + totalN + ' | spread: ' + spread.toFixed(2);
}

async function fetchTheoryHistory() {
    try {
        const resp = await fetch('/api/theory_history');
        if (!resp.ok) return;
        histData = await resp.json();
        if (!reviewMode) buildHistoryScene();
    } catch(e) {}
}

// ── Review mode (visibility-based, no rebuild) ──────────────
let reviewMode = false;
let reviewTick = 1;
let reviewPlaying = false;
let reviewInterval = null;

const reviewBar = document.getElementById('review-bar');
const reviewSlider = document.getElementById('review-slider');
const reviewLabel = document.getElementById('review-label');

document.getElementById('btn-review').addEventListener('click', () => {
    if (reviewMode) { exitReview(); } else { enterReview(); }
});

function enterReview() {
    if (!data) return;
    reviewMode = true;
    reviewTick = 1;
    reviewPlaying = false;
    const total = data.n_ticks || 1;
    reviewSlider.max = total;
    reviewSlider.value = 1;
    reviewBar.style.display = 'flex';
    document.getElementById('btn-review').classList.add('active');
    // Build full scene once, then just toggle visibility
    buildScene();
    applyReviewVisibility();
    buildHistoryScene(reviewTick);
    buildMiniSceneForTick(reviewTick);
}

function exitReview() {
    reviewMode = false;
    reviewPlaying = false;
    if (reviewInterval) { clearTimeout(reviewInterval); reviewInterval = null; }
    reviewBar.style.display = 'none';
    document.getElementById('btn-review').classList.remove('active');
    // Restore visibility from layer toggles
    for (const s of tickSpheres) s.visible = true;
    for (const t of tickTubes) t.mesh.visible = true;
    for (const t of tickTransitions) t.mesh.visible = true;
    for (const h of tickHalos) h.light.visible = true;
    if (currentSphere) currentSphere.visible = true;
    if (glowSphere) glowSphere.visible = true;
    trajectoryGroup.visible = document.getElementById('chk-trajectory').checked;
    attractorGroup.visible = document.getElementById('chk-attractors').checked;
    connectorGroup.visible = document.getElementById('chk-connectors').checked;
    applyTickWindow();
    buildHistoryScene();
    buildMiniScene();
}

function applyReviewVisibility() {
    const rt = reviewTick;
    // Spheres: show only up to reviewTick
    for (let i = 0; i < tickSpheres.length; i++) {
        tickSpheres[i].visible = (i < rt);
    }
    // Tubes: show if endTick <= reviewTick
    for (const t of tickTubes) {
        t.mesh.visible = (t.endTick <= rt);
    }
    // Transitions: show if tick < reviewTick
    for (const t of tickTransitions) {
        t.mesh.visible = (t.tick < rt);
    }
    // Halos: show if tick < reviewTick
    for (const h of tickHalos) {
        h.light.visible = (h.tick < rt);
    }
    // Attractors + connectors: hide during review (they need full data)
    attractorGroup.visible = (rt >= tickSpheres.length);
    connectorGroup.visible = (rt >= tickSpheres.length) && document.getElementById('chk-connectors').checked;
    // Move current sphere to reviewTick position
    if (currentSphere && rt > 0 && rt <= pointPositions.length) {
        currentSphere.visible = true;
        glowSphere.visible = true;
        const pos = pointPositions[rt - 1];
        currentSphere.position.copy(pos);
        glowSphere.position.copy(pos);
    }
    // Update UI
    const total = data ? (data.n_ticks || 0) : 0;
    reviewLabel.textContent = 'Review: tick ' + rt + '/' + total;
    reviewSlider.value = rt;
}

document.getElementById('rev-start').addEventListener('click', () => {
    reviewTick = 1; applyReviewVisibility(); buildHistoryScene(reviewTick); buildMiniSceneForTick(reviewTick);
});
document.getElementById('rev-end').addEventListener('click', () => {
    const total = data ? (data.n_ticks || 1) : 1;
    reviewTick = total; applyReviewVisibility(); exitReview();
});
function reviewStep() {
    if (!reviewMode || !reviewPlaying) return;
    const total = data ? (data.n_ticks || 1) : 1;
    reviewTick++;
    if (reviewTick > total) { exitReview(); return; }
    applyReviewVisibility();
    // Only rebuild mini panels every 10 ticks (heavy)
    if (reviewTick % 10 === 0) {
        buildHistoryScene(reviewTick);
        buildMiniSceneForTick(reviewTick);
    }
    reviewInterval = setTimeout(reviewStep, 300);
}

document.getElementById('rev-play').addEventListener('click', () => {
    if (reviewPlaying) return;
    if (!reviewMode) return;
    reviewPlaying = true;
    console.log('Review PLAY started at tick', reviewTick, '/ total', data ? data.n_ticks : '?');
    reviewStep();
});
document.getElementById('rev-pause').addEventListener('click', () => {
    reviewPlaying = false;
    if (reviewInterval) { clearTimeout(reviewInterval); reviewInterval = null; }
    console.log('Review PAUSED at tick', reviewTick);
});
reviewSlider.addEventListener('input', (e) => {
    reviewTick = parseInt(e.target.value) || 1;
    applyReviewVisibility();
    buildHistoryScene(reviewTick);
    buildMiniSceneForTick(reviewTick);
});

// ── Animation loop ─────────────────────────────────────────
let clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const t = clock.getElapsedTime();

    // Auto-rotate: plays while isPlaying, pauses only during drag
    if (isPlaying && !isDragging) {
        orbitTheta += 0.002;
        updateCamera();
    }

    // Pulse current sphere: scale 1.0 – 1.3 over 2 seconds
    if (currentSphere) {
        const pulse = 1.0 + 0.3 * Math.sin(t * Math.PI);
        currentSphere.scale.set(pulse, pulse, pulse);
    }
    if (glowSphere) {
        const glowPulse = 1.0 + 0.2 * Math.sin(t * Math.PI);
        glowSphere.scale.set(glowPulse, glowPulse, glowPulse);
        glowSphere.material.opacity = 0.10 + 0.08 * Math.sin(t * Math.PI);
    }

    // Attractor wireframes: desynchronized breathing
    for (const am of attractorMeshes) {
        am.mesh.material.opacity = 0.03 + 0.04 * Math.sin(t * (0.3 + am.index * 0.17));
    }

    // Axis labels face camera
    for (const label of axisLabels) {
        label.lookAt(camera.position);
    }

    renderer.render(scene, camera);

    // Mini-panel: auto-rotate
    miniTheta += 0.00873;
    const miniR = 8;
    miniCamera.position.set(
        miniR * Math.sin(miniTheta),
        miniR * 0.4,
        miniR * Math.cos(miniTheta)
    );
    miniCamera.lookAt(0, 0, 0);

    // Theory spheres: desynchronized breathing (scale + opacity)
    for (const ts of miniTheorySpheres) {
        const j = ts.index;
        const s = 0.95 + 0.1 * Math.sin(t * (0.4 + j * 0.13));
        ts.mesh.scale.set(s, s, s);
        ts.mesh.material.opacity = 0.7 + 0.2 * Math.sin(t * (0.5 + j * 0.19));
    }

    // Edge lines: subtle collective breathing
    for (const el of miniEdgeLines) {
        el.line.material.opacity = el.baseOpacity + 0.05 * Math.sin(t * 0.2);
    }

    miniRenderer.render(miniScene, miniCamera);

    // History panel: auto-rotate
    histTheta += 0.005;
    const histR = 10;
    histCamera.position.set(
        histR * Math.sin(histTheta),
        histR * 0.2,
        histR * Math.cos(histTheta)
    );
    histCamera.lookAt(0, 0, 0);
    histRenderer.render(histScene, histCamera);
}

// ── Run selector ────────────────────────────────────────────
const runSelect = document.getElementById('h-run-select');
let currentRunPath = '';

async function fetchRunsList() {
    try {
        const resp = await fetch('/api/runs');
        if (!resp.ok) return;
        const info = await resp.json();
        const dirs = info.dirs || [];
        currentRunPath = info.active || '';
        runSelect.innerHTML = '';
        if (dirs.length === 0) {
            runSelect.innerHTML = '<option value="">— no runs found —</option>';
            return;
        }
        for (const d of dirs) {
            const opt = document.createElement('option');
            opt.value = d.path;
            opt.textContent = d.label;
            if (d.path === currentRunPath) opt.selected = true;
            runSelect.appendChild(opt);
        }
    } catch(e) { console.error('fetchRunsList', e); }
}

runSelect.addEventListener('change', async function() {
    const newPath = this.value;
    if (!newPath || newPath === currentRunPath) return;
    try {
        const resp = await fetch('/api/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: newPath}),
        });
        if (resp.ok) {
            currentRunPath = newPath;
            refreshAll();
        }
    } catch(e) { console.error('switch', e); }
});

function refreshAll() {
    fetchData();
    fetchPolyhedron();
    fetchTheoryHistory();
}

// ── Start ──────────────────────────────────────────────────
fetchRunsList();
refreshAll();
setInterval(refreshAll, 10000);
setInterval(fetchRunsList, 30000);
animate();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


RUNS_DIR: Path = Path(".")
CONDITION: str = "organism"
ACTIVE_DIR: Optional[Path] = None  # Set via /api/switch — overrides auto-detect
LEGEND_MAP: Dict[str, str] = {}  # Optional mode label overrides (loaded from JSON)

# ── Signal dimensions extracted from events.jsonl tick_end rows ────
_SIGNAL_KEYS = ["novelty", "conflict", "cohesion", "impl_pressure", "prediction_error", "cost"]


def _list_all_run_dirs() -> List[Dict[str, str]]:
    """List all directories containing events.jsonl under RUNS_DIR, sorted by mtime desc."""
    results = []
    if not RUNS_DIR.exists():
        return results

    for events_file in sorted(RUNS_DIR.rglob("events.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = events_file.parent
        # Build a human-readable label from the path relative to RUNS_DIR
        try:
            rel = d.relative_to(RUNS_DIR)
        except ValueError:
            rel = d
        # Count ticks for extra info
        n_lines = 0
        try:
            with open(events_file) as f:
                for line in f:
                    if '"tick_end"' in line:
                        n_lines += 1
        except Exception:
            pass
        results.append({
            "path": str(d),
            "label": f"{rel} ({n_lines}t)" if n_lines else str(rel),
        })
    return results


@app.route("/api/runs")
def api_runs():
    """List all available run directories with events.jsonl."""
    dirs = _list_all_run_dirs()
    # Determine which is currently active
    active_path = ""
    if ACTIVE_DIR and (ACTIVE_DIR / "events.jsonl").exists():
        active_path = str(ACTIVE_DIR)
    else:
        ev = _find_latest_events()
        if ev:
            active_path = str(ev.parent)
    return jsonify({"dirs": dirs, "active": active_path})


@app.route("/api/switch", methods=["POST"])
def api_switch():
    """Switch the active data directory."""
    global ACTIVE_DIR
    data = {}
    try:
        data = json.loads(request.get_data(as_text=True))
    except Exception:
        pass
    new_path = data.get("path", "")
    if not new_path:
        return jsonify({"error": "no path"}), 400
    p = Path(new_path)
    if not (p / "events.jsonl").exists():
        return jsonify({"error": "events.jsonl not found in " + new_path}), 404
    ACTIVE_DIR = p
    log.info("Switched to: %s", p)
    return jsonify({"ok": True, "active": str(p)})


def _find_latest_events() -> Optional[Path]:
    """Find events.jsonl — uses ACTIVE_DIR if set, then explicit DATA_DIR, then auto-detect."""
    # 1. User-selected directory via UI
    if ACTIVE_DIR is not None:
        af = ACTIVE_DIR / "events.jsonl"
        if af.exists():
            return af

    # 2. Explicit --data-dir
    explicit = DATA_DIR / "events.jsonl"
    if explicit.exists():
        return explicit

    # 3. Auto-detect latest bench
    candidates = []
    for run_dir in RUNS_DIR.glob("*_bench"):
        ef = run_dir / CONDITION / "events.jsonl"
        if ef.exists():
            try:
                candidates.append((ef.stat().st_mtime, ef))
            except OSError:
                continue
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def _parse_events(events_path: Path) -> Dict[str, Any]:
    """Read events.jsonl + metrics.jsonl and compute PCA + trajectory analysis live."""
    tick_data: List[Dict] = []
    with open(events_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "tick_end":
                tick_data.append(ev)

    run_label = events_path.parent.parent.name + "/" + events_path.parent.name
    n = len(tick_data)

    if n == 0:
        return {"n_ticks": 0, "pca_2d": [], "pca_3d": [], "modes": [],
                "tick_ids": [], "velocities": [], "phase_transitions": [],
                "attractors": [], "theory_scores": [], "run_label": run_label}

    # Build signal matrix (n_ticks x n_dims)
    # Auto-detect signal keys: use default CRISTAL keys if present,
    # otherwise detect from first tick (supports COGITATE band powers etc.)
    signal_keys = _SIGNAL_KEYS
    first = tick_data[0]
    if not any(first.get(k) is not None and first.get(k) != 0.0 for k in _SIGNAL_KEYS):
        # CRISTAL keys absent — detect alternative signal keys
        _ALT_SIGNAL_KEYS = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]
        if any(first.get(k) is not None for k in _ALT_SIGNAL_KEYS):
            signal_keys = _ALT_SIGNAL_KEYS

    modes = [t.get("mode", "Idle") for t in tick_data]
    tick_ids = [t.get("tick_id", i + 1) for i, t in enumerate(tick_data)]
    vectors = []
    for t in tick_data:
        vec = [float(t.get(k, 0.0)) for k in signal_keys]
        vectors.append(vec)
    X = np.array(vectors, dtype=np.float64)

    # ── Theory scores from metrics.jsonl ──
    theory_scores: List[Dict] = []
    metrics_path = events_path.parent / "metrics.jsonl"
    if metrics_path.exists():
        tick_theories: Dict[int, Dict] = {}
        try:
            with open(metrics_path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if "theory_scores" in row:
                            tid = row.get("tick_id", row.get("tick", 0))
                            tick_theories[tid] = row["theory_scores"]
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        for i, td in enumerate(tick_data):
            tid = td.get("tick_id", i + 1)
            raw = tick_theories.get(tid, {})
            theory_scores.append({k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                                  for k, v in raw.items()})
    else:
        for td in tick_data:
            raw = td.get("theory_scores", {})
            theory_scores.append({k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                                  for k, v in raw.items()})

    # PCA
    pca_2d = []
    pca_3d = []
    explained_variance = [0.0, 0.0, 0.0]
    eff_dim = 0.0
    eigvecs = None

    if n >= 3 and X.shape[1] >= 2:
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

        for i in range(n):
            pca_2d.append({"x": round(float(proj[i, 0]), 4),
                           "y": round(float(proj[i, 1]), 4) if n_comp >= 2 else 0.0})
            pca_3d.append({"x": round(float(proj[i, 0]), 4),
                           "y": round(float(proj[i, 1]), 4) if n_comp >= 2 else 0.0,
                           "z": round(float(proj[i, 2]), 4) if n_comp >= 3 else 0.0})

        total_var = max(eigvals.sum(), 1e-12)
        for c in range(n_comp):
            explained_variance[c] = round(float(eigvals[c] / total_var), 4)

        normed = eigvals / total_var
        normed = normed[normed > 1e-12]
        eff_dim = float(np.exp(-np.sum(normed * np.log(normed + 1e-30))))
    else:
        for i in range(n):
            x_val = float(X[i, 0]) if X.shape[1] > 0 else 0.0
            y_val = float(X[i, 1]) if X.shape[1] > 1 else 0.0
            z_val = float(X[i, 2]) if X.shape[1] > 2 else 0.0
            pca_2d.append({"x": round(x_val, 4), "y": round(y_val, 4)})
            pca_3d.append({"x": round(x_val, 4), "y": round(y_val, 4), "z": round(z_val, 4)})

    # Velocities
    velocities = [0.0]
    for i in range(1, len(pca_3d)):
        dx = pca_3d[i]["x"] - pca_3d[i - 1]["x"]
        dy = pca_3d[i]["y"] - pca_3d[i - 1]["y"]
        dz = pca_3d[i]["z"] - pca_3d[i - 1]["z"]
        velocities.append(round(math.sqrt(dx * dx + dy * dy + dz * dz), 4))

    # Phase transitions
    phase_transitions = []
    for i in range(1, n):
        if modes[i] != modes[i - 1]:
            phase_transitions.append({
                "tick_id": tick_ids[i],
                "velocity": velocities[i],
                "from_mode": modes[i - 1],
                "to_mode": modes[i],
            })

    # Attractors
    attractors = []
    if n >= 5:
        i = 0
        while i < n:
            j = i
            while j < n and modes[j] == modes[i]:
                j += 1
            dwell = j - i
            if dwell >= 3:
                cx = sum(pca_3d[k]["x"] for k in range(i, j)) / dwell
                cy = sum(pca_3d[k]["y"] for k in range(i, j)) / dwell
                cz = sum(pca_3d[k]["z"] for k in range(i, j)) / dwell
                r = max(
                    math.sqrt((pca_3d[k]["x"] - cx) ** 2 +
                              (pca_3d[k]["y"] - cy) ** 2 +
                              (pca_3d[k]["z"] - cz) ** 2)
                    for k in range(i, j)
                )
                # Mean speed inside attractor
                mean_speed = sum(velocities[k] for k in range(i, j)) / dwell if dwell > 0 else 0.0
                attractors.append({
                    "center_2d": {"x": round(cx, 4), "y": round(cy, 4)},
                    "center_3d": {"x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4)},
                    "dwell_ticks": dwell,
                    "dominant_mode": modes[i],
                    "radius": round(max(r, 0.3), 4),
                    "mean_speed": round(mean_speed, 4),
                    "entry_count": 1 if len(attractors) > 0 else 0,
                    "exit_count": 0,
                    "purity": 1.0,
                })
            i = j

        # Fix exit_count for all but last attractor
        for ai in range(len(attractors) - 1):
            attractors[ai]["exit_count"] = 1

    # PCA loading labels
    pca_labels = ["", "", ""]
    if n >= 3 and eigvecs is not None and X.shape[1] >= 2:
        for pc in range(min(3, eigvecs.shape[1])):
            loadings = eigvecs[:, pc]
            top_idx = np.argsort(np.abs(loadings))[::-1][:2]
            parts = []
            for idx in top_idx:
                sign = "+" if loadings[idx] > 0 else "-"
                parts.append(f"{sign}{signal_keys[idx]}")
            pca_labels[pc] = ", ".join(parts)

    return {
        "n_ticks": n,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "explained_variance": explained_variance,
        "pca_labels": pca_labels,
        "effective_dim": round(eff_dim, 2),
        "modes": modes,
        "tick_ids": tick_ids,
        "velocities": velocities,
        "phase_transitions": phase_transitions,
        "attractors": attractors,
        "theory_scores": theory_scores,
        "run_label": run_label,
        "signal_dims": signal_keys,
    }


_THEORY_KEYS = ["MDM", "GWT", "HOT", "FEP", "IIT", "DYN", "RPT"]


@app.route("/api/polyhedron")
def api_polyhedron():
    """Theory-space polyhedron: 7 vertices (theories) + 21 edges with correlations."""
    # Fixed heptagram 3D positions (unit sphere)
    _POSITIONS = {
        "MDM": (0.0, 1.0, 0.0),
        "GWT": (0.87, 0.5, 0.0),
        "HOT": (0.87, -0.5, 0.0),
        "IIT": (0.0, -1.0, 0.0),
        "DYN": (-0.87, -0.5, 0.0),
        "RPT": (-0.87, 0.5, 0.0),
        "FEP": (0.0, 0.0, 1.0),
    }

    events_path = _find_latest_events()
    empty = {"vertices": [], "edges": [], "scores": {}, "correlations": {}}
    if events_path is None:
        return jsonify(empty), 200

    metrics_path = events_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        return jsonify(empty), 200

    # Read all theory scores per tick (ordered)
    tick_rows: List[Dict[str, float]] = []
    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "theory_scores" not in row:
                        continue
                    ts = row["theory_scores"]
                    clean = {}
                    for k in _THEORY_KEYS:
                        v = ts.get(k, 0.0)
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            v = 0.0
                        clean[k] = float(v)
                    tick_rows.append(clean)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return jsonify(empty), 200

    if not tick_rows:
        return jsonify(empty), 200

    # Last tick scores → vertex positions
    last_scores = tick_rows[-1]
    vertices = []
    for k in _THEORY_KEYS:
        s = last_scores.get(k, 0.0)
        bx, by, bz = _POSITIONS[k]
        vertices.append({"name": k, "x": round(bx * s, 4), "y": round(by * s, 4), "z": round(bz * s, 4), "score": round(s, 4)})

    # Sliding-window correlations (last 20 ticks) between all theory pairs
    window = tick_rows[-20:] if len(tick_rows) >= 20 else tick_rows
    n_win = len(window)
    correlations = {}
    edges = []

    for ia in range(len(_THEORY_KEYS)):
        for ib in range(ia + 1, len(_THEORY_KEYS)):
            ka, kb = _THEORY_KEYS[ia], _THEORY_KEYS[ib]
            if n_win >= 3:
                a_vals = [w.get(ka, 0.0) for w in window]
                b_vals = [w.get(kb, 0.0) for w in window]
                a_arr = np.array(a_vals)
                b_arr = np.array(b_vals)
                a_std = a_arr.std()
                b_std = b_arr.std()
                if a_std > 1e-9 and b_std > 1e-9:
                    r = float(np.corrcoef(a_arr, b_arr)[0, 1])
                else:
                    r = 0.0
            else:
                r = 0.0
            pair_key = f"{ka}-{kb}"
            correlations[pair_key] = round(r, 4)
            edges.append({"i": ia, "j": ib, "a": ka, "b": kb, "r": round(r, 4)})

    # Count weak correlations
    n_weak = sum(1 for e in edges if abs(e["r"]) < 0.5)

    # Stability halos: std of last 5 ticks per theory
    stability = {}
    if len(tick_rows) >= 5:
        last5 = tick_rows[-5:]
        for k in _THEORY_KEYS:
            vals = [r.get(k, 0.0) for r in last5]
            std_val = float(np.std(vals))
            stab = max(0.0, min(1.0, 1.0 - std_val * 10))
            stability[k] = round(stab, 4)
    else:
        for k in _THEORY_KEYS:
            stability[k] = 0.0

    return jsonify({
        "vertices": vertices,
        "edges": edges,
        "scores": {k: round(last_scores.get(k, 0.0), 4) for k in _THEORY_KEYS},
        "correlations": correlations,
        "stability": stability,
        "n_ticks": len(tick_rows),
        "n_weak": n_weak,
        "n_edges": len(edges),
    })


@app.route("/api/theory_history")
def api_theory_history():
    """Per-tick theory scores for the Theory History panel."""
    events_path = _find_latest_events()
    if events_path is None:
        return jsonify({"ticks": []}), 200

    metrics_path = events_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        return jsonify({"ticks": []}), 200

    ticks = []
    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "theory_scores" not in row:
                        continue
                    tid = row.get("tick_id", row.get("tick", len(ticks) + 1))
                    ts = row["theory_scores"]
                    clean = {}
                    for k in _THEORY_KEYS:
                        v = ts.get(k, 0.0)
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            v = 0.0
                        clean[k] = round(float(v), 4)
                    ticks.append({"tick_id": tid, "scores": clean})
                except json.JSONDecodeError:
                    continue
    except OSError:
        return jsonify({"ticks": []}), 200

    # Compute spread (std of theory means)
    if ticks:
        means = []
        for k in _THEORY_KEYS:
            vals = [t["scores"].get(k, 0.0) for t in ticks]
            means.append(float(np.mean(vals)))
        spread = round(float(np.std(means)), 4)
    else:
        spread = 0.0

    return jsonify({"ticks": ticks, "spread": spread, "n_ticks": len(ticks)})


@app.route("/api/stem")
def api_stem():
    events_path = _find_latest_events()
    if events_path is None:
        return jsonify({
            "n_ticks": 0, "pca_2d": [], "pca_3d": [], "modes": [],
            "attractors": [], "phase_transitions": [], "theory_scores": [],
            "run_label": "", "hint": "No bench run found",
        }), 200

    try:
        data = _parse_events(events_path)
        return jsonify(data)
    except Exception as exc:
        log.exception("Error parsing events")
        return jsonify({"error": str(exc), "run_label": str(events_path)}), 500


@app.route("/api/metrics")
def api_metrics():
    """Return full STEM metrics pack JSON."""
    events_path = _find_latest_events()
    if events_path is None:
        return jsonify({"error": "no_data"}), 200

    metrics_path = events_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        metrics_path = None

    window = request.args.get("window", None, type=int)
    try:
        m = compute_stem_metrics(events_path, metrics_path, window)
        return jsonify(m)
    except Exception as exc:
        log.exception("Error computing metrics")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/export/csv")
def api_export_csv():
    """Return per-tick CSV file download."""
    events_path = _find_latest_events()
    if events_path is None:
        return Response("no data", mimetype="text/plain", status=404)

    metrics_path = events_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        metrics_path = None

    try:
        csv_str = export_per_tick_csv(events_path, metrics_path)
        return Response(
            csv_str,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=stem_ticks.csv"},
        )
    except Exception as exc:
        log.exception("Error exporting CSV")
        return Response(str(exc), mimetype="text/plain", status=500)


@app.route("/api/export/markdown")
def api_export_markdown():
    """Return markdown summary."""
    events_path = _find_latest_events()
    if events_path is None:
        return Response("no data", mimetype="text/plain", status=404)

    metrics_path = events_path.parent / "metrics.jsonl"
    if not metrics_path.exists():
        metrics_path = None

    try:
        m = compute_stem_metrics(events_path, metrics_path)
        md = export_summary_markdown(m)
        return Response(md, mimetype="text/markdown")
    except Exception as exc:
        log.exception("Error exporting markdown")
        return Response(str(exc), mimetype="text/plain", status=500)


@app.route("/api/legend")
def api_legend():
    """Return legend label mapping (empty dict if no mapping loaded)."""
    return jsonify(LEGEND_MAP)


def main():
    global DATA_DIR, RUNS_DIR, CONDITION, ACTIVE_DIR, LEGEND_MAP

    parser = argparse.ArgumentParser(description="STEM Live Visualizer")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Explicit directory containing events.jsonl (optional — auto-detects latest bench)",
    )
    parser.add_argument(
        "--condition", type=str, default="organism",
        help="Bench condition to watch (organism, random_judge, single_agent)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--legend-map", type=str, default=None,
        help="JSON file mapping mode names to display labels (optional)",
    )
    args = parser.parse_args()

    CONDITION = args.condition
    RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
    DATA_DIR = Path(args.data_dir) if args.data_dir else Path("/nonexistent")
    if args.data_dir:
        ACTIVE_DIR = Path(args.data_dir)

    # ── Legend mapping: explicit > auto-detect > none ──
    legend_path = None
    if args.legend_map:
        legend_path = Path(args.legend_map)
    elif args.data_dir:
        auto = Path(args.data_dir) / "legend_mapping.json"
        if auto.exists():
            legend_path = auto
    if legend_path and legend_path.exists():
        try:
            with open(legend_path) as f:
                LEGEND_MAP = json.load(f)
            print(f"  Legend   : {legend_path} ({len(LEGEND_MAP)} mappings)")
        except Exception as exc:
            print(f"  Legend   : FAILED to load {legend_path}: {exc}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ev = _find_latest_events()
    mode_str = f"explicit: {args.data_dir}" if args.data_dir else f"auto-detect (condition={CONDITION})"

    print(f"STEM Live Visualizer 3D")
    print(f"  Source   : events.jsonl + metrics.jsonl (live PCA)")
    print(f"  Mode     : {mode_str}")
    if ev and ev.exists():
        print(f"  Watching : {ev}")
    else:
        print(f"  Watching : (waiting for bench to start...)")
    print(f"  URL      : http://localhost:{args.port}")
    print(f"  Refresh  : every 10s")
    print()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
