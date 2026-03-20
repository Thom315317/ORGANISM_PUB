# Organism V2 — Bench & Viewer

## Bench (`bench_v2.py`)

Runs 4 conditions × N seeds × 50 ticks.

| Condition | Description |
|-----------|-------------|
| A | Multi-agent, no perturbation — baseline |
| B | Multi-agent + neutral perturbation (rephrase) at t15/t35 |
| C | Multi-agent + strong perturbation (compression t15, inversion t35) |
| E | Single-agent + strong perturbation — controls for multi-agent contribution. SingleDraftJudge is active. |

### Tick indexing convention

- `bench loop`: `tick in range(total_ticks)` → **0-indexed** (0 to 49)
- `orchestrator`: `self._tick_id` → **1-indexed** (1 to 50)
- `perturbation_log["tick"]` → 0-indexed (bench loop)
- `events.jsonl tick_id` → 1-indexed (orchestrator)
- `sim_curves` keys (`"tick_15"`, `"tick_35"`) → 0-indexed (bench loop)
- All analysis code must use bench loop index as reference.

### Usage

```bash
python organism_v2/bench_v2.py                          # Full (4 cond × 3 seeds)
python organism_v2/bench_v2.py --conditions C,E --seeds 7,77,777 --ticks 50
python organism_v2/bench_v2.py --dry-run                # 10 ticks, no LLM
```

## Viewer (`viewer_v3.py`)

Flask + Three.js 3D trajectory viewer.

```bash
python organism_v2/viewer_v3.py --runs-dir runs/bench_v2/ --port 8767
```

Features: PCA/UMAP projection, connectome, metric coloring, trail fading, distance matrix, mouse controls (drag=rotate, wheel=zoom).

## Known limitations

### Connectome is not comparable across conditions with different agent counts

The connectome computes cosine similarity between `state_vector_mean` at different ticks. In multi-agent conditions (A, B, C), `state_vector_mean` is the average of 3 agent embeddings. This averaging smooths the vector, producing higher inter-tick similarity (~0.76 mean) and dense connectome graphs (400+ edges at threshold 0.80).

In single-agent condition (E), `state_vector_mean` is the raw embedding of a single agent. No smoothing occurs, leading to lower inter-tick similarity (~0.50 mean) and sparse connectomes (30-60 edges at threshold 0.80).

This is a structural artefact of the averaging, not a signal about consciousness or resilience. The connectome is a valid intra-condition visualization tool (comparing seeds within C, or tracking temporal cohesion within a single run), but **not a valid inter-condition metric** when agent counts differ.

Do not normalize or lower the threshold for E — this would mask a real structural difference.

### CCV is zero for single-agent condition

Claim Cosine Variance (CCV) measures pairwise cosine distance between agent drafts within a tick. With a single agent (condition E), CCV is always 0.0 by definition. CCV is only meaningful for multi-agent conditions.

### Perturbation output varies across seeds

Perturbation operators (compression, inversion) call the LLM on the winning draft from the previous tick. Since draft content depends on the full conversation trajectory, perturbation output length and content vary significantly across seeds (e.g., inversion output ranges from 4000 to 5400 chars). This is expected — use multiple seeds and average to isolate the perturbation effect from trajectory variance.
