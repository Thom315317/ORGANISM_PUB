# Organism V2 — Bench & Viewer

## Bench (`bench_v2.py`)

Runs 7 conditions × 6 seeds × 50 ticks = 42 runs.

| Condition | Description |
|-----------|-------------|
| A | Multi-agent, no perturbation — baseline |
| B | Multi-agent + neutral perturbation (rephrase) at t15/t35 |
| C | Multi-agent + strong perturbation (compression t15, inversion t35) |
| E | Single-agent A (glm-5:cloud) + strong perturbation + no-LLM judge |
| E_B | Single-agent B (kimi-k2.5:cloud) + strong perturbation + no-LLM judge |
| E_C | Single-agent C (qwen3.5:397b-cloud) + strong perturbation + no-LLM judge |
| R | Multi-agent + random winner + strong perturbation — isolates selection from averaging |

### Bench V3 audit fixes

- Judge temperature fixed at 0.5 (no adaptive drift)
- Anti-stagnation disabled for all conditions
- num_ctx=32768 for all models
- SingleDraftJudge replaced by NoLLMSingleDraftJudge (no LLM call)
- All model families distinct across roles
- judge_temp_history and analysis_notes in output JSON

### Tick indexing convention

- `bench loop`: `tick in range(total_ticks)` → **0-indexed** (0 to 49)
- `orchestrator`: `self._tick_id` → **1-indexed** (1 to 50)
- `perturbation_log["tick"]` → 0-indexed (bench loop)
- `events.jsonl tick_id` → 1-indexed (orchestrator)
- `sim_curves` keys (`"tick_15"`, `"tick_35"`) → 0-indexed (bench loop)
- All analysis code must use bench loop index as reference.

### Usage

```bash
# Full bench V3 (42 runs)
python organism_v2/bench_v2.py --conditions A,B,C,E,E_B,E_C,R \
  --seeds 42,123,456,7,77,777 --ticks 50 --output-dir runs/bench_v3/

# Dry-run
python organism_v2/bench_v2.py --dry-run --conditions A,C,E,R --seeds 42 \
  --output-dir runs/bench_v3/
```

## Viewer (`viewer_v3.py`)

Flask + Three.js 3D trajectory viewer.

```bash
python organism_v2/viewer_v3.py --runs-dir runs/bench_v3/ --port 8767
```

Features: PCA/UMAP/DIST-3D projection, connectome, metric coloring, trail fading, distance matrix, mouse controls (drag=rotate, wheel=zoom), Mean-Selected gap indicator.

## Known limitations

### Connectome is not comparable across conditions with different agent counts

The connectome computes cosine similarity between `state_vector_mean` at different ticks. In multi-agent conditions (A, B, C), `state_vector_mean` is the average of 3 agent embeddings. This averaging smooths the vector, producing higher inter-tick similarity (~0.76 mean) and dense connectome graphs (400+ edges at threshold 0.80).

In single-agent conditions (E, E_B, E_C), `state_vector_mean` is the raw embedding of a single agent. No smoothing occurs, leading to lower inter-tick similarity (~0.50 mean) and sparse connectomes (30-60 edges at threshold 0.80).

This is a structural artefact of the averaging, not a signal about consciousness or resilience. The connectome is a valid intra-condition visualization tool, but **not a valid inter-condition metric** when agent counts differ.

### CCV is zero for single-agent conditions

Claim Cosine Variance (CCV) measures pairwise cosine distance between agent drafts within a tick. With a single agent (E, E_B, E_C), CCV is always 0.0 by definition.

### sv_selected metrics are winner-dependent

draft_velocity and PSV use sv_selected which depends on winner identity. In condition R (random winner), these reflect random switches, not semantic evolution. Use for intra-condition analysis only.

### Perturbation output varies across seeds

Perturbation operators call the LLM on the winning draft from the previous tick. Output varies significantly across seeds. Use multiple seeds and average to isolate the perturbation effect.
