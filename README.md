# CRISTAL / Organism

Multi-agent LLM consciousness benchmarking system.
Publication repository for [paper title TBD].

## Architecture

Three-agent deliberative organism with competitive judge selection.
Agents generate free-form text each tick. A judge pipeline (summarizer + ranker)
selects a winner. The winning draft propagates into shared memory (L0R),
influencing future prompts. Perturbation operators inject transformed text
at scheduled ticks to test system resilience.

## Bench V3 — Model Lineup

| Role | Model | Temp | num_ctx |
|------|-------|------|---------|
| Agent A | glm-5:cloud | 0.7 | 32768 |
| Agent B | kimi-k2.5:cloud | 0.7 | 32768 |
| Agent C | qwen3.5:397b-cloud | 0.7 | 32768 |
| Judge | gemini-3-flash-preview:cloud | 0.5 (fixed) | 32768 |
| Summarizer | nemotron-3-super:cloud | 0.3 | 32768 |
| Perturbation | minimax-m2.7:cloud | 0.0 | 32768 |

All models served via Ollama cloud endpoints. No two roles share a model family.

## Conditions

| Condition | Agents | Judge | Perturbation | Purpose |
|-----------|--------|-------|-------------|---------|
| A | A,B,C | Competitive | None | Baseline |
| B | A,B,C | Competitive | Neutral t15/t35 | Injection control |
| C | A,B,C | Competitive | Strong t15/t35 | Resilience |
| E | A only | No-LLM auto | Strong t15/t35 | Single-agent control (slot A) |
| E_B | B only | No-LLM auto | Strong t15/t35 | Single-agent control (slot B) |
| E_C | C only | No-LLM auto | Strong t15/t35 | Single-agent control (slot C) |
| R | A,B,C | Random winner | Strong t15/t35 | Isolates selection from averaging |

Seeds: 42, 123, 456, 7, 77, 777 (6 per condition, 42 total runs, 50 ticks each)

## Audit Fixes (V3 vs V2)

- Judge temperature fixed at 0.5 (no adaptive drift)
- Anti-stagnation disabled (no bias correction asymmetry between C and R)
- num_ctx=32768 explicit for all models (prevents silent truncation)
- SingleDraftJudge: no LLM call (eliminates confound in E conditions)
- All model families distinct (no role sharing)
- judge_temp_history logged in output JSON
- analysis_notes embedded in every results.json

## Installation

```bash
pip install -r requirements.txt
```

Optional:
```bash
pip install umap-learn   # UMAP projection in viewer
pip install pyphi        # exact IIT computation
```

Requires Ollama running locally: https://ollama.com

## Usage

```bash
# Full bench (42 runs)
python organism_v2/bench_v2.py --conditions A,B,C,E,E_B,E_C,R \
  --seeds 42,123,456,7,77,777 --ticks 50 --output-dir runs/bench_v3/

# Viewer
python organism_v2/viewer_v3.py --runs-dir runs/bench_v3/ --port 8767
```

## Key Files

- `organism/orchestrator.py` — tick execution, agent calls, judge integration
- `organism/judge.py` — summarizer + judge pipeline, fixed temperature
- `organism/scheduler.py` — mode selection (Idle/Explore/Debate/Implement/Recover)
- `organism_v2/bench_v2.py` — bench runner, conditions, metrics collection
- `organism_v2/metrics_v2.py` — embedding-based time series (CCV, velocity, PSV)
- `organism_v2/perturbation.py` — LLM transform operators (neutral, compression, inversion)
- `organism_v2/viewer_v3.py` — 3D trajectory viewer (PCA/UMAP/DIST-3D)
- `causal_audit_O2.md` — full causal audit of the pipeline
