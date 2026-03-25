# CRISTAL / Organism

Multi-agent LLM consciousness benchmarking system.
Publication repository for [paper title TBD].

## Architecture

Three-agent deliberative organism with competitive judge selection.
Agents generate free-form text each tick. A judge pipeline (summarizer + ranker)
selects a winner. The winning draft propagates into shared memory (L0R),
influencing future prompts. Perturbation operators inject transformed text
at scheduled ticks to test system resilience.

### Tick Execution Order

1. Perturbation injection (if scheduled at this tick)
2. Prompt building (evidence pack from L0R + world model + user messages)
3. Agent calls (A, B, C sequentially, `think=False`)
4. `strip_thinking()` safety net (removes any residual reasoning traces)
5. Draft discard: drafts <20 chars post-strip = forfeit (agent skips tick)
6. Judge (gemini) ranks and selects winner (raw drafts, no summarizer)
7. Winner propagates to L0R at salience=1.0
8. Metrics computed (embeddings, CCV, velocity, PSV, sim_curves)
9. Results flushed to JSON

### Memory Systems

- **L0R** (Layer 0 Register) — short-term working memory, winner draft at salience=1.0
- **World Model (WM)** — accumulated factual claims
- **Message Registry (MR)** — blockchain-like event log with hash chain

Feedback loop: winner -> L0R -> next prompt -> future winner.
This is the mechanism under study, not a confound.

## Bench V4 — Model Lineup

| Role | Model | Family | Temp | num_ctx | num_predict |
|------|-------|--------|------|---------|-------------|
| Agent A | minimax-m2.7:cloud | MiniMax | 0.7 | 65536 | 1500 |
| Agent B | kimi-k2.5:cloud | Moonshot | 0.7 | 65536 | 1500 |
| Agent C | glm-5:cloud | Zhipu | 0.7 | 65536 | 1500 |
| Judge | gemini-3-flash-preview:cloud | Google | 0.5 (fixed) | 131072 | 4000 |
| Perturbation | gpt-oss:120b-cloud | OpenAI-oss | 0.0 | 65536 | 3000 |

All models served via Ollama cloud endpoints. 5 distinct model families.
All agents use `repeat_penalty=1.2`, `think=False`.
Summarizer (nemotron-3-super:cloud) disabled — judge receives raw drafts.

### Thinking Mode

All LLM calls use `think=False`. Safety net:
1. `strip_thinking()` removes any leaked reasoning traces from content
2. Drafts <20 chars post-strip are discarded (agent forfeits tick)
3. System prompts explicitly penalize non-conforming responses

Discarded drafts are logged in `discarded_drafts` field and `events.jsonl`.

## Conditions

| Condition | Agents | Judge | Perturbation | Purpose |
|-----------|--------|-------|-------------|---------|
| A | A,B,C | Competitive (gemini) | None | Baseline — natural multi-agent dynamics |
| B | A,B,C | Competitive (gemini) | Neutral t15/t35 | Injection control — does injection itself perturb? |
| C | A,B,C | Competitive (gemini) | Strong t15/t35 | Resilience — response to destructive perturbation |
| E | A only | No-LLM auto | Strong t15/t35 | Single-agent control (slot A, minimax-m2.7) |
| E_B | B only | No-LLM auto | Strong t15/t35 | Single-agent control (slot B, kimi-k2.5) |
| E_C | C only | No-LLM auto | Strong t15/t35 | Single-agent control (slot C, glm-5) |
| R | A,B,C | Random winner | Strong t15/t35 | Isolates competitive selection from geometric averaging |

### Perturbation Schedule

- **Neutral (B)**: rephrase — same meaning, different words, same length
- **Compression (C/E/E_B/E_C/R, t15)**: reduce to 1 sentence
- **Inversion (C/E/E_B/E_C/R, t35)**: argue the opposite

Perturbation applied to previous winning draft, injected as user message
BEFORE agent calls at the perturbation tick.

### Seeds and Runs

Seeds: 42, 123, 456, 7, 77, 777 (6 per condition)
Total: 7 conditions x 6 seeds = 42 runs, 80 ticks each

## Key Comparisons

- **A vs B** — does injection itself have an effect? (if B ~ A, injection is neutral)
- **B vs C** — does perturbation content matter? (if C != B, content matters)
- **C vs E/E_B/E_C** — does multi-agent resist better than single-agent?
- **C vs R** — does competitive selection add value beyond geometric averaging?
- **E vs E_B vs E_C** — model size/family effect on single-agent resilience

## Metrics

| Metric | Description | Type |
|--------|-------------|------|
| state_vector_mean | Mean embedding of 3 agent drafts (768-dim) | DIRECT |
| state_vector_selected | Embedding of winning draft only | DIRECT |
| claim_cosine_variance | Cosine variance between agent embeddings | DERIVED |
| draft_velocity | Cosine distance between consecutive selected embeddings | DERIVED |
| post_selection_variance | Distance between selected and mean vectors | DERIVED |
| quality_per_tick | Judge quality score (0-1) | DIRECT |
| judge_score_dispersion | Variance of judge scores across agents | DERIVED |
| sim_curves | Cosine similarity decay after perturbation (k=1..k_max) | DERIVED |

### Analysis Caveats

- `draft_velocity` C vs R: NOT directly comparable (random winner switches in R vs coherent sequence in C)
- `PSV` C vs R: same limitation
- `sv_selected` C vs R: winner-dependent by construction, account for structural difference
- Condition E: `sv_mean == sv_selected` by construction (single agent)
- Condition E: `CCV = 0` always (single agent, no variance)

## Audit Fixes (V4 vs V3 vs V2)

### V3 Fixes (retained in V4)
- Judge temperature fixed at 0.5 (no adaptive drift)
- Anti-stagnation disabled (no asymmetry between C and R)
- SingleDraftJudge: no LLM call (eliminates confound in E conditions)
- All model families distinct (no role sharing)
- judge_temp_history logged in output JSON
- analysis_notes embedded in every results.json

### V4 Fixes
- `think=False` for all LLM calls (agents, judge, perturbation)
- `strip_thinking()` safety net + draft discard (<20 chars = forfeit)
- Summarizer disabled — judge receives raw drafts directly
- System prompts: competitive debate framing, sanctions for non-conforming responses
- `_anon_map` in tick_end events for anonymization audit
- `num_ctx` explicit: 65536 (agents/perturbation), 131072 (judge)
- `num_predict=1500` for agents (was 4000), 4000 for judge
- 80 ticks per run (was 50)
- Injection at t42 removed (was present in V3 DEFAULT_INJECTIONS)
- Perturbation neutral prompt enforces same-length output
- Perturbation model: gpt-oss:120b-cloud (was minimax-m2.7)
- Agent A: minimax-m2.7:cloud (was glm-5:cloud — 503 errors)
- bench_version field: "v4" in all output JSON
- k_max=44 for sim_curves (was 14)

## Connectome (Viewer)

Cosine similarity edges between tick pairs in raw 768-dim embedding space.
Threshold: > 0.80, |i-j| > 1. Computed on state_vector_mean only.

**Important**: connectome reflects temporal semantic cohesion of the collective
mean vector — not directly comparable across conditions with different agent
counts (multi-agent averaging produces structurally higher inter-tick similarity).

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
# Full bench V4 (42 runs, 80 ticks)
python organism_v2/bench_v2.py --conditions A,B,C,E,E_B,E_C,R \
  --seeds 42,123,456,7,77,777 --ticks 80 --output-dir runs/bench_v4/

# Viewer
python organism_v2/viewer_v3.py --runs-dir runs/bench_v4/ --port 8767
```

## Key Files

| File | Description |
|------|-------------|
| `organism/orchestrator.py` | Tick execution, agent calls, strip_thinking(), judge integration |
| `organism/agent_wrapper.py` | LLM calls (think=False), system prompts, response parsing |
| `organism/judge.py` | Judge pipeline (summarizer disabled), fixed temperature, anti-stagnation guard |
| `organism/scheduler.py` | Mode selection (Idle/Explore/Debate/Implement/Recover) |
| `organism/l0r.py` | Layer 0 Register — short-term memory, winner propagation |
| `organism/mr.py` | Message Registry — blockchain event log |
| `organism/world_model.py` | Accumulated factual claims |
| `organism_v2/bench_v2.py` | Bench runner, 7 conditions, metrics collection, skip-existing |
| `organism_v2/metrics_v2.py` | Embedding model (all-mpnet-base-v2), CCV, velocity, PSV |
| `organism_v2/perturbation.py` | LLM transform operators + file cache |
| `organism_v2/viewer_v3.py` | 3D trajectory viewer (PCA/UMAP/connectome/DIST-3D) |
| `consciousness/theories/` | 8 consciousness theories (MDM, GWT, HOT, IIT, FEP, DYN, RPT, Hybrid) |
| `tools/bench_latin.py` | Organism V1 Latin Square bench, RandomJudge, SingleDraftJudge |
| `causal_audit_O2.md` | Full causal audit of the pipeline |

## Known Limitations

- Connectome not comparable across conditions with different agent counts (averaging artifact)
- SingleDraftJudge returns confidence=1.0 always (architectural artifact, not quality signal)
- LLM non-determinism at temp=0.7 means runs are not exactly reproducible
- Perturbation at temp=0.0 is near-deterministic but not guaranteed identical across reruns
- `random.seed()` controls experimental design (RandomJudge) but not LLM sampling
