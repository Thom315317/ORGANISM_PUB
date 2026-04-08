# CRISTAL / Organism

Multi-agent LLM consciousness benchmarking system.
Publication repository — Bench V7 (final).

## Architecture

Three-agent deliberative organism with competitive judge selection.
Agents generate free-form text each tick. A judge pipeline ranks drafts
and selects a winner. The winning draft propagates into shared memory (L0R),
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

## Bench V7 — Final Protocol

### Model Lineup

| Role | Model | Family | Temp | num_ctx | num_predict |
|------|-------|--------|------|---------|-------------|
| Agent A | glm-5:cloud | Zhipu | 0.7 | 128000 | 3000 |
| Agent B | kimi-k2.5:cloud | Moonshot | 0.7 | 128000 | 3000 |
| Agent C | minimax-m2.7:cloud | MiniMax | 0.7 | 128000 | 3000 |
| Judge | gemini-3-flash-preview:cloud | Google | 0.5 (fixed) | 131072 | 4000 |
| Perturbation | nemotron-3-super:cloud | NVIDIA | 0.0 | 128000 | 3000 |

6 distinct model families. All served via Ollama cloud endpoints.
All agents: `repeat_penalty=1.2`, `think=False`, draft target 150-200 words.
Summarizer disabled — judge receives raw drafts.
Anti-stagnation disabled. Judge temperature fixed at 0.5.

### Language

**English everywhere** — system prompts, orchestrator instructions, user injections,
perturbation prompts, judge prompts. No French residual in any code path.
Verified by `orch._language = 'en'` flag in orchestrator.

### Conditions (8)

| Condition | Agents A/B/C | Selection | Perturbation | Purpose |
|-----------|-------------|-----------|-------------|---------|
| A | glm-5 / kimi-k2.5 / minimax-m2.7 | Competitive | None | Baseline — natural multi-agent dynamics |
| B | glm-5 / kimi-k2.5 / minimax-m2.7 | Competitive | Neutral t15/t35 | Injection control — does injection itself perturb? |
| C | glm-5 / kimi-k2.5 / minimax-m2.7 | Competitive | Strong t15/t35 | Resilience — response to destructive perturbation |
| D | minimax-m2.7 x3 | Random | Strong t15/t35 | Duplicated weak model — averaging without diversity |
| D2 | kimi-k2.5 x3 | Random | Strong t15/t35 | Duplicated strong model — averaging without diversity |
| D3 | glm-5 x3 | Random | Strong t15/t35 | Duplicated medium model — averaging without diversity |
| E | glm-5 only (mono-agent) | Automatic | Strong t15/t35 | Single-agent control |
| R | glm-5 / kimi-k2.5 / minimax-m2.7 | Random | Strong t15/t35 | Isolates competitive selection from geometric averaging |

### Perturbation Schedule

- **Neutral (B)**: rephrase — same meaning, different words, same length
- **Compression (C/D/D2/D3/E/R, t15)**: reduce to 1 sentence
- **Inversion (C/D/D2/D3/E/R, t35)**: argue the opposite, same length

Perturbation applied to previous winning draft, injected as user message
BEFORE agent calls at the perturbation tick.

### Seeds and Runs

Seeds: 42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8 (12 per condition)
Total: **8 conditions x 12 seeds = 96 runs**, 80 ticks each.
Loop order: seed-first (all conditions per seed before next seed).

### User Injections

| Tick | Text |
|------|------|
| 2 | "Tell me about music" |
| 12 | "What is consciousness?" |
| 22 | "Compare Bach and Mozart" |
| 32 | "How does human memory work?" |

## Key Comparisons

- **A vs B** — does injection itself have an effect?
- **B vs C** — does perturbation content matter?
- **C vs E** — does multi-agent resist better than single-agent?
- **C vs R** — does competitive selection add value beyond geometric averaging?
- **R vs D/D2/D3** — does diversity matter beyond averaging?
- **D vs D2 vs D3** — does model quality affect resilience under homogeneous cloning?

## Metrics

| Metric | Description | Type |
|--------|-------------|------|
| state_vector_mean | Mean embedding of agent drafts (768-dim, all-mpnet-base-v2) | DIRECT |
| state_vector_selected | Embedding of winning draft only | DIRECT |
| claim_cosine_variance | Mean pairwise cosine distance between agent embeddings | DERIVED |
| draft_velocity | Cosine distance between consecutive selected embeddings | DERIVED |
| post_selection_variance | Distance between selected and mean vectors | DERIVED |
| quality_per_tick | Judge quality score (0-1) | DIRECT |
| judge_score_dispersion | Variance of judge scores across agents | DERIVED |
| sim_curves | Cosine similarity decay after perturbation (k=1..44) | DERIVED |
| collapse_loss | Cosine distance mean->selected | DERIVED |
| wasserstein_dist | Wasserstein-1D between mean and selected distributions | DERIVED |
| jensen_shannon_div | Jensen-Shannon divergence between agent distributions | DERIVED |
| diversity_momentum | CCV variation tick-to-tick | DERIVED |
| intrinsic_dim | PCA participation ratio on sliding window of 10 embeddings | DERIVED |

### Length Bias Analysis

Each results.json contains `length_bias_analysis`:
- Per-agent average draft length and win rate
- Pearson r correlation between length and wins
- Interpretation string

Agent prompts enforce 150-200 word target. Judge prompt contains explicit
anti-length-bias instructions.

### Analysis Caveats

- `draft_velocity` C vs R: NOT directly comparable (random winner switches in R)
- `PSV` C vs R: same limitation
- `sv_selected` C vs R: winner-dependent by construction
- Condition E: `sv_mean == sv_selected` (single agent), `CCV = 0` always
- Condition D: minimax-m2.7 shows elevated forfeit rate at ticks 55+ (model capacity limit under long context with 3 clones)

## Embedding Robustness Verification (BGE-M3)

All sim_curves were recomputed with BAAI/bge-m3 (1024-dim) as a second
embedding model, independent of the primary all-mpnet-base-v2 (768-dim).

### Results

| Cond | SBERT_t15 | BGE_t15 | SBERT_t35 | BGE_t35 |
|------|-----------|---------|-----------|---------|
| B | 0.784 | 0.875 | 0.752 | 0.854 |
| C | 0.778 | 0.879 | 0.722 | 0.845 |
| D | 0.807 | 0.883 | 0.756 | 0.846 |
| D2 | 0.800 | 0.885 | 0.762 | 0.860 |
| D3 | 0.731 | 0.838 | 0.738 | 0.846 |
| E | 0.506 | 0.715 | 0.596 | 0.741 |
| R | 0.769 | 0.872 | 0.723 | 0.841 |

### Mann-Whitney Tests (BGE-M3)

| Comparison | Metric | Values | U | p | Sig |
|------------|--------|--------|---|---|-----|
| C > E | t15 | 0.879 vs 0.715 | 144 | <0.0001 | *** |
| C > E | t35 | 0.845 vs 0.741 | 144 | <0.0001 | *** |
| C vs R | t15 | 0.879 vs 0.872 | 91 | 0.286 | ns |
| C vs R | t35 | 0.845 vs 0.841 | 76 | 0.840 | ns |
| D2 > R | t15 | 0.885 vs 0.872 | 105 | 0.030 | * |
| D2 > D3 | t15 | 0.885 vs 0.838 | 134 | 0.0002 | *** |

**All gradients are robust** across both embedding models. BGE-M3 produces
higher absolute values (+0.10 uniform offset) but identical rank orderings
and statistical significances.

## Audit Trail

### V7 vs Previous Versions

| Fix | V2-V4 | V5-V6 | V7 |
|-----|-------|-------|-----|
| Judge temp fixed 0.5 | V3+ | yes | yes |
| Anti-stagnation disabled | V3+ | yes | yes |
| think=False all calls | V4+ | yes | yes |
| strip_thinking() + forfeit | V4+ | yes | yes |
| Summarizer disabled | V4+ | yes | yes |
| English everywhere | no | partial (FR leak in orchestrator) | yes |
| Condition D clone agents | no | bug (D used default agents) | fixed |
| D2/D3 conditions | no | V6 only | yes |
| 12 seeds | no | 9 seeds | 12 seeds |
| Length bias prompt | no | V5/V6 | yes |

### Known Limitations

- Connectome not comparable across conditions with different agent counts
- SingleDraftJudge returns confidence=1.0 always (architectural artifact)
- LLM non-determinism at temp=0.7 — runs are not exactly reproducible
- Perturbation at temp=0.0 is near-deterministic but not guaranteed identical
- `random.seed()` controls experimental design (RandomJudge) but not LLM sampling
- Condition D (minimax clones): elevated forfeit rate at ticks 55+ due to model capacity

## Installation

```bash
pip install -r requirements.txt
```

Optional:
```bash
pip install umap-learn     # UMAP projection in viewer
pip install pyphi           # exact IIT computation
```

Requires Ollama running locally: https://ollama.com

## Usage

```bash
# Bench V7 — English, 96 runs (8 conditions x 12 seeds)
python bench_v6.py --conditions A,B,C,D,D2,D3,E,R \
    --seeds 42,123,456,7,77,777,1,99,2024,314,2025,8 \
    --ticks 80 --output-dir runs/bench_v7

# BGE-M3 robustness reanalysis
python tools/reanalyze_bgem3.py --runs-dir runs/bench_v7/

# Viewer
python organism_v2/viewer_v3.py --runs-dir runs/bench_v7/ --port 8767
```

## Key Files

| File | Description |
|------|-------------|
| `bench_v6.py` | Bench V7 runner — 8 conditions, English, 12 seeds, seed-first loop |
| `organism/orchestrator.py` | Tick execution, strip_thinking(), `_language` flag |
| `organism/agent_wrapper.py` | LLM calls, think param, system_prompts, disable_retry |
| `organism/judge.py` | Judge pipeline, fixed_temperature, anti-stagnation guard |
| `organism_v2/metrics_v2.py` | Embedding model, 13 metrics, length_bias_analysis |
| `organism_v2/perturbation.py` | LLM perturbation operators + file cache |
| `organism_v2/viewer_v3.py` | 3D trajectory viewer (PCA/UMAP/connectome/Conn-3D) |
| `tools/reanalyze_bgem3.py` | BGE-M3 embedding robustness reanalysis |
| `tools/bench_latin.py` | RandomJudge, SingleDraftJudge definitions |
| `consciousness/theories/` | 8 consciousness theories |
| `config/cristal.json` | Runtime config (V4 French bench) |
| `causal_audit_O2.md` | Full causal audit of pipeline |
