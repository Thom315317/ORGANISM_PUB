# CRISTAL / Organism

Multi-agent LLM consciousness benchmarking system.
Publication repository — Bench V9 (final).

## Research question

Does structured persistent memory in a multi-agent LLM pipeline slow semantic drift
after an exogenous perturbation, beyond what is explained by multi-sampling +
shared context + selection?

## Design

Three-agent deliberative organism. At each tick, agents generate free-form text.
A judge ranks drafts and selects a winner. The winning draft propagates into shared
memory (L0R ring buffer + World Model) and influences future prompts. At scheduled
ticks, perturbations inject exogenous content to test system resilience.

## V9 vs V7 — Key changes

| Change | V7 | V9 |
|---|---|---|
| Perturbation | LLM compression/inversion | OOD pre-selected texts (no LLM) |
| Perturbation ticks | t15 / t35 | t20 / t40 |
| Measurement windows | 16-34 / 36-79 (contains injections) | 21-29 / 41-49 (zero injection) |
| Injection ticks | t2, t12, t22, t32 | t2, t10, t30 (all outside windows) |
| Condition I (new) | — | 3× glm-5, L0R/WM recreated each tick, raw context (last 3 drafts) |
| Seeds | 12 | 20 |
| Conditions | 8 | 9 (+ I) |
| Total runs | 96 | 180 |
| OOD cosine distance validation | — | >0.30 required |
| k_max | 44 | 9 |

## Tick execution order

1. Perturbation injection (if scheduled at this tick)
2. Prompt building (evidence pack from L0R + world model + user messages)
3. Agent calls (A, B, C sequentially, `think=False`)
4. `strip_thinking()` safety net (removes residual reasoning traces)
5. Draft discard: drafts <20 chars post-strip = forfeit
6. Judge (gemini) ranks and selects winner
7. Winner propagates to L0R at salience=1.0
8. Metrics computed (embeddings, CCV, velocity, PSV, sim_curves)
9. Results flushed to JSON

For condition I: L0R ring and WM claims are cleared before each tick. Agents
receive the last 3 winning drafts as raw text via `inject_user_message`, never
as structured L0R slots.

## Model lineup

| Role | Model | Family |
|------|-------|--------|
| Agent A | glm-5:cloud | Zhipu |
| Agent B | kimi-k2.5:cloud | Moonshot |
| Agent C | minimax-m2.7:cloud | MiniMax |
| Judge | gemini-3-flash-preview:cloud | Google |
| Perturbation neutral (condition B) | nemotron-3-super:cloud | NVIDIA |
| OOD perturbation | pre-selected texts (no LLM) | — |
| D clones | minimax-m2.7:cloud × 3 | MiniMax |
| D2 clones | kimi-k2.5:cloud × 3 | Moonshot |
| D3 clones | glm-5:cloud × 3 | Zhipu |
| I clones | glm-5:cloud × 3 | Zhipu |

All models via Ollama cloud. num_ctx=128000, num_predict=3000 (agents) / 4000 (judge).
Temperature=0.7 (agents), 0.5 fixed (judge). `think=False` everywhere.
Summarizer disabled. Anti-stagnation disabled.

## Conditions (9)

| Cond | Agents | Selection | Perturbation | Purpose |
|------|--------|-----------|-------------|---------|
| A | glm-5 / kimi / minimax | Competitive | None | Baseline |
| B | glm-5 / kimi / minimax | Competitive | Neutral (rephrase) t20/t40 | Injection control |
| C | glm-5 / kimi / minimax | Competitive | OOD t20/t40 | Resilience |
| D | minimax × 3 | Random | OOD | Weak model clones |
| D2 | kimi × 3 | Random | OOD | Strong model clones |
| D3 | glm-5 × 3 | Random | OOD | Medium model clones |
| E | glm-5 only | Automatic | OOD | Mono-agent control |
| R | glm-5 / kimi / minimax | Random | OOD | Random selection control |
| **I** | glm-5 × 3 | Random | OOD | **No structured memory — raw context only** |

## Seeds

Pre-registered 12: 42, 123, 456, 7, 77, 777, 1, 99, 2024, 314, 2025, 8
Supplementary 8: 13, 55, 101, 256, 512, 1000, 1337, 2026

Total: 9 × 20 = **180 runs**, 80 ticks each.

## OOD perturbation

At t20 and t40, the previous winning draft is replaced by a pre-selected text
from `ood_texts_final.json`. 20 expository + 20 narrative texts on topics
with no semantic overlap with music, consciousness, Bach, Mozart, or memory.

Assignment is deterministic per seed (see `OOD_ASSIGNMENT` in `bench_v9.py`).
Each seed receives a unique (P1, P2) pair. No recycling.

Validation: cosine distance between pre-perturbation draft and OOD text must
be > 0.30. All 360 perturbations (180 runs × 2) passed this check.

## Metrics

| Metric | Description |
|--------|-------------|
| `state_vector_mean` | Mean embedding of agent drafts (768-dim, all-mpnet-base-v2) |
| `state_vector_selected` | Embedding of winning draft only |
| `sim_curves` | Cosine similarity to R_pre = mean(sv[t-3], sv[t-2], sv[t-1]) for k=1..9 |
| `claim_cosine_variance` | Pairwise cosine distance between agent embeddings |
| `draft_velocity` | Cosine distance between consecutive selected embeddings |
| `post_selection_variance` | Distance between selected and mean vectors |
| `perturbation_log` | OOD text ID, cosine_distance_to_draft validation |
| `winner_log` | Per-tick winner (for random selection auditing) |

## Primary endpoints

**Spearman ρ on t40_sel (slope):** does the selected draft drift back toward
pre-perturbation reference over ticks 41-49? ρ>0 = recovery, ρ<0 = drift.

**Mean t40_sel (level):** average cosine similarity across k=1..9.

## Statistical plan (pre-registered)

Wilcoxon signed-rank paired by seed, one-sided. Matched-pairs rank-biserial r.
Holm-Bonferroni per family:

- **Family A (primary):** 7 tests on slope (D3>E, C>E, R>E, D>E, D2>E, D3>I, I>E)
- **Family B (confirmatory):** 7 tests on level
- **Family C (competition):** 2 tests (C>R slope + level)

## Results

See `runs/bench_v9/v9_final_report.txt` for the full output.

### Family B — Level (all significant after Holm)

| Comparison | mean_diff | r_rb | p_corr |
|------------|-----------|------|--------|
| C > E | +0.260 | -0.99 | <0.001 *** |
| D > E | +0.266 | -0.97 | <0.001 *** |
| D2 > E | +0.299 | -0.99 | <0.001 *** |
| R > E | +0.237 | -0.99 | <0.001 *** |
| I > E | +0.205 | -0.98 | <0.001 *** |
| D3 > E | +0.175 | -0.80 | 0.001 *** |
| **D3 > I** | **-0.030** | +0.15 | **0.727 ns** |

### Family A — Slope

No comparison significant after Holm (p_corr 0.27–0.49). Directional signal
present (E drifts most: ρ=-0.41; C least: ρ=-0.25) but variance too large.

### Family C — Competition

- C vs R slope: diff=0.009, p=0.47 **ns**
- C vs R level: diff=0.023, p=0.47 **ns**

### Interpretation

**H1 confirmed:** all multi-agent conditions outperform mono-agent (E) on the
level endpoint, p<0.001 after correction. Medium-to-large effects (r_rb ≈ 0.8-0.99).

**H2 rejected:** D3 ≤ I on level (p=0.727 ns). Structured memory (L0R + WM)
does not add measurable value above raw context injection (last 3 winning drafts
as plain text).

**Family C:** competitive selection (judge) does not outperform random selection
on either level or slope.

**Main claim (defensible):**

> The multi-agent advantage on post-perturbation persistence comes from
> multi-sampling + shared context propagation, not from structured memory
> or competitive selection. A trivial pipeline (3 independent calls + raw
> context history + random selection) reproduces the effect.

## Installation

```bash
pip install -r requirements.txt
```

Optional: `pip install umap-learn` for UMAP in the viewer.

Requires Ollama running locally: https://ollama.com

Required cloud models (accessed via `:cloud` suffix, no local download):
- glm-5:cloud
- kimi-k2.5:cloud
- minimax-m2.7:cloud
- gemini-3-flash-preview:cloud
- nemotron-3-super:cloud

## Usage

```bash
# Full bench V9 (180 runs)
python bench_v9.py \
    --conditions A,B,C,D,D2,D3,E,R,I \
    --seeds 42,123,456,7,77,777,1,99,2024,314,2025,8,13,55,101,256,512,1000,1337,2026 \
    --ticks 80 \
    --output-dir runs/bench_v9

# Analysis scripts (in tools/)
python tools/stats_reanalysis_v7.py --runs-dir runs/bench_v9    # adapt paths as needed
python tools/publication_figures.py --runs-dir runs/bench_v9

# Viewer
python organism_v2/viewer_v3.py --runs-dir runs/bench_v9 --port 8767
```

## Key files

| File | Description |
|------|-------------|
| `bench_v9.py` | Main runner — 9 conditions, 20 seeds, OOD perturbation, condition I |
| `ood_texts_final.json` | 40 pre-selected OOD texts (20 expository + 20 narrative) |
| `organism/orchestrator.py` | Tick execution, strip_thinking(), `_language` flag |
| `organism/agent_wrapper.py` | LLM calls, think param, system_prompts, disable_retry |
| `organism/judge.py` | Judge pipeline, fixed_temperature, anti-stagnation guard |
| `organism_v2/metrics_v2.py` | Embedding model, sim_curves, all tick-level metrics |
| `organism_v2/perturbation.py` | Neutral rephrase operator (condition B) |
| `organism_v2/viewer_v3.py` | 3D trajectory viewer (PCA/UMAP/connectome/Conn-3D) |
| `tools/` | Statistical reanalysis, baselines, figures, BGE-M3 robustness |
| `config/cristal.json` | Runtime config (V4 French bench, legacy) |

## Known limitations

- LLM non-determinism at temp=0.7 — runs are not exactly reproducible even with fixed seeds
- Slope endpoint (rho) underpowered at n=20; level is the reliable readout
- SingleDraftJudge (condition E) returns confidence=1.0 always (architectural artifact)
- Ollama cloud occasional DNS/CUDA failures during long runs; skip-existing handles recovery
