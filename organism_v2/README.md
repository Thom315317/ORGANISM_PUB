# Organism V2 — Bench & Viewer

## Bench (`bench_v2.py`)

Runs 7 conditions x N seeds x 80 ticks.

| Condition | Description |
|-----------|-------------|
| A | Multi-agent, no perturbation — baseline |
| B | Multi-agent + neutral perturbation (rephrase) at t15/t35 |
| C | Multi-agent + strong perturbation (compression t15, inversion t35) |
| E | Single-agent A (minimax-m2.7:cloud) + strong perturbation + no-LLM judge |
| E_B | Single-agent B (kimi-k2.5:cloud) + strong perturbation + no-LLM judge |
| E_C | Single-agent C (glm-5:cloud) + strong perturbation + no-LLM judge |
| R | Multi-agent + random winner + strong perturbation — isolates selection from averaging |

### Bench V4 changes

- **think=False for all LLM calls**: agents, judge, perturbation. No thinking mode.
- **strip_thinking() pipeline**: safety net — strips residual reasoning traces before judge/embeddings/L0R. Thinking text preserved in events.jsonl for audit.
- **Draft discard**: drafts <20 chars post-strip treated as forfeit (thinking_only). Agent skips tick, not sent to judge. Logged in `discarded_drafts` field.
- **Summarizer disabled**: judge receives raw drafts directly (num_ctx=131072 sufficient). Nemotron code preserved but bypassed via `disable_summarizer=True`.
- **num_predict=1500 agents, 4000 judge**. num_ctx=65536 agents, 131072 judge.
- **80 ticks per run** (was 50). Injection at t42 removed. 4 standard injections remain (t2, t12, t22, t32).
- **bench_version=v4**, output directory bench_v4.
- **Condition F removed**: 7 conditions remain (A, B, C, E, E_B, E_C, R).
- **k_max=44** for sim_curves computation (was 30).
- **Perturbation model**: gpt-oss:120b-cloud, num_predict=3000, num_ctx=65536, think=False. Neutral prompt includes length constraint.
- **System prompts**: competitive debate framing, sanctions for non-conforming responses (identical block for A/B/C).
- **_anon_map** in tick_end events for anonymization audit.
- **Anonymization confirmed**: all judge prompts use numeric labels (1/2/3).

### Model lineup (bench V4)

| Role | Model | Family |
|------|-------|--------|
| Agent A | minimax-m2.7:cloud | MiniMax |
| Agent B | kimi-k2.5:cloud | Moonshot |
| Agent C | glm-5:cloud | Zhipu |
| Judge | gemini-3-flash-preview:cloud | Google |
| Perturbation | gpt-oss:120b-cloud | OpenAI-oss |

5 distinct families. No shared weights or architecture.

### Bench V3 audit fixes (retained)

- Judge temperature fixed at 0.5 (no adaptive drift)
- Anti-stagnation disabled for all conditions
- SingleDraftJudge replaced by NoLLMSingleDraftJudge (no LLM call)
- judge_temp_history and analysis_notes in output JSON

### Tick indexing convention

- `bench loop`: `tick in range(total_ticks)` → **0-indexed** (0 to 79)
- `orchestrator`: `self._tick_id` → **1-indexed** (1 to 80)
- `perturbation_log["tick"]` → 0-indexed (bench loop)
- `events.jsonl tick_id` → 1-indexed (orchestrator)
- `sim_curves` keys (`"tick_15"`, `"tick_35"`) → 0-indexed (bench loop)
- All analysis code must use bench loop index as reference.

### Usage

```bash
# Full bench V4 (21 runs, 80 ticks each)
python organism_v2/bench_v2.py --conditions A,B,C,E,E_B,E_C,R \
  --seeds 42,123,456 --ticks 80

# Dry-run
python organism_v2/bench_v2.py --dry-run --conditions A,C,E,R --seeds 42
```

## Viewer (`viewer_v3.py`)

Flask + Three.js 3D trajectory viewer.

```bash
python organism_v2/viewer_v3.py --runs-dir runs/bench_v4/ --port 8767
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
