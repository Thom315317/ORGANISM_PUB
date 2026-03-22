# Causal Audit — Organism O2 Pipeline

Static reasoning audit. All references are to source files in the organism/ project root.

---

## SECTION 1 — EXECUTION GRAPH (TICK-LEVEL)

Exact execution order for one tick (bench loop iteration in bench_v2.py:217-276):

### Pre-tick (bench_v2.py)

| # | Step | File:Line | Inputs | Outputs | Side Effects |
|---|------|-----------|--------|---------|--------------|
| 0a | User injection check | bench_v2.py:218-220 | `DEFAULT_INJECTIONS[tick]` | — | `orch.inject_user_message()` → appends to `_user_messages`, writes MR event, inserts L0R slot (salience=1.0) |
| 0b | Perturbation check | bench_v2.py:222-236 | `condition`, `tick`, `prev_selected_draft` | `perturbed_text` | If pert tick: calls LLM (temp=0), `orch.inject_user_message()` → same side effects as 0a, appends to `perturbation_log` |

### Inside run_tick() (orchestrator.py:311-730)

| # | Step | File:Line | Inputs | Outputs | Side Effects |
|---|------|-----------|--------|---------|--------------|
| 1 | Tick ID increment | orchestrator.py:314 | `_tick_id` | `_tick_id += 1` | Veto budget tick, reset_tick_counter, wm.tick_id set |
| 2 | tick_start event | orchestrator.py:319-325 | tick_id, mode | MR event | MR append |
| 3 | Scheduler tick | orchestrator.py:328 | `_scheduler.signals` | `mode`, `mode_changed` | Updates `_scheduler.current_mode`, `_scheduler._dwell`, appends to `_scheduler.history` |
| 4 | L0R decay | orchestrator.py:343 | L0R ring slots | `expired` count | Decrements TTL on all slots, removes expired |
| 5 | Evidence pack build | orchestrator.py:344 | L0R ring (top-K by composite_score), MR | `pack` (EvidencePack) | Reads MR to resolve chunk_ids to text |
| 6 | Agent selection | orchestrator.py:347 | `mode` | `agents_to_call` list | — |
| 7 | **For each agent** (orchestrator.py:356-465): | | | | |
| 7a | Build prompt | orchestrator.py:358 | `_user_messages`, `pack`, `mode`, `params`, `_wm.get_facts()` | `prompt` string | Reads `_user_messages` (perturbation visible here) |
| 7b | Call agent_fn | orchestrator.py:361-362 | `agent_id`, `prompt`, `params` | `AgentTurn` (text, token_in, token_out, novelty, conflict, etc.) | LLM API call |
| 7c | Log to MR | orchestrator.py:370-383 | turn data | MR event | Writes event, sets `turn.chunk_id` |
| 7d | Insert to L0R | orchestrator.py:387-393 | chunk_id, salience, novelty | L0R slot | Only if turn has text |
| 7e | Web search | orchestrator.py:396-420 | turn.text (search tags) | search results | Disabled in bench_mode. Writes MR+L0R if active |
| 7f | Claims extraction | orchestrator.py:422-442 | turn.text | WM claims | Explicit claims + heuristic extraction (max 3 per agent) |
| 7g | Veto check | orchestrator.py:444-463 | turn.veto, veto_budget | veto state | If veto accepted: break loop (remaining agents not called) |
| 8 | **Judge pipeline** | orchestrator.py:467-507 | `judgeable_turns`, `_recent_winners` | `judge_verdict` | |
| 8a | If ≥2 turns: full judge | orchestrator.py:473-479 | turns, recent_winners | JudgeVerdict | Calls summarizer LLM + judge LLM (2 API calls), random.shuffle for anonymization |
| 8b | If 1 turn: default win | orchestrator.py:480-505 | sole turn | JudgeVerdict(confidence=0.5) | No LLM call, hardcoded signals |
| 8c | If 0 turns: no verdict | orchestrator.py:506-507 | — | None | — |
| 9 | Signal aggregation | orchestrator.py:509-558 | judge_verdict.signals OR agent signals | `ControlSignals` | Updates scheduler signals, computes prediction_error from 5-tick history |
| 10 | Post-judge writes | orchestrator.py:560-628 | judge_verdict, turns | — | Winner draft → L0R (salience=1.0), judge claims → WM, fallback claim extraction if judge produced 0 claims, track `_recent_winners` and `_recent_margins` (both capped at 50) |
| 11 | Adapt judge temperature | orchestrator.py:631-632 | `_recent_margins` | `_judge_temp` | Only if ≥5 margins. Adjusts temp ±0.05 based on margin variance |
| 12 | OrganismState snapshot | orchestrator.py:634-648 | all state | OrganismState | Read-only snapshot for theories |
| 13 | Theory computation | orchestrator.py:651-674 | OrganismState | theory_scores | 7 theories + Hybrid (Hybrid reads others' scores) |
| 14 | STEM accumulation | orchestrator.py:677-681 | OrganismState | — | State vector for STEM model |
| 15 | Save prev state | orchestrator.py:683-685 | signals, mode | `_prev_signals`, `_prev_mode` | For next tick's prediction_error |
| 16 | tick_end event | orchestrator.py:687-711 | all | MR event | — |
| 17 | Drain user messages | orchestrator.py:714 | `_user_messages` | empty list | **CRITICAL**: perturbation text consumed, not available at tick t+1 |

### Post-tick (bench_v2.py)

| # | Step | File:Line | Inputs | Outputs | Side Effects |
|---|------|-----------|--------|---------|--------------|
| 18 | Extract drafts | bench_v2.py:243-248 | result.agent_turns | `agent_drafts` dict | — |
| 19 | Extract winner | bench_v2.py:250-254 | result.judge_verdict | `winner_id` | — |
| 20 | Update prev_selected_draft | bench_v2.py:256-260 | winner_id, agent_drafts | `prev_selected_draft` | **CRITICAL**: source text for next perturbation |
| 21 | Record metrics | bench_v2.py:262 | agent_drafts, winner_id, verdict | TickMetrics state | Embeds all drafts, computes CCV, velocity, PSV, quality |
| 22 | Consecutive failures check | bench_v2.py:264-275 | active turns | consecutive_failures counter | Abort if 5 consecutive empty ticks |
| 23 | Live flush | bench_v2.py:277-292 | all metrics | results.json | Overwrites every tick |

---

## SECTION 2 — GLOBAL STATE MAP

### bench_v2.py

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `prev_selected_draft` | str | `""` (line 214) | After each tick (line 256-260) | Perturbation input (line 225) | [HIDDEN_STATE] [CONDITION_DEPENDENT] [NOT_LOGGED] |
| `perturbation_log` | list | `[]` (line 215) | On pert ticks (line 228) | Output JSON | |
| `consecutive_failures` | int | `0` (line 212) | Each tick (line 264-275) | Abort check | [NOT_LOGGED] |

### orchestrator.py

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `_tick_id` | int | `0` (line 240) | +1 each tick (line 314) | Everywhere | |
| `_user_messages` | list[str] | `[]` (line 249) | inject_user_message (line 265), cleared end of tick (line 714) | _build_agent_prompt (line 801) | [HIDDEN_STATE] |
| `_recent_winners` | list[str] | `[]` (line 250) | After judge (line 622) | Judge anti-stagnation (line 476), adapt_temperature trigger | [HIDDEN_STATE] [CONDITION_DEPENDENT] [NOT_LOGGED] |
| `_recent_margins` | list[float] | `[]` (line 251) | After judge (line 626) | adapt_temperature (line 632) | [HIDDEN_STATE] [CONDITION_DEPENDENT] [NOT_LOGGED] |
| `_signal_history` | list[ControlSignals] | `[]` (line 252) | Each tick (line 531) | prediction_error (line 535-547) | [HIDDEN_STATE] [NOT_LOGGED] |
| `_prev_signals` | ControlSignals | None (line 253) | End of tick (line 684) | OrganismState (line 645) | [HIDDEN_STATE] [NOT_LOGGED] |
| `_prev_mode` | Mode | None (line 254) | End of tick (line 685) | OrganismState (line 646) | [HIDDEN_STATE] [NOT_LOGGED] |
| `_condition` | str | Constructor | Never | OrganismState | [CONDITION_DEPENDENT] |
| `_bench_mode` | bool | Constructor | Never | web_search guard (line 280) | [CONDITION_DEPENDENT] |

### judge.py — JudgePipeline

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `_judge_temp` | float | `_JUDGE_TEMP` (line 503) | adapt_temperature (line 680-683) | _call_judge options (line 749) | [HIDDEN_STATE] [CONDITION_DEPENDENT] [NOT_LOGGED] |
| `_call_count` | int | `0` (line 504) | Each evaluate() (line 568) | — | [NOT_LOGGED] |
| `_valid_json_count` | int | `0` (line 505) | On parse success | — | [NOT_LOGGED] |
| `_temp_history` | list[float] | `[]` (line 506) | adapt_temperature (line 690) | — | [NOT_LOGGED] |

### scheduler.py — Scheduler

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `signals` | ControlSignals | default (all 0) | update_signals() | tick() mode scoring | [HIDDEN_STATE] |
| `current_mode` | Mode | Mode.IDLE | tick() | agent selection, prompt | [HIDDEN_STATE] |
| `_dwell` | int | `0` | tick() +1 or reset | Mode fatigue penalty | [HIDDEN_STATE] [NOT_LOGGED] |
| `history` | list[ModeTransition] | `[]` | On mode change | — | [NOT_LOGGED] |

### L0R

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `_ring` | deque[L0RSlot] | empty | insert(), tick_decay() | build_evidence_pack() | [HIDDEN_STATE] |
| `_index` | dict | empty | insert(), decay | lookup | [HIDDEN_STATE] [NOT_LOGGED] |

### WorldModel

| Variable | Type | Initialized | Updated | Read | Tags |
|----------|------|-------------|---------|------|------|
| `_claims` | list[Claim] | `[]` | add_claim() from agents + judge | get_facts() for prompt | [HIDDEN_STATE] [NOT_LOGGED] |

### VetoBudget (orchestrator.py)

| Variable | Type | Tags |
|----------|------|------|
| `_last_veto_tick` per agent | dict | [HIDDEN_STATE] [NOT_LOGGED] |
| `_veto_count` per agent | dict | [HIDDEN_STATE] [NOT_LOGGED] |

---

## SECTION 3 — CONDITION DIFFERENCE MATRIX

| Component | A | B | C | E | E_B | R | F |
|-----------|---|---|---|---|-----|---|---|
| Active agents | A,B,C | A,B,C | A,B,C | A only | B only | A,B,C | A,B,C |
| agent_fn | OllamaAgentFn (3 models) | same | same | _make_single_agent_fn(A) | _make_single_agent_fn(B) | same as A | same as A |
| Judge type | JudgePipeline | JudgePipeline | JudgePipeline | SingleDraftJudge | SingleDraftJudge | **RandomJudge**(real_pipeline) | JudgePipeline |
| Judge temp | adaptive | adaptive | adaptive | N/A (no LLM) | N/A (no LLM) | **adaptive** (real judge called) | adaptive |
| Anti-stagnation | active | active | active | N/A | N/A | **active** (real judge prompt) | active |
| Winner selection | judge ranking | judge ranking | judge ranking | always A | always B | **random** (uniform) | judge ranking |
| Perturbation t15 | none | neutral | compression | compression | compression | compression | compression |
| Perturbation t35 | none | neutral | inversion | inversion | inversion | inversion | inversion |
| Perturbation t10,t20,t30,t40 | — | — | — | — | — | — | comp/inv/comp/inv |
| Orchestrator condition | "full" | "full" | "full" | "single_agent" | "single_agent" | "full" | "full" |
| sv_mean composition | mean(A,B,C embeddings) | same | same | A embedding only | B embedding only | mean(A,B,C) | mean(A,B,C) |
| CCV | pairwise dist(A,B,C) | same | same | always 0.0 | always 0.0 | same as A | same as A |

---

## SECTION 4 — MEMORY & FEEDBACK PATHS

### L0R (Ring Buffer)

**Written at tick t:**
- Each agent draft with text → slot (salience = 0.5 + 0.3 × novelty) [orchestrator.py:388-393]
- User injection / perturbation → slot (salience = 1.0) [orchestrator.py:273-276]
- Web search results → slot (salience = 0.9) [orchestrator.py:413-417]
- Winning draft → promoted to salience 1.0 [orchestrator.py:571-576]

**Read at tick t+1:**
- `build_evidence_pack(budget_tokens=2000)` selects top-K slots by composite_score [orchestrator.py:344]
- Evidence pack injected into agent prompts [orchestrator.py:826-831]

**Feedback loop:**
`winner at tick t` → L0R salience boost → higher rank in evidence pack at t+1 → more prominent in agent prompts → influences future drafts → influences future winner

**Condition dependency:**
- E/E_B: only 1 draft per tick inserted, always "wins" → L0R always contains that draft at salience 1.0. No diversity in evidence pack.
- R: winner is random but the WINNING draft gets L0R salience boost. Different draft promoted than C → different evidence pack at t+1 → **divergent trajectories** [ASYMMETRIC_C_R]

### World Model (WM)

**Written at tick t:**
- Agent claims (explicit + heuristic extraction, max 3/agent) [orchestrator.py:422-442]
- Judge claims [orchestrator.py:580-602]
- Fallback claims from winner draft if judge produced 0 [orchestrator.py:606-618]

**Read at tick t+1:**
- `_wm.get_facts()` → top 8 facts included in agent prompt [orchestrator.py:833-837]

**Feedback loop:**
`judge claims (based on winner)` → WM facts → next prompt → influences next drafts

**Condition dependency:**
- R: judge produces real claims based on real evaluation, but winner is random → claims may not align with promoted draft → weaker feedback coherence than C

### MR (Reality Memory / Event Log)

**Written at tick t:** every event (tick_start, agent_message, tick_end, tool_result, mode_change)
**Read at tick t+1:** by L0R to resolve chunk_ids → text for evidence pack
**Feedback:** indirect via L0R

### _recent_winners / _recent_margins

**Written at tick t:** after judge verdict [orchestrator.py:621-628]
**Read at tick t+1:** by judge anti-stagnation [judge.py:292-301] and adapt_temperature [judge.py:657-691]

**CRITICAL FEEDBACK LOOP for C vs R:**
- C: `_recent_winners` reflects quality-based selection → anti-stagnation triggers if best agent dominates
- R: `_recent_winners` reflects random selection → anti-stagnation triggers LESS (uniform distribution) [ASYMMETRIC_C_R]
- C: `_recent_margins` reflects real competitive margins → adaptive temp responds to competition quality
- R: `_recent_margins` are REAL (from judge) but winner is random → margins NOT affected by random selection. Same temp adaptation as C. [SYMMETRIC]

---

## SECTION 5 — JUDGE INTERNAL DYNAMICS

### adapt_temperature() [judge.py:657-691]

**Mechanism:**
- Window: last `_ADAPT_WINDOW` margins (default not shown, uses `recent_margins` passed in)
- Computes variance of margin window
- If var < `_TARGET_VAR_LOW` (0.02): temp += 0.05 (too uniform → explore)
- If var > `_TARGET_VAR_HIGH` (0.15): temp -= 0.05 (too chaotic → stabilize)
- Clamped between `_TEMP_MIN` and `_TEMP_MAX`

**Called at:** orchestrator.py:631-632, after each tick with ≥5 margins

**C vs R:**
- Margins are computed by the REAL judge in both C and R
- RandomJudge preserves the real judge's margins [bench_latin.py:191-193]
- Therefore `adapt_temperature()` receives the SAME margin distribution in C and R
- **SYMMETRIC** — judge temperature evolution is comparable

### Anti-stagnation [judge.py:290-301]

**Mechanism:**
- Reads `recent_winners[-10:]`
- Counts most frequent winner
- If top_count ≥ 6 out of 10: injects dominance_note in judge prompt

**C vs R:** [ASYMMETRIC_C_R]
- C: quality-based winner → if one agent consistently better, anti-stagnation fires → biases judge AGAINST dominant agent
- R: random winner → uniform distribution → top_count ≈ 3-4 out of 10 → anti-stagnation almost NEVER fires
- **This means C has an active bias correction that R does not.** C's winning distribution is artificially flattened. R's is naturally uniform.
- **Impact on sim_curves comparison:** if C > R on recovery, part of the effect may come from anti-stagnation diversifying the agent pool, not from competitive selection itself.

---

## SECTION 6 — FALLBACKS & SILENT PATHS

| Location | Trigger | Effect | Logged | Tag |
|----------|---------|--------|--------|-----|
| orchestrator.py:361-367 | No agent_fn | Empty AgentTurn (no text) | No | [SILENT_CONF] |
| orchestrator.py:478-479 | Judge exception | `judge_verdict = None` | Yes (log.exception) | |
| orchestrator.py:480-505 | 1 judgeable turn | Hardcoded verdict: confidence=0.5, margin_1v2=1.0 | Yes (log.info) | [SILENT_CONF] — affects quality_per_tick |
| orchestrator.py:526-527 | No judge verdict or no signals | `_aggregate_signals(turns)` instead of judge signals | No explicit log | [SILENT_CONF] |
| orchestrator.py:605-618 | Judge produced 0 claims | Extract claims from winner draft | No | [SILENT_CONF] |
| judge.py:536-566 | < 2 drafts with text | confidence=1.0 for E/E_B (SingleDraftJudge returns this) | Yes | Known limitation |
| judge.py:585-591 | Summarizer fails | Raw draft[:100] used as summary | Yes (log.warning) | [SILENT_CONF] — degrades judge quality silently |
| judge.py:random.shuffle | Every evaluate() | Draft presentation order randomized | Not in output JSON | [SILENT_CONF] — ordering bias mitigation, but adds noise |
| perturbation.py:87-89 | LLM fails | Returns `""` → perturbation skipped | Yes (log.error) | [SILENT_CONF] — perturbation tick becomes normal tick |
| bench_v2.py:264-275 | 5 consecutive empty ticks | RuntimeError abort | Yes | |
| metrics_v2.py:140-151 | No agent texts | Zero vector for sv_mean/sv_selected | No | [SILENT_CONF] |

---

## SECTION 7 — PROMPT & TOKEN FLOW AUDIT

### Agent prompt composition [orchestrator.py:797-847]

Order:
1. `_user_messages` (perturbation text) — "Réponds à cette question : >>> {text}" [line 800-805]
2. Mode instruction (1 sentence) [line 808-816]
3. Agent status (LEAD/OPPOSE/SUPPORT instruction) [line 818-823]
4. Evidence pack context — last exchanges from L0R, deduped [line 826-831]
5. World Model facts (top 8) [line 833-837]
6. Bootstrap if first tick with no context [line 840-845]

**Truncation:**
- Evidence pack: `budget_tokens=2000` [orchestrator.py:344]
- L0R text resolution: `event.payload.get("text", "")[:200]` per slot [l0r.py:52]
- Agent MR event: `turn.text[:1500]` [orchestrator.py:382]
- Perturbation input: `draft_text[:3000]` [perturbation.py:27]

**Condition-dependent prompt size:**
- E/E_B: fewer entries in L0R (1 agent vs 3) → smaller evidence pack → shorter prompts
- Perturbation ticks: prompt includes injected text → significantly longer
- User injection ticks (2, 12, 22, 32, 42): prompt includes user message

**Is perturbation fully included?**
- Yes. `inject_user_message()` appends full text to `_user_messages` [orchestrator.py:265]
- `_build_agent_prompt` includes ALL user messages without truncation [orchestrator.py:801-804]
- But the perturbation output itself was generated from `draft_text[:3000]` — input truncated at 3000 chars

---

## SECTION 8 — METRICS PROVENANCE

| Metric | Computed from | When | Selection dependency | Fallback | Tag |
|--------|-------------|------|---------------------|----------|-----|
| `state_vector_mean` | Mean of all agent embeddings (embed_texts on all non-empty drafts) | bench_v2.py post-tick (metrics_v2.py:156) | **Before** selection — all drafts | Zero vector if no texts | [DERIVED] |
| `state_vector_selected` | Embedding of winner's draft (or first draft as fallback) | metrics_v2.py:160-166 | **After** selection — depends on winner_id | Falls back to first agent's embedding | [DERIVED] [CONDITION_DEPENDENT] |
| `claim_cosine_variance` | Mean pairwise cosine distance between all agent embeddings | metrics_v2.py:169-171 | Before selection | 0.0 if < 2 agents | [DIRECT] |
| `judge_score_dispersion` | Std of [1.0, 1.0-m12, 1.0-m12-m23] from verdict margins | metrics_v2.py:94-107 | After selection (uses verdict) | 0.0 if no verdict | [DERIVED] |
| `draft_velocity` | Cosine distance(sv_selected[t], sv_selected[t-1]) | metrics_v2.py:179-184 | After selection — depends on winner sequence | NaN for t=0 | [DERIVED] [CONDITION_DEPENDENT] |
| `post_selection_variance` | Cosine distance(sv_selected, sv_mean) | metrics_v2.py:188-190 | After selection | — | [DERIVED] |
| `quality_per_tick` | `verdict.confidence` | metrics_v2.py:193-196 | After selection | 0.0 if no verdict | [DIRECT] |
| `sim_curves` | Cosine sim between R_pre (mean of 3 pre-pert ticks) and post-pert ticks | metrics_v2.py:198-257 | Uses sv_mean and sv_selected | None if < 3 pre-pert ticks | [DERIVED] |

---

## SECTION 9 — RANDOMNESS & NON-DETERMINISM

| Source | Affects | Controlled | Logged |
|--------|---------|------------|--------|
| `random.seed(seed)` in bench_v2.py:141 | Python random module (judge anonymization shuffle, RandomJudge winner) | Yes — per-run seed | Seed in output JSON |
| LLM sampling (temp=0.7) | All agent outputs, judge outputs | No — non-deterministic at temp>0 | No |
| LLM sampling (temp=0.0) | Perturbation operators | Partially — temp=0 is near-deterministic but not guaranteed | No |
| Judge anonymization shuffle | `random.shuffle(shuffled_ids)` in judge.py:574 | Controlled by seed | Not logged (only in _anon_map if audit) |
| RandomJudge winner | `random.shuffle(shuffled)` in bench_latin.py:175 | Controlled by seed | Logged (reason field) |
| Agent call order | Fixed: A, B, C (from `_select_agents`) | Yes — deterministic | Not explicitly logged |
| Ollama API variability | Latency, occasional failures | No | Latency logged in events |
| Judge `adapt_temperature` | Judge LLM temperature | Deterministic given margin history | temp_history not in output JSON |
| `_extract_claims_from_text` heuristics | WM claims content | Deterministic | Not logged |

---

## SECTION 10 — POTENTIAL CONFOUNDS

### A. Structural

| Confound | Mechanism | Conditions | Bias direction | Severity |
|----------|-----------|------------|----------------|----------|
| **Barycentre smoothing** | sv_mean = average of 3 embeddings → smoother trajectory than single embedding | E/E_B vs A/B/C/R/F | Multi-agent conditions appear more stable | **HIGH** |
| **CCV structural zero** | E/E_B have CCV=0 by construction | E/E_B | Incomparable with multi-agent | **HIGH** |
| **Single-agent prompt difference** | E/E_B prompts are shorter (fewer L0R entries) | E/E_B | Different context → different behavior | **MEDIUM** |

### B. Judge-related

| Confound | Mechanism | Conditions | Bias direction | Severity |
|----------|-----------|------------|----------------|----------|
| **Anti-stagnation asymmetry** | Anti-stagnation fires in C (quality selection → dominant agent) but rarely in R (random → uniform) | C vs R | C winning distribution artificially flattened vs R | **HIGH** |
| **Adaptive temperature drift** | Judge temp drifts over 50 ticks based on margin variance. Not logged in output JSON | All multi-agent | Unknown — affects judge discrimination quality | **MEDIUM** |
| **SingleDraftJudge confidence=1.0** | E/E_B always report confidence=1.0 — architectural artifact | E/E_B | quality_per_tick meaningless for E/E_B | **MEDIUM** |
| **Sole-draft confidence=0.5** | If only 1 judgeable turn (post-veto), confidence hardcoded to 0.5 | All multi-agent (rare) | Depresses quality_per_tick on veto ticks | **LOW** |
| **Judge shuffle noise** | `random.shuffle` on draft order each tick adds ordering noise | All multi-agent | Adds variance to winner selection | **LOW** |

### C. Memory-related

| Confound | Mechanism | Conditions | Bias direction | Severity |
|----------|-----------|------------|----------------|----------|
| **Winner → L0R → prompt feedback** | Winning draft gets salience=1.0, dominates next evidence pack | C vs R | C: best draft reinforced → coherent trajectory. R: random draft reinforced → noisier trajectory | **HIGH** |
| **WM claim accumulation** | Claims from winner + judge accumulate in WM → influence prompts | All | Winner-dependent claim seeding affects future discourse | **MEDIUM** |
| **L0R decay not logged** | TTL-based decay removes old slots silently | All | Context window composition changes over time without record | **LOW** |

### D. Prompt-related

| Confound | Mechanism | Conditions | Bias direction | Severity |
|----------|-----------|------------|----------------|----------|
| **Perturbation truncation at 3000 chars** | Input to perturbation LLM truncated at 3000 chars | B/C/E/E_B/R/F | Long winning drafts lose tail content before perturbation | **LOW** |
| **Evidence pack 200-char truncation** | L0R resolves text to 200 chars max per slot | All | Long drafts lose information in cross-tick memory | **MEDIUM** |
| **User injections as confound** | Ticks 2,12,22,32,42 inject French questions regardless of condition | All equally | Fixed schedule → comparable, but creates periodic disruption | **LOW** |

### E. Measurement-related

| Confound | Mechanism | Conditions | Bias direction | Severity |
|----------|-----------|------------|----------------|----------|
| **sv_selected depends on winner** | draft_velocity and PSV use sv_selected which depends on winner identity | C vs R | C: coherent winner sequence → smooth velocity. R: random winner → noisy velocity | **HIGH** |
| **sim_curves use sv_mean** | Recovery measured on sv_mean (barycentre) → smoothed by construction | All multi-agent | Overestimates recovery (smoothing absorbs perturbation impact) | **MEDIUM** |
| **quality_per_tick = judge confidence** | Not a content quality metric — reflects judge certainty about ranking | All | High disagreement → low confidence, not low quality | **MEDIUM** |
| **JSON key inconsistency across batches** | Batch 1 has `ranking_disagreement`, batch 2 has `judge_score_dispersion` | Cross-batch | Analysis code must handle both keys | **LOW** |
| **C_seed42 is 10-tick dry-run** | C_seed42 was never rerun live after dry-run overwrote it | C | Missing 1 of 6 seeds for condition C | **MEDIUM** |

---

## Summary of Critical Findings

1. **Anti-stagnation fires asymmetrically between C and R** — this is the most serious confound for C vs R comparison. C's winning distribution is actively corrected, R's is not. Any C > R result on sim_curves could partly reflect this bias.

2. **Winner → L0R → prompt feedback loop** — the selected winner shapes the next tick's context. In C, the best draft propagates. In R, a random draft propagates. This creates fundamentally different trajectory dynamics beyond just "selection vs averaging."

3. **sv_selected metrics are winner-dependent** — draft_velocity and PSV in R reflect random winner switches, not semantic evolution. These metrics are NOT directly comparable between C and R.

4. **Judge temperature is adaptive and not logged** — impossible to verify post-hoc whether C and R had similar judge temperature trajectories. Add `_temp_history` to output JSON.

5. **C_seed42 is a 10-tick dry-run** — must be rerun or excluded from analysis.
