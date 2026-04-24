# CRISTAL V9 — Preregistration of Multi-Judge Evaluation

**Document purpose:** Lock analysis decisions before results are observed.
**Timestamp (to be filled at git commit):** `YYYY-MM-DD HH:MM UTC`
**Git commit hash (to be filled):** `________________`
**Author:** Julien DEFRANOUX, independent researcher
**Project:** CRISTAL — multi-agent LLM benchmark under perturbation
**Document version:** 1.0 (2026-04-24)

---

## 1. Rationale

The CRISTAL V9 benchmark (180 runs, 20 seeds × 9 conditions) has produced results on the `sv_selected` persistence metric: multi-agent conditions outperform mono-agent (p<0.001), structured memory does not improve over raw context (D3 ≈ I), and competitive selection does not improve over random (C ≈ R).

These claims are metric-dependent. `sv_selected` measures cosine proximity in the Sentence-BERT `all-mpnet-base-v2` space between the selected draft's embedding and a reference embedding over time. It is defensible as a proxy for geometric continuity. Whether it is also a proxy for judged quality is an open empirical question.

This document preregisters the multi-judge evaluation designed to **situate** (not validate) the relation between `sv_selected` and quality-related judgments. All thresholds, analysis choices, and publication branches are locked before any judge call is made.

## 2. Primary research question

**RQ:** What is the relationship between `sv_selected` (geometric persistence) and LLM-judged continuity and general quality, on drafts from V9 post-perturbation windows?

Two candidate framings for the resulting paper are defined in Section 10. The framing selected will be determined by the branching criteria in Section 9, not by post-hoc narrative choice.

## 3. Data

- **Drafts:** outputs from V9 runs, 20 seeds × 9 conditions. For judging, restrict to conditions {E, I, D3, C, R}.
- **Ticks evaluated:** {21, 25, 29, 41, 45, 49} — three points within each post-perturbation window (P1 = t21-29, P2 = t41-49).
- **Triplet unit:** at each (seed, condition, tick), the three pre-selection agent drafts form one triplet.
- **Total:** 20 seeds × 5 conditions × 6 ticks = **600 triplets**.

## 4. Judges

### 4.1 Included (principal panel)

| Judge | Ollama identifier | Role |
|---|---|---|
| J1 | `gemma4:31b-cloud` | Principal judge |
| J2 | `qwen3.5:cloud` | Principal judge |
| J3 | `deepseek-v4-flash:cloud` | Principal judge |

### 4.2 Excluded (with reasons)

| Model | Reason |
|---|---|
| `glm-5:cloud`, `kimi-k2.5:cloud`, `minimax-m2.7:cloud` | Generated V9 drafts (self-preference bias, Panickssery et al. 2024) |
| `nemotron-3-super:cloud` | Involved in V9 perturbation B |
| `gemini-3-flash-preview:cloud` | Already judged V9 drafts in bench pipeline (conditions A, B, C); self-consistency bias risk |
| `deepseek-v3.2:cloud` | Held in reserve; activated only if J1/J2/J3 pilot fails |

### 4.3 Judge call parameters

- Temperature: 0.0
- Thinking mode: disabled (no reasoning traces in output)
- Response order: A/B/C presentation randomized per call; order is logged
- Input language: English only. All prompts, all draft texts for judging, all judge responses in English

## 5. Judgment prompts (two independent evaluations per triplet)

Both prompts are in **English**. Each judge sees each triplet twice (continuity prompt, then quality prompt) in separate API calls with fresh context.

### 5.1 Continuity prompt (aligned with `sv_selected`)

```
You will see the start of a discussion, followed by three possible continuations labeled A, B, and C.

Discussion so far:
"""
{context}
"""

Continuation A:
"""
{draft_a}
"""

Continuation B:
"""
{draft_b}
"""

Continuation C:
"""
{draft_c}
"""

Evaluate which continuation is MOST coherent with the initial topic and LEAST disrupted by any off-topic elements. Rank from best to worst.

Output EXACTLY this JSON object, nothing else:
{
  "ranking": ["X", "Y", "Z"],
  "best": "X",
  "confidence": <integer 1-5>
}
```

### 5.2 Quality prompt (designed to diverge from continuity)

```
You will see the start of a discussion, followed by three possible continuations labeled A, B, and C.

Discussion so far:
"""
{context}
"""

Continuation A:
"""
{draft_a}
"""

Continuation B:
"""
{draft_b}
"""

Continuation C:
"""
{draft_c}
"""

Evaluate each continuation on these criteria:
- Novel informational contribution beyond the discussion context
- Argumentative depth (not merely surface coherence)
- Originality of perspective
- Specificity of claims (vs. generic statements)

Rank from best to worst. Ignore surface smoothness and length; reward substance.

Output EXACTLY this JSON object, nothing else:
{
  "ranking": ["X", "Y", "Z"],
  "best": "X",
  "scores_per_criterion": {
    "A": {"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>},
    "B": {"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>},
    "C": {"novelty": <1-5>, "depth": <1-5>, "originality": <1-5>, "specificity": <1-5>}
  },
  "confidence": <integer 1-5>
}
```

Note: the quality prompt includes `ignore surface smoothness and length; reward substance` to reduce length bias at the judgment stage. This does not eliminate the need for statistical length control (Section 8).

## 6. Human calibration anchor

Before any judge LLM is called on the full dataset, Julien will judge **50 randomly sampled triplets** (stratified by condition), in complete blind:
- No access to judge LLM outputs before finishing his 50 judgments.
- No access to which condition produced which draft.
- Order A/B/C randomized per triplet.
- Using the continuity prompt format only (quality prompt is too cognitively demanding for 50 triplets).

**This anchor is a sanity check of non-divergence between human and LLM judgment at the order-of-magnitude level.** It is explicitly NOT framed as ground truth. If LLM panel and Julien diverge massively (Fleiss kappa Julien-vs-LLM-majority < 0.2), this is reported as a limitation and the LLM panel results are downgraded in the paper.

## 7. Preregistered hypotheses

**H1 (primary):** Within-triplet Spearman ρ between judge ranking (continuity prompt, majority vote across J1/J2/J3) and `sv_selected` ranking of the three drafts, aggregated across the 600 triplets.

**H2 (secondary):** Same as H1 but with the quality prompt.

**H3 (architectural):** Under the quality prompt, pairwise preferences for D3 over I and for C over R, tested by Borda score differences and by within-triplet win rate. Tests whether V9 conclusions (D3 ≈ I, C ≈ R on `sv_selected`) transfer to judged quality.

## 8. Length control (primary analysis, not secondary)

Judge LLMs are known to exhibit verbosity bias. To prevent a length confound from masquerading as an architectural effect, length control is a **primary** analysis, not a post-hoc robustness check.

### 8.1 Measurements per draft

- Token count (tiktoken cl100k_base)
- Number of structural markers (bullets, numbered lists, headers via regex)

### 8.2 Statistical model

Ordinal mixed-effects regression:

```
judge_rank ~ condition + log(token_count) + structural_markers + tick + (1 | seed) + (1 | judge)
```

Fit once per prompt version (continuity, quality).

### 8.3 Primary length-adjustment test

Compare the coefficient for `condition` (e.g., C vs. E) between two models:
- M0: `judge_rank ~ condition + tick + (1 | seed) + (1 | judge)`
- M1: M0 + `log(token_count) + structural_markers`

Define **magnitude preservation ratio** = |β_condition(M1)| / |β_condition(M0)|.

If magnitude preservation ratio < **0.70** (i.e., loss > 30%), the apparent architectural effect is classified as **length-confounded**.

## 9. Preregistered decision branches

All thresholds below are locked. No modification post-hoc.

Let ρ_cont = median within-triplet Spearman ρ between continuity-judge-majority ranking and `sv_selected` ranking, bootstrap 95% CI clustered by seed.

Let ρ_qual = same for quality prompt.

Let Kappa = Fleiss' kappa for inter-judge agreement on the `best` label, averaged across both prompts.

Let MPR = magnitude preservation ratio defined in 8.3.

### Branch A1 — Claim A principal, Claim B in support

- **Trigger:** ρ_cont ≥ 0.65 AND ρ_qual ≥ 0.65 AND Kappa ≥ 0.30 AND MPR ≥ 0.70
- **Paper:** `sv_selected` is validated as a proxy for judged preference. Architectural conclusions (D3 ≈ I, C ≈ R) transfer.
- **Title A used (see 10).**

### Branch A2 — Differential case (the most informative)

- **Trigger:** ρ_cont ≥ 0.65 AND ρ_qual < 0.65 AND Kappa ≥ 0.30
- **Paper:** `sv_selected` captures continuity but not general quality. Architectural conclusions hold under a continuity framing, are suspended under a quality framing. The gap is quantified.
- **Title B used, with emphasis on continuity ≠ quality.**

### Branch B1 — Claim B principal

- **Trigger:** ρ_cont < 0.35 AND Kappa ≥ 0.30
- **Paper:** Geometric persistence is dissociable from judged continuity on this benchmark. Architectural conclusions from V9 are presented as illustrations of a metric-dependent gap.
- **Title B used.**

### Branch M — Mixed / ambiguous zone

- **Trigger:** ρ_cont ∈ [0.35, 0.65) AND Kappa ≥ 0.30
- **Paper:** Claim A conditional. Results reported with explicit caveat about moderate correlation. Both titles used; editorial choice deferred to co-authors.

### Branch L — Length confounded

- **Trigger:** MPR < 0.70 in either prompt (independent of ρ values)
- **Paper:** Architectural effects classified as confounded by verbosity. V9 conclusions reframed: "the apparent multi-agent advantage on `sv_selected` is partially attributable to length differences between conditions."

### Branch K — Judges diverge

- **Trigger:** Kappa < 0.30 (independent of ρ)
- **Paper:** Multi-judge axis is degraded to an appendix. Main claim reverts to Claim A with an explicit limitation that judge agreement was insufficient to draw quality conclusions.

**Branch precedence:** If multiple branches trigger, precedence is K > L > B1 > A1 > A2 > M. The most limiting branch wins.

## 10. Candidate paper framings (both written before results)

### 10.1 Title A

**"Multi-sampling, not coordination: what drives semantic persistence in multi-agent LLM pipelines under out-of-distribution perturbation"**

Abstract (draft):
> We study a multi-agent LLM benchmark under controlled out-of-distribution perturbation and find that the collective advantage of multi-agent pipelines over single-agent baselines is reproduced by a trivial pipeline combining independent sampling, raw context propagation, and random selection. Structured memory and competitive judge selection add no measurable benefit. A multi-judge evaluation across three independent LLMs confirms that these conclusions transfer from our geometric persistence metric to judged draft preference.

### 10.2 Title B

**"Persistence is not quality: dissociating geometric and semantic evaluation in multi-agent LLM benchmarks"**

Abstract (draft):
> Benchmarks for multi-agent LLM systems increasingly rely on embedding-based persistence metrics as proxies for semantic quality under perturbation. We show, on a controlled multi-agent benchmark, that geometric persistence (cosine stability in Sentence-BERT space) and judged quality (via a three-LLM panel with length controls) measure separable dimensions. Architectural conclusions about the contribution of structured memory and competitive selection differ depending on which metric is chosen. We quantify the gap and discuss implications for multi-agent evaluation methodology.

## 11. Analysis plan (in execution order)

1. Phase 0 sensitivity results already collected (separate document). GO to Phase 1 conditional on sensitivity verdict.
2. Compile triplet dataset from V9 runs.
3. Human anchor: Julien judges 50 triplets blind.
4. Pilot run: 5 seeds × 5 conditions × 6 ticks × 3 judges × 2 prompts.
5. Pilot diagnostics: Kappa, positional balance per judge, length correlation with ranking. If any fails acceptance (Kappa < 0.30, positional imbalance > 45/55 for any letter, length correlation > 0.5), diagnose and revise prompts once. If revision still fails, activate Branch K.
6. Full run: 20 seeds.
7. Compute ρ_cont, ρ_qual, Kappa, MPR with CIs.
8. Apply branch precedence rules (Section 9).
9. Write paper along the selected branch using the pre-written title/abstract.

## 12. Deviations protocol

Any deviation from this preregistration must be:
- Documented in a separate file `DEVIATION_LOG.md` at commit time.
- Justified (e.g., API unavailable, dataset issue).
- Timestamped with git hash.

Deviations do not invalidate the preregistration, but must be reported explicitly in the paper's limitations section. Post-hoc analyses are permitted but must be labeled as such in the paper; they cannot be upgraded to confirmatory claims.

## 13. What is explicitly NOT preregistered

- Exploratory analyses on additional tick positions or conditions (A, B, D, D2).
- Cross-embedding validation (BGE-M3). Planned as a separate study.
- Qualitative analysis of individual examples.
- Extensions to other perturbation types.

These may be reported but only as exploratory, not confirmatory.

## 14. Commitment

By committing this file to the project repository under git, I commit to honor the thresholds, branches, and analysis plan above. I acknowledge that post-hoc modification of thresholds, re-selection of judges, or narrative reframing that contradicts branch precedence constitutes HARKing and will be disclosed as such if it occurs.

**Signature:** Julien DEFRANOUX
**Date:** ________________
**Git commit hash:** ________________
**SHA-256 of this file at commit:** ________________ (to be computed)

---

*End of preregistration.*
