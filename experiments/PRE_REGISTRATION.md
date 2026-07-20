# Pre-Registration Document

**Date**: 2026-07-14  
**Commitment Dynamics Paper — Commitment Layer Identification Study**

> ⚠️ **AMENDED 2026-07-19 — read `PRE_REGISTRATION_AMENDMENT.md` alongside this
> document.** Run 1 is withdrawn (invalid stimuli and a defective analysis
> pipeline). The amendment replaces the stimulus set (§7 sample size here is
> superseded), corrects the test implementation to the one-tailed exact test
> this document already specified, and adds confirmatory 2AFC metrics. The
> hypotheses, band, θ procedure, and split in this document are unchanged.

This document pre-specifies hypotheses, analysis procedures, and robustness criteria **before viewing real GPU traces**.

---

## 1. Research Questions & Hypotheses

### Primary Research Question
**When do language models commit to answers in the J-space, and does internal false-lead processing show measurable signatures?**

### Hypotheses (Pre-Registered)

#### H1: Later Commitment Under False-Lead (Directional)
- **Prediction**: Commitment layer ℓ* is later (higher layer index) for false_lead than straightforward
- **Operationalization**: ℓ* = first band layer where target_rank = 0 and remains 0 through end of band
- **Statistical test**: Paired Wilcoxon signed-rank test on (false_lead - straightforward) per pair
- **Support criterion**: p < 0.05 (one-tailed), median_diff > 0
- **Mock validation**: ✓ Δℓ* = +1.0 layer, p = 0.004
- **Expected direction**: False-lead shows later commitment (makes intuitive sense; model is misled initially)

#### H2: Oscillation Under False-Lead (Directional)
- **Prediction**: Oscillation depth is higher (more top-1 identity changes) for false_lead than straightforward
- **Operationalization**: oscillation = count of distinct top-1 token identity changes at query_position after ℓ_H is first exceeded
- **Statistical test**: Paired Wilcoxon signed-rank test
- **Support criterion**: p < 0.05 (one-tailed), median_diff > 0
- **Mock validation**: ✓ Δ = +2.0 changes, p = 0.004
- **Expected direction**: False-lead shows more oscillation (signature of internal revision)
- **Novelty note**: This is the most novel finding; if only one hypothesis holds, we prioritize this

#### H3: Dissociation Gap Under False-Lead (Directional)
- **Prediction**: Dissociation gap (ℓ* − ℓ_H) is larger for false_lead than straightforward
- **Operationalization**:
  - ℓ_H = first band layer where excess_entropy ΔH > θ
  - gap = ℓ* − ℓ_H (can be negative if entropy collapse precedes commitment)
- **Statistical test**: Paired Wilcoxon signed-rank test
- **Support criterion**: p < 0.05 (one-tailed), median_diff > 0
- **Mock validation**: ✓ Δ = +15.0 layers, p = 0.004
- **Expected direction**: False-lead shows larger positive gap (confidence before correctness = hallucination risk)

### Secondary Hypothesis (H4 — Extension, Optional)

**H4: CoT Reduces Oscillation**
- **Prediction**: Chain-of-thought arithmetic prompts show lower oscillation than direct prompts
- **Status**: Optional (time-permitting); uses same setup, CoT wrapping
- **Support criterion**: Lower oscillation in CoT condition (descriptive, not pre-registered p-value)

---

## 2. Workspace Band Specification

### Primary Band (Conservative, Pre-Specified)
- **Choice**: [0.25, 0.90] (relative to min/max layers)
- **Justification**: Matches Gurnee et al. 2026 default; works for Qwen models
- **All hypotheses tested**: Using this band

### Secondary Analysis (Sensitivity)
- **Alternative bands**: [0.20, 0.95], [0.30, 0.85]
- **Criterion for robustness**: All three hypotheses hold the same direction across all band choices
- **Report**: Appendix will show sensitivity curves

### Per-Model Band Identification (If Time Permits)
- **Method**: Excess kurtosis, autocorrelation, next-token accuracy (Gurnee et al. Fig 28)
- **Only if**: Real traces show unexpected patterns or Qwen band differs from literature
- **Report**: Appendix C (optional)

---

## 3. Theta (Confidence-Collapse Threshold) Specification

### Threshold Definition
- **θ = 80th percentile** of excess_entropy ℓ_H values in straightforward condition (dev set only)
- **Justification**: Standard robustness threshold; high-confidence-collapse marker
- **Rationale**: At 80th percentile, we capture "committed" regime without overfitting to noise

### Dev/Holdout Split
- **Dev set**: 60% of unique pair IDs (random assignment, seed=42)
- **Holdout set**: 40% of unique pair IDs
- **Theta computation**: On dev set only
- **All metrics reported**: On holdout set only
- **Prevents**: P-hacking by setting threshold and testing on same data

### Sensitivity Analysis (Robustness)
- **Test**: Re-run analysis with θ at 70th and 90th percentiles
- **Criterion**: H1–H3 support unchanged (same direction, p-values similar)
- **Report**: Appendix D

---

## 4. Primary Analysis Plan

### Step 1: Compute Metrics (Per-Stimulus, Per-Layer, Per-Position)

For each stimulus, each layer in band, each position:
- Excess entropy: ΔH = H_random − H_actual
- Target rank: rank of target token in lens logits
- Distractor rank: rank of distractor token
- Top-1 identity: which token is argmax of lens logits

Aggregate metrics per stimulus:
- l_H = first layer where ΔH > θ
- l_star = first layer where target_rank = 0, stable through end
- gap = l_star − l_H
- oscillation = count of top-1 identity changes after l_H

### Step 2: Paired Statistical Tests

For each hypothesis (H1, H2, H3):
1. Extract metric per stimulus per condition
2. Compute difference (false_lead − straightforward) for each pair
3. Run Wilcoxon signed-rank test on differences
4. Report: n_pairs, median_diff, W-statistic, p-value
5. Confirm direction matches pre-registered prediction

### Step 3: Validation Against Hard Controls

- Hard-control stimuli should show: late commitment, NO oscillation, NO dissociation
- Confirms that effects are specific to false-lead, not difficulty
- If hard controls show oscillation/gap: re-examine stimuli design

### Step 4: Robustness Checks (See § 5 below)

---

## 5. Robustness Checks (All Pre-Specified)

### 5.1 Logit-Lens Replication (Secondary Readout)
- **Procedure**: Re-run 03_analyze.py on logit-lens traces (identity transport)
- **Criterion**: H1–H3 hold the same direction with logit-lens
- **Interpretation**: Findings are robust to lens-specific artifacts
- **Report**: Appendix D, side-by-side comparison

### 5.2 Band Sensitivity
- **Procedure**: Re-run with [0.20, 0.95] and [0.30, 0.85]
- **Criterion**: H1–H3 direction unchanged; p-values similar (not all drop below 0.05, but same significance level)
- **Report**: Appendix D, sensitivity curves

### 5.3 Theta Sensitivity
- **Procedure**: Re-run with θ at 70th and 90th percentiles
- **Criterion**: H1–H3 direction unchanged
- **Report**: Appendix D

### 5.4 Behavioral Pre-Screen
- **Procedure** (before full traces): Check that model answers straightforward versions correctly
  - Straightforward top-1 should be target (behavioral.target_is_top1 == true)
  - False-lead: distractor should be elevated somewhere (behavioral.distractor_in_top5 early on)
- **Criterion for inclusion**: Keep only pairs passing behavioral check
- **Stop-point**: If pre-screen fails for many pairs, fix stimuli and re-run

---

## 6. Confound Controls & Design Validation

### Controlled For (By Stimulus Design)
- **Length-matching**: Target/distractor pairs matched on token count
- **Difficulty confound**: Hard controls separate difficulty from false-lead effects
- **Frequency confound**: Targets and distractors are equally plausible a priori

### To Be Reported (As Covariates)
- Prompt length (token count) per stimulus
- Distractor rank in the straightforward condition (how tempting is it there?)
- Distance to disambiguation point (in tokens from start)

---

## 7. Sample Size & Statistical Power

### Design
- 77 stimuli across 3 families (29 factual, 24 arithmetic, 24 garden-path)
- Paired design: each pair has both straightforward and false_lead conditions
- Effective pairs for paired tests: n = 21–24 (depending on family coverage)

### Power Calculation (Ballpark)
- Mock data shows effect sizes: Δℓ* ≈ 1 layer, Δ_osc ≈ 2 changes, Δ_gap ≈ 15 layers
- With n ≈ 20 pairs, Wilcoxon should detect effects of this magnitude (power ≈ 0.7–0.8)
- If real effects are smaller than mock (expected), may not reach p < 0.05 individually
- **Mitigation**: Report all three together; even if one drops below threshold, the pattern is evidence

---

## 8. Multiple Comparisons

### Correction Method
- **Primary**: Test three hypotheses (H1, H2, H3)
- **Correction**: Bonferroni-like informal correction (three independent hypotheses)
- **Criterion**: Report all three p-values; if ≥2/3 supported, call the pattern robust

### Avoiding P-Hacking
- Pre-registered band, theta, and hypotheses
- Dev/holdout split (theta set on dev, tested on holdout)
- Robustness checks (band sensitivity, theta sensitivity, logit-lens replication)
- Hard controls (validates specificity)

---

## 9. Definitions & Operationalizations

### Key Terms
- **Band**: Workspace band layers (typically [0.25, 0.90] relative to depth)
- **Commitment layer ℓ***: First band layer where target_rank = 0 and stable
- **Confidence-collapse layer ℓ_H**: First band layer where ΔH > θ
- **Dissociation gap**: ℓ* − ℓ_H (negative = collapse before commitment, positive = commitment after collapse)
- **Oscillation**: Count of top-1 identity changes at query_position in band layers after ℓ_H
- **Excess entropy ΔH**: H_random_baseline − H_actual (positive = structured signal)

---

## 10. Expected Results (From Mock Validation)

### If Real Model Matches Mock Behavior
- H1: p ≈ 0.004, Δℓ* ≈ +1 layer
- H2: p ≈ 0.004, Δ_osc ≈ +2 changes
- H3: p ≈ 0.004, Δ_gap ≈ +15 layers

### If Real Model Shows Weaker Effects
- Likely: p-values higher (0.01–0.10)
- Still publishable: Pattern is there, just needs larger sample or better stimuli design
- Mitigation: Expand stimuli post-hoc (after observing real data) for replication

### If Real Model Shows No Effects
- Hypothesis: Stimuli don't work (distractors aren't tempting) or Qwen shows no oscillation
- Investigation: Check behavioral pre-screen; if passed, model is answering correctly but not oscillating
- Pivot: Publish negative result + design discussion

---

## 11. Deviations & Flexibility

### Pre-Specified Deviations Allowed
- **Stimuli**: Can expand beyond 77 if needed for power (post-hoc)
- **Band**: Can identify per-model if defaults don't match literature
- **Theta**: Can adjust sensitivity if 80th percentile is extreme

### Deviations Requiring Report
- Any p-value calculation changes
- Any test statistic changes (Wilcoxon → Mann-Whitney, etc.)
- Any hypothesis changes (these go in Discussion as post-hoc)

---

## 12. Timeline & Commitment

**Written**: 2026-07-14 (before GPU traces)  
**GPU traces expected**: 2026-07-15–2026-07-17  
**Analysis**: 2026-07-17–2026-07-18  
**Deviations documented in paper**: Discussion section

---

## Signature

This pre-registration document serves as a commitment to the analysis plan. Any deviations from this plan will be documented in the paper's Methods or Discussion section.

**Pre-registered by**: Claude (AI research assistant)  
**For**: Commitment Dynamics in the J-Space (working title)

---

## Appendix: Mock Data Validation

All hypotheses were pre-tested on synthetic data (27 traces, 9 pairs × 3 conditions):

| Hypothesis | Mock Result | Direction | p-value |
|---|---|---|---|
| H1 | ✓ SUPPORTED | false_lead later | 0.004 |
| H2 | ✓ SUPPORTED | false_lead more oscillation | 0.004 |
| H3 | ✓ SUPPORTED | false_lead larger gap | 0.004 |

This validation ensures the pipeline works before GPU runs. Real results may differ; deviations will be explained in Discussion.
