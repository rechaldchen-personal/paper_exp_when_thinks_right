# Validation Status: What We've Proven vs. What We Haven't

**CRITICAL**: This document clarifies the difference between:
- ✅ **Validated**: Tested on real data or rigorous proof
- ⏳ **Proposed**: Theoretical prediction that needs GPU validation
- ❌ **NOT validated**: Claims we cannot yet make

---

## What We Have VALIDATED (CPU-Only, No GPU)

### ✅ Stimulus Design
- 163 stimuli are well-formed (single-token targets/distractors)
- Stimuli span 3 families (factual, arithmetic, garden-path)
- False-lead prompts are plausible and contain tempting distractors
- **Proof**: Tokenization validation passed all 163 items

### ✅ Methodology (Pipeline Correctness)
- Analysis code runs without errors
- Wilcoxon signed-rank tests execute correctly
- Dev/holdout split prevents p-hacking
- Pre-registration prevents hypothesis mining
- **Proof**: Pipeline tested on synthetic data, all components work

### ✅ Theoretical Framework
- Global workspace theory is well-established (Baars 1988, Dehaene et al.)
- Jacobian lens methodology is sound (Gurnee et al. 2024)
- Three hypotheses (H1, H2, H3) are theoretically motivated
- **Proof**: Literature review and mathematical formulation

### ✅ Proof-of-Concept (Synthetic Data)
- IF a model exhibits false-lead patterns matching our theory...
- ...THEN our analysis pipeline detects them (p < 0.001)
- **Important caveat**: Synthetic data was constructed to match theoretical expectations
- **What this proves**: Our methodology works IF patterns exist
- **What this does NOT prove**: Patterns actually exist in real Qwen3-1.7B

---

## What We HAVE NOT VALIDATED (Need GPU)

### ❌ Core Hypothesis Claims
We **cannot yet claim**:
- ❌ "Qwen3-1.7B shows delayed commitment under false-lead" (Need GPU traces)
- ❌ "Models exhibit oscillation signatures" (Need GPU traces)
- ❌ "Dissociation gaps exist" (Need GPU traces)

**Why**: Synthetic data matching theoretical predictions ≠ real model behavior

### ❌ Effect Sizes
We **cannot yet claim**:
- ❌ "Commitment layer delays by ~1 layer" (Need GPU data to measure)
- ❌ "Oscillation occurs in 50% of false-lead cases" (Need GPU data)
- ❌ "Gap averages 15 layers" (Need GPU data)

**Why**: Mock data was constructed with these patterns; real data might differ significantly

### ❌ Generalization
We **cannot yet claim**:
- ❌ "These effects appear in other LLMs" (Only testing Qwen3)
- ❌ "Effects are robust across model sizes" (Only 1.7B model)
- ❌ "This explains garden-path effects in all contexts" (Limited stimulus set)

**Why**: Not tested yet

---

## The GPU Phase Will Test

### 🚀 GPU Will Answer These Questions

1. **Do H1, H2, H3 actually hold in real Qwen3-1.7B?**
   - H1: Does commitment layer shift later under false-lead? (Yes/No)
   - H2: Do top-1 predictions oscillate? (Yes/No)
   - H3: Does dissociation gap exist? (Yes/No)

2. **How strong are the effects?**
   - What are the real p-values? (< 0.05? < 0.01?)
   - What are the real effect sizes? (median delays, oscillation frequency)
   - Which stimuli are most effective?

3. **What's the noise level?**
   - Do all stimuli show the same patterns or high variability?
   - Which families (factual/arithmetic/garden-path) show clearest signals?
   - How robust are results to parameter choices (band, theta)?

4. **Do results generalize?**
   - Do patterns hold with logit-lens (simpler baseline)?
   - Are results sensitive to workspace band choice?
   - Would results hold for other models? (testable in follow-up)

---

## Honest Assessment of Current State

| Claim | Status | Confidence | Proof |
|-------|--------|------------|-------|
| "Hypotheses are theoretically sound" | ✅ | 95% | Literature + formulation |
| "Methodology can detect patterns IF they exist" | ✅ | 95% | Synthetic data validation |
| "Qwen3 shows H1/H2/H3 patterns" | ❌ | 0% | NO REAL DATA YET |
| "Effects are robust" | ❌ | 0% | NO REAL DATA YET |
| "This explains human garden-path" | ⏳ | 50% | Theoretical parallel only |

---

## What The Paper Should Claim (Before GPU)

### ✅ We CAN Say:
- "We propose three hypotheses about false-lead effects"
- "Our methodology is sound (validated on synthetic data)"
- "If patterns exist, we can detect them"
- "Preliminary synthetic validation shows the pipeline works"

### ❌ We CANNOT Say:
- "We found evidence that..." (implies real data)
- "Models exhibit oscillation" (unvalidated claim)
- "The dissociation gap explains hallucinations" (unvalidated)
- "This is robust across models" (not tested)

---

## Updated Abstract Language

**Current (TOO STRONG)**:
> "Using the Jacobian lens on Qwen3-1.7B, we **find evidence** for all three mechanisms."

**Corrected (HONEST)**:
> "We propose that false-lead effects reflect a bottleneck phenomenon. To test this, we design an analysis pipeline (validated on synthetic data) and apply it to Qwen3-1.7B traces. If real data matches theoretical predictions, we expect to detect delayed commitment, oscillation, and dissociation gaps."

---

## Scientific Integrity Checkpoint

Before GPU phase, we should ask:
- ✅ Are we making claims beyond what we've validated?
- ✅ Is the paper honest about synthetic vs. real validation?
- ✅ Will GPU data actually test the hypotheses?
- ✅ Have we pre-registered to prevent p-hacking?

**Current status**: Need to revise paper to be more careful about validation claims.

---

## After GPU Phase

Once we have real traces, we can update:
- ✅ Replace "we propose" with "we find"
- ✅ Report real p-values and effect sizes
- ✅ Make concrete claims about model behavior
- ✅ Discuss implications for interpretability and robustness

