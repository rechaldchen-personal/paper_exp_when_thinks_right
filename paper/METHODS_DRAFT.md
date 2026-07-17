# Methods (Draft for Paper)

## 4.1 Models and Setup

**Models**: Qwen3 family (1.7B, 4B, [14B if compute permits])
- Chosen for alignment with reference repo (Gurnee et al. 2026) and robust open-weight availability
- Loaded in float16 (bfloat16 for fitting) via Hugging Face Transformers
- Eval mode throughout; no gradient updates during lens application

**Lens fitting**:
- Corpus: FineWeb-10BT (Web text corpus; swap C4 / OpenWebText as needed)
- Sequences: 1,000 × 128-token chunks (paper spec; 100-prompt ablation in appendix)
- Jacobian: Averaged ∂h_final / ∂h_{ℓ,t} over the corpus
- Tool: Official anthropics/jacobian-lens reference implementation

**Workspace band**: [0.25, 0.90] as default; per-model identification via Gurnee et al. Fig 28 criteria (excess kurtosis, autocorrelation, next-token accuracy) in Appendix C.

## 4.2 Stimuli

**Design**: Matched minimal pairs (false_lead vs. straightforward) across three families:

### Factual (29 items → ~15 pairs)
Single-fact questions where a plausible distractor can mislead:
- Example: "The capital of France is **[Paris vs. Amsterdam]**" (straightforward)
- vs. "The capital of the country famous for tulips — no wait, the Eiffel Tower — is **[Paris vs. Amsterdam]**" (false_lead)
- Design: Lure initiates plausible parse; "wait" cue triggers revision

### Arithmetic (16 items → 8 pairs)
Order-of-operations ambiguity: left-to-right (naive) vs. PEMDAS (correct)
- Example: "calc: ( 4 + 17 ) * 2 = **[42]**" (straightforward)
- vs. "calc: 4 + 17 * 2 = **[38 vs. 42]**" (false_lead: target shifts!)
- Design: Parentheses vs. bare expression; answers swap per condition

### Garden-Path (10 items → 5 pairs)
Syntactic ambiguity from psycholinguistics literature (Natural Stories anchor):
- Example: "The horse **that was** raced past the barn fell. The thing that fell was the **[horse vs. barn]**" (straightforward)
- vs. "The horse raced past the barn fell. The thing that fell was the **[horse vs. barn]**" (false_lead: reduced relative clause)
- Design: Omitting "that was" triggers garden-path garden-path effect in humans

**Hard controls** (3 items):
Difficult problems WITHOUT a tempting distractor, to isolate false-lead effect from difficulty confound.
- Example: "The first president of a nation that declared independence in 1776 was **[Washington vs. Jefferson]**"

**Tokenization validation**: All targets and distractors are single tokens (with leading space) under the target model's tokenizer. Validated via `02_run_experiment.py --validate` before GPU runs.

## 4.3 Metrics (Per-stimulus, per-layer, per-position)

Let h_{ℓ,t} = residual stream at layer ℓ, position t.  
J_ℓ = averaged Jacobian (∂h_final,t' / ∂h_{ℓ,t}).

**Lens readout**: softmax( W_U · norm(J_ℓ h_{ℓ,t}) ) → vocabulary distribution.

### 4.3.1 Entropy & Baseline

**Entropy**: H(ℓ,t) = −Σ p log p (nats) over lens readout.

**Random-direction baseline**: For each (ℓ, t):
1. Sample K random vectors v ~ 𝒩(0, I) with norm(v) = norm(h_{ℓ,t})
2. Compute H_rand(v) = lens-readout entropy for each random v
3. H_rand(ℓ,t) = median of K samples

**Excess entropy reduction**: ΔH(ℓ,t) = H_rand(ℓ,t) − H(ℓ,t)  
(Positive = structured signal; isolates from early-layer degeneracy & vocabulary priors)

### 4.3.2 Commitment Metrics

All computed at **query position** (−1 ≡ last token) within the **workspace band**.

**Commitment layer** ℓ*: 
- First band layer where target rank = 0 (top-1)
- AND target remains top-1 through end of band (stability requirement)
- Return None if never achieved

**Confidence-collapse layer** ℓ_H:
- First band layer where ΔH > θ
- θ = 80th percentile of ΔH over straightforward-condition band positions
- (Set on dev split; holdout used for testing)

**Oscillation depth**:
- Number of distinct top-1 token identity changes at query position
- Counted in band layers after ℓ_H is first exceeded
- Signature of backtracking: model revises its confident prediction

**Dissociation gap**: ℓ* − ℓ_H  
- "Confidently-wrong window" (Zhang et al. 2023 framing)
- Positive = model achieves low entropy before locking onto correct answer
- Indicates a confident-but-wrong phase

### 4.3.3 Distractor Lead

Fraction of band layers where distractor rank < target rank.  
Measures how much the distractor was tempting throughout.

## 4.4 Analysis Pipeline

### Step 1: Lens Fitting (GPU, ~2–4 hours)
```bash
python 01_fit_lens.py --model Qwen/Qwen3-1.7B \
  --n-prompts 1000 --seq-len 128 \
  --out out/lens_qwen3_1p7b.pt
```

### Step 2: Trace Collection (GPU, ~10–20 min)
```bash
python 02_run_experiment.py --model Qwen/Qwen3-1.7B \
  --lens out/lens_qwen3_1p7b.pt \
  --stimuli stimuli.json --out out/traces.json
```
Output: Per-stimulus traces (entropy, ranks, logits, behavioral) as JSON.

### Step 3: Analysis & Statistics (CPU)
```bash
python 03_analyze.py --traces out/traces.json \
  --outdir out/analysis \
  --band 0.25 0.90 \
  --dev-split 0.6 \
  --theta-pct 80.0
```

**Dev/holdout split**:
- 60% of unique pair IDs → dev (θ set on this)
- 40% of pair IDs → holdout (metrics computed on this)
- Prevents p-hacking and ensures valid inference

**Output**: 
- `report.json`: Theta, condition summaries, paired Wilcoxon tests
- `per_stimulus_metrics.json`: Per-item l_star, l_H, gap, oscillation, etc.
- `heatmaps/`: (layer × position) images for representative pairs

## 4.5 Statistical Tests

**Paired Wilcoxon signed-rank test** on false_lead − straightforward per pair.
- Appropriate for paired, non-Gaussian data
- Report: n, median difference, W, p-value
- Significance: α = 0.05 (two-tailed)
- Correction: Bonferroni across 3 hypotheses (H1, H2, H3)

**Fallback**: Mann-Whitney U if not all pairs present in both conditions.

**Caution**: θ set and tested on same (or overlapping) data in this analysis.  
Split to dev/holdout before reporting p-values for paper submission.

## 4.6 Robustness Checks (Tier 2)

### Secondary Readout: Logit-Lens (Identity Transport)
Run all analyses with logit-lens (J_ℓ = I) as alternative:
- If main findings hold under both J-lens and logit-lens, robust to lens reliability issues
- Appendix D: side-by-side comparison for all figures

### Model Ablations
- Band sensitivity: rerun 03_analyze.py with --band [0.15, 0.95], [0.30, 0.85], etc.
- Theta sensitivity: rerun with --theta-pct 70, 90
- Report: do H1–H3 results hold across parameters?

## 4.7 External Validity: Natural Stories

(See natural_stories_plan.md for details)

Correlate internal metrics with human reading times on garden-path sentences.  
Expected: oscillation_depth and dissociation_gap correlate with human RT slowdown (r > 0.4, p < 0.05).

---

## Data & Code Availability

All code, stimuli (stimuli.json), fitted lens checkpoints, and per-prompt traces released at [repo URL].  
Built on anthropics/jacobian-lens (Apache 2.0).
