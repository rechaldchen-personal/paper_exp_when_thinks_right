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

**Workspace band**: [0.25, 0.90] as default (all run-2 results use this). Per-model identification via Gurnee et al. Fig 28-style criteria (excess kurtosis, layer-to-layer rank autocorrelation, next-token accuracy) is implemented in `04_identify_band.py`/`band_analysis.py` (Appendix C) and unit-tested (`test_band_analysis.py`), but has not yet been run — it requires a GPU pass over a held-out corpus. See `experiments/workspace_band_guide.md` for the exact commands and for a note on where our autocorrelation operationalization deviates from the original guide's ambiguous pseudocode.

## 4.2 Stimuli

**Design**: 156 stimuli — **72 strictly matched pairs** (24 per family) +
**12 hard controls**, built by `build_stimuli.py`. "Strictly matched" is a
hard requirement, enforced by `validate_stimuli.py` (rule R1): within a pair,
target and distractor must be IDENTICAL across the straightforward and
false_lead conditions, so only the framing differs — see
`build_stimuli.py`'s module docstring for why an earlier version of this
stimulus set (run 1) violated this for all 28 of its arithmetic pairs, which
invalidated its paired tests.

### Factual (24 pairs)
Single-fact questions where a plausible distractor can mislead:
- Example: "The capital of France is **[Paris vs. Amsterdam]**" (straightforward)
- vs. "The capital of the country famous for tulips — no wait, the Eiffel Tower — is **[Paris vs. Amsterdam]**" (false_lead)
- Design: Lure initiates plausible parse; "wait" cue triggers revision; target/distractor identical across conditions

### Arithmetic (24 pairs)
Order-of-operations ambiguity: left-to-right (naive) vs. PEMDAS (correct).
Answer is a **fixed** English number word (' word' form) across both
conditions — only the framing (parenthesized vs. bare, then corrected)
changes; the expression and answer never swap.
- Example: "Q: ( 1 + 4 ) * 2\nA:" → **ten** (straightforward)
- vs. "Q: 1 + 4 * 2 — no wait, the sum in brackets comes first: ( 1 + 4 ) * 2\nA:" → **ten** (false_lead, same target)
- Few-shot frame required: Qwen3 tokenizes digit answers (' 10' → [space, '1', '0']) as multiple tokens, and the model defaults to a digit continuation unless number words are demonstrated first

### Garden-Path (24 pairs)
Reduced relative clause syntactic ambiguity:
- Example: "The horse **that was** raced past the barn fell. The thing that fell was the" → **horse** (straightforward)
- vs. "The horse raced past the barn fell. The thing that fell was the" → **horse** (false_lead: reduced relative clause, same target)
- Design: Omitting "that was" triggers the garden-path effect documented in humans; the probe sentence always ends on "the" so the query position is a genuine prediction, not an already-answered slot

**Hard controls** (12 items):
Difficult factual retrieval WITHOUT a tempting distractor, to isolate
false-lead effect from difficulty confound.
- Example: "Fact: The first president of the United States was" → **George** vs. **Thomas**

**Validation**: `validate_stimuli.py` enforces 7 design rules (matched pairs,
no answer leakage, single-token targets, no answer-in-context among them) on
every edit to `stimuli.json`, using the real tokenizer (CPU-only, no torch
needed). `02_run_experiment.py --validate` remains available as a quick
tokenization-only pre-GPU check but does not replace it — tokenization-OK is
not the same as design-sound, which is exactly how run 1's defects passed
undetected.

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

**Oscillation depth** (primary; see also the confirmatory 2AFC variant below):
- Number of distinct top-1 token identity changes at query position
- Counted in band layers after ℓ_H is first exceeded
- Originally hypothesized as a signature of backtracking (model revising
  between the target and distractor); run 2's confirmatory test does not
  support that reading — see Results §5.2 (H2) and Discussion §6.2.2. The
  metric itself (top-1 identity churn) is defined and computed as above
  regardless of interpretation.

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

**Paired, one-tailed, exact signed-rank test** on false_lead − straightforward
per pair (`03_analyze.py`), for the directional hypotheses H1-H3 as specified
in `PRE_REGISTRATION.md`.
- Zero differences dropped (standard `zero_method="wilcox"`); exact null
  distribution computed via subset-sum DP over signed mid-ranks, not
  scipy's `wilcoxon()` — scipy's `method="auto"` silently falls back to a
  normal approximation under ties/zeros at small n while warning that the
  approximation is invalid, and reproducibility across scipy versions is not
  guaranteed; see `out/ANALYSIS_NOTE.md` for the concrete case where this
  produced two different p-values (0.0625 vs 0.0339) for identical data.
- Report: n_pairs, n_nonzero, median difference, W₊, p (one-tailed, with the
  two-tailed value also recorded)
- Significance: α = 0.05
- **No Bonferroni correction applied.** Per `PRE_REGISTRATION.md` §8, all
  three hypotheses (and, per the amendment, their confirmatory counterparts)
  are reported together and the pattern across all six is what's interpreted,
  rather than correcting each test individually.

**Confirmatory readout** (`PRE_REGISTRATION_AMENDMENT.md` §4): the same three
tests, re-computed over a readout restricted to {target, distractor} rather
than the full vocabulary, reported alongside the primary tests with the
interpretation rule for agreement/disagreement fixed in advance (amendment §5).

**Dev/holdout split**: θ set on 60% of pairs (dev), all tests run on the
remaining 40% (holdout), seed fixed at 42 for reproducibility — this is
implemented, not a caution for future work.

## 4.6 Robustness Checks (Tier 2)

### Secondary Readout: Logit-Lens (Identity Transport)
**Status: run** (`out/traces_run2_logitlens.json`, `experiments/LOGIT_LENS_REPORT.md`).
`02_run_experiment.py --readout logit_lens` collects traces via identity
transport (residual → norm → lm_head directly, no fitted lens or `jlens`
dependency), output schema identical to the primary `--readout jlens`
traces. Result: **H3 confirmatory holds under both readouts** (jlens
p=0.0021, logit-lens p=0.040), ruling out J-lens-specific reliability issues
(§2.7) as an explanation for the headline finding. H1 confirmatory holds
under both but more marginally under logit-lens. H2 primary loses
significance under logit-lens (p=0.453) — a third line of evidence against
treating it as a real, lens-independent phenomenon, alongside its
θ-fragility and the confirmatory null. Full comparison table: Results §5.2 /
`LOGIT_LENS_REPORT.md`; figure/side-by-side: Appendix D.

### Model Ablations
- **Band sensitivity: done.** `experiments/SENSITIVITY_REPORT.md` — θ ∈
  {70,80,90} pct × band ∈ {[.20,.95],[.25,.90],[.30,.85]} on run-2 traces.
  Direction never flips; confirmatory metrics are significant at every
  setting, primary metrics only near the pre-registered default (see report
  for the mechanism).
- **Per-model band identification: run**, all 4 model×readout combinations
  (`experiments/BAND_IDENTIFICATION_REPORT.md`) — distinct from band
  *sensitivity* above; this identifies a data-driven band per model
  rather than perturbing the pre-registered default. See §4.1 and
  `experiments/workspace_band_guide.md`.
- Theta sensitivity: done, part of the same sensitivity report.

## 4.7 External Validity: Natural Stories

(See natural_stories_plan.md for details)

Correlate internal metrics with human reading times on garden-path sentences.
**Updated target (2026-07-22, per Discussion §6.4)**: correlate against the
dissociation gap, not oscillation depth — oscillation's mechanism is
unresolved (§4.3.2), so a correlation with human RT would not be
interpretable as a revision-cost analog the way the gap would be. If run,
expected: dissociation_gap correlates with human RT slowdown (r > 0.4, p < 0.05).

---

## Data & Code Availability

All code, stimuli (stimuli.json), fitted lens checkpoints, and per-prompt traces released at [repo URL].  
Built on anthropics/jacobian-lens (Apache 2.0).
