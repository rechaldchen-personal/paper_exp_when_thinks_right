# Section 5: Results

**Status**: Filled from Qwen3-1.7B run 2 (2026-07-22) and the Qwen3-4B
replication + logit-lens + band-identification follow-ups (2026-07-24), on
the rebuilt stimulus set, per `experiments/PRE_REGISTRATION_AMENDMENT.md`
(whose scope already covers "any later model"). Run 1 (2026-07-17) is
withdrawn — see `out/ANALYSIS_NOTE.md` — and its numbers do not appear here.

**Validation status**:
- ✅ Stimuli: 156 items, 72 strictly matched pairs (24/family) + 12 hard
  controls, all pass `validate_stimuli.py` (7 design rules) for both Qwen3-1.7B
  and Qwen3-4B tokenizers (shared 151,936-token vocabulary)
- ✅ Behavioral pre-screen passed on all three families, both models, both
  readouts (amendment §8 gate): see §5.2/§5.8 for exact rates
- ✅ Real data: `out/traces_run2.json` / `out/traces_run2_logitlens.json`
  (1.7B), `out/traces_4b.json` / `out/traces_4b_logitlens.json` (4B)
- ✅ Figures in `out/figures/`, including the model-comparison figure
  (`model_comparison_1p7b_vs_4b.png`)
- ✅ Sensitivity checks (θ/band, 1.7B/jlens): `experiments/SENSITIVITY_REPORT.md`
- ✅ Logit-lens secondary readout (1.7B): `experiments/LOGIT_LENS_REPORT.md`
- ✅ Qwen3-4B replication (own lens fit, both readouts): **§5.8** below, full
  detail in `experiments/4B_REPLICATION_REPORT.md` — **does not replicate
  H1/H3**, the most consequential finding in this section
- ✅ Per-model workspace band identification (all 4 model×readout
  combinations): `experiments/BAND_IDENTIFICATION_REPORT.md` — not adopted
  as a new default (see that report's caveats)
- ⏸ Natural Stories external-validity analysis not run

**Analysis settings (pre-registered, identical across models)**:
- Models: Qwen/Qwen3-1.7B (27 lens layers) and Qwen/Qwen3-4B (35 lens layers),
  each with its own freshly fitted Jacobian lens on 1000×128 FineWeb prompts
- Dev/holdout split: 60/40 by pair (`--dev-split 0.6`, seed 42) → 50 dev / 34
  holdout pairs, same split procedure for both models
- Theta (ΔH threshold): 80th percentile of straightforward condition, dev
  only — 3.975 nats (1.7B jlens), 4.946 (1.7B logit_lens), 5.107 (4B jlens),
  5.874 (4B logit_lens); different values are expected (raw entropy scale
  differs by model), the *procedure* is identical
- Workspace band: [0.25, 0.90], pre-registered default, used throughout
  (data-driven band identification was run but not adopted — see
  `BAND_IDENTIFICATION_REPORT.md`)
- Two readouts per hypothesis: **primary** (global argmax, originally
  specified) and **confirmatory** (restricted to {target, distractor}),
  `PRE_REGISTRATION_AMENDMENT.md` §4 — and, orthogonally, two **lens
  methods**: jlens (fitted Jacobian lens) and logit_lens (identity
  transport), giving four independent readings per hypothesis per model

---

## 5.1 Sanity Check: Excess-Entropy Profile

**Figure 5.0**: `out/figures/entropy_curves.png` — layer-wise median excess
entropy ΔH for Qwen3-1.7B (jlens) by condition on holdout items.
- Workspace band shaded at fractional depth [0.25, 0.90]; θ threshold overlaid

| Model | Readout | Band used | Theta (nats) |
|-------|---------|-----------|--------------|
| Qwen3-1.7B | jlens | [0.25, 0.90] | 3.975 |
| Qwen3-1.7B | logit_lens | [0.25, 0.90] | 4.946 |
| Qwen3-4B | jlens | [0.25, 0.90] | 5.107 |
| Qwen3-4B | logit_lens | [0.25, 0.90] | 5.874 |

---

## 5.2 Main Findings: Commitment Dynamics (Qwen3-1.7B)

All tests compare **false_lead − straightforward** on holdout pairs, one-tailed
exact signed-rank (`03_analyze.py`; see `out/ANALYSIS_NOTE.md` for why the
test is exact and one-tailed rather than the scipy default). Each hypothesis
is reported as a pair: **primary** (global argmax, originally pre-registered)
and **confirmatory** (restricted to {target, distractor}), per amendment §4.
The permitted claim for each combination of outcomes was fixed *before* this
run (amendment §5) and is applied here without adjustment. Numbers below are
the primary jlens readout unless noted; the logit-lens comparison for each
hypothesis is folded in from `experiments/LOGIT_LENS_REPORT.md`. **Whether
each effect replicates on Qwen3-4B is reported separately in §5.8 — read
that section before treating any of the following as a general property of
these models rather than a Qwen3-1.7B-specific finding.**

### H1: Later Commitment Under False Lead

**Prediction**: ℓ* (commitment layer) larger for false_lead than straightforward.

**Figure 5A**: `out/figures/H1_commitment_layer.png`

**Primary** (target must beat the entire vocabulary):
- Median paired difference (FL − SF): **0.0** layers
- Exact one-tailed signed-rank: W₊=**22.5**, p=**0.109**, n_pairs=**14** (7 nonzero)
- **Result**: H1 primary **NOT SIGNIFICANT**. Logit-lens agrees (p=0.145,
  also null) — consistent null under both readouts.

**Confirmatory** (target need only beat the distractor):
- Median paired difference (FL − SF): **+2.0** layers
- Exact one-tailed signed-rank: W₊=**108.0**, p=**0.0021**, n_pairs=**19** (15 nonzero)
- **Result**: H1 confirmatory **SUPPORTED**, robust across every θ
  (70th/80th/90th pct) and band setting tested (`SENSITIVITY_REPORT.md`).
  **Logit-lens also significant but far more marginal** (p=0.040 vs. jlens'
  0.0021) — same direction, weaker effect (`LOGIT_LENS_REPORT.md`).

**Interpretation (amendment §5, H1 row)**: primary null / confirmatory
significant → narrow-readout claim only. False-lead prompts reliably delay
the point where the target overtakes the distractor specifically, but this
does not translate into a later *global* commitment layer. This claim should
be qualified as "robust in direction across θ/band, but readout-dependent in
strength" rather than unconditionally robust — see the logit-lens caveat
above, and §5.8 for whether it holds on 4B at all (it does not).

---

### H2: Oscillation Under False Lead

**Prediction**: Oscillation depth (top-1 identity changes after ℓ_H) higher
for false_lead.

**Figure 5B**: `out/figures/H2_oscillation.png` · **Figure 5B-detail**:
`out/figures/H2_oscillation_hist.png`

**Primary** (global top-1 identity changes):
- Median paired difference (FL − SF): **+0.5** changes
- Exact one-tailed signed-rank: W₊=**182.0**, p=**0.0077**, n_pairs=**28** (21 nonzero)
- **Result**: significant at the pre-registered default, but **not robust to
  θ** (null at θ=70, p=0.147; θ=90, p=0.480) **and not robust to lens
  method** (logit_lens p=0.453 — null). Two independent axes of fragility.

**Confirmatory** (target-vs-distractor preference flips only):
- Median paired difference (FL − SF): **0.0** changes
- Exact one-tailed signed-rank: W₊=**41.0**, p=**0.46**, n_pairs=**28** (12 nonzero)
- **Result**: **NOT SIGNIFICANT**, and the most consistently null result in
  the entire study: p ranges 0.27–0.62 across every θ/band setting *and*
  both lens methods *and* — per §5.8 — both models.

**Hard controls** (no tempting distractor, difficulty-only): median
oscillation **2.0** changes (n=6 holdout), comparable to false_lead's 2.5 and
well above straightforward's 1.0.

**Interpretation (amendment §5, "unattributed churn" rule)**: more top-1
churn under false-lead by the pre-registered global metric, at the
pre-registered default only — but not attributable to wavering between
target and distractor. Four independent lines of evidence now converge on
this: the confirmatory-metric null, the θ-fragility, the logit-lens null,
and the hard-control comparison. **We do not claim internal revision between
candidate answers**, and this reading is the most secure conclusion in the
whole study precisely because it is a stable null under every stress test
applied.

---

### H3: Dissociation Gap Under False Lead

**Prediction**: Gap ℓ* − ℓ_H larger for false_lead (confidently-wrong window).

**Figure 5C**: `out/figures/H3_dissociation_gap.png`

**Primary**:
- Median paired difference (FL − SF): **+1.0** layers
- Exact one-tailed signed-rank: W₊=**59.0**, p=**0.0112**, n_pairs=**14** (11 nonzero)
- **Result**: significant at the pre-registered default, but fragile to θ
  (null at θ=70, p=0.184; θ=90, p=0.217) **and to lens method** (logit_lens
  p=0.193 — null).

**Confirmatory**:
- Median paired difference (FL − SF): **+4.0** layers
- Exact one-tailed signed-rank: W₊=**135.5**, p=**0.0016**, n_pairs=**18** (17 nonzero)
- **Result**: **SUPPORTED, and the most robust result on Qwen3-1.7B** —
  significant at every θ/band setting (p 0.0013–0.033) *and* under the
  independent logit-lens readout (p=0.034). Three independent robustness
  axes, all passed.

**Interpretation (amendment §5)**: this is the strongest claim the 1.7B data
support. We lead with the confirmatory number (median Δ=+4.0 layers,
p=0.0016) and report the primary metric (median Δ=+1.0, p=0.011) as
corroborating at the default setting but not independently robust.
**Critically, this robustness is scoped to Qwen3-1.7B — see §5.8: the same
effect, measured identically, is null on Qwen3-4B under both readouts.**
"Robust" in this section means robust across analysis choices *within* one
model; it does not mean robust across model scale, and the two should not be
conflated when this result is cited.

**Hard controls**: median gap **1.5** layers (n=2 holdout — too few to test).

---

## 5.3 Case Studies: Heatmaps

Representative Qwen3-1.7B holdout heatmaps in `out/figures/heatmaps/`:

| Pair | Family | Conditions | Path |
|------|--------|------------|------|
| capital_cl | factual | SF / FL | `capital_cl_*.png` |
| capital_eg | factual | SF / FL | `capital_eg_*.png` |
| legs_spider | factual | SF / FL | `legs_spider_*.png` |
| gp_horse | garden_path | SF / FL | `gp_horse_*.png` |

---

## 5.4 Distractor Temptation (Qwen3-1.7B)

**Table 5.1** (holdout pairs; full table in `out/figures/table_distractor_temptation.md`):

| Family | Pair | Straightforward (distractor_lead_layers) | False_lead (distractor_lead_layers) |
|--------|------|------------------------------------------|-------------------------------------|
| Factual | capital_eg | 7 | 12 |
| Factual | legs_spider | 4 | 11 |
| Garden-path | gp_horse | 8 | 11 |
| Garden-path | gp_runner | 14 | 17 |
| Arithmetic | arith_1_2_3 | 10 | 16 |

**Condition medians** (holdout, primary/jlens): straightforward **4.0**,
false_lead **8.5** (Δ = +4.5 layers).

**Behavioral temptation rate** (distractor reaches the model's final top-5):
factual 25.0%, arithmetic 29.2%, **garden-path 66.7%**. Garden-path is
tempted 2–2.5× more than the other families — the same pattern shows up, more
starkly, at 4B in §5.8, where it is central to explaining the non-replication.

---

## 5.5 Sensitivity Checks (Qwen3-1.7B, jlens)

**Status: run** (`experiments/SENSITIVITY_REPORT.md`). Pre-registered θ ∈
{70th, 80th, 90th pct} × band ∈ {[0.20,0.95], [0.25,0.90], [0.30,0.85]}.

**Direction never flips**: the minimal pre-registered robustness bar is met
everywhere. **Significance is not uniform**: confirmatory metrics
(`l_star_2afc`, `gap_2afc`) are significant at every setting; primary
metrics (`oscillation`, `gap`) are significant only near the default and
lose significance at θ=70/90 — because `ℓ_H` is the *first* band layer
crossing an arbitrary θ, so everything downstream of it inherits that
sensitivity, while the confirmatory metrics depend only on the
target/distractor comparison and are structurally insulated from it.

This section covers the θ/band axis only. §5.2 folds in the lens-method
axis (jlens vs. logit_lens); §5.8 adds the model-scale axis (1.7B vs. 4B) —
together these are the three independent robustness checks applied in this
study, and H3's confirmatory metric is the only result to pass all three.

---

## 5.6 Workspace Band Identification

**Status: run**, all 4 model×readout combinations, **not adopted** as a new
default. Full detail and the accuracy-curve caveat that motivates not
adopting it: `experiments/BAND_IDENTIFICATION_REPORT.md`. Summary: all four
combinations suggest a band starting later (54–69% of depth vs. the
pre-registered 25%) and extending to the final layer (vs. 90%), but
next-token accuracy — one of the three signals feeding the suggestion — rises
near-monotonically with depth in a way that isn't specific to a genuine
bounded workspace, so this is reported as a documented open question, not a
correction to the pre-registered analysis.

---

## 5.7 External Validity: Natural Stories

**Not run.** Deferred pending Natural Stories RT linkage
(`experiments/natural_stories_plan.md`). If run, should target the
dissociation gap (H3), not oscillation (H2) — see Discussion §6.4.

---

## 5.8 Qwen3-4B Replication — H1 and H3 Do Not Replicate

**Full detail**: `experiments/4B_REPLICATION_REPORT.md`. This is a
pre-registered replication attempt under the *identical* analysis plan as
§5.2–5.5 (same stimuli, same split procedure, same θ/band defaults, same
pre-screen gate) — a genuine second data point, not new hypothesis mining.

**Behavioral pre-screen**: passed cleanly on both readouts — arithmetic
100.0%, factual 91.7%, garden_path 83.3% (all above the 50% gate; arithmetic
and garden_path notably *higher* than 1.7B's 70.8%/75.0%).

**Table 5.3**: one-tailed exact p-values across all four model×readout
combinations (bold = p < 0.05):

| Metric | 1.7B jlens | 1.7B logit_lens | 4B jlens | 4B logit_lens |
|---|---|---|---|---|
| H1 primary | 0.109 | 0.145 | 0.469 | 0.095 |
| H1 confirmatory | **0.002** | **0.040** | 0.334 | 0.461 |
| H2 primary | **0.008** | 0.453 | 0.211 | 0.061 |
| H2 confirmatory | 0.461 | 0.266 | 0.432 | 0.348 |
| H3 primary | **0.011** | 0.193 | 0.379 | 0.224 |
| H3 confirmatory | **0.002** | **0.033** | 0.369 | 0.434 |

**Figure**: `out/figures/model_comparison_1p7b_vs_4b.png`.

**Every 4B cell is null**, including H3 confirmatory — the result that was
significant under every θ/band/lens-method combination tested on 1.7B. This
is not an underpowered null: median effect sizes at 4B are 0.0 for nearly
every metric (vs. +2 to +4 layers at 1.7B for the significant confirmatory
metrics), and 4B's sample sizes are, if anything, larger (e.g. `gap` n=17 at
4B vs. 14 at 1.7B). **H2 confirmatory remains a stable null across all four
model×readout cells**, a fourth independent line of evidence for §5.2's H2
conclusion.

**Why**: 4B is both more accurate and substantially less internally tempted
by the false-lead framing:

| | 1.7B | 4B |
|---|---|---|
| Arithmetic straightforward accuracy | 70.8% | **100.0%** |
| Arithmetic false-lead temptation rate (distractor in top-5) | 29.2% | **4.2%** |
| Internal temptation, median Δ `distractor_lead_layers` (FL−SF) | +4.5 layers | **+1.5 layers** |

The internal distractor-competition window shrinks roughly threefold at 4B.
If the false-lead framing barely perturbs the internal computation to begin
with, there is little dissociation window left for H3 to detect.

**Interpretation**: the confidence–correctness dissociation effect (H3),
where it occurs, is real and robust to every within-model analysis choice
tested — but it is evidently **scale- or capability-dependent, not a fixed
property of how these models handle false-lead prompts in general**. This
reframes rather than voids the 1.7B finding: a within-model-robust,
cross-model-fragile effect is itself a substantive and underappreciated
finding for single-model interpretability claims generally, not merely a
limitation of this particular study.

---

## 5.9 Summary Table

**Table 5.4**: Hypothesis support summary across all axes tested.

| Hypothesis (readout) | 1.7B jlens | 1.7B logit_lens | θ/band robust? | 4B (either readout) |
|---|---|---|---|---|
| H1 primary | ✗ p=0.109 | ✗ p=0.145 | n/a (null throughout) | ✗ null |
| H1 confirmatory | ✓ p=0.002 | ✓ p=0.040 (weak) | ✅ all settings | ✗ **does not replicate** |
| H2 primary | ✓ p=0.008 | ✗ p=0.453 | ❌ fails θ70/θ90 | ✗ null |
| H2 confirmatory | ✗ p=0.461 | ✗ p=0.266 | ✅ stable null | ✗ stable null |
| H3 primary | ✓ p=0.011 | ✗ p=0.193 | ❌ fails θ70/θ90/[.20,.95] | ✗ null |
| H3 confirmatory | ✓ p=0.002 | ✓ p=0.033 | ✅ all settings | ✗ **does not replicate** |

---

## Interpretation & Scope

**Key findings**: within **Qwen3-1.7B**, false-lead prompts produce a
**confidence–correctness dissociation (H3, confirmatory)** that is robust to
every θ, band, and lens-method choice tested, and a corresponding delay in
target-vs-distractor commitment (**H1, confirmatory**) that is directionally
robust but readout-magnitude-sensitive. **Neither effect replicates on
Qwen3-4B**, under any readout — the single most important qualifier on every
claim in this section. **Global-vocabulary oscillation (H2 primary)** is
significant only at the 1.7B/jlens/default-θ setting and fails every other
robustness check (θ, band, lens method, and model scale); the confirmatory
test that would show actual candidate-level revision is a stable null across
all four model×readout combinations, so **H2 is reported as an open question
about general instability, not a resolved backtracking signature**, and this
is now the best-supported conclusion in the paper precisely because it is
consistently null.

**Limitations**:
- **The dissociation-gap effect is Qwen3-1.7B-specific among the two model
  sizes tested; it does not replicate at 4B** (§5.8) — the previous framing
  of "single model, generalization untested" is no longer accurate or
  sufficient; this is now a positive finding of non-replication, not an
  open gap
- Primary H1/H3 metrics rest on 14 complete holdout pairs at 1.7B (17–18 at
  4B); confirmatory metrics on 18–26 across models — larger than run 1's
  n=6, but a third, larger model would still strengthen any scale claim
- One step in scale (1.7B → 4B, ≈2× compute) establishes non-replication at
  this specific step, not a monotonic trend; a third model size is needed to
  distinguish "the effect fades somewhere in this range" from "1.7B has some
  specific property a smooth scaling story wouldn't predict"
- All results reflect a single pre-registered dev/holdout split (seed 42)
  per model; we did not resample the split itself
- Arithmetic answers use single-token English number words with a few-shot
  frame for Qwen3 tokenization; a residual minority of arithmetic items
  still default to a digit continuation at 1.7B (resolved at 4B — 100%
  accuracy)
- No Natural Stories external validation in this run
- Hard-control n is small (2–6 in holdout) — suggestive, not independently
  conclusive
- Band identification suggests a different band than the pre-registered
  default, but was not adopted (§5.6) — an open methodological question, not
  resolved here

**Implications**:
- The dissociation gap is a usable, robust-*within-model* signature: models
  can be measurably confident before they are correct, detectable without an
  auxiliary classifier — but its presence should not be assumed to transfer
  across model scale without testing, which is itself a broader caution for
  single-model interpretability claims
- Delayed target-vs-distractor resolution is real within 1.7B, framed at the
  resolution the data support (narrow readout), and likewise does not
  transfer to 4B
- Oscillation's interpretation remains unresolved; treating it as a
  "revision" signature would overclaim relative to four independent null
  checks (confirmatory metric, θ-sensitivity, lens-method, and model scale)
- Connects to agentic-auditing agenda with an important caveat: readable
  internal uncertainty signatures may need per-model (re-)validation before
  being trusted as general auditing tools, rather than assumed portable
  across a model family
