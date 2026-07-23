# Section 5: Results

**Status**: Filled from Qwen3-1.7B run 2 (2026-07-22), on the rebuilt stimulus
set, per `experiments/PRE_REGISTRATION_AMENDMENT.md`. Run 1 (2026-07-17) is
withdrawn — see `out/ANALYSIS_NOTE.md` — and its numbers do not appear here.

**Validation status**:
- ✅ Stimuli: 156 items, 72 strictly matched pairs (24/family) + 12 hard
  controls, all pass `validate_stimuli.py` (7 design rules)
- ✅ Behavioral pre-screen passed on all three families (amendment §8 gate):
  factual 95.8%, garden-path 75.0%, arithmetic 70.8% straightforward top-1
  accuracy (all ≥ 50% threshold; run 1's arithmetic was 3.6%)
- ✅ Real data collected: `out/traces_run2.json` (156 stimuli),
  `out/analysis_real/report.json`
- ✅ Figures in `out/figures/`
- ✅ Sensitivity checks run: `experiments/SENSITIVITY_REPORT.md`
- ⏸ Qwen 4B replication deferred
- ⏸ Natural Stories external-validity analysis not run
- ⏸ Logit-lens secondary readout not integrated

**Analysis settings (pre-registered)**:
- Model: Qwen/Qwen3-1.7B (27 lens layers)
- Lens: Jacobian lens fitted on 1000 × 128 FineWeb prompts (`out/lens_qwen3_1p7b.pt`)
- Dev/holdout split: 60/40 by pair (`--dev-split 0.6`, seed 42) → 50 dev pairs, 34 holdout pairs
- Theta (ΔH threshold): 3.975 nats (80th percentile of straightforward condition, computed on **dev only**)
- Workspace band: [0.25, 0.90]
- Two readouts per hypothesis, both pre-registered (`PRE_REGISTRATION_AMENDMENT.md` §4):
  **primary** (global argmax, as originally specified) and **confirmatory**
  (restricted to {target, distractor})

---

## 5.1 Sanity Check: Excess-Entropy Profile

**Figure 5.0**: `out/figures/entropy_curves.png` — layer-wise median excess
entropy ΔH for Qwen3-1.7B by condition on holdout items.
- X-axis: layer index (0–26; 27 source layers in the fitted lens)
- Y-axis: excess entropy ΔH (nats)
- Workspace band shaded at fractional depth [0.25, 0.90]
- Theta threshold overlaid at 3.975 nats

| Model | Band used | Theta (nats) | Notes |
|-------|-----------|--------------|-------|
| Qwen3-1.7B | [0.25, 0.90] | 3.975 | Pre-registered fractional band; θ from SF 80th pct on dev |
| Qwen 4B | — | — | Deferred |

---

## 5.2 Main Findings: Commitment Dynamics

All tests compare **false_lead − straightforward** on holdout pairs, one-tailed
exact signed-rank (`03_analyze.py`; see `out/ANALYSIS_NOTE.md` for why the
test is now exact and one-tailed rather than the scipy default). Each
hypothesis is reported as a pair: **primary** (global argmax over the full
vocabulary, the originally pre-registered metric) and **confirmatory**
(restricted to {target, distractor}), per amendment §4. The permitted claim
for each combination of outcomes was fixed *before* this run (amendment §5)
and is applied here without adjustment.

### H1: Later Commitment Under False Lead

**Prediction**: ℓ* (commitment layer) larger for false_lead than straightforward.

**Figure 5A**: `out/figures/H1_commitment_layer.png`

**Primary** (target must beat the entire vocabulary):
- Median ℓ* (straightforward): **21.0** layers (n=21 with defined ℓ*)
- Median ℓ* (false_lead): **21.0** layers (n=14)
- Median paired difference (FL − SF): **0.0** layers
- Exact one-tailed signed-rank: W₊=**22.5**, p=**0.109**, n_pairs=**14** (7 nonzero)
- **Result**: H1 primary **NOT SIGNIFICANT**

**Confirmatory** (target need only beat the distractor):
- Median paired difference (FL − SF): **+2.0** layers
- Exact one-tailed signed-rank: W₊=**108.0**, p=**0.0021**, n_pairs=**19** (15 nonzero)
- **Result**: H1 confirmatory **SUPPORTED**, and robust — significant at
  p≈0.002 across every pre-registered θ (70th/80th/90th pct) and band
  ([0.20,0.95]/[0.25,0.90]/[0.30,0.85]) setting tested
  (`experiments/SENSITIVITY_REPORT.md`)

**Interpretation (per amendment §5, H1 row)**: primary null / confirmatory
significant → narrow-readout claim only. False-lead prompts reliably delay
the point where the target overtakes the distractor specifically, but this
does not translate into a later *global* commitment layer — plausibly because
by the time the target beats the distractor, it has often already beaten
everything else too, so the extra delay is absorbed before the primary metric
would register it. We do not claim general delayed commitment; we claim
delayed target-over-distractor resolution.

---

### H2: Oscillation Under False Lead

**Prediction**: Oscillation depth (top-1 identity changes after ℓ_H) higher
for false_lead.

**Figure 5B**: `out/figures/H2_oscillation.png`
**Figure 5B-detail**: `out/figures/H2_oscillation_hist.png`

**Primary** (global top-1 identity changes):
- Median oscillation (straightforward): **1.0** changes (n=28)
- Median oscillation (false_lead): **2.5** changes (n=28)
- Median paired difference (FL − SF): **+0.5** changes
- Exact one-tailed signed-rank: W₊=**182.0**, p=**0.0077**, n_pairs=**28** (21 nonzero)
- **Result**: H2 primary **SUPPORTED** at the pre-registered default, but
  **not robust to θ**: significant at θ=80 (default) and band=[0.30,0.85]
  (p=0.004) and [0.20,0.95] (p=0.028), but **not significant** at θ=70
  (p=0.147) or θ=90 (p=0.480) — see `experiments/SENSITIVITY_REPORT.md`

**Confirmatory** (target-vs-distractor preference flips only):
- Median paired difference (FL − SF): **0.0** changes
- Exact one-tailed signed-rank: W₊=**41.0**, p=**0.46**, n_pairs=**28** (12 nonzero)
- **Result**: H2 confirmatory **NOT SIGNIFICANT**, and consistently so:
  p ranges 0.46–0.62 across every θ/band sensitivity setting — the most
  stable null in the entire analysis

**Hard controls** (no tempting distractor, difficulty-only): median
oscillation **2.0** changes (n=6 holdout), comparable to false_lead's 2.5 and
well above straightforward's 1.0.

**Interpretation (per amendment §5, H2 row — primary significant /
confirmatory null → "unattributed churn" rule)**: there is more top-1 churn
under false-lead by the pre-registered global metric, but this is **not**
attributable to the model wavering between the target and the distractor —
the confirmatory metric that would show that directly is null and stable
under every setting tested. The hard-control comparison corroborates this:
items with no tempting distractor at all oscillate almost as much as
false-lead items, meaning oscillation tracks something closer to general
difficulty or instability than to false-lead-specific revision. **We do not
claim internal revision between candidate answers.** The primary effect is
real but of unresolved origin, and its own significance is fragile to the θ
threshold, which is additional evidence against treating it as a robust
phenomenon.

---

### H3: Dissociation Gap Under False Lead

**Prediction**: Gap ℓ* − ℓ_H larger for false_lead (confidently-wrong window).

**Figure 5C**: `out/figures/H3_dissociation_gap.png`

**Primary**:
- Median gap (straightforward): **1.0** layers (n=21)
- Median gap (false_lead): **1.0** layers (n=14)
- Median paired difference (FL − SF): **+1.0** layers
- Exact one-tailed signed-rank: W₊=**59.0**, p=**0.0112**, n_pairs=**14** (11 nonzero)
- **Result**: H3 primary **SUPPORTED** at the pre-registered default, but
  fragile to θ in the same way as H2: significant at θ=80 (p=0.011) and
  band=[0.30,0.85] (p=0.012), **not significant** at θ=70 (p=0.184), θ=90
  (p=0.217), or band=[0.20,0.95] (p=0.085)

**Confirmatory**:
- Median paired difference (FL − SF): **+4.0** layers
- Exact one-tailed signed-rank: W₊=**135.5**, p=**0.0016**, n_pairs=**18** (17 nonzero)
- **Result**: H3 confirmatory **SUPPORTED, and the most robust result in the
  study** — significant at every θ/band setting tested, p ranging
  0.0013–0.033, never approaching the primary metric's fragility

**Interpretation (per amendment §5)**: both readouts agree in direction and
both are significant at the pre-registered default, which is the strongest
joint outcome available. Unlike H1/H2, H3's primary metric is directionally
consistent with the confirmatory one everywhere (median_diff never negative
across any sensitivity setting), and the confirmatory metric is significant
everywhere. **This is the headline finding**: false-lead prompts open a
larger confidence–correctness dissociation window, and this holds up under
every robustness check we ran. We lead with the confirmatory number
(median Δ=+4.0 layers, p=0.0016) as the primary evidentiary claim and report
the pre-registered global metric (median Δ=+1.0, p=0.011) as corroborating
but θ-sensitive.

**Hard controls**: median gap **1.5** layers (n=2 holdout — too few to test,
but not obviously larger than straightforward's 1.0), consistent with the
gap being a false-lead-specific phenomenon rather than a general
difficulty artifact, though this is suggestive only given n=2.

---

## 5.3 Case Studies: Heatmaps

Representative holdout heatmaps in `out/figures/heatmaps/` (also in
`out/analysis_real/heatmaps/`, 102 items, gitignored as regenerable):

| Pair | Family | Conditions | Path |
|------|--------|------------|------|
| capital_cl | factual | SF / FL | `capital_cl_*.png` |
| capital_eg | factual | SF / FL | `capital_eg_*.png` |
| legs_spider | factual | SF / FL | `legs_spider_*.png` |
| gp_horse | garden_path | SF / FL | `gp_horse_*.png` |

**Caption (example)**: Layer × position heatmaps for factual pairs capital_cl
/ capital_eg and garden-path pair gp_horse. False-lead panels typically show
longer distractor-lead stretches and a wider confidence–correctness gap
relative to straightforward prompts.

---

## 5.4 Distractor Temptation

**Table 5.1** (holdout pairs; full table in `out/figures/table_distractor_temptation.md`):

| Family | Pair | Straightforward (distractor_lead_layers) | False_lead (distractor_lead_layers) |
|--------|------|------------------------------------------|-------------------------------------|
| Factual | capital_eg | 7 | 12 |
| Factual | legs_spider | 4 | 11 |
| Factual | element_lightest | 2 | 12 |
| Garden-path | gp_horse | 8 | 11 |
| Garden-path | gp_runner | 14 | 17 |
| Garden-path | gp_soldier | 5 | 9 |
| Arithmetic | arith_1_2_3 | 10 | 16 |
| Arithmetic | arith_2_2_4 | 2 | 7 |

**Condition medians** (holdout, primary metric readout):
- Straightforward distractor_lead_layers: **4.0** (n=28)
- False_lead distractor_lead_layers: **8.5** (n=28)

**Behavioral temptation rate** (distractor reaches the model's final top-5,
from the pre-screen; a diagnostic of manipulation strength, not a gate):
factual 25.0%, arithmetic 29.2%, **garden-path 66.7%**. Garden-path stimuli
tempt the model roughly 2–2.5× more often than the other two families, which
is consistent with any effects concentrating there rather than being uniform
across families — worth a per-family breakdown of H1–H3 as future work.

---

## 5.5 Sensitivity Checks

**Status: run** (see `experiments/SENSITIVITY_REPORT.md` for the full table
and commands). Pre-registered θ ∈ {70th, 80th, 90th percentile} and band ∈
{[0.20,0.95], [0.25,0.90], [0.30,0.85]}, same dev/holdout split throughout.

**Direction never flips**: median_diff is non-negative for every
primary/confirmatory metric at every setting tested — the minimal
pre-registered robustness bar is met everywhere.

**Significance is not uniform**, and this differs sharply by metric:

- **Confirmatory metrics are robust.** `l_star_2afc` (H1) and `gap_2afc`
  (H3) are significant at every θ/band setting tested (p ranging
  0.0013–0.033), independent of the specific threshold chosen.
- **Primary metrics are θ-fragile.** `oscillation` (H2) and `gap` (H3) are
  significant only near the pre-registered θ=80 default and lose
  significance at θ=70 or θ=90. `l_star` (H1) is null at every setting.

This asymmetry is expected given how the metrics are constructed: `l_H`
(confidence-collapse layer, used by `gap` and `oscillation`) is the *first*
band layer crossing an arbitrary θ, so its definition — and hence everything
downstream of it — is sensitive to where θ is drawn. The confirmatory metrics
depend only on the target/distractor logit comparison and are structurally
insulated from this. Practically: **the confirmatory readout is the one to
trust for robustness claims**; the primary readout, where it agrees, should
be read as corroborating at the specific pre-registered setting rather than
as independently robust.

---

## 5.6 External Validity: Natural Stories

**Not run.** Deferred pending Natural Stories RT linkage
(`experiments/natural_stories_plan.md`).

---

## 5.7 Summary Table

**Table 5.2**: Hypothesis support summary (Qwen3-1.7B, holdout, run 2).

| Hypothesis | Readout | n_pairs | Median Δ | p (one-tailed) | Robust to sensitivity? | Result |
|---|---|---|---|---|---|---|
| H1: later commitment | primary | 14 | 0.0 | 0.109 | — | ✗ not sig. |
| H1: later commitment | confirmatory | 19 | +2.0 | **0.0021** | ✅ all settings | ✓ **supported (narrow)** |
| H2: more oscillation | primary | 28 | +0.5 | **0.0077** | ❌ fails at θ70/θ90 | ✓ sig. but fragile |
| H2: more oscillation | confirmatory | 28 | 0.0 | 0.46 | ✅ stable null | ✗ **not supported** |
| H3: larger gap | primary | 14 | +1.0 | **0.0112** | ❌ fails at θ70/θ90/[.20,.95] | ✓ sig. but fragile |
| H3: larger gap | confirmatory | 18 | +4.0 | **0.0016** | ✅ all settings | ✓ **supported — headline** |

---

## Interpretation & Scope

**Key findings**: On Qwen3-1.7B with a fitted Jacobian lens and a rebuilt,
strictly matched stimulus set, false-lead prompts produce a **robust
confidence–correctness dissociation (H3)**: models become confident before
they are correct, and this window widens under false lead in a way that
survives every θ and band we tested. False-lead prompts also delay the point
where the target overtakes the distractor specifically (**H1**, confirmatory
reading only, likewise robust). **Global-vocabulary top-1 oscillation
increases under false lead (H2 primary), but this is not evidence of
revision between the candidate answers** — the targeted confirmatory test is
null and stable, and hard-control items without any tempting distractor
oscillate almost as much as false-lead items. We report H2 as an open
question about general instability, not as a resolved backtracking
signature.

**Limitations**:
- Single model (Qwen3-1.7B); Qwen 4B deferred
- Primary H1/H3 metrics rest on 14 complete holdout pairs; confirmatory
  metrics on 18–19 — larger than run 1's n=6, but still modest
- All results reflect a single pre-registered dev/holdout split (seed 42);
  we did not resample the split itself
- Arithmetic answers use single-token English number words with a few-shot
  frame for Qwen3 tokenization; a residual minority of arithmetic items
  still default to a digit continuation (see `experiments/README_GPU_PHASE.md`
  troubleshooting)
- No Natural Stories external validation in this run
- Hard-control n is small (2–6 in holdout) — suggestive for H2's reframing,
  not independently conclusive for H3

**Implications**:
- The dissociation gap is a usable, robust, pre-output signature: models can
  be measurably confident before they are correct, and this is detectable
  without any auxiliary classifier
- Delayed target-vs-distractor resolution is real and robust, but framed at
  the resolution the data support (narrow readout), not as general delayed
  commitment
- Oscillation is real but its interpretation is unresolved; treating it as a
  "revision" signature would overclaim relative to the confirmatory test and
  the hard-control comparison
- Connects to agentic-auditing agenda: readable internal uncertainty outside
  the final answer token, via the dissociation gap specifically
