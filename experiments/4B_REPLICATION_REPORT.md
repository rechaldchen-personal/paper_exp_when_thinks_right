# Qwen3-4B Replication Report

**Date**: 2026-07-24
**Status**: Complete. This is a pre-registered replication attempt under the
exact same analysis plan as run 2 (`PRE_REGISTRATION.md` +
`PRE_REGISTRATION_AMENDMENT.md`, whose "Applies to" clause already covers
"any later model" — no new amendment was required to run this).

## Headline finding

**H3 (dissociation gap) — the most robust result in the 1.7B study, significant
under every θ, band, and lens-readout combination tested — does not replicate
on Qwen3-4B, under either readout.** Neither does H1 (confirmatory). This is
reported as a genuine, important negative result, not a setback to explain
away: it means the 1.7B finding does not generalize past a single small model,
and the paper's claims must be scoped accordingly.

## Setup

Same stimuli (156 items, 72 matched pairs + 12 hard controls), same
pre-registered pipeline, same behavioral pre-screen gate, same dev/holdout
split procedure (60/40, seed 42), same θ/band defaults ([0.25, 0.90], 80th
percentile). Two readouts collected, exactly mirroring the 1.7B comparison:
`out/traces_4b.json` (jlens, fresh lens fit — `out/lens_qwen3_4b.pt`, 458MB)
and `out/traces_4b_logitlens.json` (identity transport, no lens needed).

**Behavioral pre-screen**: passed cleanly on both readouts (behavioral
accuracy is a property of the model's own output, identical regardless of
readout) — arithmetic 100.0%, factual 91.7%, garden_path 83.3%. All comfortably
above the 50% gate, and arithmetic/garden_path are notably *higher* than
1.7B's 70.8%/75.0%.

## Results: full comparison across model x readout

One-tailed exact signed-rank p-values (bold = p < 0.05; `*` marks results
discussed further below):

| Metric | 1.7B jlens | 1.7B logit_lens | 4B jlens | 4B logit_lens |
|---|---|---|---|---|
| H1 primary (`l_star`) | 0.109 | 0.145 | 0.469 | 0.095 |
| H1 confirmatory (`l_star_2afc`) | **0.002** | **0.040** | 0.334 | 0.461 |
| H2 primary (`oscillation`) | **0.008** | 0.453 | 0.211 | 0.061 |
| H2 confirmatory (`oscillation_2afc`) | 0.461 | 0.266 | 0.432 | 0.348 |
| H3 primary (`gap`) | **0.011** | 0.193 | 0.379 | 0.224 |
| H3 confirmatory (`gap_2afc`) | **0.002** | **0.033** | 0.369 | 0.434 |

Figure: `out/figures/model_comparison_1p7b_vs_4b.png`.

Every 4B cell is null. Effect sizes collapse alongside significance — median
differences at 4B are 0.0 for nearly every metric (vs. +2 to +4 layers at
1.7B for the confirmatory H1/H3 metrics), so this is not merely underpowered:
the point estimate of the effect itself is near zero, not just noisy. Sample
sizes if anything favor 4B (e.g. `gap` n_pairs=17 at 4B vs 14 at 1.7B;
`l_star_2afc` n=26 vs 19), so this isn't a coverage/power artifact either.

`H2 confirmatory` (`oscillation_2afc`) is the one metric that is a stable
null everywhere — all four conditions, consistent with the 1.7B finding
already in the paper that oscillation does not reflect target-vs-distractor
revision. 4B adds a third and fourth independent line of evidence for that
conclusion.

## Why: behavioral and internal evidence for scale-dependence

This isn't an unexplained void — the data point at a specific, coherent
mechanism: **4B is both more accurate and substantially less internally
tempted by the false-lead framing than 1.7B**, especially for arithmetic.

**Behavioral temptation rate** (false-lead distractor reaching the model's
final top-5 — how much does the framing actually mislead the model's output):

| Family | 1.7B | 4B |
|---|---|---|
| factual | 25.0% | 29.2% |
| arithmetic | 29.2% | **4.2%** |
| garden_path | 66.7% | 54.2% |

**Internal temptation** (median `distractor_lead_layers`, primary/jlens —
how many band layers the distractor internally outranks the target):

| Condition | 1.7B jlens | 4B jlens |
|---|---|---|
| straightforward | 4.0 | 5.5 |
| false_lead | 8.5 | 7.0 |
| **Δ (FL − SF)** | **+4.5** | **+1.5** |

The false-lead framing widens internal distractor competition by 4.5 layers
at 1.7B but only 1.5 at 4B — roughly a 3x shrinkage, tracking the collapse
in the downstream H1/H3 effects. Reading these together: **4B is simply
harder to fool**, both in what it outputs and in what its internal
representations do along the way. If the false-lead manipulation barely
perturbs the internal computation to begin with, there is little dissociation
window left for H3 to detect — the mechanism the hypothesis describes may
require a model susceptible enough to actually be tempted internally, which
1.7B was and 4B largely is not.

This reframes the contribution rather than voiding it: **the confidence–
correctness dissociation effect, where it occurs, is real and robust to every
analysis choice we tested — but it is evidently scale- or capability-
dependent, not a fixed property of how these models process false-lead
prompts in general.** That is itself a substantive, publishable finding, and
a more interesting one than uniform replication would have been.

## What this changes in the paper

- **Abstract**: the headline claim must be scoped to Qwen3-1.7B explicitly,
  with the 4B null and the temptation-rate explanation reported alongside it
  as a primary finding, not a footnote.
- **Results**: needs a new major section reporting this table and the
  behavioral evidence above.
- **Discussion**: the "robustness" framing throughout (built on the θ/band/
  lens-method sensitivity checks) needs an explicit new axis — robust *within*
  a model is not the same claim as robust *across* models, and the paper
  must not conflate them. The scale-dependence explanation belongs in §6.2 or
  a new subsection.
- **Limitations**: "single model (Qwen3-1.7B)" is no longer accurate — it
  should become "the effect is specific to Qwen3-1.7B among the two model
  sizes tested; it did not replicate at 4B" — a stronger and more honest
  limitation than the placeholder it replaces.

## Caveats on this replication itself

- Band identification (see `BAND_IDENTIFICATION_REPORT.md`) suggests a
  different band than the pre-registered [0.25, 0.90] default, consistently
  across all four model/readout combinations — this analysis used the
  pre-registered default throughout, as required for a same-plan replication.
  Whether the null at 4B would look different under a data-driven band is an
  open question, not addressed here (would require a fresh amendment and
  re-run, since changing the band post-hoc after seeing this null would be
  exactly the kind of tuning pre-registration exists to prevent).
- This is one replication at one larger size (4B vs 1.7B, ~2x compute). It
  establishes non-replication at this specific step up in scale, not a
  monotonic trend — a third model size would be needed to distinguish
  "the effect disappears somewhere between 1.7B and 4B" from
  "the effect requires something specific to 1.7B that a smooth scaling
  story wouldn't predict."
