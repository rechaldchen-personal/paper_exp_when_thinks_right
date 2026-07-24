# Logit-Lens Robustness Report (Appendix D)

**Date**: 2026-07-22 (1.7B); see `4B_REPLICATION_REPORT.md` for the 4B
comparison, which supersedes this report's scope for that model.

**Status**: Complete for Qwen3-1.7B. Implements Methods §4.6's promised
secondary readout: `02_run_experiment.py --readout logit_lens` (identity
transport, no fitted lens or `jlens` dependency) against the primary
`--readout jlens` traces from run 2, same stimuli, same pre-registered
analysis settings.

## Result (Qwen3-1.7B, holdout, one-tailed exact p)

| Metric | jlens (primary readout) | logit_lens | Both significant? |
|---|---|---|---|
| H1 primary (`l_star`) | 0.109 | 0.145 | no — consistent null |
| H1 confirmatory (`l_star_2afc`) | **0.0021** | **0.0402** | **yes**, though far weaker under logit_lens |
| H2 primary (`oscillation`) | **0.0077** | 0.453 | no — does not survive lens change |
| H2 confirmatory (`oscillation_2afc`) | 0.461 | 0.266 | no — consistent stable null |
| H3 primary (`gap`) | **0.0112** | 0.193 | no — does not survive lens change |
| H3 confirmatory (`gap_2afc`) | **0.0016** | **0.0335** | **yes** — the only metric significant everywhere |

## Interpretation

**H3 confirmatory is now the best-attested result in the study**: significant
under every θ (70/80/90 pct), every band ([0.20,0.95]/[0.25,0.90]/[0.30,0.85])
in `SENSITIVITY_REPORT.md`, *and* under both lens readouts here. That is
robustness across three independent axes of analysis choice — this is what
"robust" should mean when used to describe H3 in the paper, and the strongest
claim the 1.7B result supports.

**H1 confirmatory holds under both readouts but far more marginally under
logit_lens** (p=0.040, barely under the 0.05 line, vs. p=0.0021 under jlens).
The direction is consistent, so the qualitative claim (delayed target-vs-
distractor commitment) survives, but the paper should not describe H1 as
"robust across settings" without noting this readout-dependent weakening —
that phrase should be reserved for H3.

**H2 primary loses significance entirely under logit_lens** (0.453 vs. 0.0077
under jlens) — a third line of evidence (alongside the θ-fragility in
`SENSITIVITY_REPORT.md` and the confirmatory null already in the paper)
against treating H2's primary result as a real, lens-independent phenomenon.
**H2 confirmatory remains a stable null under both readouts**, reinforcing
that the "revision between candidates" reading was correctly rejected.

## What this changes in the paper

- The Results/Discussion "robust across every setting" language for H3 can
  now honestly cite lens-method as a third axis, not just θ/band.
- H1's robustness claim needs the same qualifier added: robust in direction,
  but the logit_lens p-value (0.040) is close enough to the threshold that
  it should be reported explicitly rather than folded into a blanket "robust"
  statement.
- No change to H2's status — this adds evidence, doesn't alter the verdict.

## Caveat

This report covers Qwen3-1.7B only. The 4B replication (own lens fit, own
logit-lens traces) is the more consequential finding and is reported
separately in `4B_REPLICATION_REPORT.md` — read that report for the current
overall picture; this one documents the lens-method axis in isolation.
