# Sensitivity Report (run 2, Qwen3-1.7B)

**Date**: 2026-07-22
**Data**: `out/traces_run2.json` (same traces as the primary analysis)
**Checks**: pre-registered in `PRE_REGISTRATION.md` §2 (band) and §3 (theta):
θ ∈ {70th, 80th (primary), 90th} percentile; band ∈ {[0.20,0.95], [0.25,0.90]
(primary), [0.30,0.85]}. Same 60/40 dev/holdout split (seed 42) throughout —
only θ and band vary.

Commands: `03_analyze.py --traces out/traces_run2.json --dev-split 0.6` with
`--theta-pct {70,90}` or `--band {0.20 0.95 | 0.30 0.85}`. Full reports in
`out/sensitivity/{theta70,theta90,band_wide,band_narrow}/report.json`.

## Result

| Setting | θ (nats) | `l_star` p | `gap` p | `osc` p | `l_star_2afc` p | `gap_2afc` p | `osc_2afc` p |
|---|---|---|---|---|---|---|---|
| primary (θ80, [.25,.90]) | 3.975 | 0.109 | **0.011** | **0.008** | **0.002** | **0.002** | 0.461 |
| θ70 | 3.130 | 0.109 | 0.184 | 0.147 | **0.002** | **0.008** | 0.610 |
| θ90 | 5.038 | 0.109 | 0.217 | 0.480 | **0.002** | **0.033** | 0.625 |
| band [0.20,0.95] | 4.300 | 0.109 | 0.085 | **0.028** | **0.002** | **0.006** | 0.530 |
| band [0.30,0.85] | 3.672 | 0.109 | **0.012** | **0.004** | **0.002** | **0.001** | 0.515 |

Bold = p < 0.05. Direction (sign of median_diff) is non-negative in every cell
of every setting — never flips — which is the letter of the pre-registered
criterion ("H1–H3 direction unchanged").

## Interpretation

**The confirmatory 2AFC metrics are the robust ones.** `l_star_2afc` is
significant at p≈0.002 in all five settings; `gap_2afc` is significant
everywhere, weakening only at θ90 (p=0.033, still <0.05). Neither depends on
which θ or band was chosen.

**The pre-registered primary metrics are not robust to θ**, though they
satisfy the letter of the pre-registration (direction never flips). `gap`
(H3) and `oscillation` (H2) are significant at the primary θ=80 and at
θ-adjacent band settings, but lose significance at θ70 and θ90 — i.e., moving
the confidence-collapse threshold by ±10 percentile points is enough to flip
the primary H2/H3 verdicts from significant to not. `l_star` (H1) is
uniformly non-significant regardless of setting, consistent with the primary
result.

This is a real fragility, not a rounding artifact: `l_H` (confidence-collapse
layer) is defined as the *first* band layer where ΔH crosses θ, so a
percentile-point change in θ can shift which items even have `l_H` defined,
which propagates into `gap` (which needs `l_H`) and `oscillation` (whose
window starts at `l_H`). `l_star_2afc` and `gap_2afc` are comparatively
insulated because `l_star_2afc` only depends on the target/distractor logit
comparison, not on where entropy crosses an arbitrary line — this is precisely
the coverage/robustness advantage the confirmatory metrics were added for in
`PRE_REGISTRATION_AMENDMENT.md` §4, now confirmed empirically rather than by
argument alone.

## What this changes about the headline claim

H3 (dissociation gap) is still the best-supported hypothesis, but the
**confirmatory reading is now the one to lead with**, not the primary one: the
2AFC gap is significant at every θ and band tested, while the primary gap is
significant only near the pre-registered default. Report both, but do not
describe the primary H3 result as "robust to sensitivity checks" without this
caveat — describe the *confirmatory* result that way instead.

H2 (oscillation) is unaffected in substance: the primary metric was already
unsupported once you require agreement with the confirmatory readout
(amendment §5, "significant / null" row), and that null confirmatory result
(`oscillation_2afc`, p≈0.5–0.6 across all five settings) is itself the most
stable number in this entire table. The θ-sensitivity of the primary
oscillation metric is now moot for the paper's claims, but it is additional
evidence that the primary oscillation signal was measuring something threshold
-dependent rather than a robust phenomenon.

H1 is unchanged: null under the primary metric at every setting, significant
under the confirmatory metric at every setting — the narrow-readout-only
reading holds regardless of θ/band.

## Caveat

θ-sensitivity was only run on the pre-registered percentile grid
(70/80/90) and the pre-registered band grid; it is not a continuous stability
curve. A denser sweep would show exactly where the primary metrics cross
significance, which is future work, not required for the current claims.
