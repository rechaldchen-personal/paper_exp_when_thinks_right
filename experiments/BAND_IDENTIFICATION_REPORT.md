# Workspace Band Identification Report (Appendix C)

**Date**: 2026-07-24. Ran `04_identify_band.py` on all four model×readout
combinations (200 held-out FineWeb prompts each, `--skip-prompts 1000`).

## Result

| Model / readout | n_layers | Suggested band (layers) | Suggested band (fraction) |
|---|---|---|---|
| 1.7B jlens | 27 | [14, 26] | [0.538, 1.0] |
| 1.7B logit_lens | 28 | [18, 27] | [0.667, 1.0] |
| 4B jlens | 35 | [23, 34] | [0.676, 1.0] |
| 4B logit_lens | 36 | [24, 35] | [0.686, 1.0] |

Pre-registered default (all main analyses in this project): **[0.25, 0.90]**.

All four suggestions point the same direction relative to the default: start
noticeably later (54–69% of depth vs. the default's 25%) and extend all the
way to the final layer (fraction 1.0 vs. the default's 0.90 cutoff).

## Caveat: read before treating this as a correction

I inspected the raw per-layer curves (not just the composite score) before
writing this up, because a data-driven "band" is only as good as its inputs.
Next-token accuracy on natural text — one of the three signals feeding the
composite score — rises close to monotonically with depth (e.g. 1.7B jlens:
0.000 at layer 0 → 0.006 at layer 9 → 0.237 by layer 24), which is expected
of any competent language model and is not by itself evidence of a bounded
"workspace" the way Gurnee et al.'s criterion intends. A monotonically rising
accuracy curve will always push the composite score, and therefore the
suggested band, toward the deepest layers — independent of whether there is
a genuine sparse workspace there or the signal is just "later is better."

Kurtosis, the second signal, is *not* simply monotonic (1.7B jlens: 0.47 →
7.41 → 2.54 → 1.02 → 0.31 → −0.86 → −0.04 → 6.90 → 13.18 across layers 0–24),
so the late-band suggestion isn't purely an artifact of the accuracy curve —
there is real non-monotonic structure supporting a genuine late-layer
concentration. But the accuracy curve's contribution to the late bias should
be discounted somewhat, not read at face value.

4B's PNG plots (`out/figures/band_identification_*.png` for 1.7B only) were
not synced before the pod was shut down, so the 4B curves have not been
visually inspected — only the JSON summary statistics above. This is a real
gap; regenerating the 4B plots would need another GPU session (cheap, ~10 min,
data is already collected) or could be done from the raw `out/band_identification_4b_*.json`
files without a GPU if the plotting code is factored out (it currently isn't
— see `04_identify_band.py`'s plotting block).

## Decision: not adopted as a new default

Given the accuracy-curve caveat above and that 4B's plots haven't been
visually checked, this report does **not** recommend amending the
pre-registered band. All main analyses (run 2, sensitivity checks, 4B
replication) correctly and intentionally used the pre-registered default
throughout — adopting a different band now, after already knowing the 4B
result, would be exactly the kind of post-hoc tuning pre-registration exists
to prevent. If a data-driven band is adopted in future work, it should be
via a fresh pre-registration amendment, applied prospectively to new data
collection, not retrofitted onto results already in hand.

## What this does NOT change

The main findings (1.7B H3 robust across θ/band/lens-method; H3/H1 do not
replicate at 4B) are unaffected — none of the sensitivity or replication
analysis used a band this report would suggest changing to.
