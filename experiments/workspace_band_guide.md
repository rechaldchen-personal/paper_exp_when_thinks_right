# Workspace Band Identification Guide

**Status (2026-07-22)**: implemented as `04_identify_band.py` +
`band_analysis.py` (math unit-tested in `test_band_analysis.py`, all CPU-only
and passing). Execution requires GPU — not yet run. This guide's "tools TBD"
note below is resolved; see "How to Run" for the actual commands.

Per Gurnee et al. (2026), the "workspace" is a sparse band of layers where verbalizable information surfaces. Early and late layers don't reliably represent answers.

The paper uses **hardcoded band [0.25, 0.90]** as a default. For proper results, **identify the band per-model** using the criteria below.

## Criteria (from Gurnee et al. Fig 28)

Three complementary signals identify the band:

### 1. Excess Kurtosis

Excess kurtosis of lens readout entropy: peaks within the workspace.

```python
# Pseudo-code; adapt to your data
kurtosis_by_layer = []
for layer_idx in range(n_layers):
    ents = [entropy[layer_idx][t] for t in range(n_positions)]
    k = scipy.stats.kurtosis(ents)
    kurtosis_by_layer.append(k)

band_start = argmax(kurtosis_by_layer[:middle])
band_end = argmax(kurtosis_by_layer[middle:]) + middle
```

### 2. Autocorrelation of Rank Trajectories

Within the workspace, target rank stays stable; outside, it's noisy.

> **Implementation note**: the pseudocode originally sketched here (fixed
> layer, autocorrelating ranks across query positions) has no natural
> ordering to autocorrelate against — query positions aren't sequential in
> any sense the autocorrelation would be measuring. `band_analysis.py`'s
> `layer_autocorr()` instead computes, for each pair of adjacent layers, the
> Pearson correlation ACROSS the held-out corpus between a token's rank at
> layer L-1 and at layer L — this is genuinely layer-indexed and captures
> the same intent ("rank stays stable within the workspace") in a
> well-defined way. See that function's docstring for the full rationale;
> this is a documented operationalization choice, not a verified replication
> of Gurnee et al.'s exact procedure (their code was not available to us).

### 3. Next-Token Accuracy on Held-Out Examples

Models' top-1 predictions on held-out text: peaks in the workspace.

```python
# Run lens on a small corpus of held-out examples
# Compute fraction of correct next-token predictions per layer
correct_by_layer = []
for layer_idx in range(n_layers):
    preds = argmax(logits[layer_idx]) for each position
    acc = mean(preds == true_tokens)
    correct_by_layer.append(acc)
```

## How to Run

**On the GPU box** (~5-15 min for 200 prompts — far cheaper than lens
fitting, since this is forward-passes-only, same as trace collection):

```bash
# 1. Primary readout (needs the fitted lens)
python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout jlens \
    --lens out/lens_qwen3_1p7b.pt --n-prompts 200 \
    --out out/band_identification_jlens.json

# 2. Cross-check with logit-lens (no fitted lens needed) — should suggest a
#    similar band; if it doesn't, that's itself a robustness finding
python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout logit_lens \
    --n-prompts 200 --out out/band_identification_logitlens.json

# 3. Inspect out/figures/band_identification_{jlens,logit_lens}.png and the
#    "suggested_band_*" fields in each JSON. The suggested band is a starting
#    point (longest contiguous layer run above a composite-score threshold,
#    default 40th percentile) — look at the plot before trusting it.

# 4. If you adopt a new band, update the pre-registered default everywhere
#    it's used: 03_analyze.py --band, and re-run analysis + sensitivity
#    checks with the new band as the new default (this is a pre-registration
#    amendment, not a post-hoc tweak — see PRE_REGISTRATION_AMENDMENT.md for
#    the amendment format).
```

`--n-prompts 200` is a reasonable default; raise it if the curves look noisy
in the plot. `--skip-prompts` (default 1000) keeps this corpus held-out from
whatever corpus size was used to fit the lens being evaluated — match it to
that value if you fit with a different `--n-prompts`.

## Default Fallback

If you don't run this, the hardcoded [0.25, 0.90] works for most Qwen models
per the original Gurnee et al. default, but may miss model-specific effects —
this is exactly what run 2's sensitivity checks
(`experiments/SENSITIVITY_REPORT.md`) probe indirectly, by testing whether
results hold under nearby bands, without formally identifying the "true" one.

## Output for Paper

In Appendix C, report (all produced by `04_identify_band.py`):
- Kurtosis curve (by layer) — `out.json["kurtosis"]`
- Autocorrelation curve (by layer) — `out.json["autocorrelation"]`
- Next-token accuracy curve (by layer) — `out.json["accuracy"]`
- The combined plot — `out/figures/band_identification_<readout>.png`
- The band you selected + justification (the tool's suggestion is a
  starting point, not a substitute for looking at the plot)
