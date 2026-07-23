"""band_analysis.py — pure-math half of workspace band identification.

Split out from 04_identify_band.py so the curve/band-selection logic can be
unit-tested on CPU with synthetic data (see test_band_analysis.py) —
independent of the GPU-dependent corpus collection loop, which is the part
most likely to silently produce a wrong band if it has a bug.

Implements the three signals from experiments/workspace_band_guide.md,
operationalized precisely (the guide's original pseudocode is genuinely
ambiguous about the autocorrelation criterion — see the note on
`layer_autocorr` below):

1. Excess kurtosis of per-layer entropy (Gurnee et al.'s described criterion,
   as-is from the guide).
2. Layer-to-layer rank autocorrelation (this script's operationalization —
   see `layer_autocorr` docstring).
3. Next-token top-1 accuracy per layer (as-is from the guide).
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def excess_kurtosis_by_layer(entropy_by_layer: Dict[int, Sequence[float]]) -> Dict[int, float]:
    """Fisher (excess) kurtosis of the entropy distribution at each layer.

    entropy_by_layer[L] = entropy values pooled over every held-out corpus
    position scored at layer L. Excess kurtosis of 0 matches a normal
    distribution; workspace layers are expected to show elevated kurtosis
    (a mix of very-low-entropy "confident" positions and a heavier tail),
    per Gurnee et al.'s Fig 28 criterion.
    """
    out = {}
    for L, vals in entropy_by_layer.items():
        vals = np.asarray(vals, dtype=float)
        out[L] = float(stats.kurtosis(vals, fisher=True)) if len(vals) >= 4 else float("nan")
    return out


def layer_autocorr(rank_by_layer: Dict[int, Sequence[float]],
                   layers: Sequence[int]) -> Dict[int, float]:
    """Lag-1 autocorrelation of next-token rank ACROSS the layer axis.

    Operationalization (deviates from workspace_band_guide.md's original
    pseudocode — documented here since it's a design choice, not a verified
    replication of an unavailable reference implementation): the guide's
    snippet computed, for a fixed layer, the autocorrelation of ranks across
    different query positions — but query positions have no natural order,
    so there is nothing for that computation to mean. Instead, for each pair
    of adjacent layers (L-1, L), this computes the Pearson correlation,
    across the held-out corpus, between a token's rank at L-1 and its rank
    at L. High correlation means rank moves smoothly across that layer
    transition (workspace-like stability, matching the guide's intent —
    "within the workspace, target rank stays stable"); low correlation means
    the layer scrambles the representation. This IS layer-indexed and well
    defined, unlike the original pseudocode.

    rank_by_layer[L] and rank_by_layer[L'] must be positionally aligned:
    index i in both lists must refer to the same underlying corpus sample.
    The first layer in `layers` has no predecessor and gets nan.
    """
    out = {layers[0]: float("nan")}
    for i in range(1, len(layers)):
        prev_L, L = layers[i - 1], layers[i]
        a = np.asarray(rank_by_layer[prev_L], dtype=float)
        b = np.asarray(rank_by_layer[L], dtype=float)
        if len(a) < 2 or len(b) < 2 or len(a) != len(b):
            out[L] = float("nan")
            continue
        if np.std(a) == 0 or np.std(b) == 0:
            out[L] = float("nan")  # degenerate (e.g. all ranks identical)
            continue
        out[L] = float(np.corrcoef(a, b)[0, 1])
    return out


def accuracy_by_layer(correct_by_layer: Dict[int, Sequence[int]]) -> Dict[int, float]:
    """Fraction of held-out positions where argmax(layer logits) == true next token."""
    out = {}
    for L, hits in correct_by_layer.items():
        hits = np.asarray(hits, dtype=float)
        out[L] = float(hits.mean()) if len(hits) else float("nan")
    return out


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.zeros_like(values)
    lo, hi = finite.min(), finite.max()
    out = (values - lo) / (hi - lo)
    out[~np.isfinite(values)] = np.nan
    return out


def longest_run_at_or_above(mask: np.ndarray):
    """Return (start_idx, end_idx) of the longest contiguous True run in `mask`.

    end_idx is inclusive. Returns None if no True entries.
    """
    best = None
    run_start = None
    for i, v in enumerate(mask):
        if v and run_start is None:
            run_start = i
        if (not v or i == len(mask) - 1) and run_start is not None:
            run_end = i if v else i - 1
            if best is None or (run_end - run_start) > (best[1] - best[0]):
                best = (run_start, run_end)
            run_start = None
    return best


def identify_band(layers: Sequence[int],
                  entropy_by_layer: Dict[int, Sequence[float]],
                  rank_by_layer: Dict[int, Sequence[float]],
                  correct_by_layer: Dict[int, Sequence[int]],
                  threshold_pct: float = 40.0) -> dict:
    """Combine the three signals into per-layer curves + a suggested band.

    Returns a dict with the raw and normalized curves (for plotting/manual
    inspection — always look at the plot, this is a starting point) and the
    suggested band as both layer indices and fractional depth, in the
    [lo, hi] convention 03_analyze.py's --band flag expects.
    """
    layers = list(layers)
    kurt = excess_kurtosis_by_layer(entropy_by_layer)
    ac = layer_autocorr(rank_by_layer, layers)
    acc = accuracy_by_layer(correct_by_layer)

    kurt_arr = np.array([kurt[L] for L in layers])
    ac_arr = np.array([ac[L] for L in layers])
    acc_arr = np.array([acc[L] for L in layers])

    kurt_n = _minmax_normalize(kurt_arr)
    ac_n = _minmax_normalize(ac_arr)
    acc_n = _minmax_normalize(acc_arr)

    stacked = np.vstack([kurt_n, ac_n, acc_n])
    composite = np.nanmean(stacked, axis=0)

    finite_composite = composite[np.isfinite(composite)]
    threshold = (float(np.percentile(finite_composite, threshold_pct))
                if len(finite_composite) else float("nan"))
    mask = composite >= threshold

    run = longest_run_at_or_above(mask)
    if run is None:
        band_layers, band_frac = None, None
    else:
        start_i, end_i = run
        band_layers = [layers[start_i], layers[end_i]]
        lmin, lmax = min(layers), max(layers)
        span = max(lmax - lmin, 1)
        band_frac = [round((band_layers[0] - lmin) / span, 3),
                    round((band_layers[1] - lmin) / span, 3)]

    return {
        "layers": layers,
        "kurtosis": kurt_arr.tolist(),
        "autocorrelation": ac_arr.tolist(),
        "accuracy": acc_arr.tolist(),
        "kurtosis_normalized": kurt_n.tolist(),
        "autocorrelation_normalized": ac_n.tolist(),
        "accuracy_normalized": acc_n.tolist(),
        "composite": composite.tolist(),
        "threshold_pct": threshold_pct,
        "threshold_value": threshold,
        "suggested_band_layers": band_layers,
        "suggested_band_fraction": band_frac,
    }
