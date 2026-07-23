#!/usr/bin/env python
"""test_band_analysis.py — CPU-only sanity tests for band_analysis.py.

Run: venv/bin/python test_band_analysis.py

These tests exist because band_analysis.py's curve computations run
entirely on GPU-collected data — a bug here would silently mis-select the
workspace band with no error, only wrong downstream numbers. Testing with
synthetic data where we know the right answer catches that before any GPU
time is spent.
"""

import numpy as np

from band_analysis import (
    accuracy_by_layer,
    excess_kurtosis_by_layer,
    identify_band,
    layer_autocorr,
    longest_run_at_or_above,
)

rng = np.random.default_rng(0)
failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        failures.append(name)


print("excess_kurtosis_by_layer")
normal_vals = rng.normal(size=5000)
peaked_vals = rng.laplace(size=5000)  # heavier-tailed / more peaked than normal
k = excess_kurtosis_by_layer({0: normal_vals, 1: peaked_vals})
check("normal distribution has ~0 excess kurtosis", abs(k[0]) < 0.3)
check("laplace distribution has positive excess kurtosis", k[1] > 1.0)
print()

print("layer_autocorr")
n = 2000
smooth_walk = np.cumsum(rng.normal(scale=0.1, size=n))  # highly autocorrelated
noise_series = rng.normal(size=n)  # independent draws each layer
layers = [0, 1, 2, 3]
rank_by_layer = {
    0: smooth_walk,
    1: smooth_walk + rng.normal(scale=0.05, size=n),   # near-identical to layer 0
    2: noise_series,
    3: rng.normal(size=n),                             # independent of layer 2
}
ac = layer_autocorr(rank_by_layer, layers)
check("first layer autocorr is NaN (no predecessor)", np.isnan(ac[0]))
check("smooth-walk transition has high autocorr", ac[1] > 0.9)
check("independent-noise transition has near-zero autocorr", abs(ac[3]) < 0.1)
print()

print("accuracy_by_layer")
acc = accuracy_by_layer({0: [1, 1, 1, 0], 1: [0, 0, 0, 0], 2: []})
check("3/4 correct -> 0.75", abs(acc[0] - 0.75) < 1e-9)
check("0/4 correct -> 0.0", abs(acc[1] - 0.0) < 1e-9)
check("empty layer -> nan", np.isnan(acc[2]))
print()

print("longest_run_at_or_above")
mask = np.array([False, True, True, True, False, True, False])
run = longest_run_at_or_above(mask)
check("finds the length-3 run at [1,3]", run == (1, 3))
check("no True entries -> None",
     longest_run_at_or_above(np.array([False, False])) is None)
check("all True -> full range",
     longest_run_at_or_above(np.array([True, True, True])) == (0, 2))
print()

print("identify_band (synthetic workspace bump in the middle)")
n_layers = 20
layers = list(range(n_layers))
n_samples = 1500
entropy_by_layer, rank_by_layer, correct_by_layer = {}, {}, {}
workspace = set(range(6, 15))  # true synthetic workspace: layers 6-14
base_walk = np.cumsum(rng.normal(scale=0.1, size=n_samples))
for L in layers:
    if L in workspace:
        entropy_by_layer[L] = rng.laplace(size=n_samples)         # peaked
        rank_by_layer[L] = base_walk + rng.normal(scale=0.05, size=n_samples)  # smooth
        correct_by_layer[L] = (rng.random(n_samples) < 0.8).astype(int)       # high acc
    else:
        entropy_by_layer[L] = rng.normal(size=n_samples)          # ~0 excess kurtosis
        rank_by_layer[L] = rng.normal(size=n_samples)              # independent noise
        correct_by_layer[L] = (rng.random(n_samples) < 0.1).astype(int)       # low acc

result = identify_band(layers, entropy_by_layer, rank_by_layer, correct_by_layer,
                       threshold_pct=40.0)
lo_layer, hi_layer = result["suggested_band_layers"]
check(f"suggested band {result['suggested_band_layers']} overlaps true workspace [6,14]",
     lo_layer <= 14 and hi_layer >= 6)
check("suggested band doesn't span the whole range (signal is localized)",
     (hi_layer - lo_layer) < n_layers - 1)
check("fractional band values are within [0,1]",
     0 <= result["suggested_band_fraction"][0] <= 1
     and 0 <= result["suggested_band_fraction"][1] <= 1)
print(f"  (suggested: layers {result['suggested_band_layers']}, "
     f"fraction {result['suggested_band_fraction']}, true workspace was [6,14])")
print()

if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("ALL TESTS PASSED")
