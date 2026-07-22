# Analysis note: why the numbers changed on 2026-07-19

> **Superseded 2026-07-22.** `out/analysis_real/` now holds the **run-2** result
> (rebuilt stimuli, `out/traces_run2.json`), which is the current canonical
> analysis. This note documents the earlier diagnostic re-analysis of the
> withdrawn **run-1** traces (`out/traces.json`); those run-1 numbers live in git
> history (commit 6e01b5c). The three pipeline defects below were fixed before
> run 2, which is why run 2 is trustworthy.

This note (below) concerns the run-1 re-analysis. `out/analysis_real/` was
regenerated after fixing three defects in the analysis pipeline. The run-1
traces (`out/traces.json`) and the fitted lens are unchanged — the model data is
the same; only the analysis of it was wrong.

## Defects found and fixed

**1. The dev/holdout split was not reproducible.** `03_analyze.py` built the
pair list with `list(set(...))` before shuffling. Python randomizes string
hashing per process (`PYTHONHASHSEED`), so set iteration order — and therefore
the split — differed on every run despite `random.seed(42)`. Three consecutive
runs produced three different splits. The run-1 numbers came from one
unreproducible draw. Fixed by sorting before shuffling.

**2. The test was two-tailed; the pre-registration specifies one-tailed.**
`experiments/PRE_REGISTRATION.md` states "p < 0.05 (one-tailed)" for H1–H3 with
directional predictions, but the code called `scipy.stats.wilcoxon(diffs)`,
whose default is two-sided. The reported p-values were therefore 2× the
pre-registered test. Fixed by testing `alternative="greater"`, matching the
registered directional prediction. Both one- and two-tailed p-values are now
recorded in `report.json` for transparency.

**3. The p-value depended on the scipy version.** `wilcoxon(method="auto")`
changed behaviour across releases: with zeros/ties at small n, scipy 1.13.1
falls back to a normal approximation it simultaneously warns is invalid, while
the GPU box's scipy used the exact method. The same diffs gave p=0.0625 there
and p=0.0339 locally. `03_analyze.py` now computes the exact signed-rank null
itself via subset-sum DP over mid-ranks — deterministic, version-independent,
and correct under ties (verified against brute-force enumeration and against
scipy's exact method on tie-free data up to n=30; see `verify_env.py`).

Related: the repo shipped no `venv/`, so `source venv/bin/activate` silently
did nothing and analysis ran under a stray anaconda Python 3.7 / scipy 1.7.3.
The environment is now pinned (`requirements.txt`) and checked
(`verify_env.py`, which fails outside the repo venv).

## Effect on results

Same traces, same pre-registered settings (band [0.25, 0.90], θ = 80th pct of
straightforward ΔH on dev, 60/40 split, seed 42) — but a reproducible split and
the registered one-tailed exact test:

| Test | Run 1 (as published) | Corrected | Change |
|---|---|---|---|
| H1 (ℓ*) | median Δ=+2.0, p=0.0625, n=6 | median Δ=**0.0**, p=**0.3125**, n=11 (6 nonzero) | effect disappears |
| H2 (oscillation) | median Δ=+2.0, p=4.89×10⁻⁵, n=32 | median Δ=**+1.0**, p=**0.0038**, n=30 (25 nonzero) | survives, much weaker |
| H3 (gap) | median Δ=+4.0, p=0.0625, n=6 | median Δ=**+2.0**, p=**0.0078**, n=11 (9 nonzero) | now significant |

θ also moved (3.517 → 3.285 nats) because it is estimated on the dev split,
which changed.

## How to read this

The swing between two arbitrary splits of the same data is large enough that
**neither set of numbers should be treated as a stable estimate**. That
instability is itself the finding: at n≈6–11 complete pairs, the result is
dominated by which pairs happen to land in the holdout. This is a power problem
that more/better stimuli must fix — see the matched-pair confound documented in
`CLAUDE.md`, where only 31 of 78 pairs hold target and distractor constant
across conditions.

Do not quote any of these numbers in the paper until the stimuli are rebuilt
and the run is repeated.
