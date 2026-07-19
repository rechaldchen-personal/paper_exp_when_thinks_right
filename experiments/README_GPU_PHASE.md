# GPU Phase Guide

**Status**: Run 1 on Qwen3-1.7B completed 2026-07-17 (results in
`out/analysis_real/`). This guide now covers **re-runs** — needed after fixing
the arithmetic stimuli / expanding hard controls, or for a Qwen3-4B replication.

The lens for Qwen3-1.7B is already fitted (`out/lens_qwen3_1p7b.pt`, git-lfs) —
a 1.7B re-run skips lens fitting and takes ~30 min of GPU time.

---

## Setup (fresh GPU box, ~15 min)

```bash
git clone https://github.com/anthropics/jacobian-lens
pip install -e ./jacobian-lens
pip install torch transformers datasets scipy matplotlib pandas
git lfs pull   # fetch out/lens_qwen3_1p7b.pt (~226 MB)
```

## Step 0: Validate stimuli (CPU, always run after editing stimuli.json)

```bash
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --validate
# Expect: "Tokenization OK: 163 stimuli, all targets single-token."
```

Also spot-check behavioral plausibility of any reformatted prompts — run 1
failed on arithmetic because the model's top-1 after "... equals" was a bare
space token, even though tokenization validation passed. Tokenization OK ≠
behaviorally answerable.

## Step 1: Fit lens (GPU, ~2 h — SKIP for Qwen3-1.7B, already done)

```bash
python 01_fit_lens.py --model Qwen/Qwen3-4B --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_4b.pt --checkpoint out/fit_ckpt_4b.pt
```

Resumable via `--checkpoint`. Output ~200–500 MB depending on d_model.

## Step 2: Collect traces (GPU, ~20 min)

```bash
python 02_run_experiment.py --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt --out out/traces.json
```

Watch the `top1=...` column in the log: if a whole family shows `top1=' '` or
another junk token, stop and fix prompts before analyzing.

## Step 3: Analyze (CPU, ~5 min)

```bash
python 03_analyze.py --traces out/traces.json --outdir out/analysis_real \
    --dev-split 0.6 --band 0.25 0.90 --theta-pct 80.0
```

Settings are pre-registered (`experiments/PRE_REGISTRATION.md`) — do not tune.
Note the flags: `--outdir` (not `--out`), `--theta-pct`.

Outputs: `report.json` (tests), `per_stimulus_metrics.json`, `heatmaps/`.
Check `n_pairs` in each test — run 1 had only 6 complete pairs for H1/H3
(p floor 0.0625 at n=6); a healthy re-run needs behavioral accuracy high enough
that ℓ* is defined for most pairs in both conditions.

## Step 4: Figures + tables (CPU, ~2 min)

```bash
python generate_figures.py   # reads out/analysis_real/, writes out/figures/
```

## Step 5: Update paper

- `paper/RESULTS_TEMPLATE.md` — refresh all numbers from `report.json`
- `paper/ABSTRACT.md` — refresh the results paragraph
- `paper/DISCUSSION_OUTLINE.md` — reconcile interpretation with outcomes

## Optional robustness passes (promised in Methods 4.7)

- **Logit-lens secondary readout**: integrate `lens_utils.py` into
  `02_run_experiment.py` (identity transport), re-run Step 3 on those traces.
- **Per-model workspace band**: `experiments/workspace_band_guide.md`
  (kurtosis / autocorrelation / next-token accuracy criteria).
- **Natural Stories**: `experiments/natural_stories_plan.md`.

## Troubleshooting

- **jlens API mismatch**: call sites are marked `[API]` in `01_fit_lens.py` and
  `02_run_experiment.py`; signatures were verified 2026-07-15 and worked in run 1.
- **Multi-token targets**: fix the item in `stimuli.json` (leading space) and
  re-run Step 0.
- **OOM during fitting**: reduce `--n-prompts` to 500 or `--seq-len` to 64.
