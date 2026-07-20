# Commitment Dynamics Paper — Project Guide for Claude Code

**What this is**: Research paper on when LMs commit to answers under false-lead
temptation, probed with a Jacobian lens on Qwen3-1.7B. See `README.md` for
current status, results, and known issues; see git history for the full
session-by-session log (the old CLAUDE.md tracker was condensed on 2026-07-19).

## Current state (2026-07-19)

- **Run 1 is superseded — do not quote its numbers.** Diagnostics found defects
  in both the analysis pipeline and the stimulus set. Both are now fixed; the
  GPU run needs repeating on the rebuilt stimuli.
- **Analysis fixed** (`out/ANALYSIS_NOTE.md`): the dev/holdout split was
  non-reproducible (`list(set(...))` + per-process string hashing defeated
  `seed=42`), the test was two-tailed while the pre-registration says
  one-tailed, and p-values depended on the scipy version. `03_analyze.py` now
  computes the exact signed-rank null itself. Re-analyzing the *same* traces
  moved every result (H1 vanished, H3 became significant, H2 weakened from
  4.9e-5 to 0.0038), showing the outcome was split-dominated at n≈6–11.
- **Stimuli rebuilt** (`build_stimuli.py`): 72/72 strictly matched pairs,
  24 per family, + 12 hard controls, all passing `validate_stimuli.py`.
  Previously only 31/78 pairs were valid; arithmetic swapped target and
  distractor between conditions, and 15 garden-path prompts ended with the
  answer word.
- **Lens is unchanged and still valid** (`out/lens_qwen3_1p7b.pt`, git-lfs) —
  a re-run skips fitting and costs ~30 min GPU.

## Priority queue (in order)

1. ✅ **Pre-registration amended and locked** —
   `experiments/PRE_REGISTRATION_AMENDMENT.md` (2026-07-19). Primary metrics
   unchanged; confirmatory 2AFC metrics added and implemented in
   `03_analyze.py`; interpretation rules for every primary×confirmatory outcome
   fixed in advance; run 1 formally withdrawn.
2. **Re-run GPU traces** on the new stimuli, then re-analyze and regenerate
   figures. Run the **per-family behavioural pre-screen first** — amendment §8
   makes it a stop condition, not a formality.
3. **Rewrite Results/Abstract** from the new numbers — not before.
4. **Robustness checks promised in Methods 4.7** — logit-lens secondary readout
   (`lens_utils.py` not yet integrated into `02_run_experiment.py`) and
   per-model workspace band (`experiments/workspace_band_guide.md`).
5. **Reconcile `paper/DISCUSSION_OUTLINE.md`** — remove "X flips", "r ≈ ?", and
   the unrun Natural Stories claims; H2's "revision between candidates" reading
   is contradicted by the 2AFC check (p≈0.67) and must be softened or retested.
6. Only once 1.7B is clean: Qwen3-4B replication, CoT variant
   (`02_run_experiment_cot.py`), Natural Stories.

## Ground rules

- **Scientific honesty**: never claim support for a hypothesis the data don't
  show. H1/H3 are "directional, not significant" until a higher-powered re-run
  says otherwise. Mock-data validation proves the pipeline, not the science.
- **Pre-registration is locked** (`experiments/PRE_REGISTRATION.md`): band
  [0.25, 0.90], θ = 80th percentile of straightforward ΔH on dev only, 60/40
  split by pair, seed 42, one-sided Wilcoxon. Don't tune post hoc; any deviation
  must be reported as exploratory.
- **Before burning GPU time**: run `venv/bin/python validate_stimuli.py` after
  any change to `stimuli.json`. It enforces matched pairs, no answer leakage,
  single-token targets, and no answer-in-context. The old
  `02_run_experiment.py --validate` only checked tokenization, which *passed*
  on the broken run-1 set — tokenization OK does not mean the design is sound.
- **Always invoke `venv/bin/python`**, never bare `python` (a stray anaconda
  3.7 previously produced different p-values than the committed results).
- `*.pt` files are git-lfs; `out/*.log` are kept for provenance.
- Environment: `source venv/bin/activate`; CPU deps in `requirements.txt`,
  GPU deps (`torch transformers datasets` + jlens) installed on the GPU box.

## Key facts

- Stimuli: 156 items — 72 matched pairs (24 factual / 24 arithmetic /
  24 garden-path) + 12 hard controls. Regenerate with `build_stimuli.py`.
- Qwen3 tokenization gotcha: ' 10' is [220, 16, 15] (space + digits), so number
  answers must be English words in the ' word' form. Bare 'fifteen' is 3
  tokens. Token 220 (' ') topping the readout in run 1 was the model correctly
  starting a *digit* answer, not junk.
- Traces schema: see `02_run_experiment.py` docstring (entropy, ranks, logp,
  top1_id per layer×position + behavioral top-5 check).
- Metrics: ℓ_H (entropy collapse), ℓ* (stable target top-1 in band),
  gap = ℓ* − ℓ_H, oscillation (top-1 changes after ℓ_H) — `03_analyze.py`.
- Analysis flags: `--outdir` (not `--out`), `--theta-pct`, `--dev-split 0.6`.
