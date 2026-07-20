# Commitment Dynamics in the J-Space

When do language models commit to answers, and what happens under false-lead
temptation? We probe Qwen3 with a Jacobian lens on matched stimulus pairs
(straightforward vs. false-lead) and test three pre-registered hypotheses about
commitment layer (H1), internal oscillation (H2), and the confidence–correctness
dissociation gap (H3).

**Status**: Run 1 (Qwen3-1.7B, 2026-07-17) is **superseded**. Diagnostics on
2026-07-19 found defects in both the analysis pipeline and the stimulus set
that invalidate its numbers. Both are now fixed; the run needs repeating.
**Do not quote run-1 results.**

---

## Where things stand

Run 1's traces were analyzed with a non-reproducible dev/holdout split, a
two-tailed test where the pre-registration specifies one-tailed, and a
scipy-version-dependent p-value. Separately, only 31 of 78 stimulus pairs were
valid matched pairs. Details: `out/ANALYSIS_NOTE.md` (analysis) and
`build_stimuli.py`'s docstring (stimuli).

Re-analyzing the same traces with the corrected pipeline moved every result —
H1's effect vanished, H3 became significant, H2 weakened by two orders of
magnitude — which shows the outcome was dominated by an arbitrary split at
n≈6–11 pairs. The corrected numbers live in `out/analysis_real/report.json`,
but they still rest on the old broken stimuli, so they are diagnostic only.

**Fixed so far**

- Analysis is reproducible, exact, and one-tailed as pre-registered
  (`03_analyze.py`); environment pinned and self-checking (`verify_env.py`).
- Stimuli rebuilt as **72/72 strictly matched pairs**, balanced 24 per family,
  plus 12 hard controls (`build_stimuli.py`), all passing `validate_stimuli.py`.

**Still open**

1. **Re-run the GPU traces** on the new stimuli (lens already fitted, ~30 min).
2. **Amend the pre-registration** before that run — see "Next steps".
3. **H2's mechanism is unverified**: a target-vs-distractor (2AFC) oscillation
   metric does not reach significance where the primary metric does, so the
   measured top-1 churn may be among arbitrary tokens rather than wavering
   between the candidate answers. Confirmatory metrics are now implemented and
   pre-registered, with the permitted claim for each outcome fixed in advance
   (`experiments/PRE_REGISTRATION_AMENDMENT.md` §5). Until run 2 settles it, the
   "revision between candidates" reading in `paper/ABSTRACT.md` and
   `paper/DISCUSSION_OUTLINE.md` is not supported.
4. **Promised robustness checks not yet run**: logit-lens secondary readout
   (`lens_utils.py`, not integrated) and per-model workspace-band
   identification (band is the pre-registered default [0.25, 0.90]).

## Repository layout

```
stimuli.json               156 stimuli: 72 matched pairs (24/family) + 12 hard controls
build_stimuli.py           Regenerates stimuli.json; documents the run-1 defects
validate_stimuli.py        Enforces the 6 design rules (CPU, tokenizer only)
00_generate_mock_traces.py Synthetic traces for CPU-only pipeline validation
01_fit_lens.py             Fit Jacobian lens (GPU, ~2 h)
02_run_experiment.py       Collect lens traces on stimuli (GPU, ~20 min)
02_run_experiment_cot.py   CoT variant scaffold (H4, not yet run)
03_analyze.py              Metrics + exact signed-rank tests (CPU)
generate_figures.py        Publication figures from out/analysis_real/ (CPU)
lens_utils.py              Logit-lens utilities (pending integration)
paper/                     Paper sections (see paper/INDEX.md)
experiments/               PRE_REGISTRATION.md (locked), GPU re-run guide,
                           workspace band guide, Natural Stories plan
out/                       Lens (git-lfs), traces, analysis, figures, logs
```

## Environment

```bash
./setup_env.sh                        # builds venv/ with pinned versions
venv/bin/python verify_env.py         # must print "analysis environment OK"
```

Always invoke scripts as `venv/bin/python …`. Versions are pinned because an
unpinned scipy silently changed p-values between machines; `verify_env.py`
checks the interpreter, the pins, and the statistics themselves, and fails if
run outside the repo venv.

## Reproducing the pipeline

```bash
# Stimuli: rebuild and validate (CPU; needs the tokenizer, not torch)
venv/bin/python build_stimuli.py --out stimuli.json
venv/bin/python validate_stimuli.py --stimuli stimuli.json

# Analysis pipeline smoke test (CPU)
venv/bin/python 00_generate_mock_traces.py --out out/traces_mock.json
venv/bin/python 03_analyze.py --traces out/traces_mock.json --outdir out/analysis_mock

# Full pipeline (GPU) — see experiments/README_GPU_PHASE.md
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --out out/lens_qwen3_1p7b.pt
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens_qwen3_1p7b.pt --out out/traces.json
venv/bin/python 03_analyze.py --traces out/traces.json --outdir out/analysis_real --dev-split 0.6
venv/bin/python generate_figures.py
```

The analysis plan (band, θ procedure, dev/holdout split, tests) is locked in
`experiments/PRE_REGISTRATION.md` — do not tune it post hoc.

## Next steps

1. ✅ **Pre-registration amended and locked** before the re-run —
   `experiments/PRE_REGISTRATION_AMENDMENT.md`.
2. **Re-run traces on the new stimuli** (Qwen3-1.7B; lens already fitted).
   Run the per-family behavioural pre-screen first: amendment §8 makes a family
   whose straightforward condition mostly misses the target a stop condition,
   not something to analyze around.
3. Re-analyze, regenerate figures, and only then rewrite Results/Abstract.
4. Integrate logit-lens robustness readout; band identification (Appendix C).
5. Reconcile `paper/DISCUSSION_OUTLINE.md` with whatever the re-run shows.
6. Only after 1.7B is clean: Qwen3-4B replication; Natural Stories.
