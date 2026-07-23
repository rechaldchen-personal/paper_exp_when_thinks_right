# Commitment Dynamics in the J-Space

When do language models commit to answers, and what happens under false-lead
temptation? We probe Qwen3 with a Jacobian lens on matched stimulus pairs
(straightforward vs. false-lead) and test three pre-registered hypotheses about
commitment layer (H1), internal oscillation (H2), and the confidence–correctness
dissociation gap (H3).

**Status**: Run 2 (Qwen3-1.7B, 2026-07-22) is the current result — rebuilt
stimuli, corrected analysis, passed the behavioural gate on all three families,
θ/band sensitivity checked, and the paper (Abstract/Results/Discussion)
rewritten to match. Run 1 is withdrawn; do not quote its numbers
(`out/ANALYSIS_NOTE.md` explains why).

---

## Results (run 2, Qwen3-1.7B holdout)

Pre-registered one-tailed exact signed-rank tests, false-lead − straightforward.
Full report: `out/analysis_real/report.json`; figures: `out/figures/`.

| Hypothesis | Primary (pre-registered) | Confirmatory (2AFC) | Verdict |
|---|---|---|---|
| H1 delayed commitment | +0.0 layers, p=0.109 | +2.0 layers, p=0.002 | narrow-readout only |
| H2 oscillation | +0.5, p=0.008 ✓ | 0.0, p=0.46 | churn, **not** candidate revision |
| H3 dissociation gap | +1.0 layers, p=0.011 ✓ | +4.0 layers, p=0.002 ✓ | **robust — headline** |

Reading (per `experiments/PRE_REGISTRATION_AMENDMENT.md` §5): **H3 is the robust
finding** — models grow confident before they are correct, and that window
widens under false lead, under both readouts. **H2's "internal revision between
candidate answers" framing is refuted** — the oscillation is real but is churn
among unrelated tokens (confirmatory null; hard controls oscillate too), so that
language is out of the Abstract and Discussion. **H1** holds only for the
target-vs-distractor readout, not global commitment.

Behavioural gate (straightforward top-1 accuracy): factual 95.8%, garden-path
75.0%, arithmetic 70.8% — all clear 50%. Run 1 had arithmetic at 3.6%.

**Sensitivity checks** (`experiments/SENSITIVITY_REPORT.md`, pre-registered θ ∈
{70,80,90} pct × band ∈ {[.20,.95],[.25,.90],[.30,.85]}): direction never flips
sign anywhere. But significance is not uniform — the **confirmatory** metrics
(`l_star_2afc`, `gap_2afc`) are significant at *every* setting tested, while
the **primary** `oscillation` and `gap` are significant only near the
pre-registered θ=80 default and lose significance at θ=70/90. This is why the
Results and Discussion lead with the confirmatory numbers for H1/H3 rather
than the primary ones — the confirmatory readout is the one that's actually
robust, not just correctly directioned.

Caveats: H1/H3 primary rest on 14 pairs (confirmatory on 18–19); this is the
one registered split (seed 42); single model. 4B replication is the next
credibility step.

**Repository background**

- Analysis is reproducible, exact, and one-tailed as pre-registered
  (`03_analyze.py`); environment pinned and self-checking (`verify_env.py`).
- Stimuli rebuilt as **72/72 strictly matched pairs**, balanced 24 per family,
  plus 12 hard controls (`build_stimuli.py`), all passing `validate_stimuli.py`.

**Still open**

1. **Expand the writing templates** in `paper/DISCUSSION_OUTLINE.md` into
   final submission prose — structurally complete, still template-shaped.
   Free/CPU.
2. **Logit-lens secondary readout** — code-complete
   (`02_run_experiment.py --readout logit_lens`, `lens_utils.py`), unrun.
   ~20-30 min GPU, no fresh lens needed.
3. **Per-model workspace band identification** — code-complete
   (`04_identify_band.py`, `band_analysis.py`), math unit-tested
   (`test_band_analysis.py`, CPU, passing), unrun. ~5-15 min GPU per readout.
4. **Qwen3-4B replication** — stimuli pre-verified to need no rework
   (`validate_stimuli.py --model Qwen/Qwen3-4B` passes all 156; both models
   share a 151,936-token vocabulary). Needs its own lens fit; ~5-7 h GPU
   estimated (~2× 1.7B's compute, from published layer/hidden-size configs).

Full commands and sequencing for all three: `experiments/README_GPU_PHASE.md`
§"After run 2".

## Repository layout

```
stimuli.json               156 stimuli: 72 matched pairs (24/family) + 12 hard controls
build_stimuli.py           Regenerates stimuli.json; documents the run-1 defects
validate_stimuli.py        Enforces the 7 design rules (CPU, tokenizer only)
prescreen.py               Behavioural gate on collected traces (amendment §8)
00_generate_mock_traces.py Synthetic traces for CPU-only pipeline validation
01_fit_lens.py             Fit Jacobian lens (GPU, ~2 h)
02_run_experiment.py       Collect traces; --readout jlens|logit_lens (GPU)
02_run_experiment_cot.py   CoT variant scaffold (H4, not yet run)
03_analyze.py              Metrics + exact signed-rank tests (CPU)
04_identify_band.py        Per-model workspace band identification (GPU)
band_analysis.py           Band-identification math (CPU, unit-tested)
test_band_analysis.py      Unit tests for band_analysis.py — run on CPU
generate_figures.py        Publication figures from out/analysis_real/ (CPU)
lens_utils.py              Logit-lens (identity transport) readout, integrated
paper/                     Paper sections (see paper/INDEX.md)
experiments/               PRE_REGISTRATION.md (+ AMENDMENT, locked),
                           GPU guide (run-2 + logit-lens + band-ID + 4B),
                           SENSITIVITY_REPORT.md, workspace band guide,
                           Natural Stories plan
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

# Band-identification math smoke test (CPU, synthetic data)
venv/bin/python test_band_analysis.py

# Full pipeline (GPU) — see experiments/README_GPU_PHASE.md.
# out/traces_run2.json + out/analysis_real/ are already the current, checked-in
# result (Qwen3-1.7B); only re-run this if collecting NEW data (e.g. 4B, or a
# stimuli expansion) — write to a new --out/--outdir so you don't overwrite it.
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --out out/lens_qwen3_1p7b.pt
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens_qwen3_1p7b.pt --out out/traces_run3.json
venv/bin/python 03_analyze.py --traces out/traces_run3.json --outdir out/analysis_run3 --dev-split 0.6
venv/bin/python generate_figures.py   # reads out/analysis_real/ by default — repoint or promote first

# Sensitivity checks (already run once; commands to re-verify or extend):
venv/bin/python 03_analyze.py --traces out/traces_run2.json --outdir out/sensitivity/theta70 --dev-split 0.6 --theta-pct 70
```

The analysis plan (band, θ procedure, dev/holdout split, tests) is locked in
`experiments/PRE_REGISTRATION.md` + `experiments/PRE_REGISTRATION_AMENDMENT.md`
— do not tune it post hoc.

## Next steps

1. ✅ Pre-registration amended and locked —
   `experiments/PRE_REGISTRATION_AMENDMENT.md`.
2. ✅ Run 2 traces collected and gated —
   `out/traces_run2.json`, `prescreen.py` passed all families.
3. ✅ Analyzed, figures regenerated, results promoted to `out/analysis_real/`.
4. ✅ Sensitivity checks run — `experiments/SENSITIVITY_REPORT.md`.
5. ✅ Results/Abstract/Discussion rewritten to match run 2 + sensitivity.
6. ✅ Logit-lens readout, per-model band identification, and the 4B plan are
   code-complete and CPU-side verified (see "Still open" above) — each needs
   one GPU session to execute.
7. **Next**: run those three GPU sessions (`experiments/README_GPU_PHASE.md`
   §"After run 2"); expand Discussion templates into final prose; Natural
   Stories (retarget at H3, not H2 — see Discussion §6.4).
