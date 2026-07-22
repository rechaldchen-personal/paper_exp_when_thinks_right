# Commitment Dynamics in the J-Space

When do language models commit to answers, and what happens under false-lead
temptation? We probe Qwen3 with a Jacobian lens on matched stimulus pairs
(straightforward vs. false-lead) and test three pre-registered hypotheses about
commitment layer (H1), internal oscillation (H2), and the confidence–correctness
dissociation gap (H3).

**Status**: Run 2 (Qwen3-1.7B, 2026-07-22) is the current result — rebuilt
stimuli, corrected analysis, passed the behavioural gate on all three families.
Run 1 is withdrawn; do not quote its numbers (`out/ANALYSIS_NOTE.md` explains
why). Robustness checks (θ/band sensitivity) and the paper rewrite are pending.

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
language must come out of the Abstract and Discussion. **H1** holds only for the
target-vs-distractor readout, not global commitment.

Behavioural gate (straightforward top-1 accuracy): factual 95.8%, garden-path
75.0%, arithmetic 70.8% — all clear 50%. Run 1 had arithmetic at 3.6%.

Caveats: H1/H3 primary rest on 14 pairs; this is the one registered split
(seed 42); single model. Robustness checks and 4B replication are the next
credibility steps.

**Repository background**

- Analysis is reproducible, exact, and one-tailed as pre-registered
  (`03_analyze.py`); environment pinned and self-checking (`verify_env.py`).
- Stimuli rebuilt as **72/72 strictly matched pairs**, balanced 24 per family,
  plus 12 hard controls (`build_stimuli.py`), all passing `validate_stimuli.py`.

**Still open (all CPU/free unless noted)**

1. **θ/band sensitivity checks** — pre-registered robustness (re-run
   `03_analyze.py` at θ 70/90 and bands [0.20,0.95]/[0.30,0.85] on
   `out/traces_run2.json`). No GPU.
2. **Rewrite the paper to match run 2**: H3 headline; strike the H2
   "revision between candidates" language from `paper/ABSTRACT.md` and
   `paper/DISCUSSION_OUTLINE.md` per amendment §5; H1 as narrow-readout only.
3. **Logit-lens secondary readout** (`lens_utils.py`, not integrated) — needs
   code work + a GPU re-run.
4. **Generalization (GPU, deliberate next session)**: Qwen3-4B replication
   (fresh lens fit) and/or more stimuli for power on H1/H3.

## Repository layout

```
stimuli.json               156 stimuli: 72 matched pairs (24/family) + 12 hard controls
build_stimuli.py           Regenerates stimuli.json; documents the run-1 defects
validate_stimuli.py        Enforces the 7 design rules (CPU, tokenizer only)
prescreen.py               Behavioural gate on collected traces (amendment §8)
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
2. **Re-run traces on the new stimuli** — follow
   `experiments/README_GPU_PHASE.md` (run-2 procedure, ~35 min GPU since the
   lens is already fitted). It has a hard gate at Step 4: `prescreen.py` must
   pass per family before any analysis, per amendment §8.
3. Re-analyze, regenerate figures, and only then rewrite Results/Abstract.
4. Integrate logit-lens robustness readout; band identification (Appendix C).
5. Reconcile `paper/DISCUSSION_OUTLINE.md` with whatever the re-run shows.
6. Only after 1.7B is clean: Qwen3-4B replication; Natural Stories.
