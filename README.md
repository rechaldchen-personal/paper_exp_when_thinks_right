# Commitment Dynamics in the J-Space

When do language models commit to answers, and what happens under false-lead
temptation? We probe Qwen3 with a Jacobian lens on matched stimulus pairs
(straightforward vs. false-lead) and test three pre-registered hypotheses about
commitment layer (H1), internal oscillation (H2), and the confidence–correctness
dissociation gap (H3).

**Status (2026-07-24)**: Complete for this phase. Qwen3-1.7B (run 2) shows a
robust dissociation-gap effect (H3) — robust to θ, band, *and* lens method.
**That effect does not replicate on Qwen3-4B, under either readout.** This is
now the paper's second major finding, not a caveat: a within-model-robust
interpretability signature failed to generalize one step up in scale. Run 1
is withdrawn; do not quote its numbers (`out/ANALYSIS_NOTE.md`).

---

## Headline result

**H3 (dissociation gap) is robust within Qwen3-1.7B and absent in Qwen3-4B.**
Full writeup: `experiments/4B_REPLICATION_REPORT.md`. One-tailed exact
signed-rank p-values, bold = p < 0.05:

| Metric | 1.7B jlens | 1.7B logit_lens | 4B jlens | 4B logit_lens |
|---|---|---|---|---|
| H1 confirmatory | **0.002** | **0.040** | 0.334 | 0.461 |
| H2 confirmatory | 0.461 | 0.266 | 0.432 | 0.348 |
| H3 confirmatory | **0.002** | **0.033** | 0.369 | 0.434 |

(Primary-metric rows and full table: `paper/RESULTS_TEMPLATE.md` §5.8.)

**Why it doesn't replicate**: Qwen3-4B is both more accurate and far less
internally tempted by the false-lead framing than 1.7B — e.g. on arithmetic,
straightforward accuracy is 100% vs. 70.8%, and the false-lead distractor
reaches the model's output top-5 only 4.2% of the time vs. 29.2% at 1.7B. The
internal `distractor_lead_layers` gap (how much the framing perturbs internal
computation) shrinks roughly threefold. Read together: 4B is harder to fool,
so there's less dissociation window for H3 to detect. This reframes the
paper's contribution — the effect is real but capability-gated, not a fixed
transformer property — which is itself a substantive finding about the
fragility of single-model interpretability claims.

**H2 (oscillation)** never supported a "revision between candidates" reading
at any of four independent checks: confirmatory metric, θ-sensitivity,
logit-lens, and now 4B — a consistent null everywhere.

## Repository background

- Analysis is reproducible, exact, and one-tailed as pre-registered
  (`03_analyze.py`); environment pinned and self-checking (`verify_env.py`).
- Stimuli: **72/72 strictly matched pairs**, 24 per family, plus 12 hard
  controls (`build_stimuli.py`/`validate_stimuli.py`), verified to need no
  rework across the 1.7B/4B tokenizers (shared vocabulary).
- Two readouts per hypothesis (primary: global argmax; confirmatory:
  target-vs-distractor only), two lens methods (jlens; logit-lens/identity
  transport), two models (1.7B; 4B) — every H1–H3 claim is reported across
  all of these axes, not just the one that looked best.

**Supporting reports** (all in `experiments/`):
- `PRE_REGISTRATION.md` + `PRE_REGISTRATION_AMENDMENT.md` — locked analysis plan
- `SENSITIVITY_REPORT.md` — θ/band sensitivity (1.7B/jlens)
- `LOGIT_LENS_REPORT.md` — jlens vs. logit-lens comparison (1.7B)
- `BAND_IDENTIFICATION_REPORT.md` — data-driven band identification (all 4
  model×readout combos; not adopted, see its caveats)
- `4B_REPLICATION_REPORT.md` — the headline cross-scale finding above

**Still open**:
1. Expand `paper/DISCUSSION_OUTLINE.md` writing templates into final
   submission prose — structurally complete, still template-shaped. Free/CPU.
2. **Qwen3-8B replication — prepped, next up.** Stimuli pre-verified
   (`validate_stimuli.py --model Qwen/Qwen3-8B` passes all 156, shared
   vocab). Grounded estimate (from the actual 4B fit time, not just config
   math): **~14h GPU for the lens fit**, ~15h total with traces/band-ID.
   One resumable, backgrounded script covers the whole pipeline:
   `out/run_8b_pipeline.sh`. Full plan: `experiments/README_GPU_PHASE.md` §D.
3. Natural Stories external validity, retargeted at H3 (not H2, given its
   non-replication) — see Discussion §6.4.
4. 4B's band-identification plots weren't synced before pod shutdown
   (JSON summaries were) — cosmetic gap, see `BAND_IDENTIFICATION_REPORT.md`.

## Repository layout

```
stimuli.json               156 stimuli: 72 matched pairs (24/family) + 12 hard controls
build_stimuli.py           Regenerates stimuli.json; documents the run-1 defects
validate_stimuli.py        Enforces the 7 design rules (CPU, tokenizer only)
prescreen.py               Behavioural gate on collected traces (amendment §8)
00_generate_mock_traces.py Synthetic traces for CPU-only pipeline validation
01_fit_lens.py             Fit Jacobian lens (GPU, ~2-6 h depending on model)
02_run_experiment.py       Collect traces; --readout jlens|logit_lens (GPU)
02_run_experiment_cot.py   CoT variant scaffold (H4, not yet run)
03_analyze.py              Metrics + exact signed-rank tests (CPU)
04_identify_band.py        Per-model workspace band identification (GPU)
band_analysis.py           Band-identification math (CPU, unit-tested)
test_band_analysis.py      Unit tests for band_analysis.py — run on CPU
generate_figures.py        Publication figures from out/analysis_real/ (CPU)
make_comparison_figure.py  1.7B-vs-4B comparison figure (CPU)
lens_utils.py              Logit-lens (identity transport) readout, integrated
paper/                     Paper sections (see paper/INDEX.md)
experiments/               Pre-registration + amendment (locked), GPU guide,
                           sensitivity/logit-lens/band-ID/4B reports
out/                       Lenses (git-lfs), traces, analysis, figures, logs
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
venv/bin/python validate_stimuli.py --stimuli stimuli.json          # 1.7B
venv/bin/python validate_stimuli.py --model Qwen/Qwen3-4B            # 4B

# Analysis pipeline smoke test (CPU)
venv/bin/python 00_generate_mock_traces.py --out out/traces_mock.json
venv/bin/python 03_analyze.py --traces out/traces_mock.json --outdir out/analysis_mock
venv/bin/python test_band_analysis.py

# Full pipeline (GPU) — see experiments/README_GPU_PHASE.md.
# out/traces_run2.json (1.7B) and out/traces_4b.json (4B), with their
# out/analysis_*/ directories, are already the current, checked-in results.
# Only re-run if collecting NEW data — write to new --out/--outdir paths so
# you don't overwrite them.
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --out out/lens_qwen3_1p7b.pt
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --readout jlens \
    --lens out/lens_qwen3_1p7b.pt --out out/traces_run3.json
venv/bin/python 03_analyze.py --traces out/traces_run3.json --outdir out/analysis_run3 --dev-split 0.6
venv/bin/python generate_figures.py   # reads out/analysis_real/ by default — repoint or promote first
venv/bin/python make_comparison_figure.py   # regenerate the 1.7B-vs-4B figure
```

The analysis plan (band, θ procedure, dev/holdout split, tests) is locked in
`experiments/PRE_REGISTRATION.md` + `experiments/PRE_REGISTRATION_AMENDMENT.md`
— do not tune it post hoc, including in light of the 4B null.

## Status log

1. ✅ Pre-registration amended and locked.
2. ✅ Run 2 (Qwen3-1.7B): traces collected, gated, analyzed, sensitivity-checked.
3. ✅ Results/Abstract/Discussion rewritten to match run 2 + sensitivity.
4. ✅ Logit-lens readout (1.7B): implemented, run, analyzed —
   strengthens H3's robustness claim, weakens H1's.
5. ✅ Per-model band identification: implemented, run on all 4 model×readout
   combinations — not adopted as a new default (see caveats).
6. ✅ Qwen3-4B replication: fresh lens fit, both readouts, full pre-screen
   and analysis — **H1/H3 do not replicate**; this is now the paper's
   second headline finding, fully integrated into Abstract/Results/Discussion.
7. **Next**: expand Discussion templates into final prose; consider a third
   model size to disambiguate smooth-scaling from a 1.7B-specific property;
   Natural Stories (retarget at H3).
