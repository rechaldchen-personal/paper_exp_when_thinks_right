# Commitment Dynamics Paper — Project Guide for Claude Code

**What this is**: Research paper on when LMs commit to answers under false-lead
temptation, probed with a Jacobian lens on Qwen3-1.7B. See `README.md` for
current status, results, and known issues; see git history for the full
session-by-session log (the old CLAUDE.md tracker was condensed on 2026-07-19).

## Current state (2026-07-24)

- **Run 2 (Qwen3-1.7B) is complete and trustworthy.** Rebuilt stimuli (156
  items, 72 matched pairs + 12 hard controls), gated by `prescreen.py`,
  analyzed with the corrected pipeline, sensitivity-checked, and cross-checked
  with an independent logit-lens readout. `out/traces_run2.json` +
  `out/analysis_real/report.json` are canonical for 1.7B. Run 1 (2026-07-17)
  is withdrawn — see `out/ANALYSIS_NOTE.md` — do not quote it.
- **Headline finding #1 (within 1.7B): H3 (dissociation gap), not H2.**
  Confirmatory metric significant at every θ/band setting *and* under
  logit-lens (`experiments/SENSITIVITY_REPORT.md`,
  `experiments/LOGIT_LENS_REPORT.md`). H2's "internal revision" reading is
  refuted at every setting tested. H1 holds only under the confirmatory
  readout.
- **Headline finding #2 (THE key update this session): H3 does NOT replicate
  on Qwen3-4B, under either readout.** Fresh lens fit
  (`out/lens_qwen3_4b.pt`), full pre-screen/analysis pass, both readouts —
  every test significant on 1.7B is null on 4B (p 0.061–0.469, near-zero
  effect sizes, not underpowered). Explanation: 4B is more accurate and far
  less internally tempted by the false-lead framing (arithmetic: 100% vs.
  70.8% accuracy; 4.2% vs. 29.2% behavioral temptation; internal
  `distractor_lead_layers` gap shrinks ~3×). Full report:
  `experiments/4B_REPLICATION_REPORT.md`. **This is reported as a finding
  about scale-dependence, not minimized as a limitation** — see that report
  and `paper/DISCUSSION_OUTLINE.md` §6.2.2 for how it's framed.
- **Paper fully rewritten to match** (2026-07-24): `paper/ABSTRACT.md`,
  `paper/RESULTS_TEMPLATE.md` (new §5.8 for the 4B replication),
  `paper/DISCUSSION_OUTLINE.md` (new §6.2.2 "Why the Effect Doesn't
  Generalize"), `paper/INDEX.md` all updated. Every numeric claim across all
  docs was verified programmatically against the underlying report.json
  files before committing — see the verification commands in this session's
  history if you need to re-check.
- All raw GPU outputs (traces, lens, analyses, band-ID JSONs) are local —
  pod is shut down. `out/PIPELINE_PAUSED.md` and `out/SYNC_COMPLETE.md`
  document what was and wasn't synced (4B's band-ID PNG plots were not).

## Priority queue (in order)

1. ✅ Pre-registration amended and locked — `experiments/PRE_REGISTRATION_AMENDMENT.md`.
2. ✅ Run-2 GPU traces (1.7B) collected, gated, analyzed, sensitivity-checked.
3. ✅ Results/Abstract/Discussion rewritten from the real 1.7B numbers.
4. ✅ Logit-lens readout: implemented AND run (1.7B) —
   `experiments/LOGIT_LENS_REPORT.md`.
5. ✅ Per-model band identification: implemented AND run (all 4
   model×readout combos) — `experiments/BAND_IDENTIFICATION_REPORT.md`,
   not adopted as a new default (accuracy-curve caveat documented there).
6. ✅ Qwen3-4B replication: fresh lens fit, both readouts, full pipeline —
   `experiments/4B_REPLICATION_REPORT.md` — **H1/H3 do not replicate**.
7. ✅ Paper (Abstract/Results/Discussion/INDEX) rewritten to integrate all
   of the above, numbers verified against source JSON.
8. ✅ **Qwen3-8B replication prepped, not yet run** (2026-07-24): stimuli
   pre-verified against the real 8B tokenizer, no rework needed. Grounded
   compute estimate (from actual 4B fit timing, not just config math): ~14h
   for the lens fit, ~15h total. One resumable script covers the whole
   pipeline: `out/run_8b_pipeline.sh` — run it via `nohup ... & disown` on
   the pod given the length. Full plan: `experiments/README_GPU_PHASE.md` §D.
9. **Next**: run the 8B GPU session above; expand
   `paper/DISCUSSION_OUTLINE.md` writing templates into final submission
   prose. CoT variant (`02_run_experiment_cot.py`) and Natural Stories
   remain optional; if Natural Stories is run, correlate against H3, not
   H2, and treat as per-model (§6.4).

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
  Confirmatory 2AFC variants (`l_star_2afc`, `gap_2afc`, `oscillation_2afc`)
  read out over {target, distractor} only.
- Analysis flags: `--outdir` (not `--out`), `--theta-pct`, `--dev-split 0.6`.
- Qwen3-1.7B config: hidden=2048, layers=28, vocab=151936. Qwen3-4B: hidden=2560,
  layers=36, same 151936 vocab (so stimuli need no rework across sizes).
- `02_run_experiment.py --readout {jlens,logit_lens}`: both produce identical
  record schemas, so `03_analyze.py` runs on either unmodified — that's the
  whole point of the robustness comparison.
- **4B replication result** (memorize this — it's the paper's second
  headline): H1/H3 confirmatory, robust across every θ/band/lens-method
  setting on 1.7B, are null on 4B under both readouts. Do not describe H3 as
  "robust" without the "within Qwen3-1.7B" qualifier — that's the single
  most important phrasing rule in the whole paper now.
- 4B traces/analysis: `out/traces_4b.json` (jlens), `out/traces_4b_logitlens.json`,
  `out/analysis_4b/`, `out/analysis_4b_logitlens/`. Comparison figure:
  `out/figures/model_comparison_1p7b_vs_4b.png` (`make_comparison_figure.py`).
