# Commitment Dynamics Paper — Project Guide for Claude Code

**What this is**: Research paper on when LMs commit to answers under false-lead
temptation, probed with a Jacobian lens on Qwen3-1.7B. See `README.md` for
current status, results, and known issues; see git history for the full
session-by-session log (the old CLAUDE.md tracker was condensed on 2026-07-19).

## Current state (2026-07-22)

- **Run 2 is the current, trustworthy result.** Rebuilt stimuli (156 items,
  72 matched pairs + 12 hard controls) on GPU, gated by `prescreen.py` (all
  three families cleared 70%+ accuracy), analyzed with the corrected pipeline,
  and sensitivity-checked. `out/traces_run2.json` +
  `out/analysis_real/report.json` are canonical. Run 1 (2026-07-17) is
  withdrawn — see `out/ANALYSIS_NOTE.md` — do not quote it.
- **Headline finding: H3 (dissociation gap), not H2.** Confirmatory metric
  significant at every θ/band setting tested (`experiments/SENSITIVITY_REPORT.md`).
  **H2's "internal revision" reading is refuted**: primary oscillation metric
  is significant but θ-fragile (null at θ70/θ90), the confirmatory
  target-vs-distractor test is a stable null (p≈0.46–0.62 everywhere), and
  hard controls (no tempting distractor) oscillate almost as much as
  false-lead items. **H1** holds only under the confirmatory readout
  (robust), not the primary global-argmax one.
- **Paper rewritten to match**: `paper/ABSTRACT.md`, `paper/RESULTS_TEMPLATE.md`,
  `paper/DISCUSSION_OUTLINE.md` (esp. §6.2.2 oscillation and §6.4
  psycholinguistics, both originally built on the now-refuted reading),
  `paper/INDEX.md`, `paper/INTRODUCTION_AND_RELATED_WORK.md` (stimulus count,
  §2.8 summary table) all updated 2026-07-22.
- **Lens unchanged and still valid** (`out/lens_qwen3_1p7b.pt`, git-lfs).

## Priority queue (in order)

1. ✅ Pre-registration amended and locked — `experiments/PRE_REGISTRATION_AMENDMENT.md`.
2. ✅ Run-2 GPU traces collected and gated.
3. ✅ Analyzed; `out/analysis_real/` promoted to run-2; figures regenerated.
4. ✅ Sensitivity checks (θ 70/80/90 × band 3 settings) —
   `experiments/SENSITIVITY_REPORT.md`.
5. ✅ Results/Abstract/Discussion rewritten from the real numbers.
6. **Next**: expand `paper/DISCUSSION_OUTLINE.md` writing templates into
   final submission prose (structurally complete, still template-shaped).
7. **Robustness checks still promised in Methods 4.7 but not run**: logit-lens
   secondary readout (`lens_utils.py` not yet integrated into
   `02_run_experiment.py`) and per-model workspace band *identification*
   (`experiments/workspace_band_guide.md`) — distinct from the band
   *sensitivity* check already done.
8. Deliberate next GPU session: Qwen3-4B replication (fresh lens fit) and/or
   more stimuli for power (H1/H3 primary rest on 14 pairs). CoT variant
   (`02_run_experiment_cot.py`) and Natural Stories optional; if Natural
   Stories is run, correlate against H3 (dissociation gap), not H2 — see
   Discussion §6.4.

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
