# Commitment Dynamics Paper — Project Guide for Claude Code

**What this is**: Research paper on when LMs commit to answers under false-lead
temptation, probed with a Jacobian lens on Qwen3-1.7B. See `README.md` for
current status, results, and known issues; see git history for the full
session-by-session log (the old CLAUDE.md tracker was condensed on 2026-07-19).

## Current state (2026-07-19)

- **GPU run 1 complete** (Qwen3-1.7B, 2026-07-17): lens fitted on 1000×128
  FineWeb prompts (`out/lens_qwen3_1p7b.pt`, git-lfs), 163 traces collected,
  pre-registered 60/40 dev/holdout analysis run (`out/analysis_real/`).
- **Findings**: H2 (oscillation) supported at p = 4.89×10⁻⁵; H1/H3 directional
  but not significant (p = 0.0625, only 6 complete pairs — power problem).
- **Paper**: Abstract and Results already filled with real, honest numbers.
  `paper/DISCUSSION_OUTLINE.md` still contains mock-era placeholders.

## Priority queue (in order)

1. **Fix arithmetic stimuli** — model's top-1 after `"... equals"` is a bare
   space token; behavioral accuracy is 1/56. Reformat prompts, re-validate with
   `02_run_experiment.py --validate`, then re-run traces.
2. **Expand hard controls** — only 3 items, all fell in the dev split, so the
   holdout has zero hard-control data.
3. **Re-run GPU traces + analysis** (lens already fitted; ~30 min GPU).
4. **Robustness checks promised in Methods 4.7** — logit-lens secondary readout
   (`lens_utils.py` not yet integrated into `02_run_experiment.py`) and
   per-model workspace band (`experiments/workspace_band_guide.md`).
5. **Reconcile `paper/DISCUSSION_OUTLINE.md`** with real results (H1/H3 not
   significant; remove "X flips", "r ≈ ?", unrun Natural Stories claims).
6. Optional: Qwen3-4B replication, CoT variant (`02_run_experiment_cot.py`),
   Natural Stories (`experiments/natural_stories_plan.md`).

## Ground rules

- **Scientific honesty**: never claim support for a hypothesis the data don't
  show. H1/H3 are "directional, not significant" until a higher-powered re-run
  says otherwise. Mock-data validation proves the pipeline, not the science.
- **Pre-registration is locked** (`experiments/PRE_REGISTRATION.md`): band
  [0.25, 0.90], θ = 80th percentile of straightforward ΔH on dev only, 60/40
  split by pair, seed 42, one-sided Wilcoxon. Don't tune post hoc; any deviation
  must be reported as exploratory.
- **Before burning GPU time**: always run `02_run_experiment.py --validate`
  (tokenization check, CPU) after touching `stimuli.json`.
- `*.pt` files are git-lfs; `out/*.log` are kept for provenance.
- Environment: `source venv/bin/activate`; CPU deps in `requirements.txt`,
  GPU deps (`torch transformers datasets` + jlens) installed on the GPU box.

## Key facts

- Stimuli: 163 items, 78 pairs (51 factual / 56 arithmetic / 56 garden-path;
  conditions: 80 straightforward / 80 false_lead / 3 hard_control).
- Traces schema: see `02_run_experiment.py` docstring (entropy, ranks, logp,
  top1_id per layer×position + behavioral top-5 check).
- Metrics: ℓ_H (entropy collapse), ℓ* (stable target top-1 in band),
  gap = ℓ* − ℓ_H, oscillation (top-1 changes after ℓ_H) — `03_analyze.py`.
- Analysis flags: `--outdir` (not `--out`), `--theta-pct`, `--dev-split 0.6`.
