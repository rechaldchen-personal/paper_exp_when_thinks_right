# Commitment Dynamics in the J-Space

When do language models commit to answers, and what happens under false-lead
temptation? We probe Qwen3 with a Jacobian lens on matched stimulus pairs
(straightforward vs. false-lead) and test three pre-registered hypotheses about
commitment layer (H1), internal oscillation (H2), and the confidence–correctness
dissociation gap (H3).

**Status**: First GPU run on Qwen3-1.7B complete (2026-07-17). Results are in
the paper draft. H2 strongly supported; H1/H3 directional but underpowered —
see "Known issues" below for what needs fixing before the next run.

---

## Results so far (Qwen3-1.7B, holdout set)

| Hypothesis | Median Δ (FL − SF) | Wilcoxon | n pairs | Verdict |
|---|---|---|---|---|
| H1: later commitment (ℓ*) | +2.0 layers | p = 0.0625 | 6 | Directional, not significant |
| H2: more oscillation | +2.0 changes | p = 4.89×10⁻⁵ | 32 | **Supported** |
| H3: larger gap (ℓ* − ℓ_H) | +4.0 layers | p = 0.0625 | 6 | Directional, not significant |

Full numbers: `out/analysis_real/report.json`; figures: `out/figures/`;
narrative: `paper/RESULTS_TEMPLATE.md` (filled with real data).

## Known issues blocking stronger claims

1. **Arithmetic stimuli behaviorally fail** (1/56 correct): the model's top-1
   after `"... equals"` is a bare space token, so ℓ* is undefined for nearly all
   arithmetic items. Fix the prompt format (or query position) and re-run.
2. **H1/H3 are power-limited, not effect-limited**: only 6 complete pairs have
   ℓ* defined in both conditions; p = 0.0625 is the theoretical floor at n = 6.
   All 6 pairs moved in the predicted direction. Fixing (1) should fix this.
3. **Hard controls contribute no data**: only 3 items exist and all landed in
   the dev split (holdout n = 0). Expand to ~10+ or drop the specificity claim.
4. **Promised robustness checks not yet run**: logit-lens secondary readout
   (`lens_utils.py`, not integrated) and per-model workspace-band
   identification (band is the pre-registered default [0.25, 0.90]).

## Repository layout

```
stimuli.json               163 stimuli (51 factual, 56 arithmetic, 56 garden-path)
00_generate_mock_traces.py Synthetic traces for CPU-only pipeline validation
01_fit_lens.py             Fit Jacobian lens (GPU, ~2 h)
02_run_experiment.py       Collect lens traces on stimuli (GPU, ~20 min)
02_run_experiment_cot.py   CoT variant scaffold (H4, not yet run)
03_analyze.py              Metrics + Wilcoxon tests (CPU)
generate_figures.py        Publication figures from out/analysis_real/ (CPU)
lens_utils.py              Logit-lens utilities (pending integration)
paper/                     Paper sections (see paper/INDEX.md)
experiments/               PRE_REGISTRATION.md (locked), GPU re-run guide,
                           workspace band guide, Natural Stories plan
out/                       Lens (git-lfs), traces, analysis, figures, logs
```

## Environment

```bash
./setup_env.sh                     # or: python3 -m venv venv && pip install -r requirements.txt
source venv/bin/activate
python verify_env.py               # checks CPU deps; GPU deps only needed for 01/02
pip install torch transformers datasets   # GPU phases only
```

## Reproducing the pipeline

```bash
# CPU-only sanity check of the analysis pipeline
python 00_generate_mock_traces.py
python 03_analyze.py --traces out/traces_mock.json --outdir out/analysis_mock

# Full pipeline (GPU) — see experiments/README_GPU_PHASE.md
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --validate   # tokenization check, CPU
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --out out/lens_qwen3_1p7b.pt
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens_qwen3_1p7b.pt --out out/traces.json
python 03_analyze.py --traces out/traces.json --outdir out/analysis_real --dev-split 0.6
python generate_figures.py
```

The analysis plan (band, θ procedure, dev/holdout split, tests) is locked in
`experiments/PRE_REGISTRATION.md` — do not tune it post hoc.

## Next steps

1. Fix arithmetic prompt format; re-validate behavioral accuracy per family.
2. Expand hard controls (~10+ items).
3. Re-run traces + analysis on Qwen3-1.7B (lens already fitted, ~30 min GPU).
4. Integrate logit-lens robustness readout; band identification (Appendix C).
5. Optional: Qwen3-4B replication; Natural Stories external validity.
6. Update `paper/DISCUSSION_OUTLINE.md` placeholders with final numbers.
