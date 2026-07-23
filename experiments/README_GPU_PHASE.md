# GPU Phase Guide — Run 2 (complete) / template for future runs

**Status**: Run 2 completed 2026-07-22 — `out/traces_run2.json`, gated
(`prescreen.py` passed all families), analyzed (`out/analysis_real/`), and
sensitivity-checked (`SENSITIVITY_REPORT.md`). Results are in
`paper/RESULTS_TEMPLATE.md`. This guide remains the procedure to follow for
**any future GPU run** (4B replication, expanded stimuli, etc.) — swap the
model/output paths and follow the same steps, especially the Step 4 gate.
Run 1 (2026-07-17) is withdrawn — see `PRE_REGISTRATION_AMENDMENT.md` §1; do
not quote its numbers.

**Budget**: ~35 min GPU if the fitted lens is available, ~2.5 h if it must be
refitted. Nothing here needs a fresh lens unless you are changing model.

**The one rule**: there is a hard gate at Step 4. If the pre-screen fails, fix
the stimuli and recollect — do not proceed to analysis. Run 1's arithmetic
scored 1/56 and was analyzed anyway; that is the failure this gate exists to
prevent.

---

## Step 0 — Get the code and the lens onto the GPU box

```bash
git clone <your-remote> paper_exp_when_thinks_right
cd paper_exp_when_thinks_right
git lfs install && git lfs pull          # REQUIRED — see the warning below
pip install torch transformers datasets scipy numpy matplotlib pandas
pip install -e ./jacobian-lens           # provides `jlens`
```

> ⚠️ **Check the lens is real, not a pointer.** `out/lens_qwen3_1p7b.pt` is
> tracked by git-lfs. Without `git lfs pull` it is a 134-byte text stub and
> `JacobianLens.load()` will fail or, worse, behave oddly. Verify:
>
> ```bash
> ls -la out/lens_qwen3_1p7b.pt      # expect ~226 MB, NOT 134 bytes
> file  out/lens_qwen3_1p7b.pt       # expect "data", NOT "ASCII text"
> ```
>
> If it comes back as ASCII text and `git lfs pull` cannot fetch it (the bytes
> may never have reached the remote), you must refit — jump to Step 2.

## Step 1 — Confirm the inputs (CPU, 1 min)

```bash
python validate_stimuli.py --stimuli stimuli.json
```

Expected: `156 stimuli, 72 pairs` and `✅ all rules pass`. This enforces matched
pairs, no answer leakage, single-token targets, no answer-in-context, and clean
prompt+target concatenation. **Do not skip it** — the old
`02_run_experiment.py --validate` only checked tokenization and passed happily
on the broken run-1 set.

## Step 2 — Fit the lens (GPU, ~2 h) — SKIP unless Step 0 said otherwise

Only needed for a new model, or if the fitted lens is unavailable.

```bash
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_1p7b.pt --checkpoint out/fit_ckpt.pt
```

Resumable: re-running the same command picks up from the checkpoint.

## Step 3 — Collect traces (GPU, ~20 min)

```bash
python 02_run_experiment.py --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt \
    --stimuli stimuli.json \
    --out out/traces_run2.json
```

> Write to `traces_run2.json`, **not** `traces.json`. Run-1 traces are kept as
> the evidence behind `out/ANALYSIS_NOTE.md`; overwriting them destroys the
> provenance of the withdrawal.

Watch the live `top1=` column. Healthy output looks like `top1=' Paris'`,
`top1=' horse'`, `top1=' ten'`. If an entire family prints `top1=' '` or
`top1=' the'`, stop now — that is the run-1 failure mode recurring, and Step 4
is about to tell you so anyway.

## Step 4 — Behavioural pre-screen (CPU, seconds) — **HARD GATE**

```bash
python prescreen.py --traces out/traces_run2.json
```

Required: every family at **≥50%** straightforward top-1 accuracy. For
reference, run 1 produced factual 95.8%, garden-path 42.9%, arithmetic 3.6%.

- **Exit 0 / all ✅** → continue to Step 5.
- **Exit 1 / any ❌** → **stop.** Per amendment §8, fix that family's stimuli in
  `build_stimuli.py`, regenerate, re-validate, and redo Step 3. The failing
  examples printed by the tool tell you what the model wanted to say instead.
  Do not analyze around a failing family and do not lower `--min-accuracy` to
  make it pass.

The false-lead temptation percentages printed underneath are diagnostic, not a
gate: they say how often the distractor reached the final top-5, i.e. whether
the manipulation actually tempted the model. Very low numbers across the board
would mean the false-lead framing is too weak to be testing anything, which is
worth knowing before you interpret a null.

## Step 5 — Analysis (CPU, ~5 min)

```bash
python 03_analyze.py --traces out/traces_run2.json \
    --outdir out/analysis_run2 --dev-split 0.6 --band 0.25 0.90 --theta-pct 80.0
```

All settings are pre-registered — do not tune them. Note the flag names:
`--outdir` (not `--out`) and `--theta-pct`.

Output prints two blocks:

- **PRIMARY (pre-registered)** — `l_star` (H1), `oscillation` (H2), `gap` (H3),
  one-tailed exact signed-rank. These are the headline tests.
- **CONFIRMATORY 2AFC** — the same hypotheses read out over {target,
  distractor} only. These decide what may be *claimed* about mechanism.

Check `n_pairs` and `n_nonzero`. Run 1 collapsed to n=6 for H1/H3; the rebuilt
set should give tens of pairs. If n is still tiny, the metrics are undefined for
most items and something upstream is still wrong.

## Step 6 — Figures and tables (CPU, ~2 min)

```bash
python generate_figures.py     # reads out/analysis_real/ by default
```

If you analyzed into `out/analysis_run2`, either point the script at it or copy
the results into `out/analysis_real/` first. Outputs land in `out/figures/`.

## Step 7 — Interpret, using the table you committed to in advance

Read `PRE_REGISTRATION_AMENDMENT.md` §5 **before** writing any prose. The
permitted claim is fixed by the joint outcome:

| `oscillation` | `oscillation_2afc` | You may claim |
|---|---|---|
| sig | sig | revision **between the candidate answers** — current framing holds |
| sig | null | unattributed top-1 churn; strip "revision between candidates" from Abstract and Discussion |
| null | sig | sub-threshold preference effect; reframe |
| null | null | H2 not supported |

Report all six tests in Results whichever way they fall (§6), alongside the
run-1 pilot values disclosed in §0.

Then, in order: rewrite `paper/RESULTS_TEMPLATE.md` from the new numbers,
update `paper/ABSTRACT.md`, and reconcile `paper/DISCUSSION_OUTLINE.md` (it
still holds mock-era placeholders such as "X top-1 flips" and "r ≈ ?").

---

## After run 2 is clean

Only once 1.7B passes the gate and the analysis is stable:

- **Robustness** promised in Methods §4.7: logit-lens secondary readout
  (`lens_utils.py`, not yet wired into `02_run_experiment.py`) and per-model
  workspace band (`workspace_band_guide.md`, Appendix C).
- **Qwen3-4B replication** — needs its own lens fit (Step 2, ~2 h).
- **Optional**: CoT variant (`02_run_experiment_cot.py`, H4), Natural Stories
  (`natural_stories_plan.md`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `JacobianLens.load()` fails; lens is 134 bytes | git-lfs stub — `git lfs pull` (Step 0) |
| A whole family shows `top1=' '` | Model is answering in digits: Qwen3 tokenizes ' 10' as [220,16,15]. Answers must be single-token English number words in the ' word' form. |
| A family shows `top1=' the'` | Probe ends at a verb; it must end on "the" so the next token is the answer noun. |
| `KeyError: 'wilcoxon_stat'` | Stale `generate_figures.py` against a new report; both are updated in this repo — `git pull`. |
| p-values differ from the committed ones | Wrong interpreter. Use `venv/bin/python` and run `verify_env.py`. |
| `jlens` API mismatch | Call sites are marked `[API]` in `01_fit_lens.py` / `02_run_experiment.py`; the signatures worked in run 1. |
| OOM while fitting | Lower `--n-prompts` to 500 or `--seq-len` to 64. |
