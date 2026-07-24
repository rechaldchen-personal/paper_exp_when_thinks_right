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

**Next up** (2026-07-22): three follow-ups are code-complete and CPU-side
tested, GPU-side unrun — see "After run 2" below: (A) logit-lens robustness
readout, ~20-30 min; (B) per-model band identification, ~5-15 min; (C)
Qwen3-4B replication, ~5-7 h (needs its own lens fit). None require further
code changes; A and B don't even need a fresh lens fit.

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

## After run 2

**Status (2026-07-24): A, B, C below are all DONE** — see
`experiments/LOGIT_LENS_REPORT.md`, `BAND_IDENTIFICATION_REPORT.md`, and
`4B_REPLICATION_REPORT.md` for results (headline: H3 robust within
Qwen3-1.7B across every axis in A/B, but does not replicate at 4B in C).
Kept below as the reference procedure — **D (Qwen3-8B) is next**, planned
2026-07-25, not yet run.

### A. Logit-lens robustness readout (~20-30 min GPU, no lens fit needed) — DONE

`02_run_experiment.py --readout logit_lens` is implemented (identity
transport, no `jlens` dependency, output schema identical to the primary
readout). Run it, analyze it, and compare directions against run 2:

```bash
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --readout logit_lens \
    --out out/traces_run2_logitlens.json
python prescreen.py --traces out/traces_run2_logitlens.json
venv/bin/python 03_analyze.py --traces out/traces_run2_logitlens.json \
    --outdir out/analysis_run2_logitlens --dev-split 0.6
```

Compare `out/analysis_run2_logitlens/report.json` against
`out/analysis_real/report.json`: do the same hypotheses come out significant,
in the same direction, especially H3 (the headline)? Report as Appendix D.

### B. Per-model workspace band identification (~5-15 min GPU per readout) — DONE

`04_identify_band.py` + `band_analysis.py` are implemented and unit-tested
(`venv/bin/python test_band_analysis.py`, CPU-only, passes). Needs a GPU pass
over a held-out corpus (cheaper than lens fitting — forward passes only):

```bash
python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout jlens \
    --lens out/lens_qwen3_1p7b.pt --n-prompts 200 \
    --out out/band_identification_jlens.json
python 04_identify_band.py --model Qwen/Qwen3-1.7B --readout logit_lens \
    --n-prompts 200 --out out/band_identification_logitlens.json
```

Inspect `out/figures/band_identification_*.png` and the `suggested_band_*`
fields before trusting them (§ full guidance in `workspace_band_guide.md`).
If you adopt a different band, that's a pre-registration amendment (rerun
sensitivity + main analysis with the new default), not a silent swap.

### C. Qwen3-4B replication (~5-7 h GPU total, budget accordingly) — DONE

Verified CPU-side before spending any GPU time: **stimuli need no rework** —
`validate_stimuli.py --model Qwen/Qwen3-4B` passes all 156 stimuli, all 7
rules (Qwen3-1.7B and 4B share the same 151,936-token vocabulary, confirmed
via each model's `config.json`). The scripts are already model-agnostic
(`--model` is a real parameter everywhere, nothing hardcoded to 1.7B) — no
code changes needed, only new output paths so nothing overwrites the 1.7B
run.

Rough compute estimate (layers × hidden_size²: 28×2048² for 1.7B vs
36×2560² for 4B ≈ **2.0×** the compute per token) — expect **lens fitting
roughly 2× longer than 1.7B's** (which took a few hours), so budget ~4-6h for
fitting alone, plus ~30-40 min for trace collection. This is an estimate from
published configs, not a measurement — confirm early and adjust.

```bash
# 1. Fit a fresh lens (4B needs its own; do NOT reuse the 1.7B lens)
python 01_fit_lens.py --model Qwen/Qwen3-4B --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_4b.pt --checkpoint out/fit_ckpt_4b.pt

# 2. Behavioral pre-screen still applies — same stimuli, different model,
#    accuracy could differ meaningfully
python 02_run_experiment.py --model Qwen/Qwen3-4B --readout jlens \
    --lens out/lens_qwen3_4b.pt --out out/traces_4b.json
python prescreen.py --traces out/traces_4b.json   # HARD GATE, same as run 2

# 3. Analyze under the SAME pre-registered plan (band, theta, split, tests) —
#    this is what makes it a real replication rather than a fresh fishing
#    expedition. PRE_REGISTRATION_AMENDMENT.md §"Applies to" already covers
#    "any later model" under the same plan; no new amendment needed unless
#    you want to change something (e.g. adopt a band from step B above).
venv/bin/python 03_analyze.py --traces out/traces_4b.json \
    --outdir out/analysis_4b --dev-split 0.6
venv/bin/python generate_figures.py   # repoint at out/analysis_4b or promote it
```

Interpretation: this is a genuine replication attempt (same plan, second
model), not new hypothesis mining. Report whichever way it comes out — H3
replicating on 4B is the strongest possible credibility signal this paper can
get; H3 *not* replicating is equally important to report honestly, and would
mean the 1.7B finding doesn't generalize past a single small model.

**Result: H1/H3 do NOT replicate on 4B, under either readout** — every test
significant on 1.7B is null on 4B (p 0.061–0.469), not underpowered (effect
sizes ~0, sample sizes if anything larger). Explanation: 4B is more accurate
and far less internally tempted by the false-lead framing. Full detail:
`experiments/4B_REPLICATION_REPORT.md`.

### D. Qwen3-8B replication — planned, next up (~14-15 h GPU, budget accordingly)

Verified CPU-side before spending any GPU time, same as C: **stimuli need no
rework** — `validate_stimuli.py --model Qwen/Qwen3-8B` passes all 156
stimuli, all 7 rules (1.7B/4B/8B share the same 151,936-token vocabulary,
confirmed via each model's `config.json`). Scripts remain model-agnostic, no
code changes needed.

**Motivation**: C establishes that H3 (robust across every within-model axis
on 1.7B) is absent on 4B. That's one replication attempt at one step in
scale — it can't distinguish "the effect fades smoothly with scale" from
"1.7B has some specific property a smooth story wouldn't predict." 8B is the
natural next data point: same family (isolates the scale variable — Qwen3
has no 7B; the dense lineup is 0.6B/1.7B/4B/8B/14B/32B), no new code.

**Compute estimate — grounded in the actual 4B run, not just config math**:
the resumed 4B fit went from checkpoint n_done=349 to completion in ≤3h40m
for 651 prompts, extrapolating to **~5.6h for a full 1000-prompt 4B fit**
(matches the earlier config-based ~2× prediction from A/B/C). Scaling that
by the same layers×hidden² method (36×4096² for 8B vs 36×2560² for 4B ≈
**2.56×** the compute per token): **8B lens fit ≈ 5.6h × 2.56 ≈ 14h**, plus
~30-40 min for jlens traces, ~10-15 min for band identification (both
readouts), ~20-30 min for logit-lens traces. **Budget ~15h GPU total.**
VRAM is not a concern — 8B is ~16GB in bf16, comfortable on an 80GB card
(same H100 SXM class as the 1.7B/4B runs).

**Given the length, run this as one resumable, backgrounded script** rather
than issuing commands interactively — a dropped SSH session over 15h would
otherwise kill the fit. `out/run_8b_pipeline.sh` is prepared: it runs fit →
jlens traces → prescreen → analyze → band ID (both readouts) → logit-lens
traces → prescreen → analyze, with every step idempotent (skips work whose
output already exists), so it's safe to re-run after any interruption.

```bash
# On the pod, after the usual bootstrap (git pull, git lfs pull, pip installs
# — see Step 0-2 above):
nohup bash out/run_8b_pipeline.sh > /dev/null 2>&1 &
disown
# Reconnect any time; check progress with:
tail -f out/run_8b_pipeline.log
# Or check whether it's still running at all:
pgrep -af run_8b_pipeline.sh
```

When `out/ALL_8B_GPU_DONE.txt` appears, everything is done. Then sync down
(`scp`/`git lfs push` the new lens + traces + analysis + band-ID files) same
as the 4B round, and re-run `make_comparison_figure.py` locally with an
added 8B column.

**Interpretation plan (fixed now, before seeing the data)**: if H3
replicates on 8B, that's strong evidence for "fades somewhere past 8B, or
doesn't fade monotonically" and meaningfully changes the scale-dependence
story. If it's null on 8B too (matching 4B), that's a second consistent data
point for smooth scale-dependence and strengthens the existing 4B finding
rather than changing it. Either way, report both readouts and both
hypotheses (H1, H3) exactly as done for 4B — no cherry-picking which test to
lead with after seeing the result.

### Optional (lower priority, not scoped in detail here)

CoT variant (`02_run_experiment_cot.py`, H4), Natural Stories
(`natural_stories_plan.md` — target the correlation at the dissociation gap,
not oscillation; see `paper/DISCUSSION_OUTLINE.md` §6.4).

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
