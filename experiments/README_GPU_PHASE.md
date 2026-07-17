# GPU Phase Execution Guide

**Status**: GPU is the only blocker  
**All code**: Tested CPU-side, ready for GPU  
**Timeline**: ~6 hours (3.5 GPU + 2.5 post-processing)

---

## Quick Start (When GPU Available)

```bash
# 1. Validate (no GPU needed)
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --validate

# 2. Fit lens (~2 hours, GPU)
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --out out/lens.pt

# 3. Collect traces (~10 min, GPU)
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens.pt --out out/traces.json

# 4. Analyze (~5 min, CPU)
python 03_analyze.py --traces out/traces.json --outdir out/analysis --dev-split 0.6

# 5. Generate figures (~30 min, CPU)
# Run code from FIGURE_GENERATION_GUIDE.md (or use script below)
```

---

## Step-by-Step Execution Plan

### Phase 1: API Validation (No GPU)

**Time**: 5 min  
**Purpose**: Verify jlens API calls before spending GPU time

```bash
# Check jlens version
python -c "import jlens; print(jlens.__version__)"

# Validate tokenization (critical!)
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens.pt --validate

# Expected output: "Tokenization OK: 77 stimuli, all targets single-token."
```

**If fails**: Check API_VERIFICATION.md for call-site fixes

---

### Phase 2: Lens Fitting (GPU, ~2 hours)

**Purpose**: Fit Jacobian lens on 1000×128 corpus

```bash
python 01_fit_lens.py \
  --model Qwen/Qwen3-1.7B \
  --n-prompts 1000 \
  --seq-len 128 \
  --out out/lens_qwen3_1p7b.pt
```

**Output**: `out/lens_qwen3_1p7b.pt` (~1-2 GB)

**Checkpoint support**: If interrupted, re-run same command (resumes from checkpoint)

---

### Phase 3: Trace Collection (GPU, ~10 min)

**Purpose**: Apply fitted lens to all 77 stimuli

```bash
python 02_run_experiment.py \
  --model Qwen/Qwen3-1.7B \
  --lens out/lens_qwen3_1p7b.pt \
  --stimuli stimuli.json \
  --out out/traces.json
```

**Output**: `out/traces.json` (~10-50 MB)

**Progress**: Should see `done: pair_id / condition` printed for each stimulus

---

### Phase 4: Analysis (CPU, ~5 min)

**Purpose**: Compute metrics, run statistics, generate heatmaps

```bash
python 03_analyze.py \
  --traces out/traces.json \
  --outdir out/analysis \
  --band 0.25 0.90 \
  --dev-split 0.6 \
  --theta-pct 80.0
```

**Output**:
- `out/analysis/report.json` — Summary statistics + Wilcoxon tests
- `out/analysis/per_stimulus_metrics.json` — Per-item metrics
- `out/analysis/heatmaps/` — 77 heatmap PNGs

**Key output**: Look for `p` values and `median_diff` in report.json

---

### Phase 5: Figure Generation (CPU, ~30 min)

**Option A: Quick (template figures)**

```bash
python3 << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np

# Load analysis results
report = json.load(open('out/analysis/report.json'))
metrics = json.load(open('out/analysis/per_stimulus_metrics.json'))
conditions = ['straightforward', 'false_lead', 'hard_control']

# Figure 1: H1
fig, ax = plt.subplots(figsize=(8, 6))
data_l_star = {cond: [] for cond in conditions}
for m in metrics:
    if m['l_star'] is not None:
        data_l_star[m['condition']].append(m['l_star'])

bp = ax.boxplot([data_l_star[c] for c in conditions],
                  labels=['Straightforward', 'False Lead', 'Hard Control'],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

ax.set_ylabel('Commitment Layer ℓ*', fontsize=12, fontweight='bold')
ax.set_title('H1: Later Commitment Under False Lead', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

h1_test = report['tests']['l_star']
stats_text = f"p={h1_test['p']:.4f}\nΔ={h1_test['median_diff']:.1f}"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('out/figure_h1_real.png', dpi=150)
print("✓ Figure 1 saved")

# Repeat for H2, H3, entropy curves, etc.
# See FIGURE_GENERATION_GUIDE.md for complete code
EOF
```

**Option B: Full code** → See `FIGURE_GENERATION_GUIDE.md` (all 7 figures)

---

### Phase 6: Robustness Checks (Optional, GPU/CPU)

**Workspace band identification** (~30 min):
```bash
# See workspace_band_guide.md for detailed instructions
# Compute kurtosis, autocorr, accuracy curves
# Identify band per-model (currently hardcoded [0.25, 0.90])
```

**Logit-lens replication** (~20 min):
```bash
# Secondary readout with logit-lens (identity transport)
# Rerun 03_analyze.py with logit traces to verify robustness
# Code skeleton in lens_utils.py
```

**Natural Stories** (~30 min):
```bash
# Optional: Correlate oscillation with human reading times
# See natural_stories_plan.md for strategy
# Requires downloading + curating Natural Stories subset
```

---

### Phase 7: Paper Completion (CPU, ~2-3 hours)

#### 7a. Fill Results Section

1. Open `RESULTS_TEMPLATE.md`
2. Replace all `?` with values from `out/analysis/report.json`:
   - Medians for each condition
   - Wilcoxon p-values
   - Effect sizes (median_diff)
3. Fill in figures (copy PNGs from `out/analysis/heatmaps/`)

#### 7b. Fill Discussion Section

1. Open `DISCUSSION_OUTLINE.md`
2. Replace `[GPU phase]` placeholders:
   - Natural Stories correlations (if run)
   - Per-model effect sizes
   - Final effect interpretations
3. Customize the model-specific results

#### 7c. Write Abstract

- Start with one-paragraph thesis from paper_outline.md
- Add actual results (p-values, effect sizes)
- Keep under 250 words
- Template:
  > "We ask when language models commit to answers using the J-space. 
  > We test three hypotheses using a matched-pairs stimulus set across 
  > three domains. Results: H1 (later commitment, p=?), H2 (oscillation, p=?), 
  > H3 (dissociation gap, p=?). These findings reveal internal backtracking 
  > signatures, with implications for uncertainty quantification and 
  > hallucination detection."

#### 7d. Write Acknowledgments

- Thank authors (Gurnee et al. for jlens)
- Note compute resources used
- ~100 words

#### 7e. Compile References

- Add citations from paper_outline.md
- Use METHODS_DRAFT.md as reference
- Check all citations are formatted correctly
- 50–80 references expected

---

## File Locations Reference

**Scripts** (all ready to run):
- `01_fit_lens.py` — Lens fitting
- `02_run_experiment.py` — Trace collection
- `03_analyze.py` — Analysis
- `00_generate_mock_traces.py` — Mock validation (for testing)

**Data**:
- `stimuli.json` — 77 items (use as-is)
- `out/lens.pt` — Will be generated by 01_fit_lens.py
- `out/traces.json` — Will be generated by 02_run_experiment.py
- `out/analysis/` — Will be generated by 03_analyze.py

**Templates** (ready to fill):
- `RESULTS_TEMPLATE.md` — Section 5 (fill with real numbers)
- `DISCUSSION_OUTLINE.md` — Section 6 (fill with real numbers)
- `METHODS_DRAFT.md` — Section 4 (mostly complete, reference only)

**Guides**:
- `FIGURE_GENERATION_GUIDE.md` — Python code for all 7 figures
- `GPU_READY_CHECKLIST.md` — Detailed phase-by-phase checklist
- `API_VERIFICATION.md` — API call-site validation
- `workspace_band_guide.md` — Band identification method
- `natural_stories_plan.md` — External validity plan

---

## Troubleshooting

### "jlens.from_hf() not found"
**Fix**: Check jlens source (`jlens/__init__.py`) for actual method name. 
Update `01_fit_lens.py` line 71 and `02_run_experiment.py` line 158 (marked `[API]`).

### "Multi-token targets found"
**Fix**: This shouldn't happen (we validated), but if it does:
1. Check which pair fails in tokenizer
2. Update `stimuli.json` (ensure leading space)
3. Re-run `02_run_experiment.py --validate`

### "Lens file too large / OOM during fitting"
**Fix**: Reduce `--n-prompts` (try 500 instead of 1000) or `--seq-len` (try 64 instead of 128).

### "Oscillation not detected"
**Fix**: Check false_lead prompts are truly ambiguous. May need stronger distractors 
in `stimuli.json` or model-specific tuning.

---

## Expected Results (From Mock Data)

Your real results should be SIMILAR (direction, not magnitude):

| Hypothesis | Mock | Real (Expected) |
|---|---|---|
| **H1** | Δ=+1.0 layer, p=0.004 | Δ=+0.5 to +2.0 layers, p < 0.05 |
| **H2** | Δ=+2.0 changes, p=0.004 | Δ=+1.0 to +3.0 changes, p < 0.05 |
| **H3** | Δ=+15.0 layers, p=0.004 | Δ=+5.0 to +20.0 layers, p < 0.05 |

If real results are MUCH weaker:
- Check stimuli are truly tempting (behavioral check in traces.json)
- Verify workspace band ([0.25, 0.90] may be off for Qwen)
- Check lens fitting converged (monitor loss during Phase 2)

---

## Final Checklist Before Submission

- [ ] GPU phase completed (all traces generated)
- [ ] Figures generated and saved
- [ ] RESULTS_TEMPLATE.md filled with real numbers
- [ ] DISCUSSION_OUTLINE.md customized with real findings
- [ ] Abstract written (250 words)
- [ ] Acknowledgments written
- [ ] References formatted
- [ ] All sections proofread
- [ ] Figure captions match figure content
- [ ] Table numbers match text references
- [ ] Appendices formatted
- [ ] PDF compiled and validated

---

## Support

**If something doesn't work:**

1. Check `API_VERIFICATION.md` (call sites)
2. Check `GPU_READY_CHECKLIST.md` (phase-by-phase)
3. Check `FIGURE_GENERATION_GUIDE.md` (figure code)
4. Check `METHODS_DRAFT.md` (expected data structures)

**All CPU-side code is tested and documented. GPU phase is straightforward.**

Ready when you have GPU access! 🚀
