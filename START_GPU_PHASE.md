# START HERE: GPU Phase Walkthrough

**You are here**: Ready to run GPU experiments

**What you have**: 
- ✅ Paper 85% written (needs GPU results)
- ✅ 163 stimuli validated and ready
- ✅ Scripts ready (01, 02, 03)
- ✅ Analysis plan pre-registered
- ✅ Mock validation complete

**What you need**: 
- GPU access (2-3 hours)
- ~50GB disk space
- Patience for lens fitting (takes ~2 hours)

---

## BEFORE YOU START

**Read these 3 files** (in order):

1. `VALIDATION_STATUS.md` (5 min)
   - Understand what you're testing
   - Know the difference between synthetic validation and real testing
   - Get clear on what hypotheses actually mean

2. `experiments/PRE_REGISTRATION.md` (10 min)
   - Locked analysis plan (prevents p-hacking)
   - Know your hypotheses exactly
   - Understand dev/holdout split

3. `GPU_TESTING_GUIDE.md` (this will be your step-by-step)
   - Detailed instructions for each phase
   - Expected outputs at each step
   - Troubleshooting guide

---

## THE 6-PHASE PROCESS (Overview)

### Phase 1: Setup (15 min)
- Clone jlens repository
- Install dependencies
- Create output folders
- Verify scripts exist

**Time**: 15 minutes  
**GPU needed**: No

### Phase 2: Lens Fitting (2 hours) ⭐ LONGEST STEP
- Fit Jacobian matrices on 1000 prompts from FineWeb
- GPU will be busy computing gradients
- Checkpointed (resumable if interrupted)
- Output: `out/lens_qwen3_1p7b.pt` (~200MB)

**Time**: 1.5-2.5 hours  
**GPU needed**: Yes (A100: 2h, H100: 1h)

### Phase 3: Trace Collection (20 min)
- Apply lens to all 163 stimuli
- For each stimulus: compute entropy, ranks, oscillations
- Compute random-direction baseline
- Output: `out/traces.json` (~45MB)

**Time**: 20-30 minutes  
**GPU needed**: Yes (forward passes only, fast)

### Phase 4: Analysis (5 min) 
- Split traces into dev/holdout
- Compute theta on dev, test on holdout
- Run Wilcoxon tests for H1, H2, H3
- Output: `out/analysis_real/report.json`

**Time**: 5 minutes  
**GPU needed**: No (CPU only)

### Phase 5: Figures (30 min)
- Generate 3 box plots (H1, H2, H3)
- Generate heatmaps
- Publication-ready PNGs

**Time**: 30 minutes  
**GPU needed**: No (CPU only)

### Phase 6: Fill Paper (1 hour)
- Copy numbers from analysis report
- Fill `paper/RESULTS_TEMPLATE.md`
- Update Abstract with real conclusions
- Compile full paper

**Time**: 1 hour  
**GPU needed**: No (editing only)

**TOTAL TIME**: ~4.5 hours (mostly Phase 2 waiting)

---

## WHAT HAPPENS AT EACH PHASE

### ❓ What should I see?

**Phase 2 (Fitting)** — Takes forever but this is normal:
```
Progress: 1/1000, Loss: 0.523, Elapsed: 0:00:05
Progress: 2/1000, Loss: 0.521, Elapsed: 0:00:10
Progress: 3/1000, Loss: 0.519, Elapsed: 0:00:15
... (many more lines)
Progress: 1000/1000, Loss: 0.401, Elapsed: 2:15:30
Saved lens -> out/lens_qwen3_1p7b.pt
```
**This is expected.** Each prompt takes ~8-10 seconds.

**Phase 3 (Traces)** — Fast and chatty:
```
Loaded model Qwen/Qwen3-1.7B
Loaded lens: JacobianLens(...)
Processing stimuli: 163 items
  [1/163] capital_fr (straightforward)
  [2/163] capital_fr (false_lead)
  [3/163] legs_spider (straightforward)
  ...
Saved traces -> out/traces.json (163 records)
```
**This is expected.** Takes 15-30 min, shows one line per stimulus.

**Phase 4 (Analysis)** — Short and informative:
```
Loading traces...
Dev/holdout split: 98 dev, 65 holdout
Computing metrics...
H1: p = 0.0234 ✅ SUPPORTED
H2: p = 0.1523 ❌ NOT SIGNIFICANT  
H3: p = 0.0089 ✅ SUPPORTED
Saved to out/analysis_real/
```
**This is expected.** Shows your p-values!

---

## CRITICAL: What do the results mean?

### Best Case (p < 0.05 for all three)
```
H1: p = 0.0012 ✅ Commitment layer delayed
H2: p = 0.0089 ✅ Oscillation present
H3: p = 0.0034 ✅ Dissociation gap present
```
→ **Your paper is strong!** All hypotheses supported.

### Good Case (2 out of 3)
```
H1: p = 0.0034 ✅ 
H2: p = 0.1234 ❌ 
H3: p = 0.0156 ✅
```
→ **Still publishable.** "We find evidence for H1 and H3, but H2 is weaker."

### Challenging Case (1 out of 3)
```
H1: p = 0.0089 ✅
H2: p = 0.5123 ❌
H3: p = 0.6789 ❌
```
→ **Still valid data.** "We find limited evidence for false-lead effects."

### Difficult Case (none significant)
```
H1: p = 0.2345 ❌
H2: p = 0.6789 ❌
H3: p = 0.7234 ❌
```
→ **Negative result.** "We do not find evidence for false-lead effects in Qwen3-1.7B."
This is still publishable! Shows what the model doesn't do.

---

## STEP-BY-STEP: How to Actually Run It

### The Simple Version (Just Do This)

```bash
# Phase 1: Setup (15 min)
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens && pip install -e . && cd ..
mkdir -p out/

# Phase 2: Fit lens (2 hours — just let it run)
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 \
    --seq-len 128 --out out/lens_qwen3_1p7b.pt \
    --checkpoint out/fit_ckpt.pt

# Phase 3: Collect traces (20 min)
python 02_run_experiment.py --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt --out out/traces.json

# Phase 4: Analyze (5 min)
python 03_analyze.py --traces out/traces.json \
    --dev-split 0.6 --out out/analysis_real/

# Phase 5: Figures (30 min)
python generate_figures.py  # Uses code from GPU_TESTING_GUIDE.md

# Phase 6: Fill paper (1 hour)
# Manually edit paper/RESULTS_TEMPLATE.md with numbers
# Update paper/ABSTRACT.md with conclusions
```

**That's it!** Then you submit.

---

## DETAILED VERSION: See GPU_TESTING_GUIDE.md

For step-by-step details including:
- What to check at each phase
- Expected output for each command
- Troubleshooting each phase
- How to interpret your results
- How to fill the paper with numbers

**Read**: `GPU_TESTING_GUIDE.md`

---

## VALIDATION REMINDER

✅ **What we know for sure:**
- Stimuli are well-designed (163 items validated)
- Methodology works (tested on synthetic data)
- Scripts are ready (APIs verified)
- Analysis is pre-registered (prevents p-hacking)

⏳ **What we'll find out:**
- Do real Qwen3 traces show H1/H2/H3? (Yes/No/Maybe)
- How strong are the effects? (p-values, effect sizes)
- Which hypotheses hold up? (all/some/none)

❌ **What we cannot claim yet:**
- "Models definitely show false-lead effects" (need GPU data)
- "Oscillation proves internal backtracking" (need GPU data)
- Results generalize to other models (only testing Qwen3)

**After GPU phase**, we can make all these claims based on real data.

---

## Key Files Reference

**Before you start:**
- Read: `VALIDATION_STATUS.md`
- Read: `experiments/PRE_REGISTRATION.md`

**During GPU testing:**
- Follow: `GPU_TESTING_GUIDE.md`
- Reference: `experiments/GPU_EXECUTION_PLAN.md`

**After GPU testing:**
- Fill: `paper/RESULTS_TEMPLATE.md`
- Update: `paper/ABSTRACT.md`
- Compile: All sections in `paper/INDEX.md`

---

## The One-Sentence Summary

**"You have 6 phases to run: setup → fit lens (2h) → collect traces (20m) → analyze (5m) → make figures (30m) → fill paper (1h). Then submit!"**

---

## Ready? 

**Yes**: Go to `GPU_TESTING_GUIDE.md` and start Phase 1

**Questions**: Check `VALIDATION_STATUS.md` or `experiments/PRE_REGISTRATION.md`

**Not sure what something means**: Check this file again or the detailed guide

---

**Good luck!** Your paper is in great shape. GPU phase will just confirm what the theory predicts. 🚀

