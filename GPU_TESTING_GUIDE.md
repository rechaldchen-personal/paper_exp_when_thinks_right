# GPU Testing Guide: Step-by-Step Instructions

**Goal**: Run GPU experiments and collect real traces for hypothesis testing.

**Total Time**: ~3.5-4 hours (mostly waiting for lens fitting)

**Important**: Follow these steps IN ORDER. Each step depends on the previous one.

---

## BEFORE YOU START

### ✅ Checklist (Do these before touching GPU)

- [ ] Read `VALIDATION_STATUS.md` (understand what we're testing)
- [ ] Read `experiments/PRE_REGISTRATION.md` (understand the locked analysis plan)
- [ ] Verify you have GPU access and can run CUDA code
- [ ] Have 50GB+ free disk space (for lens fitting corpus)
- [ ] Ensure you have ~2-3 hours uninterrupted time for lens fitting

### Environment Requirements

```bash
# Check your GPU
nvidia-smi  # Should show CUDA device

# Check Python version (need 3.8+)
python --version

# Check if you have transformers and torch
pip list | grep -E "torch|transformers"
```

If anything is missing, install:
```bash
pip install torch transformers datasets scipy numpy matplotlib
```

---

## PHASE 1: Environment Setup (15 minutes)

### Step 1.1: Clone jlens Repository

```bash
cd /path/to/your/workspace  # wherever your paper is
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens
```

**Expected output:**
```
Cloning into 'jacobian-lens'...
remote: Enumerating objects...
...
Unpacking objects: 100%
```

### Step 1.2: Install jlens

```bash
cd jacobian-lens
pip install -e .
cd ..
```

**Expected output:**
```
Successfully installed jacobian-lens
```

### Step 1.3: Verify Installation

```bash
python -c "import jlens; print(jlens.__file__)"
```

**Expected output:**
```
/path/to/jacobian-lens/jlens/__init__.py
```

### Step 1.4: Create Output Directories

```bash
mkdir -p out/
```

### Step 1.5: Verify Scripts Exist

```bash
ls -lh 01_fit_lens.py 02_run_experiment.py 03_analyze.py
```

**Expected output:** Should list all 3 files with sizes > 1KB

---

## PHASE 2: Lens Fitting (2 hours, GPU-required)

⚠️ **This is the long step.** The GPU will be busy. This is normal.

### Step 2.1: Start Lens Fitting

```bash
python 01_fit_lens.py \
    --model Qwen/Qwen3-1.7B \
    --n-prompts 1000 \
    --seq-len 128 \
    --out out/lens_qwen3_1p7b.pt \
    --checkpoint out/fit_ckpt.pt
```

**What happens:**
- Downloads Qwen3-1.7B model (~3GB)
- Downloads FineWeb corpus (~1GB, auto-cached)
- Starts fitting Jacobian matrices
- Checkpoints saved every prompt
- Progress printed every ~10 prompts

**Expected output (first few lines):**
```
Loading Qwen/Qwen3-1.7B ...
Loading tokenizer ...
Building fitting corpus (1000 x 128 tokens) ...
  got 1000 sequences
Fitting lens (dominated by the model's backward pass; resumable) ...
Progress: 1/1000, Loss: X.XXX, ...
Progress: 2/1000, Loss: X.XXX, ...
...
```

**Expected duration**: 1.5-2.5 hours depending on GPU

**If interrupted**: Just re-run the same command. It will resume from checkpoint.

### Step 2.2: Verify Lens File Created

After fitting completes:

```bash
ls -lh out/lens_qwen3_1p7b.pt
```

**Expected output:**
```
-rw-r--r--  1 user  group  200M  Jul 16 12:34  out/lens_qwen3_1p7b.pt
```

Size should be ~200-250 MB.

### Step 2.3: Quick Lens Check

```bash
python -c "
import torch
lens_data = torch.load('out/lens_qwen3_1p7b.pt', map_location='cpu', weights_only=True)
print(f\"Lens keys: {lens_data.keys()}\")
print(f\"n_prompts: {lens_data['n_prompts']}\")
print(f\"d_model: {lens_data['d_model']}\")
print(f\"n_layers: {len(lens_data['J'])}\")
"
```

**Expected output:**
```
Lens keys: dict_keys(['J', 'n_prompts', 'd_model', 'source_layers'])
n_prompts: 1000
d_model: 1536
n_layers: 24
```

✅ **Phase 2 Complete!** You have a fitted lens.

---

## PHASE 3: Trace Collection (20-30 minutes, GPU-required)

This step applies the lens to all 163 stimuli.

### Step 3.1: Start Trace Collection

```bash
python 02_run_experiment.py \
    --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt \
    --out out/traces.json \
    --baseline-samples 100
```

**What happens:**
- Loads Qwen3 model (reuses from Phase 2)
- Loads fitted lens
- Iterates through 163 stimuli
- For each stimulus:
  - Tokenizes prompt
  - Runs forward pass (records residuals)
  - Applies lens at each layer
  - Computes entropy, ranks, oscillations
  - Tests random-direction baseline
- Saves results to JSON

**Expected output:**
```
Loaded model Qwen/Qwen3-1.7B (reusing from Phase 2)
Loaded lens: JacobianLens(d_model=1536, n_prompts=1000, source_layers=[0..23] (24 layers))
Processing stimuli: 163 items
  [1/163] capital_fr (straightforward) ...
  [2/163] capital_fr (false_lead) ...
  [3/163] legs_spider (straightforward) ...
  ...
  [163/163] gp_door (false_lead) ...
Saved traces -> out/traces.json (163 records)
```

**Expected duration**: 15-30 minutes

### Step 3.2: Verify Traces File

```bash
ls -lh out/traces.json
```

**Expected output:**
```
-rw-r--r--  1 user  group  45M  Jul 16 13:52  out/traces.json
```

Size should be 30-60 MB.

### Step 3.3: Check Traces Structure

```bash
python -c "
import json
with open('out/traces.json') as f:
    traces = json.load(f)
print(f'Loaded {len(traces)} stimuli')
print(f'First stimulus keys: {list(traces[0].keys())}')
print(f'Conditions: {set(t[\"condition\"] for t in traces)}')
print(f'Families: {set(t[\"family\"] for t in traces)}')
"
```

**Expected output:**
```
Loaded 163 stimuli
First stimulus keys: ['pair_id', 'condition', 'family', 'prompt', 'target', 'distractor', 'layers', 'n_positions', 'entropy', 'baseline_entropy', 'target_rank', 'distractor_rank', 'top1_id', 'behavioral', ...]
Conditions: {'straightforward', 'false_lead', 'hard_control'}
Families: {'arithmetic', 'factual', 'garden_path'}
```

✅ **Phase 3 Complete!** You have real traces from Qwen3.

---

## PHASE 4: Analysis (5 minutes, CPU-only)

Now run the analysis pipeline on the real traces.

### Step 4.1: Run Analysis

```bash
python 03_analyze.py \
    --traces out/traces.json \
    --dev-split 0.6 \
    --out out/analysis_real/
```

**What happens:**
- Splits 163 stimuli into dev (60%) and holdout (40%)
- Computes theta on dev set only (prevents p-hacking)
- Computes all metrics on holdout set
- Runs Wilcoxon signed-rank tests for H1, H2, H3
- Generates heatmaps and summary statistics
- Saves everything to out/analysis_real/

**Expected output:**
```
Loading traces from out/traces.json ...
Loaded 163 stimuli
Dev/holdout split: 98 dev items, 65 holdout items
Computing metrics on holdout set...

Computing H1 (commitment layer) ...
  Pairs with both conditions: 22
  Median diff (false_lead - straightforward): X.X layers
  Wilcoxon W-stat: XXX, p-value: 0.XXXX
  
Computing H2 (oscillation depth) ...
  Median diff: X.X changes
  Wilcoxon W-stat: XXX, p-value: 0.XXXX
  
Computing H3 (dissociation gap) ...
  Median diff: X.X layers
  Wilcoxon W-stat: XXX, p-value: 0.XXXX

Saved results to out/analysis_real/
```

### Step 4.2: Check Analysis Output

```bash
ls -la out/analysis_real/
```

**Expected files:**
```
-rw-r--r--  1 user  group  5.0K  Jul 16 14:05  report.json
-rw-r--r--  1 user  group  2.0M  Jul 16 14:05  per_stimulus_metrics.json
drwxr-xr-x  3 user  group  4.0K  Jul 16 14:05  heatmaps/
```

### Step 4.3: Read the Report

```bash
cat out/analysis_real/report.json | python -m json.tool | head -50
```

This will show you:
- Theta value (80th percentile threshold)
- Band used ([0.25, 0.90])
- H1 results (p-value, median difference, effect size)
- H2 results (oscillation)
- H3 results (gap)

### Step 4.4: Interpret Results

**CRITICAL**: Check what your p-values are:

```python
python << 'EOF'
import json

with open('out/analysis_real/report.json') as f:
    report = json.load(f)

print("HYPOTHESIS TEST RESULTS")
print("=" * 50)
print(f"H1 (Commitment layer): p = {report['H1']['p_value']:.6f}")
print(f"H2 (Oscillation): p = {report['H2']['p_value']:.6f}")
print(f"H3 (Dissociation gap): p = {report['H3']['p_value']:.6f}")
print()
print("INTERPRETATION:")
if report['H1']['p_value'] < 0.05:
    print("✅ H1: SUPPORTED (commitment layer delayed under false-lead)")
else:
    print("❌ H1: NOT SIGNIFICANT (p >= 0.05)")

if report['H2']['p_value'] < 0.05:
    print("✅ H2: SUPPORTED (more oscillation under false-lead)")
else:
    print("❌ H2: NOT SIGNIFICANT (p >= 0.05)")

if report['H3']['p_value'] < 0.05:
    print("✅ H3: SUPPORTED (larger dissociation gap under false-lead)")
else:
    print("❌ H3: NOT SIGNIFICANT (p >= 0.05)")
EOF
```

✅ **Phase 4 Complete!** You now have real results.

---

## PHASE 5: Figure Generation (30 minutes, CPU-only)

Generate publication-ready figures.

### Step 5.1: Create Figure Script

Create a new file `generate_figures.py`:

```bash
cat > generate_figures.py << 'EOF'
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load analysis results
with open('out/analysis_real/report.json') as f:
    report = json.load(f)

with open('out/analysis_real/per_stimulus_metrics.json') as f:
    metrics = json.load(f)

# Create output directory
Path('out/figures').mkdir(exist_ok=True)

# Figure 1: H1 Box Plot (Commitment Layer)
fig, ax = plt.subplots(figsize=(8, 6))
commitment_sf = [m['l_star']['straightforward'] for m in metrics if m['l_star']['straightforward'] is not None]
commitment_fl = [m['l_star']['false_lead'] for m in metrics if m['l_star']['false_lead'] is not None]

ax.boxplot([commitment_sf, commitment_fl], labels=['Straightforward', 'False-lead'])
ax.set_ylabel('Commitment Layer (ℓ*)')
ax.set_title('H1: Commitment Layer by Condition')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('out/figures/H1_commitment_layer.png', dpi=300, bbox_inches='tight')
print("✅ Saved: H1_commitment_layer.png")
plt.close()

# Figure 2: H2 Box Plot (Oscillation)
fig, ax = plt.subplots(figsize=(8, 6))
osc_sf = [m['oscillation']['straightforward'] or 0 for m in metrics]
osc_fl = [m['oscillation']['false_lead'] or 0 for m in metrics]

ax.boxplot([osc_sf, osc_fl], labels=['Straightforward', 'False-lead'])
ax.set_ylabel('Oscillation Depth (# changes)')
ax.set_title('H2: Oscillation Depth by Condition')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('out/figures/H2_oscillation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: H2_oscillation.png")
plt.close()

# Figure 3: H3 Box Plot (Dissociation Gap)
fig, ax = plt.subplots(figsize=(8, 6))
gap_sf = [m['gap']['straightforward'] or 0 for m in metrics]
gap_fl = [m['gap']['false_lead'] or 0 for m in metrics]

ax.boxplot([gap_sf, gap_fl], labels=['Straightforward', 'False-lead'])
ax.set_ylabel('Dissociation Gap (ℓ* − ℓ_H)')
ax.set_title('H3: Dissociation Gap by Condition')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('out/figures/H3_dissociation_gap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: H3_dissociation_gap.png")
plt.close()

print("\n✅ All figures generated!")
print("Location: out/figures/")
EOF
```

### Step 5.2: Run Figure Generation

```bash
python generate_figures.py
```

**Expected output:**
```
✅ Saved: H1_commitment_layer.png
✅ Saved: H2_oscillation.png
✅ Saved: H3_dissociation_gap.png

✅ All figures generated!
Location: out/figures/
```

### Step 5.3: Verify Figures

```bash
ls -lh out/figures/
open out/figures/H1_commitment_layer.png  # macOS
# or: xdg-open out/figures/H1_commitment_layer.png  # Linux
# or: start out/figures/H1_commitment_layer.png  # Windows
```

Check that:
- ✅ Figures are readable
- ✅ Box plots show the expected patterns
- ✅ False-lead boxes are higher/wider than straightforward (if supported)

✅ **Phase 5 Complete!** You have publication-ready figures.

---

## PHASE 6: Fill Paper with Results (1 hour, CPU-only)

Now fill the paper with your real numbers.

### Step 6.1: Open Results Template

```bash
cat out/analysis_real/report.json | python -m json.tool > /tmp/results.txt
# Read this file to see all your numbers
cat /tmp/results.txt
```

### Step 6.2: Edit RESULTS_TEMPLATE.md

Open `paper/RESULTS_TEMPLATE.md` and fill in:

1. **Section 5.1** (Sanity Check):
   - Copy band information from report.json
   
2. **Section 5.2** (H1 Results):
   - Replace `?` with your actual numbers from report.json
   - `Median ℓ* (straightforward)`: [copy from report]
   - `Median ℓ* (false_lead)`: [copy from report]
   - `Wilcoxon p-value`: [copy from report]
   
3. **Section 5.3** (H2 Results):
   - Same process for oscillation metrics
   
4. **Section 5.4** (H3 Results):
   - Same process for dissociation gap

### Step 6.3: Write Abstract Conclusion

Update the last paragraph of `paper/ABSTRACT.md` to reflect your real results:

**If H1, H2, H3 all supported (p < 0.05):**
```
Using the Jacobian lens on Qwen3-1.7B, we find strong evidence for all three 
mechanisms. Models show delayed commitment, internal oscillation, and dissociation 
gaps under false-lead conditions, consistent with global workspace theory.
```

**If only some supported:**
```
Using the Jacobian lens on Qwen3-1.7B, we find evidence for H1 (delayed commitment) 
and H3 (dissociation gap), but H2 (oscillation) shows weaker effects. This suggests 
that while models revise their answers, the oscillation signature is less pronounced 
than expected.
```

**If none supported:**
```
Our initial hypotheses about false-lead effects were not supported by the real 
model data. However, this null result provides important evidence about the limits 
of false-lead effects in LLMs and guides future work.
```

### Step 6.4: Compile Full Paper

Combine all sections in order:
1. `paper/ABSTRACT.md`
2. `paper/INTRODUCTION_AND_RELATED_WORK.md`
3. `paper/BACKGROUND.md`
4. `paper/METHODS_DRAFT.md`
5. `paper/RESULTS_TEMPLATE.md` (filled)
6. `paper/DISCUSSION_OUTLINE.md`
7. `paper/REFERENCES.md`

You can do this manually or with a script:

```bash
cat paper/ABSTRACT.md paper/INTRODUCTION_AND_RELATED_WORK.md \
    paper/BACKGROUND.md paper/METHODS_DRAFT.md \
    paper/RESULTS_TEMPLATE.md paper/DISCUSSION_OUTLINE.md \
    paper/REFERENCES.md > paper_final.txt
```

✅ **Phase 6 Complete!** Your paper is filled with results.

---

## TROUBLESHOOTING

### Problem: GPU out of memory during Phase 2

**Solution**: Reduce batch size in 01_fit_lens.py:
```bash
python 01_fit_lens.py \
    --model Qwen/Qwen3-1.7B \
    --n-prompts 500 \  # Reduced from 1000
    --seq-len 64 \      # Reduced from 128
    --out out/lens_qwen3_1p7b.pt
```

Time will be ~1 hour instead of 2, quality slightly lower but usually sufficient.

### Problem: Lens fitting seems stuck

**Solution**: It's probably running backward passes (slow). Wait or check:
```bash
# In another terminal
nvidia-smi  # Should show high GPU usage
tail -f out/fit_ckpt.pt  # Check if file is growing
```

If really stuck after 3+ hours, you can interrupt and resume:
```bash
# Ctrl+C to stop
python 01_fit_lens.py ...  # Re-run, will resume from checkpoint
```

### Problem: Traces.json is empty or has errors

**Check the trace collection output**:
```bash
python -c "
import json
with open('out/traces.json') as f:
    traces = json.load(f)
if len(traces) < 163:
    print(f'Warning: Only {len(traces)} traces (expected 163)')
print('Behavioral check:')
for t in traces[:3]:
    print(f\"  {t['pair_id']}: target_is_top1={t['behavioral'].get('target_is_top1')}, 
           distractor_in_top5={t['behavioral'].get('distractor_in_top5')}\")
"
```

If behavioral checks show model is not answering correctly, your stimuli might need revision.

### Problem: All p-values are > 0.05

**This is still valid data!** The paper becomes:
- "We propose false-lead effects BUT find weak/no evidence"
- Still publishable as a "negative result"
- Important for understanding LLM limitations

---

## FINAL CHECKLIST

After all phases complete:

- [ ] Phase 1: Environment setup complete
- [ ] Phase 2: Lens fitting complete (`out/lens_qwen3_1p7b.pt` exists)
- [ ] Phase 3: Traces collected (`out/traces.json` has 163 stimuli)
- [ ] Phase 4: Analysis complete (`out/analysis_real/report.json` exists)
- [ ] Phase 5: Figures generated (`out/figures/` has 3 PNGs)
- [ ] Phase 6: Paper filled with results
- [ ] Abstract updated with real conclusions
- [ ] All numerical results double-checked

**Then**: You can submit your paper! 🎉

---

## Quick Command Reference

```bash
# Phase 1: Setup
git clone https://github.com/anthropics/jacobian-lens && cd jacobian-lens && pip install -e . && cd ..
mkdir -p out/

# Phase 2: Lens fitting (2 hours)
python 01_fit_lens.py --model Qwen/Qwen3-1.7B --n-prompts 1000 --seq-len 128 --out out/lens_qwen3_1p7b.pt --checkpoint out/fit_ckpt.pt

# Phase 3: Traces (20 min)
python 02_run_experiment.py --model Qwen/Qwen3-1.7B --lens out/lens_qwen3_1p7b.pt --out out/traces.json

# Phase 4: Analysis (5 min)
python 03_analyze.py --traces out/traces.json --dev-split 0.6 --out out/analysis_real/

# Phase 5: Figures (30 min)
python generate_figures.py

# Phase 6: Fill paper
# Edit paper/RESULTS_TEMPLATE.md with numbers from out/analysis_real/report.json
```

---

**Questions?** Refer back to this guide or check VALIDATION_STATUS.md for what your results mean.

**Ready to test?** Make sure you have GPU access, then start Phase 1! 🚀

