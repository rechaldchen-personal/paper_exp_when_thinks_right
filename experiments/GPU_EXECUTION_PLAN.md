# GPU Execution Plan — Step-by-Step Instructions

**Status**: All CPU prep complete. APIs verified. Ready to run GPU experiments.

**Total GPU Time**: ~3.5 hours (mostly lens fitting)

**Prerequisites**: 
- ✅ APIs verified (see `API_VERIFIED.md`)
- ✅ Pre-registration locked (this file: `PRE_REGISTRATION.md`)
- ✅ Stimuli validated (77 items in `stimuli.json`)
- ✅ Scripts ready to run (in root directory)

---

## Phase 1: Environment Setup (15 min, no GPU needed)

### 1.1 Clone jlens

```bash
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens
pip install -e .
cd ..
```

### 1.2 Install dependencies

```bash
pip install datasets torch transformers
```

### 1.3 Create output directories

```bash
mkdir -p out/
```

---

## Phase 2: Check for Pretrained Lenses (OPTIONAL, could save 2 hours)

**Goal**: See if a pretrained Jacobian lens exists for Qwen3-1.7B (or 4B).

If found, skip Phase 3 (lens fitting) and jump to Phase 4.

### 2.1 Search for Anthropic-published lenses

```bash
# Check Anthropic's GitHub for published lenses
# Look for: https://github.com/anthropics/...
# Or: huggingface.co/anthropics or anthropic (if they publish there)
```

### 2.2 Search HuggingFace for community lenses

```bash
# Check if solarkyle/jspace-lenses or similar exists
# If found: wget or git clone the repo
# Instructions will be in their README
```

### 2.3 Decision

**If lens found:**
```bash
# Download/copy to out/lens_qwen3_1p7b.pt
# Skip to Phase 4 (Trace Collection)
```

**If not found:**
```bash
# Proceed to Phase 3 (Lens Fitting)
# Estimated time: 2 hours
```

---

## Phase 3: Fit Lens (2 hours, GPU-required)

**Only if pretrained lens not found.**

### 3.1 Run lens fitting

```bash
python 01_fit_lens.py \
    --model Qwen/Qwen3-1.7B \
    --n-prompts 1000 \
    --seq-len 128 \
    --out out/lens_qwen3_1p7b.pt \
    --checkpoint out/fit_ckpt.pt
```

**What happens:**
- Loads Qwen3-1.7B model (requires ~13GB VRAM)
- Loads FineWeb corpus (auto-downloaded, ~1GB)
- Computes Jacobian matrices for each layer (1000 prompts × 128 tokens)
- Saves lens to `out/lens_qwen3_1p7b.pt` (~200MB)
- Checkpoint saved every prompt (resumable)

**Estimated time**: ~2 hours on single A100 (varies by GPU)

**If interrupted**: Re-run same command; will resume from checkpoint.

**Expected output**:
```
Fitting lens (dominated by the model's backward pass; resumable) ...
```
Then progress updates. When done:
```
Saved lens -> out/lens_qwen3_1p7b.pt
```

### 3.2 Verify lens file

```bash
ls -lh out/lens_qwen3_1p7b.pt
# Should be ~200MB
```

---

## Phase 4: Collect Traces (20 min, GPU-required)

**Goal**: Apply lens to all 77 stimuli and collect per-layer entropy/ranks.

### 4.1 Run trace collection

```bash
python 02_run_experiment.py \
    --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt \
    --out out/traces.json
```

**What happens:**
- Loads model + lens
- Iterates through all 77 stimuli
- Applies lens at every layer, every position
- Records:
  - Entropy at each (layer, position)
  - Target/distractor ranks
  - Top-1 token identity oscillation
  - Behavioral checks (target as top-1, distractor in top-5)
- Outputs JSON with per-stimulus records

**Expected output**:
```
Loaded model Qwen/Qwen3-1.7B
Loaded lens: JacobianLens(...)
Processing stimuli: 77 items
  ...
Saved traces -> out/traces.json (77 records)
```

**Expected file size**: ~10–50MB depending on vocab size and layer count

### 4.2 Verify traces

```bash
python3 << 'EOF'
import json
with open('out/traces.json') as f:
    traces = json.load(f)
print(f"✓ Loaded {len(traces)} stimuli")
print(f"✓ Conditions: {set(s['condition'] for s in traces)}")
print(f"✓ Families: {set(s['family'] for s in traces)}")
EOF
```

Expected output:
```
✓ Loaded 77 stimuli
✓ Conditions: {'straightforward', 'false_lead', 'hard_control'}
✓ Families: {'factual', 'arithmetic', 'garden-path'}
```

---

## Phase 5: Analysis (5 min, CPU-only)

**Goal**: Compute metrics (commitment layer, oscillation, gap) and run Wilcoxon tests.

### 5.1 Run analysis with dev/holdout split

```bash
python 03_analyze.py \
    --traces out/traces.json \
    --dev-split 0.6 \
    --out out/analysis_real/
```

**What happens:**
- Pre-splits stimuli: 60% dev, 40% holdout (prevents p-hacking)
- Computes theta (80th percentile) on dev set only
- Computes all metrics on holdout set
- Runs paired Wilcoxon signed-rank tests for H1, H2, H3
- Outputs JSON report + visualizations

**Expected output**:
```
Loading traces...
Dev/holdout split: 46 dev items, 31 holdout items
Computing metrics...
H1 (commitment layer): median_diff=... p=...
H2 (oscillation): median_diff=... p=...
H3 (dissociation gap): median_diff=... p=...
Saved -> out/analysis_real/
```

### 5.2 Examine results

```bash
cat out/analysis_real/report.json | python3 -m json.tool | head -100
```

Will show:
- Theta value
- Workspace band
- Per-hypothesis test results
- Per-stimulus metrics

---

## Phase 6: Generate Figures (30 min, CPU-only)

**Goal**: Create publication-ready figures for paper.

### 6.1 Run figure generation

```bash
python3 paper/FIGURE_GENERATION_GUIDE.md
# This is a markdown file with Python code snippets
# Copy the code sections into a new script, e.g.:
```

Create `generate_figures.py`:
```python
# Copy all Python code from paper/FIGURE_GENERATION_GUIDE.md
# into this file, then run:
```

```bash
python generate_figures.py
```

**What happens:**
- Reads analysis results from `out/analysis_real/report.json`
- Generates 7 figures:
  - H1 box plot (commitment layer by condition)
  - H2 box plot (oscillation depth by condition)
  - H3 box plot (dissociation gap by condition)
  - Heatmap of entropy progression
  - Heatmap of oscillation signature
  - Entropy collapse curve
  - Natural Stories correlation (if applicable)
- Saves as PNG/SVG to `out/figures/`

**Expected output**:
```
Generating figures...
✓ Figure 1: H1_commitment_layer.png
✓ Figure 2: H2_oscillation_depth.png
✓ Figure 3: H3_dissociation_gap.png
...
✓ All figures saved to out/figures/
```

### 6.2 Review figures

```bash
ls -lh out/figures/
# Should see 7 PNG files
```

---

## Phase 7: Results Compilation (1 hour, CPU-only)

**Goal**: Fill Results section with real numbers and write Abstract.

### 7.1 Fill Results section

Open: `paper/RESULTS_TEMPLATE.md`

For each hypothesis (H1, H2, H3):
- Copy numbers from `out/analysis_real/report.json`
- Paste into corresponding table in Results template
- Replace figure placeholders with real figure filenames
- Update statistical test results (p-values, effect sizes)

### 7.2 Write Abstract

Open: `paper/RESULTS_TEMPLATE.md` → Section "Abstract"

Template provided; fill in:
- One-sentence summary of core finding
- Why it matters (connection to global workspace)
- Main results (which hypotheses supported)

### 7.3 Compile full paper

Combine all sections in order:
1. `paper/INTRODUCTION_AND_RELATED_WORK.md` (Sections 1–2)
2. Background (Section 3, reference in outline)
3. `paper/METHODS_DRAFT.md` (Section 4)
4. `paper/RESULTS_TEMPLATE.md` (Section 5, filled with real numbers)
5. `paper/DISCUSSION_OUTLINE.md` (Section 6, filled with real findings)
6. Abstract (write after results are known)
7. References (compile)

---

## Troubleshooting

### "GPU out of memory" during lens fitting

**Solution**: Reduce batch size or max_seq_len
```bash
python 01_fit_lens.py \
    --model Qwen/Qwen3-1.7B \
    --n-prompts 1000 \
    --seq-len 64 \  # reduced from 128
    --out out/lens_qwen3_1p7b.pt
```

### "CUDA not available" when running scripts

**Solution**: Verify GPU availability
```bash
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
EOF
```

### Lens fitting is very slow

**Solution**: This is normal. Jacobian computation (gradient w.r.t. residual) is expensive.
- Single A100: ~2 hours for 1000 prompts
- Single H100: ~1 hour
- Estimated O(n_layers × n_prompts × seq_len) complexity

### Analysis shows no effect (p-values > 0.05)

**Possible causes**:
1. Stimuli aren't "tempting" enough (check distractor rank in straightforward condition)
2. Oscillation signature too subtle for this model size
3. Model answers perfectly (no error to recover from)

**Next steps**:
- Check behavioral pre-screen in traces (target_is_top1, distractor_in_top5)
- If distractor isn't elevated in straightforward, stimuli need redesign
- Consider whether garden-path effects exist for this model

---

## Summary Timeline

| Phase | Time | GPU? | Blocking? |
|-------|------|------|-----------|
| 1. Setup | 15 min | No | No |
| 2. Pretrained search | 30 min | No | No |
| 3. Lens fitting | 2h | YES | **YES** |
| 4. Trace collection | 20 min | YES | No (after phase 3) |
| 5. Analysis | 5 min | No | No |
| 6. Figures | 30 min | No | No |
| 7. Results writing | 1h | No | No |
| **TOTAL** | **~4.5h** | **2.5h GPU** | - |

---

## Success Criteria

After Phase 7, you should have:

✅ **out/lens_qwen3_1p7b.pt** (~200MB lens file)  
✅ **out/traces.json** (~10–50MB trace data)  
✅ **out/analysis_real/** (report.json + metrics)  
✅ **out/figures/** (7 PNG files)  
✅ **paper/** (all sections filled in with real numbers)  
✅ **Abstract** (written)  

**Then**: Final review and submit.

---

**Ready to start? Go to Phase 1.**

