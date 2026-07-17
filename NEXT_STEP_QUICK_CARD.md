# Next Steps — Action Card (Copy-Paste Ready)

**Date**: 2026-07-16  
**Status**: ✅ All CPU work complete. Ready for GPU.

---

## 🎯 What You Have Right Now

```
✅ Paper: 85% written (6/7 sections complete)
✅ Stimuli: 163 validated items ready to test
✅ Scripts: All prepared and tested on mock data
✅ Environment: Virtual environment ready, CPU packages installed
✅ Documentation: Comprehensive guides for every step
✅ Pre-registration: Analysis locked (no p-hacking)
✅ GPU instance: H100 SXM available to launch
```

---

## 🚀 Your Next 4 Steps (In Order)

### Step 1: Understand What You're Testing (5 minutes)

**Read these 3 files:**

```bash
1. VALIDATION_STATUS.md        # What's proven vs. what needs GPU
2. experiments/PRE_REGISTRATION.md   # Your locked analysis plan
3. START_GPU_PHASE.md          # Overview of GPU phase
```

**Key point**: You've proven the methodology works (on mock data). GPU phase will test if real Qwen3 shows the hypothesized patterns.

---

### Step 2: Launch GPU Instance (10 minutes)

**Go to RunPod and launch:**

```
Instance Type: PyTorch 2.8.0
GPU: 1x H100 SXM
Storage: 80GB (minimum)
Cost: $3.00/hour
```

**Setup once connected:**

```bash
ssh root@<your-runpod-ip>
cd /workspace
git clone <your-repo-url>  # or upload your paper_draft/ folder
cd paper_draft
source venv/bin/activate  # (if venv came with upload)
pip install torch transformers datasets  # GPU packages
```

---

### Step 3: Follow GPU_TESTING_GUIDE.md (3-4 hours)

**This guide has everything. Just follow it in order:**

```bash
# Phase 1: Setup (15 min) — CPU
git clone https://github.com/anthropics/jacobian-lens
cd jacobian-lens && pip install -e . && cd ..

# Phase 2: Fit Lens (2 hours) — GPU (just watch)
python 01_fit_lens.py --model Qwen/Qwen3-1.7B \
    --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_1p7b.pt \
    --checkpoint out/fit_ckpt.pt

# Phase 3: Collect Traces (20 min) — GPU
python 02_run_experiment.py --model Qwen/Qwen3-1.7B \
    --lens out/lens_qwen3_1p7b.pt --out out/traces.json

# Phase 4: Analyze (5 min) — CPU
python 03_analyze.py --traces out/traces.json \
    --dev-split 0.6 --out out/analysis_real/

# Phase 5: Figures (30 min) — CPU
# Use code from paper/FIGURE_GENERATION_GUIDE.md

# Phase 6: Fill Paper (1 hour) — Manual editing
# Copy numbers from out/analysis_real/report.json
# Into paper/RESULTS_TEMPLATE.md
```

---

### Step 4: Compile & Submit (1.5 hours)

**After GPU phase completes:**

```bash
# 1. Fill Results section with real numbers
# (paper/RESULTS_TEMPLATE.md has numbered placeholders)

# 2. Update Discussion with real findings
# (paper/DISCUSSION_OUTLINE.md has fill-in templates)

# 3. Generate all figures
# (Run FIGURE_GENERATION_GUIDE.md code with real data)

# 4. Compile full paper
# Read paper/INDEX.md for compilation order

# 5. Submit!
```

---

## 📊 Expected Timeline

| Phase | Time | GPU? | Cost |
|-------|------|------|------|
| Setup | 15 min | No | Free |
| Fit Lens | 2 hours | **Yes** | $6 |
| Traces | 20 min | **Yes** | $1 |
| Analysis | 5 min | No | Free |
| Figures | 30 min | No | Free |
| Fill Paper | 1 hour | No | Free |
| **TOTAL** | **~4.5 hours** | **~2.3h** | **~$7** |

---

## ✅ What Success Looks Like

### Best Case (All 3 hypotheses supported)
```
H1: p = 0.0089 ✅ Commitment layer delayed under false lead
H2: p = 0.0034 ✅ Oscillation present under false lead
H3: p = 0.0156 ✅ Dissociation gap larger under false lead
```
→ **Strong paper!** All mechanisms confirmed.

### Good Case (2 out of 3)
```
H1: p = 0.0034 ✅ 
H2: p = 0.1234 ❌ 
H3: p = 0.0089 ✅
```
→ **Still publishable.** "We find evidence for H1 and H3 but not H2."

### Interesting Case (1 or 0)
```
H1: p = 0.0089 ✅ 
H2: p = 0.5123 ❌
H3: p = 0.6789 ❌
```
→ **Valid negative result.** Shows what models don't do (still publishable).

**Any outcome is publishable if methodology is sound.** Your methodology IS sound.

---

## 🎯 Critical Reminders

✅ **You've already done the hard work:**
- Designed 163 stimuli (tested for ambiguity, difficulty, tokenization)
- Locked pre-registration (prevents p-hacking)
- Validated pipeline on mock data (proves it detects patterns)
- Prepared all code (all APIs verified)

❌ **Don't do these things:**
- Don't change the analysis plan (it's locked in PRE_REGISTRATION.md)
- Don't cherry-pick results (use dev/holdout split as specified)
- Don't claim patterns that don't exist in real data

✅ **Do these things:**
- Follow GPU_TESTING_GUIDE.md exactly
- Trust your pre-registration
- Report whatever results you get (positive, negative, mixed)
- Use exactly 60/40 dev/holdout split specified

---

## 📞 Troubleshooting

**"Phase 2 (lens fitting) seems stuck"**
→ Normal! It takes ~2 hours. Each prompt takes ~7-10 seconds. Check GPU utilization: `nvidia-smi`

**"I got p > 0.05 for a hypothesis"**
→ This is valid data! Publish it. Null results are publishable.

**"The GPU ran out of memory"**
→ Try reducing `--n-prompts` from 1000 to 500 (still valid).

**"Something failed in Phase 2/3"**
→ Restart from the last checkpoint. All scripts save checkpoints.

---

## 🚀 Ready?

1. ✅ Read START_GPU_PHASE.md (this tells you more)
2. ✅ Launch H100 SXM instance on RunPod
3. ✅ Follow GPU_TESTING_GUIDE.md Phase 1
4. ✅ Run Phases 2-6 in order
5. ✅ Fill paper with real results
6. ✅ Submit!

**You've got this!** 🎓

---

**Questions?** Check:
- START_GPU_PHASE.md (overview)
- GPU_TESTING_GUIDE.md (detailed walkthrough)
- VALIDATION_STATUS.md (what we know for sure)
- paper/INDEX.md (paper navigation)
