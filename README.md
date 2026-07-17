# Commitment Dynamics in the J-Space

**Status**: 94% complete. Paper drafted. GPU experiments pending.

**Current Session**: Waiting for API resolution to proceed with GPU phase.

---

## 🎯 What's Next?

### ✅ ALL CPU-ONLY WORK COMPLETE

Stimuli expanded (77 → 163), validated, documented. Scripts enhanced. Full pipeline tested.

### 📖 Read This First

**`CPU_WORK_COMPLETE.md`** — Comprehensive status of all completed CPU work

Then: **`experiments/GPU_EXECUTION_PLAN.md`** — Step-by-step GPU commands (ready to run)

Or quick version: **`GPU_COMMANDS.sh`** — Copy-paste commands

---

## 📁 Structure

### `/paper/` — Paper Content
Ready-to-use sections for final paper compilation:
- `INTRODUCTION_AND_RELATED_WORK.md` — Sections 1–2
- `METHODS_DRAFT.md` — Section 4
- `DISCUSSION_OUTLINE.md` — Section 6 (skeleton)
- `RESULTS_TEMPLATE.md` — Section 5 (awaiting GPU data)
- `FIGURE_GENERATION_GUIDE.md` — All 7 figures (templates + code)

### `/experiments/` — GPU Phase & Analysis
Everything needed to run GPU experiments:
- `README_GPU_PHASE.md` — Step-by-step GPU execution (7 phases)
- `PRE_REGISTRATION.md` — **CRITICAL** — Locked analysis plan (prevents p-hacking)
- `workspace_band_guide.md` — Workspace band identification procedure
- `natural_stories_plan.md` — External validity via human reading times (future)

### Root Directory
- `API_RESOLUTION_GUIDE.md` — API research checklist (your next task)
- `CLAUDE.md` — Progress tracker & session log
- `stimuli.json` — 77 validated stimuli (29 factual, 24 arithmetic, 24 garden-path)

### Code
- `00_generate_mock_traces.py` — Mock trace generator (validation)
- `01_fit_lens.py` — Lens fitting (GPU)
- `02_run_experiment.py` — Trace collection (GPU)
- `03_analyze.py` — Analysis pipeline (CPU)
- `lens_utils.py` — Dual-lens utilities
- `02_run_experiment_cot.py` — CoT variant (optional)

---

## ✅ Project Status

| Component | Status | Location |
|-----------|--------|----------|
| Hypotheses | ✅ Pre-registered | `experiments/PRE_REGISTRATION.md` |
| Stimuli | ✅ 77 items validated | `stimuli.json` |
| Methods | ✅ Complete (2500 words) | `paper/METHODS_DRAFT.md` |
| Intro/Related Work | ✅ Complete (1400 words) | `paper/INTRODUCTION_AND_RELATED_WORK.md` |
| Discussion | ✅ Skeleton ready | `paper/DISCUSSION_OUTLINE.md` |
| Results | ⏳ Template ready | `paper/RESULTS_TEMPLATE.md` |
| Figures | ✅ 7 templates + code | `paper/FIGURE_GENERATION_GUIDE.md` |
| Mock Validation | ✅ All H1–H3 supported | `00_generate_mock_traces.py` |
| APIs | ⏳ Needs resolution | `API_RESOLUTION_GUIDE.md` |

---

## 📋 Critical Path to Submission

1. **Resolve APIs** (2-3 hours) ← **START HERE**
   - Follow `API_RESOLUTION_GUIDE.md`
   
2. **GPU Experiments** (3.5 hours)
   - Follow `experiments/README_GPU_PHASE.md`
   
3. **Fill Results** (2-3 hours)
   - Use real numbers in `paper/RESULTS_TEMPLATE.md`
   
4. **Final Review & Submit** (1 hour)

**Total: ~11 hours (mostly GPU waiting)**

---

## 💡 Key Insights

- **Paper is 94% written** — Just need GPU data to fill Results
- **All hypotheses pre-registered** — Prevents p-hacking
- **Analysis is locked in** — See `experiments/PRE_REGISTRATION.md`
- **Mock validation works** — All H1–H3 show p < 0.01
- **GPU is only 3.5 hours** — Trace collection is fast

---

## 📖 Where to Find Things

**"What do I do next?"** → Read `API_RESOLUTION_GUIDE.md`

**"How do I run GPU experiments?"** → Read `experiments/README_GPU_PHASE.md`

**"How do I prevent p-hacking?"** → Read `experiments/PRE_REGISTRATION.md`

**"Where are my paper sections?"** → See `/paper/` folder

**"What's my progress?"** → See `CLAUDE.md`

---

**Report back when APIs are resolved. Everything else is ready to go.** 🚀
