#!/bin/bash
# GPU Experiment Commands — Copy & Paste in Order
# All prep work complete. Just run these commands sequentially.

set -e  # Exit on error

echo "=== PHASE 1: Setup ==="
mkdir -p out/
echo "✓ Output directory ready"

echo ""
echo "=== PHASE 2: (Optional) Check for pretrained lens ==="
echo "See: experiments/GPU_EXECUTION_PLAN.md Phase 2"
echo "If pretrained lens found, skip Phase 3 and jump to Phase 3 below"
echo "If not found, proceed to Phase 3"

echo ""
echo "=== PHASE 3: Lens Fitting (2 hours) ==="
echo "COMMAND:"
echo "python 01_fit_lens.py \\"
echo "    --model Qwen/Qwen3-1.7B \\"
echo "    --n-prompts 1000 \\"
echo "    --seq-len 128 \\"
echo "    --out out/lens_qwen3_1p7b.pt \\"
echo "    --checkpoint out/fit_ckpt.pt"
echo ""
echo "When done, file should be at: out/lens_qwen3_1p7b.pt (~200MB)"
read -p "Press ENTER to continue after Phase 3 is done..."

echo ""
echo "=== PHASE 4: Trace Collection (20 min) ==="
echo "COMMAND:"
echo "python 02_run_experiment.py \\"
echo "    --model Qwen/Qwen3-1.7B \\"
echo "    --lens out/lens_qwen3_1p7b.pt \\"
echo "    --out out/traces.json"
echo ""
echo "When done, file should be at: out/traces.json"
read -p "Press ENTER to continue after Phase 4 is done..."

echo ""
echo "=== PHASE 5: Analysis (5 min, CPU) ==="
python 03_analyze.py \
    --traces out/traces.json \
    --dev-split 0.6 \
    --out out/analysis_real/
echo "✓ Analysis complete"
echo "Results saved to: out/analysis_real/report.json"

echo ""
echo "=== PHASE 6: Figures (30 min, CPU) ==="
echo "See: paper/FIGURE_GENERATION_GUIDE.md"
echo "Copy the Python code into a script and run it"
echo "Figures will be saved to: out/figures/"
read -p "Press ENTER after generating figures..."

echo ""
echo "=== PHASE 7: Results Writing (1 hour, CPU) ==="
echo "1. Open: paper/RESULTS_TEMPLATE.md"
echo "2. Copy numbers from: out/analysis_real/report.json"
echo "3. Fill in Results section"
echo "4. Write Abstract"
echo "5. Compile full paper"
read -p "Press ENTER when done..."

echo ""
echo "✅ ALL GPU EXPERIMENTS COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Check out/figures/ for 7 PNG files"
echo "2. Check out/analysis_real/report.json for numerical results"
echo "3. Fill paper/RESULTS_TEMPLATE.md with real numbers"
echo "4. Write Abstract"
echo "5. Final review and submit"
