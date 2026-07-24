#!/bin/bash
# Continues after resume_4b_pipeline.sh: ensure 4B traces/analysis, then 4B band ID.
set -euo pipefail
cd /workspace/paper_exp_when_thinks_right
LOG=out/continue_all_gpu.log
exec > >(tee -a "$LOG") 2>&1
echo "===== CONTINUE_ALL START $(date -u) ====="

wait_for_fit() {
  echo "Waiting for 4B lens fit / resume pipeline ..."
  while pgrep -f "01_fit_lens.py --model Qwen/Qwen3-4B" >/dev/null 2>&1; do
    if [ -f out/fit_ckpt_4b.pt ]; then
      python3 - <<'PY' || true
import torch
c=torch.load("out/fit_ckpt_4b.pt", map_location="cpu", weights_only=False)
print(f"  progress n_done={c.get('n_done')}/1000 $(__import__('datetime').datetime.utcnow().isoformat())Z")
PY
    fi
    sleep 120
  done
  # also wait for resume orchestrator if still wrapping up
  while pgrep -f "resume_4b_pipeline.sh" >/dev/null 2>&1; do
    echo "  resume_4b_pipeline still running ..."
    sleep 60
  done
  echo "Fit/resume cleared at $(date -u)"
}

wait_for_fit

if [ ! -f out/lens_qwen3_4b.pt ]; then
  echo "ERROR: lens missing after fit; attempting resume fit once"
  PYTHONUNBUFFERED=1 python3 -u 01_fit_lens.py --model Qwen/Qwen3-4B \
    --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_4b.pt --checkpoint out/fit_ckpt_4b.pt
fi
ls -lh out/lens_qwen3_4b.pt

if [ ! -f out/traces_4b.json ]; then
  echo "===== START 4B traces $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 02_run_experiment.py --model Qwen/Qwen3-4B \
    --readout jlens --lens out/lens_qwen3_4b.pt --stimuli stimuli.json \
    --out out/traces_4b.json
fi
ls -lh out/traces_4b.json

echo "===== PRESCREEN 4B $(date -u) ====="
python3 prescreen.py --traces out/traces_4b.json | tee out/prescreen_4b.log

if [ ! -f out/analysis_4b/report.json ]; then
  echo "===== ANALYZE 4B $(date -u) ====="
  python3 03_analyze.py --traces out/traces_4b.json \
    --outdir out/analysis_4b --dev-split 0.6
fi
ls -lh out/analysis_4b/report.json

# 4B workspace band identification (Appendix C per-model)
if [ ! -f out/band_identification_4b_jlens.json ]; then
  echo "===== 4B band jlens $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 04_identify_band.py --model Qwen/Qwen3-4B \
    --readout jlens --lens out/lens_qwen3_4b.pt --n-prompts 200 \
    --out out/band_identification_4b_jlens.json
  # rename plot if script used readout name only
  if [ -f out/figures/band_identification_jlens.png ] && [ ! -f out/figures/band_identification_4b_jlens.png ]; then
    cp out/figures/band_identification_jlens.png out/figures/band_identification_4b_jlens.png || true
  fi
fi

if [ ! -f out/band_identification_4b_logitlens.json ]; then
  echo "===== 4B band logit_lens $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 04_identify_band.py --model Qwen/Qwen3-4B \
    --readout logit_lens --n-prompts 200 \
    --out out/band_identification_4b_logitlens.json
  if [ -f out/figures/band_identification_logit_lens.png ] && [ ! -f out/figures/band_identification_4b_logit_lens.png ]; then
    cp out/figures/band_identification_logit_lens.png out/figures/band_identification_4b_logit_lens.png || true
  fi
fi

# Optional: 4B logit-lens stimulus traces (robustness parity with 1.7B)
if [ ! -f out/traces_4b_logitlens.json ]; then
  echo "===== 4B logit-lens traces $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 02_run_experiment.py --model Qwen/Qwen3-4B \
    --readout logit_lens --stimuli stimuli.json \
    --out out/traces_4b_logitlens.json
  python3 prescreen.py --traces out/traces_4b_logitlens.json | tee out/prescreen_4b_logitlens.log || true
  python3 03_analyze.py --traces out/traces_4b_logitlens.json \
    --outdir out/analysis_4b_logitlens --dev-split 0.6
fi

echo "===== ALL GPU EXPERIMENTS DONE $(date -u) =====" | tee out/ALL_GPU_DONE.txt
ls -lh out/lens_qwen3_4b.pt out/traces_4b.json out/analysis_4b/report.json \
  out/band_identification_4b_*.json out/traces_4b_logitlens.json \
  out/analysis_4b_logitlens/report.json 2>&1 || true
date -u >> out/ALL_GPU_DONE.txt
