#!/bin/bash
# Qwen3-8B replication: lens fit -> jlens traces -> prescreen -> analyze ->
# band ID (jlens + logit_lens) -> logit_lens traces -> prescreen -> analyze.
#
# Mirrors the proven resume_4b_pipeline.sh + continue_all_gpu.sh pattern from
# the 4B run, consolidated into one script since 8B's fit is long enough
# (~14h estimated) that resumability matters even more this time. Every step
# is idempotent (checks for its output file before redoing), so this script
# is safe to re-run after any interruption -- just run it again.
#
# IMPORTANT: run this under nohup/tmux/screen, not a bare foreground shell.
# A dropped SSH session over a ~14h run would otherwise kill the fit.
#   nohup bash out/run_8b_pipeline.sh > /dev/null 2>&1 &
#   disown
# Then reconnect any time and check progress with:
#   tail -f out/run_8b_pipeline.log

set -euo pipefail
cd /workspace/paper_exp_when_thinks_right
LOG=out/run_8b_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "===== 8B PIPELINE START $(date -u) ====="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

if [ -f out/fit_ckpt_8b.pt ]; then
  python3 - <<'PY' || true
import torch
c = torch.load("out/fit_ckpt_8b.pt", map_location="cpu", weights_only=False)
print(f"checkpoint n_done={c.get('n_done')} next_idx={c.get('next_idx')}")
PY
fi

# ---- 1. Lens fit (the long step, ~14h estimated -- see experiments/README_GPU_PHASE.md §D) ----
if [ ! -f out/lens_qwen3_8b.pt ]; then
  echo "===== 8B lens fit $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 01_fit_lens.py \
    --model Qwen/Qwen3-8B --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_8b.pt --checkpoint out/fit_ckpt_8b.pt
fi
ls -lh out/lens_qwen3_8b.pt

# ---- 2. Primary (jlens) traces ----
if [ ! -f out/traces_8b.json ]; then
  echo "===== 8B jlens traces $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 02_run_experiment.py \
    --model Qwen/Qwen3-8B --readout jlens \
    --lens out/lens_qwen3_8b.pt --stimuli stimuli.json \
    --out out/traces_8b.json
fi
ls -lh out/traces_8b.json

echo "===== PRESCREEN 8B jlens $(date -u) ====="
python3 prescreen.py --traces out/traces_8b.json | tee out/prescreen_8b.log

if [ ! -f out/analysis_8b/report.json ]; then
  echo "===== ANALYZE 8B jlens $(date -u) ====="
  python3 03_analyze.py --traces out/traces_8b.json \
    --outdir out/analysis_8b --dev-split 0.6
fi
ls -lh out/analysis_8b/report.json

# ---- 3. Workspace band identification (Appendix C, both readouts) ----
if [ ! -f out/band_identification_8b_jlens.json ]; then
  echo "===== 8B band jlens $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 04_identify_band.py --model Qwen/Qwen3-8B \
    --readout jlens --lens out/lens_qwen3_8b.pt --n-prompts 200 \
    --out out/band_identification_8b_jlens.json
  if [ -f out/figures/band_identification_jlens.png ]; then
    cp out/figures/band_identification_jlens.png out/figures/band_identification_8b_jlens.png || true
  fi
fi

if [ ! -f out/band_identification_8b_logitlens.json ]; then
  echo "===== 8B band logit_lens $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 04_identify_band.py --model Qwen/Qwen3-8B \
    --readout logit_lens --n-prompts 200 \
    --out out/band_identification_8b_logitlens.json
  if [ -f out/figures/band_identification_logit_lens.png ]; then
    cp out/figures/band_identification_logit_lens.png out/figures/band_identification_8b_logit_lens.png || true
  fi
fi

# ---- 4. Logit-lens traces (robustness parity with 1.7B/4B) ----
if [ ! -f out/traces_8b_logitlens.json ]; then
  echo "===== 8B logit-lens traces $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 02_run_experiment.py \
    --model Qwen/Qwen3-8B --readout logit_lens \
    --stimuli stimuli.json --out out/traces_8b_logitlens.json
fi
ls -lh out/traces_8b_logitlens.json

echo "===== PRESCREEN 8B logit_lens $(date -u) ====="
python3 prescreen.py --traces out/traces_8b_logitlens.json | tee out/prescreen_8b_logitlens.log

if [ ! -f out/analysis_8b_logitlens/report.json ]; then
  echo "===== ANALYZE 8B logit_lens $(date -u) ====="
  python3 03_analyze.py --traces out/traces_8b_logitlens.json \
    --outdir out/analysis_8b_logitlens --dev-split 0.6
fi
ls -lh out/analysis_8b_logitlens/report.json

echo "===== ALL 8B GPU EXPERIMENTS DONE $(date -u) =====" | tee out/ALL_8B_GPU_DONE.txt
ls -lh out/lens_qwen3_8b.pt out/traces_8b.json out/analysis_8b/report.json \
  out/band_identification_8b_*.json out/traces_8b_logitlens.json \
  out/analysis_8b_logitlens/report.json 2>&1 || true
date -u >> out/ALL_8B_GPU_DONE.txt
