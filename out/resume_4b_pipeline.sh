#!/bin/bash
# Resume Qwen3-4B replication from fit_ckpt_4b.pt, then traces → prescreen → analyze.
set -euo pipefail
cd /workspace/paper_exp_when_thinks_right
LOG=out/resume_4b_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "===== RESUME START $(date -u) ====="
nvidia-smi --query-gpu=name,memory.used --format=csv || true

if [ -f out/fit_ckpt_4b.pt ]; then
  python3 - <<'PY'
import torch
c = torch.load("out/fit_ckpt_4b.pt", map_location="cpu", weights_only=False)
print(f"checkpoint n_done={c.get('n_done')} next_idx={c.get('next_idx')}")
PY
fi

if [ ! -f out/lens_qwen3_4b.pt ]; then
  echo "===== RESUME 4B lens fit $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 01_fit_lens.py \
    --model Qwen/Qwen3-4B --n-prompts 1000 --seq-len 128 \
    --out out/lens_qwen3_4b.pt --checkpoint out/fit_ckpt_4b.pt
  ls -lh out/lens_qwen3_4b.pt
else
  echo "lens already present: $(ls -lh out/lens_qwen3_4b.pt)"
fi

if [ ! -f out/traces_4b.json ]; then
  echo "===== START 4B traces $(date -u) ====="
  PYTHONUNBUFFERED=1 python3 -u 02_run_experiment.py \
    --model Qwen/Qwen3-4B --readout jlens \
    --lens out/lens_qwen3_4b.pt --stimuli stimuli.json \
    --out out/traces_4b.json
  ls -lh out/traces_4b.json
else
  echo "traces already present: $(ls -lh out/traces_4b.json)"
fi

echo "===== PRESCREEN 4B $(date -u) ====="
python3 prescreen.py --traces out/traces_4b.json | tee out/prescreen_4b.log

echo "===== ANALYZE 4B $(date -u) ====="
python3 03_analyze.py --traces out/traces_4b.json \
  --outdir out/analysis_4b --dev-split 0.6
ls -lh out/analysis_4b/report.json

echo "===== RESUME DONE $(date -u) ====="
ls -lh out/lens_qwen3_4b.pt out/traces_4b.json out/analysis_4b/report.json
