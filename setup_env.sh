#!/bin/bash
# Create the pinned CPU analysis environment.
#
# Always invoke scripts as venv/bin/python (or activate first). A previous
# version of this repo shipped a setup that left no venv behind, so
# `source venv/bin/activate` silently did nothing and analysis ran against a
# stray anaconda Python 3.7 with a different scipy — producing p-values that
# did not match the committed results. The check at the end guards against
# that recurring.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

echo "Using interpreter: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv venv
else
    echo "Virtual environment already exists (delete venv/ to rebuild)."
fi

echo "Installing pinned dependencies..."
venv/bin/python -m pip install --quiet --upgrade pip
venv/bin/python -m pip install --quiet -r requirements.txt

echo ""
venv/bin/python verify_env.py

cat <<'EOF'

Done. Run analysis with the venv interpreter explicitly:

    venv/bin/python 03_analyze.py --traces out/traces.json \
        --outdir out/analysis_real --dev-split 0.6
    venv/bin/python generate_figures.py

GPU phases (01_fit_lens.py / 02_run_experiment.py) run on the GPU box and
additionally need: torch, transformers, datasets, and jlens.
EOF
