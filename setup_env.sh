#!/bin/bash
# Setup script for GPU testing environment

echo "Setting up virtual environment for GPU experiments..."
echo "=================================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies (CPU-compatible)
echo "Installing core dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "Your virtual environment is ready."
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "When you have GPU access, install GPU dependencies:"
echo "  pip install torch transformers datasets"
echo ""
echo "Then you can start GPU testing:"
echo "  python 01_fit_lens.py --model Qwen/Qwen3-1.7B ..."
echo "=================================================="
