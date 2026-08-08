#!/bin/bash
# Setup script for Hop6 project — creates venv and installs all dependencies

set -e

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo ">>> venv already exists, skipping creation."
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# --- Core project dependencies (from cli.py, engine.py, train_router.py) ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install networkx
pip install questionary
pip install rich
pip install huggingface_hub
pip install accelerate       # needed by transformers for device_map
pip install safetensors      # for .safetensors model loading
pip install sentencepiece    # tokenizer backend for many models


echo ""
echo "=== All dependencies installed ==="
echo "Activate with:  source venv/bin/activate"
