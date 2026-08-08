#!/bin/bash
echo "Setting up Hop6 Engine..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found."
    echo "Please run ./setup_env.sh first."
    exit 1
fi
pip install -r requirements.txt
python src/cli.py
