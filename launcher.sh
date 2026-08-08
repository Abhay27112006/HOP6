#!/bin/bash
echo "============================================="
echo "        Hop6 Engine Launcher                 "
echo "============================================="

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found."
    echo "Please run ./setup_env.sh first."
    sleep 3
    exit 1
fi

python3 src/cli.py
