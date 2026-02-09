#!/bin/bash
set -e

echo "Setting up Chronos v3 Environment..."

# 1. Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
if [ -f "chronos_v3/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r chronos_v3/requirements.txt
    # pip install ultralytics # Explicitly ensure YOLO is there if requirements.txt is minimal
else
    echo "Error: chronos_v3/requirements.txt not found!"
    exit 1
fi

echo "Setup Complete!"
echo "Run source venv/bin/activate to enter the environment."
