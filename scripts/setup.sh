#!/bin/bash

# Install torch and related packages using the special URL
pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies from requirements.txt located in the root directory
ROOT_DIR="$(dirname "$(dirname "$0")")"
pip install -r "$ROOT_DIR/requirements.txt"
