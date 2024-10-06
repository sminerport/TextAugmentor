#!/bin/bash

# Install torch and related packages using the special URL
pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies from requirements.txt
pip install -r requirements.txt
