#!/bin/bash
set -e

echo "Setting up Mammo-CLIP Conda Environment..."

# Use the existing environment.yml to create the environment
conda env create -f /home/robit/Documents/Project/Mammo-CLIP/environment.yml -n mammo-clip || echo "Environment might already exist"

# Activate the environment (need to use source for bash scripts)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mammo-clip

# Download punkt_tab for NLTK as noted in the requirements
echo "Downloading NLTK punkt_tab..."
python -c "import nltk; nltk.download('punkt_tab')"

# Verify CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

echo "Environment setup complete! To activate: conda activate mammo-clip"
