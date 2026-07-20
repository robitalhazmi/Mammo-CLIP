#!/bin/bash
set -e

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
REPO_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"

echo "Creating directory structure at ${BASE_DIR}..."
mkdir -p "${BASE_DIR}/data/vindr/images_png"
mkdir -p "${BASE_DIR}/checkpoints/pretrained"
mkdir -p "${BASE_DIR}/checkpoints/downstream"
mkdir -p "${BASE_DIR}/outputs/zero_shot"
mkdir -p "${BASE_DIR}/outputs/linear_probe"
mkdir -p "${BASE_DIR}/outputs/fine_tune"
mkdir -p "${BASE_DIR}/logs/tensorboard"
mkdir -p "${BASE_DIR}/results"

echo "Copying CSV files from repository to data directory..."
cp "${REPO_DIR}/src/codebase/data_csv/breast-level_annotations.csv" "${BASE_DIR}/data/vindr/"
cp "${REPO_DIR}/src/codebase/data_csv/finding_annotations.csv" "${BASE_DIR}/data/vindr/"
cp "${REPO_DIR}/src/codebase/data_csv/vindr_detection_v1_folds.csv" "${BASE_DIR}/data/vindr/"

echo "Downloading Mammo-CLIP checkpoints from HuggingFace..."
# B2 model
if [ ! -f "${BASE_DIR}/checkpoints/pretrained/b2-model-best-epoch-10.tar" ]; then
    echo "Downloading B2 checkpoint..."
    wget -O "${BASE_DIR}/checkpoints/pretrained/b2-model-best-epoch-10.tar" "https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar?download=true"
else
    echo "B2 checkpoint already exists."
fi

# B5 model
if [ ! -f "${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar" ]; then
    echo "Downloading B5 checkpoint..."
    wget -O "${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar" "https://huggingface.co/shawn24/Mammo-CLIP/resolve/main/Pre-trained-checkpoints/b5-model-best-epoch-7.tar?download=true"
else
    echo "B5 checkpoint already exists."
fi

echo "For downloading the VinDr PNG dataset from Kaggle, please ensure you have kaggle credentials configured (~/.kaggle/kaggle.json)."
echo "Then run: kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png -p ${BASE_DIR}/data/vindr/ --unzip"

echo "Data setup script completed!"
