#!/bin/bash
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"
# Adjusting batch size down from 8 to 4 to fit in 20GB RTX 3080
BATCH_SIZE=4
EPOCHS=30
CHKPT="${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar"

for FRACTION in 0.1 0.5 1.0; do
    for LABEL in "Mass" "Suspicious_Calcification" "density"; do
        echo "Running LP B5 for ${LABEL} with ${FRACTION} fraction..."
        python "${CODE_DIR}/train_classifier.py" \
            --arch breast_clip_det_b5_period_n_lp \
            --dataset ViNDr \
            --label "${LABEL}" \
            --batch-size ${BATCH_SIZE} \
            --epochs ${EPOCHS} \
            --data_frac ${FRACTION} \
            --clip_chk_pt_path "${CHKPT}" \
            --output_path "${BASE_DIR}/outputs/linear_probe/b5/${LABEL}_${FRACTION}"
    done
done
echo "Linear probe B5 completed!"
