#!/bin/bash
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"
# B2 is lighter, can likely handle batch size 8 on 20GB RTX 3080
BATCH_SIZE=8
EPOCHS=30
CHKPT="${BASE_DIR}/checkpoints/pretrained/b2-model-best-epoch-10.tar"

for FRACTION in 0.1 0.5 1.0; do
    for LABEL in "Mass" "Suspicious_Calcification" "density"; do
        echo "Running LP B2 for ${LABEL} with ${FRACTION} fraction..."
        python "${CODE_DIR}/train_classifier.py" \
            --arch breast_clip_det_b2_period_n_lp \
            --dataset ViNDr \
            --label "${LABEL}" \
            --batch-size ${BATCH_SIZE} \
            --epochs ${EPOCHS} \
            --data_frac ${FRACTION} \
            --clip_chk_pt_path "${CHKPT}" \
            --output_path "${BASE_DIR}/outputs/linear_probe/b2/${LABEL}_${FRACTION}"
    done
done
echo "Linear probe B2 completed!"
