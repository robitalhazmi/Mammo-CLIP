#!/bin/bash
set -e

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"
# Fine-tuning B2, using batch size 4
BATCH_SIZE=4
EPOCHS=30
CHKPT="${BASE_DIR}/checkpoints/pretrained/b2-model-best-epoch-10.tar"
FRACTION=1.0

for LABEL in "Mass" "Suspicious_Calcification" "density"; do
    echo "Running FT B2 for ${LABEL}..."
    python "${CODE_DIR}/train_classifier.py" \
        --arch breast_clip_det_b2_period_n_ft \
        --dataset ViNDr \
        --label "${LABEL}" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --data_frac ${FRACTION} \
        --clip_chk_pt_path "${CHKPT}" \
        --output_path "${BASE_DIR}/outputs/fine_tune/b2/${LABEL}"
done
echo "Fine-tune B2 completed!"
