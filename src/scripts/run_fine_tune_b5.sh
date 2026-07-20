#!/bin/bash
set -e

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"
# Fine-tuning unfreezes the encoder, so it uses more memory. Using batch size 2.
BATCH_SIZE=2
EPOCHS=30
CHKPT="${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar"
FRACTION=1.0

for LABEL in "Mass" "Suspicious_Calcification" "density"; do
    echo "Running FT B5 for ${LABEL}..."
    python "${CODE_DIR}/train_classifier.py" \
        --arch breast_clip_det_b5_period_n_ft \
        --dataset ViNDr \
        --label "${LABEL}" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --data_frac ${FRACTION} \
        --clip_chk_pt_path "${CHKPT}" \
        --output_path "${BASE_DIR}/outputs/fine_tune/b5/${LABEL}"
done
echo "Fine-tune B5 completed!"
