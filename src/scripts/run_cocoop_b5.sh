#!/bin/bash
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"
CHKPT="${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar"

# Batch size 4 for B5 (larger model, more memory)
BATCH_SIZE=4
EPOCHS=50
LR=0.002
N_CTX=4

for LABEL in "Mass" "Suspicious_Calcification" "Malignancy" "density"; do
    echo "============================================"
    echo "Running CoCoOp B5 for ${LABEL}..."
    echo "============================================"
    python "${CODE_DIR}/train_cocoop.py" \
        --clip_chk_pt_path "${CHKPT}" \
        --label "${LABEL}" \
        --data-dir "${BASE_DIR}/data" \
        --img-dir "vindr/images_png" \
        --n_ctx ${N_CTX} \
        --ctx_init "" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --output_path "${BASE_DIR}/outputs/cocoop/b5/${LABEL}"
done

echo "CoCoOp B5 training completed!"
