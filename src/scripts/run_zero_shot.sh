#!/bin/bash
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

BASE_DIR="/data/nas07_new/PersonalData/robit/Mammo-CLIP"
CODE_DIR="${BASE_DIR}/src/codebase"

echo "Running Zero-Shot Evaluation for Mammo-CLIP B5"
python "${CODE_DIR}/eval_zero_shot_clip.py" \
    --config-name zs_clip \
    model=clip_b5_det_clinical \
    model.clip_check_point="${BASE_DIR}/checkpoints/pretrained/b5-model-best-epoch-7.tar" \
    base.output.save_path="${BASE_DIR}/outputs/zero_shot/b5"

echo "Running Zero-Shot Evaluation for Mammo-CLIP B2"
python "${CODE_DIR}/eval_zero_shot_clip.py" \
    --config-name zs_clip \
    model=clip_b2_det_clinical \
    model.clip_check_point="${BASE_DIR}/checkpoints/pretrained/b2-model-best-epoch-10.tar" \
    base.output.save_path="${BASE_DIR}/outputs/zero_shot/b2"

echo "Zero-shot evaluation completed!"
