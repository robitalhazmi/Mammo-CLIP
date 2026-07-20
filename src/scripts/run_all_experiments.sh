#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running all Mammo-CLIP experiments..."
echo "Starting Zero-Shot..."
bash "${SCRIPT_DIR}/run_zero_shot.sh"

echo "Starting Linear Probes..."
bash "${SCRIPT_DIR}/run_linear_probe_b2.sh"
bash "${SCRIPT_DIR}/run_linear_probe_b5.sh"

echo "Starting Fine-Tuning..."
bash "${SCRIPT_DIR}/run_fine_tune_b2.sh"
bash "${SCRIPT_DIR}/run_fine_tune_b5.sh"

echo "All experiments finished! Run collect_results.py to compile."
