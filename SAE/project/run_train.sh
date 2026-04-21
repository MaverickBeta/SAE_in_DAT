#!/bin/bash
# Initialize conda for this sub-shell
eval "$(conda shell.bash hook)"
conda activate rebm

# Default parameters, can be overridden when calling the script
GPU_ID=${1:-2}
STAGE_IDX=${2:-3}
K_VAL=${3:-64}
LR=${4:-3e-4}
EXPANSION_RATE=${5:-32}
DATA_PATH=${6:-}

echo "=========================================="
echo "🚀 Starting SAE training..."
echo "GPU ID:   ${GPU_ID}"
echo "Stage:    ${STAGE_IDX}"
echo "K value:  ${K_VAL}"
echo "Expansion:${EXPANSION_RATE}"
echo "Learning Rate: ${LR}"
echo "=========================================="

CMD=(
    python3 train.py
    --gpu_id ${GPU_ID}
    --stage_idx ${STAGE_IDX}
    --expansion_rate ${EXPANSION_RATE}
    --k ${K_VAL}
    --lr ${LR}
    --steps 50000
    --aux_coeff 0.1
    --dead_window 2500
)

if [[ -n "${DATA_PATH}" ]]; then
    CMD+=(--data_path "${DATA_PATH}")
fi

"${CMD[@]}"