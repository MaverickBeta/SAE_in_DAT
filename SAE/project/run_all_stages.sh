#!/bin/bash
set -euo pipefail

ACTIVE_PIDS=()

cleanup_children() {
  echo ""
  echo "[Signal] Stopping active training subprocesses..."
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done

  # Give children a moment to exit gracefully, then force kill if needed.
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done

  exit 130
}

trap cleanup_children INT TERM

# Initialize conda for this sub-shell
eval "$(conda shell.bash hook)"
conda activate rebm

GPU_LIST=${1:-6,7}
LR=${2:-3e-4}
EXPANSION_RATE=${3:-32}
STEPS=${4:-50000}
STATS_BATCHES=${9:-30}
BATCH_SIZE=${10:-8192}
FILES_PER_BATCH=${11:-1}
CACHE_SIZE=${12:-8}
HOT_FILE_POOL_SIZE=${13:-4}
POOL_REFRESH_EVERY=${14:-400}

# Optional explicit data paths for stage 0/1/2/3.
# If empty, train.py uses its internal defaults.
STAGE0_DATA_PATH=${5:-}
STAGE1_DATA_PATH=${6:-}
STAGE2_DATA_PATH=${7:-}
STAGE3_DATA_PATH=${8:-}

declare -A STAGE_DATA_PATHS
STAGE_DATA_PATHS[0]="${STAGE0_DATA_PATH}"
STAGE_DATA_PATHS[1]="${STAGE1_DATA_PATH}"
STAGE_DATA_PATHS[2]="${STAGE2_DATA_PATH}"
STAGE_DATA_PATHS[3]="${STAGE3_DATA_PATH}"

STAGES=(0 1 2 3)
K_VALUES=(32 64)

echo "=========================================="
echo "🚀 Running all stage x k jobs"
echo "GPU list: ${GPU_LIST}"
echo "LR: ${LR}"
echo "Expansion rate: ${EXPANSION_RATE}"
echo "Steps: ${STEPS}"
echo "Stats batches: ${STATS_BATCHES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Files per batch: ${FILES_PER_BATCH}"
echo "Cache size: ${CACHE_SIZE}"
echo "Hot file pool size: ${HOT_FILE_POOL_SIZE}"
echo "Pool refresh every: ${POOL_REFRESH_EVERY}"
echo "Stages: ${STAGES[*]}"
echo "K values: ${K_VALUES[*]}"
echo "=========================================="

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

if [[ ${NUM_GPUS} -eq 0 ]]; then
  echo "No GPU specified in GPU_LIST."
  exit 1
fi

echo "Total jobs: $(( ${#STAGES[@]} * ${#K_VALUES[@]} ))"
echo "Stage-wise mode: YES (finish each stage before next stage)"
echo "Max parallel per stage: 1"

for stage in "${STAGES[@]}"; do
  echo "=========================================="
  echo "Starting stage ${stage}"
  echo "=========================================="

  ACTIVE_PIDS=()

  for ((idx=0; idx<${#K_VALUES[@]}; idx++)); do
    k="${K_VALUES[$idx]}"
    gpu_id="${GPUS[$((idx % NUM_GPUS))]}"

    echo "------------------------------------------"
    echo "[GPU ${gpu_id}] Training stage=${stage}, k=${k}"

    CMD=(
      python3 train.py
      --gpu_id "${gpu_id}"
      --stage_idx "${stage}"
      --expansion_rate "${EXPANSION_RATE}"
      --k "${k}"
      --lr "${LR}"
      --steps "${STEPS}"
      --batch_size "${BATCH_SIZE}"
      --aux_coeff 0.1
      --dead_window 2500
      --stats_batches "${STATS_BATCHES}"
      --files_per_batch "${FILES_PER_BATCH}"
      --cache_size "${CACHE_SIZE}"
      --hot_file_pool_size "${HOT_FILE_POOL_SIZE}"
      --pool_refresh_every "${POOL_REFRESH_EVERY}"
    )

    if [[ -n "${STAGE_DATA_PATHS[$stage]}" ]]; then
      CMD+=(--data_path "${STAGE_DATA_PATHS[$stage]}")
    fi

    "${CMD[@]}" &
    ACTIVE_PIDS=("$!")
    wait "${ACTIVE_PIDS[0]}"
    ACTIVE_PIDS=()
  done

  echo "✅ Stage ${stage} finished."
done

echo "✅ All stage x k jobs finished."
