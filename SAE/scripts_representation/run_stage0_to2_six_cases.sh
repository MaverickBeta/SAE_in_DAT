#!/usr/bin/env bash
set -euo pipefail

# Run 6 SAE representation inspections:
#   stage0: k32, k64
#   stage1: k32, k64
#   stage2: k32, k64
#
# Defaults are intentionally small-sample friendly for quick inspection.

ROOT_DIR="/Data_share/hongyi/DAT"
INSPECT_SCRIPT="${ROOT_DIR}/SAE/scripts_representation/sae_feature_inspect.py"
BASE_CKPT="${ROOT_DIR}/checkpoints/model_bestfid.pth"
SAE_CKPT_ROOT="${ROOT_DIR}/SAE/project/checkpoints"
RESULTS_DIR="${ROOT_DIR}/SAE/results_representation"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SOURCE_WNID="${SOURCE_WNID:-n02077923}"
MAX_IMAGES="${MAX_IMAGES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TOPK_FEATURES="${TOPK_FEATURES:-300}"
ACT_THRESHOLD="${ACT_THRESHOLD:-0.4}"

if [[ ! -f "${INSPECT_SCRIPT}" ]]; then
  echo "[ERROR] Missing script: ${INSPECT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[ERROR] Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi

run_case () {
  local stage="$1"
  local ktag="$2"
  local sae_ckpt="$3"

  if [[ ! -f "${sae_ckpt}" ]]; then
    echo "[WARN] Skip missing checkpoint: ${sae_ckpt}"
    return 0
  fi

  local run_name="${SOURCE_WNID}_stage${stage}_${ktag}_step50000_n${MAX_IMAGES}"

  echo "============================================================"
  echo "Running case: stage=${stage}, k=${ktag}"
  echo "SAE CKPT: ${sae_ckpt}"
  echo "RUN NAME: ${run_name}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${INSPECT_SCRIPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --sae-ckpt "${sae_ckpt}" \
    --stage "${stage}" \
    --source-wnid "${SOURCE_WNID}" \
    --max-images "${MAX_IMAGES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --topk-features "${TOPK_FEATURES}" \
    --act-threshold "${ACT_THRESHOLD}" \
    --results-dir "${RESULTS_DIR}" \
    --run-name "${run_name}"
}

# stage0
run_case 0 k32 "${SAE_CKPT_ROOT}/stage0/k32_exp32/sae_stage0_din192_exp32_k32_step_50000.pt"
run_case 0 k64 "${SAE_CKPT_ROOT}/stage0/k64_exp32/sae_stage0_din192_exp32_k64_step_50000.pt"

# stage1
run_case 1 k32 "${SAE_CKPT_ROOT}/stage1/k32_exp32/sae_stage1_din384_exp32_k32_step_50000.pt"
run_case 1 k64 "${SAE_CKPT_ROOT}/stage1/k64_exp32/sae_stage1_din384_exp32_k64_step_50000.pt"

# stage2
run_case 2 k32 "${SAE_CKPT_ROOT}/stage2/k32_exp32/sae_stage2_din768_exp32_k32_step_50000.pt"
run_case 2 k64 "${SAE_CKPT_ROOT}/stage2/k64_exp32/sae_stage2_din768_exp32_k64_step_50000.pt"

echo ""
echo "All requested stage0-2 cases finished."
echo "Results root: ${RESULTS_DIR}"
