#!/bin/bash
# ============================================================
# 串行测试 2 个 SAE checkpoint 的 L2 PGD-CE 攻击
# 每个 checkpoint 使用自己对应的 class npz
# ============================================================

set -e

CLASS_LIST="/Data_share/hongyi/DAT/SAE/align_steering/clean_sae_results/selected_classes.txt"
CLASS_NPZ_DIR="/Data_share/hongyi/DAT/SAE/align_steering"
GPU_ID=2

N_CLASSES=50
N_SAMPLES=50
BATCH_SIZE=16
EPS=3.0
STEPS=110
STEP_SIZE=3.0

# checkpoint -> 对应的 class npz（由 sae_stat.py --auto-name 生成）
declare -A CKPT_MAP
CKPT_MAP["/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k64_exp16/sae_stage3_din1536_exp16_k64_step_20000.pt"]="${CLASS_NPZ_DIR}/sae_stage3_din1536_exp16_k64_step_20000_sae_stat_results.npz"
CKPT_MAP["/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt"]="${CLASS_NPZ_DIR}/sae_64k_k64_step_50000_sae_stat_results.npz"

echo "========================================"
echo "Starting 2-checkpoint L2 evaluation"
echo "GPU: $GPU_ID"
echo "Classes: $N_CLASSES"
echo "Samples per class: $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

for CKPT_PATH in "${!CKPT_MAP[@]}"; do
    CLASS_NPZ="${CKPT_MAP[$CKPT_PATH]}"
    CKPT_NAME=$(basename "$CKPT_PATH" .pt)
    OUT_DIR="/Data_share/hongyi/DAT/SAE/new_model/adv_samples_l2/${CKPT_NAME}"

    echo ""
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing: $CKPT_NAME"
    echo "Class NPZ: $CLASS_NPZ"
    echo "Output: $OUT_DIR"
    echo "----------------------------------------"

    if [ ! -f "$CLASS_NPZ" ]; then
        echo "WARNING: $CLASS_NPZ not found!"
        echo "Please run ./run_sae_stat.sh first to generate it."
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python batch_baseline_attack_ncm_l2.py \
        --class-list "$CLASS_LIST" \
        --n-classes $N_CLASSES \
        --n-samples $N_SAMPLES \
        --eps $EPS \
        --steps $STEPS \
        --step-size $STEP_SIZE \
        --batch-size $BATCH_SIZE \
        --gpus $GPU_ID \
        --sae-ckpt "$CKPT_PATH" \
        --class-npz "$CLASS_NPZ" \
        --output-dir "$OUT_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $CKPT_NAME"
done

echo ""
echo "========================================"
echo "All checkpoints evaluated!"
echo "========================================"
