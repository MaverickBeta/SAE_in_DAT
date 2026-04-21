#!/bin/bash
# ============================================================
# 串行测试 8 个 stage3 SAE checkpoint 的 L2 PGD-CE 攻击
# 挂载 SAE + 原始 Head（无 steering）
# 每个 checkpoint 使用 4 卡并行（GPU 4,5,6,7）
# ============================================================

set -e

CKPT_DIR="/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k64_exp16"
# 使用已有的 50 类列表（如果根目录的不存在，从子目录复制一份）
CLASS_LIST="/Data_share/hongyi/DAT/SAE/new_model/adv_samples_l2/selected_classes.txt"
if [ ! -f "$CLASS_LIST" ]; then
    FALLBACK="/Data_share/hongyi/DAT/SAE/new_model/adv_samples_l2/sae_stage3_din1536_exp16_k64_step_20000/selected_classes.txt"
    if [ -f "$FALLBACK" ]; then
        cp "$FALLBACK" "$CLASS_LIST"
        echo "Copied class list from $FALLBACK"
    else
        echo "ERROR: No class list found. Please provide --class-list manually."
        exit 1
    fi
fi
GPU_IDS=(4 5 6 7)

N_CLASSES=50
N_SAMPLES=50
BATCH_SIZE=16
EPS=3.0
STEPS=110
STEP_SIZE=3.0

# 8 个 checkpoint（串行执行）
CKPTS=(
    "sae_stage3_din1536_exp16_k64_step_10000.pt"
    "sae_stage3_din1536_exp16_k64_step_20000.pt"
    "sae_stage3_din1536_exp16_k64_step_30000.pt"
    "sae_stage3_din1536_exp16_k64_step_40000.pt"
    "sae_stage3_din1536_exp16_k64_step_50000.pt"
    "best_composite.pt"
    "best_ev.pt"
    "best_loss.pt"
)

echo "========================================"
echo "SAE-mounted L2 PGD-CE Evaluation"
echo "GPUs: ${GPU_IDS[*]}"
echo "Classes: $N_CLASSES"
echo "Samples per class: $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Total checkpoints: ${#CKPTS[@]}"
echo "========================================"

for CKPT_NAME in "${CKPTS[@]}"; do
    CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}"
    OUT_NAME="${CKPT_NAME%.pt}"
    OUT_DIR="/Data_share/hongyi/DAT/SAE/new_model/adv_samples_l2_sae_mounted/${OUT_NAME}"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH"
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing: $OUT_NAME"
    echo "Checkpoint: $CKPT_PATH"
    echo "Output: $OUT_DIR"
    echo "----------------------------------------"

    python batch_baseline_attack_sae_mounted_l2.py \
        --class-list "$CLASS_LIST" \
        --n-classes $N_CLASSES \
        --n-samples $N_SAMPLES \
        --eps $EPS \
        --steps $STEPS \
        --step-size $STEP_SIZE \
        --batch-size $BATCH_SIZE \
        --gpus "${GPU_IDS[@]}" \
        --sae-ckpt "$CKPT_PATH" \
        --sae-stage 3 \
        --output-dir "$OUT_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $OUT_NAME"
done

echo ""
echo "========================================"
echo "All checkpoints evaluated!"
echo "========================================"
