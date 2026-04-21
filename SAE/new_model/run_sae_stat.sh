#!/bin/bash
# ============================================================
# 为两个 SAE checkpoint 生成 class statistics (sae_stat.npz)
# 串行执行，避免显存冲突
# ============================================================

set -e

S2="/Data_share/hongyi/DAT/SAE/align_steering/sae_stat.py"
CKPT1="/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k64_exp16/sae_stage3_din1536_exp16_k64_step_20000.pt"
CKPT2="/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt"
GPU_ID=3

echo "========================================"
echo "Generating class statistics for 2 SAE checkpoints"
echo "GPU: $GPU_ID"
echo "========================================"

for CKPT in "$CKPT1" "$CKPT2"; do
    NAME=$(basename "$CKPT" .pt)
    echo ""
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing: $NAME"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$S2" \
        --sae-ckpt "$CKPT" \
        --images-per-class 500 \
        --batch-size 16 \
        --auto-name \
        --class-batch-size 50

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $NAME"
done

echo ""
echo "========================================"
echo "All class statistics generated!"
echo "========================================"
