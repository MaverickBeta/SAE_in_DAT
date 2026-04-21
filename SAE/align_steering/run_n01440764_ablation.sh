#!/usr/bin/env bash
# ============================================================
# 单类验证脚本：n01440764
# 测试 class stats Top64 vs Top128 对对抗样本匹配的影响
# ============================================================
# 用法：
#   1. 确认下面的路径是否正确
#   2. chmod +x run_n01440764_ablation.sh
#   3. ./run_n01440764_ablation.sh
# ============================================================

set -euo pipefail

# ---------- 配置区（按需修改） ----------
CLASS_NAME="n01440764"
GPU_ID="1"                                          # 你想用的 GPU，如 0, 1, 2, 3
BASE_MODEL="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth"
SAE_CKPT="/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k64_exp16/sae_stage3_din1536_exp16_k64_step_50000.pt"
ADV_IMG_DIR="/Data_share/hongyi/DAT/SAE/align_steering/adv_samples_eps8/no_sae/${CLASS_NAME}"
CLASS_STAT_DIR="/Data_share/hongyi/DAT/SAE/align_steering/sae_stat_npz_json"
RESULT_DIR="/Data_share/hongyi/DAT/SAE/align_steering/top64_top128"

# 如需激活 conda 环境，取消下面两行的注释并修改环境名
# source /Data_share/hongyi/miniconda3/etc/profile.d/conda.sh
# conda activate sae
# ----------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RESULT_DIR}/adv_latent"
mkdir -p "${RESULT_DIR}/eval_results"

echo "========================================"
echo "Class: ${CLASS_NAME}"
echo "GPU:   ${GPU_ID}"
echo "========================================"

# Step 1: 用新 SAE 提取对抗样本 latent（只提取一次，topk=64）
ADV_NPZ="${RESULT_DIR}/adv_latent/${CLASS_NAME}_spatial.npz"

if [ -f "${ADV_NPZ}" ]; then
    echo "[Step 1/4] Adversarial latent already exists: ${ADV_NPZ}, skipping extraction."
else
    echo "[Step 1/4] Extracting adversarial latent with new SAE..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_DIR}/extract_from_sae.py" \
        --adv-dir "${ADV_IMG_DIR}" \
        --checkpoint "${BASE_MODEL}" \
        --sae-ckpt "${SAE_CKPT}" \
        --sae-stage 3 \
        --topk 64 \
        --batch-size 16 \
        --num-workers 4 \
        --output-npz "${ADV_NPZ}"
fi

# Step 2: 评估 class_stats_top64
echo ""
echo "[Step 2/4] Evaluating with class_stats_top64.npz..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_DIR}/eval_dis_spatial_fast.py" \
    --adv-npz "${ADV_NPZ}" \
    --class-npz "${CLASS_STAT_DIR}/class_stats_top64.npz" \
    --output-json "${RESULT_DIR}/eval_results/eval_top64_${CLASS_NAME}.json"

# Step 3: 评估 class_stats_top128
echo ""
echo "[Step 3/4] Evaluating with class_stats_top128.npz..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_DIR}/eval_dis_spatial_fast.py" \
    --adv-npz "${ADV_NPZ}" \
    --class-npz "${CLASS_STAT_DIR}/class_stats_top128.npz" \
    --output-json "${RESULT_DIR}/eval_results/eval_top128_${CLASS_NAME}.json"

# Step 4: 对比结果
echo ""
echo "[Step 4/4] Comparing results..."
python "${SCRIPT_DIR}/compare_topk_eval.py" \
    "${RESULT_DIR}/eval_results/eval_top64_${CLASS_NAME}.json" \
    "${RESULT_DIR}/eval_results/eval_top128_${CLASS_NAME}.json"

echo ""
echo "========================================"
echo "All done. Results saved to:"
echo "  ${RESULT_DIR}/eval_results/"
echo "========================================"
