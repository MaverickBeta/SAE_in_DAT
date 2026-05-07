#!/usr/bin/env bash
# ============================================================
# Top-K Ablation Test for Class-wise SAE Statistics
# ============================================================
# 用法：
#   1. 修改下面 4 个路径变量
#   2. chmod +x run_topk_ablation.sh
#   3. ./run_topk_ablation.sh
# ============================================================

set -euo pipefail

# ---------- 请只修改这部分路径 ----------
BASE_MODEL="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth"
SAE_CKPT="/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k64_exp16/sae_stage3_din1536_exp16_k64_step_50000.pt"
IMAGENET_TRAIN="/Data_share/hongyi/DAT/data/ImageNet/train"
ADV_NPZ="/Data_share/hongyi/DAT/SAE/align_steering/adv_samples_eps8/sae_latent/no_sae/n01440764_spatial.npz"
# ----------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/topk_ablation_results"
mkdir -p "${RESULT_DIR}"

# 激活 conda 环境（根据你的实际环境名修改）
# source /Data_share/hongyi/miniconda3/etc/profile.d/conda.sh
# conda activate sae

echo "========================================"
echo "Top-K Ablation Test"
echo "========================================"
echo "SAE:  ${SAE_CKPT}"
echo "ADV:  ${ADV_NPZ}"
echo "OUT:  ${RESULT_DIR}"
echo ""

# 要测试的 Top-K 值列表
TOPK_LIST=(32 64 128 256)

for K in "${TOPK_LIST[@]}"; do
    CLASS_NPZ="${RESULT_DIR}/class_stats_top${K}.npz"
    CLASS_JSON="${RESULT_DIR}/class_stats_top${K}.json"
    EVAL_JSON="${RESULT_DIR}/eval_top${K}.json"

    echo ""
    echo ">>> [Top-K=${K}] Step 1/2: Generating class statistics..."

    if [ -f "${CLASS_NPZ}" ]; then
        echo "    (Found existing ${CLASS_NPZ}, skipping generation)"
    else
        python "${SCRIPT_DIR}/sae_stat.py" \
            --checkpoint "${BASE_MODEL}" \
            --sae-ckpt "${SAE_CKPT}" \
            --imagenet-train-dir "${IMAGENET_TRAIN}" \
            --sae-stage 3 \
            --topk "${K}" \
            --images-per-class 500 \
            --batch-size 16 \
            --num-workers 4 \
            --output-json "${CLASS_JSON}" \
            --output-npz "${CLASS_NPZ}"
    fi

    echo "    Class stats shape: $(python3 -c "import numpy as np; d=np.load('${CLASS_NPZ}',allow_pickle=True); print(d['spatial_indices'].shape)")"

echo ">>> [Top-K=${K}] Step 2/2: Evaluating similarity..."
    python "${SCRIPT_DIR}/eval_dis_spatial_fast.py" \
        --adv-npz "${ADV_NPZ}" \
        --class-npz "${CLASS_NPZ}" \
        --gpu 0 \
        --output-json "${EVAL_JSON}"

done

echo ""
echo ">>> Step 3: Aggregating comparison..."
python "${SCRIPT_DIR}/compare_topk_eval.py" \
    ${RESULT_DIR}/eval_top*.json

echo ""
echo "========================================"
echo "All done. Results are in: ${RESULT_DIR}"
echo "========================================"
