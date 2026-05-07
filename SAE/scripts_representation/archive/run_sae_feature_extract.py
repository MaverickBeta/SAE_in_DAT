#!/usr/bin/env python3
"""
Batch run sae_feature_extract.py for 20 representative classes.

For each class, extracts SAE latent features for:
  1. Clean images:   imagenet/val/{wnid}/
  2. Adv images:     untar_samples_20classes/{run_name}/

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_sae_feature_extract.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ── GPU Setting ─────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ── Paths ───────────────────────────────────────────────────────────
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20classes")
FEAT_SCRIPT = Path(__file__).resolve().parent / "sae_feature_extract.py"
OUTPUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")

BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"

# 20 representative classes: (wnid, class_idx, name)
CLASSES = [
    ("n01440764",   0, "tench"),
    ("n01530575",  10, "brambling"),
    ("n01641577",  30, "bullfrog"),
    ("n01806143",  84, "peacock"),
    ("n01871265", 101, "tusker"),
    ("n02077923", 150, "sea_lion"),
    ("n02123045", 281, "tabby_cat"),
    ("n02128385", 288, "leopard"),
    ("n02129604", 292, "tiger"),
    ("n02165456", 301, "ladybug"),
    ("n03063599", 504, "coffee_mug"),
    ("n03085013", 508, "computer_keyboard"),
    ("n03250847", 542, "drum"),
    ("n03445777", 574, "golf_ball"),
    ("n03770439", 655, "miniskirt"),
    ("n03888257", 701, "parachute"),
    ("n04146614", 779, "school_bus"),
    ("n04285008", 817, "sports_car"),
    ("n07720875", 945, "artichoke"),
    ("n07747607", 950, "orange"),
]

# Feature extraction config
FEAT_CONFIG = {
    "stage": 3,
    "batch_size": 8,
    "num_workers": 4,
    "device": "cuda",
}


def extract_features(input_dir: Path, run_name: str) -> bool:
    if not input_dir.is_dir():
        print(f"[SKIP] Input dir not found: {input_dir}")
        return False

    cmd = [
        sys.executable,
        str(FEAT_SCRIPT),
        "--input-dir", str(input_dir),
        "--base-ckpt", BASE_CKPT,
        "--sae-ckpt", SAE_CKPT,
        "--stage", str(FEAT_CONFIG["stage"]),
        "--batch-size", str(FEAT_CONFIG["batch_size"]),
        "--num-workers", str(FEAT_CONFIG["num_workers"]),
        "--device", FEAT_CONFIG["device"],
        "--output-dir", str(OUTPUT_DIR),
        "--run-name", run_name,
    ]

    print("\n" + "=" * 70)
    print(f"Extracting: {run_name}")
    print(f"Input: {input_dir}")
    print("=" * 70)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Failed for {run_name}, return code: {result.returncode}")
        return False
    return True


def main():
    print("Batch SAE feature extraction for 20 representative classes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Stage: {FEAT_CONFIG['stage']}")
    print(f"SAE: {SAE_CKPT}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for wnid, cls_idx, name in CLASSES:
        adv_run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
        adv_dir = ADV_ROOT / adv_run_name / "adv_succ_real"
        
        # 1. Clean features
        clean_input = IMAGENET_VAL / wnid
        clean_run_name = f"clean_{wnid}_cls{cls_idx}_stage3_k256"
        ok = extract_features(clean_input, clean_run_name)
        if ok:
            success += 1
        else:
            failed += 1

        # 2. Adv features (ONLY successfully attacked images)
        adv_run_name_feat = f"adv_{wnid}_cls{cls_idx}_stage3_k256"
        ok = extract_features(adv_dir, adv_run_name_feat)
        if ok:
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("BATCH FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Quick check: list generated .npy files
    npy_files = sorted(OUTPUT_DIR.glob("*_features.npy"))
    print(f"Generated feature files ({len(npy_files)}):")
    for f in npy_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
