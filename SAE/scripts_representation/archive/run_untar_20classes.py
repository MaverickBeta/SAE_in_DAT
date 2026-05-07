#!/usr/bin/env python3
"""
Batch run untar_sample_gen.py for 20 representative ImageNet classes.

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_untar_20classes.py

Data source:
    /Data_share/hongyi/imagenet/val/{wnid}/  (50 JPEGs per class)

Output:
    /Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20classes/
        {wnid}_cls{idx}_apgd_ce_l2_eps3_steps100/
            adv_succ_real/      # successful adversarial images
            clean_matched/      # corresponding clean images (copied)
            eval_results.csv    # metadata: pred_clean, pred_adv, conf_adv, escaped
            summary.json        # attack parameters & summary stats
"""

import os
import subprocess
import sys
from pathlib import Path

# ── GPU Setting ─────────────────────────────────────────────────────
# Use the last GPU (GPU 7 on an 8-GPU machine).
# Change this if you want a different GPU, or remove this line and use:
#   CUDA_VISIBLE_DEVICES=7 python run_untar_20classes.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ── Config ──────────────────────────────────────────────────────────
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
OUTPUT_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20classes")
UNTAR_SCRIPT = Path(__file__).resolve().parent / "untar_sample_gen.py"

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

# Attack parameters (fixed across all classes)
ATTACK_CONFIG = {
    "norm": "L2",
    "eps": 3.0,
    "steps": 100,
    "loss": "ce",
    "n_restarts": 1,
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda",
    "seed": 42,
}


def run_class(wnid: str, cls_idx: int, name: str) -> bool:
    source_dir = IMAGENET_VAL / wnid
    if not source_dir.is_dir():
        print(f"[SKIP] Source dir not found: {source_dir}")
        return False

    run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"

    cmd = [
        sys.executable,
        str(UNTAR_SCRIPT),
        "--source-dir", str(source_dir),
        "--source-cls", str(cls_idx),
        "--output-root", str(OUTPUT_ROOT),
        "--run-name", run_name,
        "--norm", ATTACK_CONFIG["norm"],
        "--eps", str(ATTACK_CONFIG["eps"]),
        "--steps", str(ATTACK_CONFIG["steps"]),
        "--loss", ATTACK_CONFIG["loss"],
        "--n-restarts", str(ATTACK_CONFIG["n_restarts"]),
        "--batch-size", str(ATTACK_CONFIG["batch_size"]),
        "--num-workers", str(ATTACK_CONFIG["num_workers"]),
        "--device", ATTACK_CONFIG["device"],
        "--seed", str(ATTACK_CONFIG["seed"]),
    ]

    print("\n" + "=" * 70)
    print(f"[{name}] wnid={wnid}  class_idx={cls_idx}")
    print(f"Run name: {run_name}")
    print("=" * 70)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Failed for {name} ({wnid}), return code: {result.returncode}")
        return False
    return True


def main():
    print("Batch adversarial sample generation for 20 representative classes")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Total classes: {len(CLASSES)}")
    print(f"Attack: APGD-{ATTACK_CONFIG['loss']} {ATTACK_CONFIG['norm']} eps={ATTACK_CONFIG['eps']} steps={ATTACK_CONFIG['steps']}")
    print()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for wnid, cls_idx, name in CLASSES:
        ok = run_class(wnid, cls_idx, name)
        if ok:
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    print(f"Success: {success}/{len(CLASSES)}")
    print(f"Failed:  {failed}/{len(CLASSES)}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print()

    # Print a quick reference table
    print("Generated runs:")
    print(f"{'#':>3} {'WNID':>12} {'Idx':>5} {'Name':>20} {'Status':>10}")
    print("-" * 60)
    for wnid, cls_idx, name in CLASSES:
        run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
        status = "OK" if (OUTPUT_ROOT / run_name / "eval_results.csv").exists() else "MISSING"
        print(f"{CLASSES.index((wnid,cls_idx,name))+1:>3} {wnid:>12} {cls_idx:>5} {name:>20} {status:>10}")


if __name__ == "__main__":
    main()
