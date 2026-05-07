#!/usr/bin/env python3
"""
Sweep probe_global_entries hyperparameters and output a CSV table.

Columns: freq, min_cls, step1, con, dir_thresh, step2, cv, step3

Total combos: 3(freq) × 3(min_cls) × 3(con) × 3(dir_thresh) × 3(cv) = 243
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path
from itertools import product
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")
OUT_CSV = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect_20/sweep_probe_results.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

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
N_CLASSES = len(CLASSES)
N_TOKENS = 49
N_CHANNELS = 12288


def load_feature_stats():
    """Load raw SAE features and compute clean_freq and delta arrays."""
    print("[Sweep] Loading raw SAE features...")
    clean_freq = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    clean_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    adv_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)

    for i, (wnid, cls_idx, name) in enumerate(CLASSES):
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"

        clean_feat = np.load(clean_path)
        adv_feat = np.load(adv_path)

        clean_freq[i] = (clean_feat != 0).mean(axis=0)
        clean_mean[i] = clean_feat.mean(axis=0)
        adv_mean[i] = adv_feat.mean(axis=0)

        del clean_feat, adv_feat

    delta = adv_mean - clean_mean
    print(f"[Sweep] Feature stats ready: {clean_freq.nbytes / 1e6:.1f} MB per array")
    return clean_freq, delta


def probe_counts(clean_freq, delta, freq, min_classes, con, dir_thresh, cv_thresh):
    """
    Run the three-step filtering and return counts.
    Mirrors probe_global_entries.py exactly.
    """
    # Step 1: freq + min_classes
    valid_mask = clean_freq >= freq
    n_valid = valid_mask.sum(axis=0)
    candidate_mask = n_valid >= min_classes
    step1 = int(candidate_mask.sum())
    candidate_indices = np.argwhere(candidate_mask)

    # Step 2+3: con + dir_thresh + cv
    step2 = 0
    step3 = 0
    for tok, ch in candidate_indices:
        vm = valid_mask[:, tok, ch]
        valid_deltas = delta[:, tok, ch][vm]

        n_suppress = int(np.sum(valid_deltas < -dir_thresh))
        n_enhance = int(np.sum(valid_deltas > dir_thresh))

        consistency = max(n_suppress, n_enhance) / vm.sum() if vm.sum() > 0 else 0.0
        if consistency < con:
            continue
        step2 += 1

        directional_mask = np.abs(valid_deltas) > dir_thresh
        directional_deltas = valid_deltas[directional_mask]

        if len(directional_deltas) < 2:
            continue

        mean_abs_delta = float(np.mean(np.abs(directional_deltas)))
        if mean_abs_delta < 1e-12:
            continue

        std_delta = float(np.std(directional_deltas, ddof=0))
        cv = std_delta / mean_abs_delta

        if cv < cv_thresh:
            step3 += 1

    return step1, step2, step3


def main():
    clean_freq, delta = load_feature_stats()

    # Parameter grids
    freq_vals = [0.92, 0.90, 0.88]
    min_cls_vals = [12, 10, 8]
    con_vals = [0.80, 0.70, 0.60]
    dir_vals = [0.5, 1.0, 2.0]
    cv_vals = [0.50, 0.80, 1.00]

    all_combos = list(product(freq_vals, min_cls_vals, con_vals, dir_vals, cv_vals))
    print(f"[Sweep] Total combinations: {len(all_combos)}")

    # Cache step1 results to avoid recomputing for same (freq, min_classes)
    step1_cache = {}

    results = []
    for idx, (freq, min_cls, con, dir_thresh, cv) in enumerate(all_combos, 1):
        cache_key = (freq, min_cls)
        if cache_key not in step1_cache:
            # We need step2/step3 too, but they depend on con/dir/cv.
            # However, step1 candidates are independent of con/dir/cv.
            # We can't fully cache step2/step3, but we can precompute candidate_indices.
            valid_mask = clean_freq >= freq
            n_valid = valid_mask.sum(axis=0)
            candidate_mask = n_valid >= min_cls
            candidate_indices = np.argwhere(candidate_mask)
            step1_cache[cache_key] = (int(candidate_mask.sum()), candidate_indices)
        else:
            candidate_indices = step1_cache[cache_key][1]

        # For step2/step3, we still need to loop candidates with the specific con/dir/cv
        step1, step2, step3 = probe_counts(
            clean_freq, delta,
            freq=freq,
            min_classes=min_cls,
            con=con,
            dir_thresh=dir_thresh,
            cv_thresh=cv,
        )

        results.append({
            "freq": freq,
            "min_cls": min_cls,
            "step1": step1,
            "con": con,
            "dir_thresh": dir_thresh,
            "step2": step2,
            "cv": cv,
            "step3": step3,
        })

        if idx % 50 == 0 or idx == len(all_combos):
            print(f"[Sweep] {idx}/{len(all_combos)} done")

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "freq", "min_cls", "step1", "con", "dir_thresh", "step2", "cv", "step3"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Sweep] Results saved: {OUT_CSV}")
    print(f"  Total rows: {len(results)}")

    # Quick summary
    max_step3 = max(r["step3"] for r in results)
    min_step3 = min(r["step3"] for r in results)
    print(f"  Step3 range: {min_step3} ~ {max_step3}")


if __name__ == "__main__":
    main()
