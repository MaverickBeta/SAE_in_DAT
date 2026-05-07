#!/usr/bin/env python3
"""
Select SAE entries related to adversarial perturbations.

Step 1 (Strict): An entry must be activated (>0) in >90% of images
        in EVERY one of the 100 classes (train clean).
Step 2 (Ranking): Among these entries, find top-30 with largest
        |adv_mean - clean_mean|, where means are computed globally
        over all train clean (10,000 images) vs all train adv
        (10,000 images).

Output: DAT/SAE/steering/selected_entries.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Select adversarial-related SAE entries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sae-latent-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples" / "sae_latent"),
        help="Root directory containing feature_index.json and .npy files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "selected_entries.json"),
    )
    parser.add_argument(
        "--freq-threshold",
        type=float,
        default=0.9,
        help="Per-class activation frequency threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top entries to select by |adv - clean|",
    )
    args = parser.parse_args()

    latent_root = Path(args.sae_latent_root)
    index_path = latent_root / "feature_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    wnids = sorted(index.keys())
    n_classes = len(wnids)
    print(f"Total classes: {n_classes}")

    # ------------------------------------------------------------------
    # Accumulate per-class statistics and global sums
    # ------------------------------------------------------------------
    freq_sum = None          # accumulated per-class freq for averaging later
    all_masks = []           # per-class boolean masks
    clean_sum = None         # accumulated sum over all train clean images
    adv_sum = None           # accumulated sum over all train adv images
    total_images = 0         # total number of clean images (== adv images)

    for i, wnid in enumerate(wnids):
        # --- train clean ---
        clean_path = index[wnid]["train_clean"]["npy"]
        clean_data = np.load(clean_path)          # (N, n_tokens, d_lat)
        n_imgs = clean_data.shape[0]

        freq = (clean_data > 0).sum(axis=0) / n_imgs   # (n_tokens, d_lat)
        mask = freq > args.freq_threshold

        if freq_sum is None:
            freq_sum = freq
        else:
            freq_sum += freq
        all_masks.append(mask)

        if clean_sum is None:
            clean_sum = clean_data.sum(axis=0)
        else:
            clean_sum += clean_data.sum(axis=0)

        # --- train adv ---
        adv_path = index[wnid]["train_adv"]["npy"]
        adv_data = np.load(adv_path)              # (N, n_tokens, d_lat)

        if adv_sum is None:
            adv_sum = adv_data.sum(axis=0)
        else:
            adv_sum += adv_data.sum(axis=0)

        total_images += n_imgs

        if (i + 1) % 10 == 0 or (i + 1) == n_classes:
            print(f"  Processed {i + 1}/{n_classes} classes")

    # ------------------------------------------------------------------
    # Step 1: global mask (activated in ALL classes)
    # ------------------------------------------------------------------
    avg_freq = freq_sum / n_classes              # (n_tokens, d_lat)
    global_mask = np.all(all_masks, axis=0)      # (n_tokens, d_lat)
    n_candidates = int(global_mask.sum())
    print(
        f"\nEntries with >{args.freq_threshold * 100:.0f}% activation "
        f"in ALL {n_classes} classes: {n_candidates}"
    )

    if n_candidates == 0:
        print("WARNING: No entries satisfy the frequency criterion!")
        result = {
            "selection_params": {
                "freq_threshold": args.freq_threshold,
                "top_k": args.top_k,
                "n_classes": n_classes,
                "split": "train",
                "n_candidates": 0,
            },
            "selected_entries": [],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Empty result saved to {args.output}")
        return

    # ------------------------------------------------------------------
    # Step 2: global means and top-k selection
    # ------------------------------------------------------------------
    clean_mean = clean_sum / total_images          # (n_tokens, d_lat)
    adv_mean = adv_sum / total_images              # (n_tokens, d_lat)
    diff = np.abs(adv_mean - clean_mean)           # (n_tokens, d_lat)

    # zero-out entries that did not pass Step 1
    masked_diff = np.where(global_mask, diff, 0.0)

    flat_indices = np.argsort(masked_diff.ravel())[-args.top_k:][::-1]
    token_indices, feature_indices = np.unravel_index(
        flat_indices, masked_diff.shape
    )

    selected = []
    for rank, (t, f) in enumerate(zip(token_indices, feature_indices), start=1):
        t = int(t)
        f = int(f)
        selected.append({
            "rank": rank,
            "feature_idx": f,
            "token_idx": t,
            "clean_mean": float(clean_mean[t, f]),
            "adv_mean": float(adv_mean[t, f]),
            "diff_abs": float(diff[t, f]),
            "activation_freq_avg_over_classes": float(avg_freq[t, f]),
        })

    result = {
        "selection_params": {
            "freq_threshold": args.freq_threshold,
            "top_k": args.top_k,
            "n_classes": n_classes,
            "split": "train",
            "n_candidates": n_candidates,
        },
        "selected_entries": selected,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\nTop-{args.top_k} selected entries:")
    print(
        f"{'Rank':>6} {'Feature':>8} {'Token':>6} "
        f"{'Clean':>10} {'Adv':>10} {'Diff':>10} {'AvgFreq':>8}"
    )
    print("-" * 68)
    for e in selected:
        print(
            f"{e['rank']:>6} {e['feature_idx']:>8} {e['token_idx']:>6} "
            f"{e['clean_mean']:>10.4f} {e['adv_mean']:>10.4f} "
            f"{e['diff_abs']:>10.4f} {e['activation_freq_avg_over_classes']:>8.4f}"
        )

    print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
