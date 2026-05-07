#!/usr/bin/env python3
"""
Channel-level statistics on SAE latent features.

Loads raw (n_images, n_tokens, d_lat) features from sae_feature_extract.py,
aggregates over the token dimension (max/mean/p90), then computes:
1. Exploration stats (distribution, thresholds, percentiles)
2. Paired clean vs adversarial comparison (effect sizes)
3. Candidate feature lists for downstream ablation/steering

No plots are generated here; only numerical outputs (.json, .csv, .npy).
"""

import argparse
import csv
import json 
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Channel-level SAE feature statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clean-npy",
        type=str,
        required=True,
        help="Path to clean features .npy from sae_feature_extract.py.",
    )
    parser.add_argument(
        "--clean-meta",
        type=str,
        required=True,
        help="Path to clean meta .json.",
    )
    parser.add_argument(
        "--adv-npy",
        type=str,
        required=True,
        help="Path to adversarial features .npy.",
    )
    parser.add_argument(
        "--adv-meta",
        type=str,
        required=True,
        help="Path to adversarial meta .json.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["max", "mean", "p90", "all"],
        default="max",
        help="How to aggregate n_tokens to a single value per feature per image.",
    )
    parser.add_argument(
        "--cohen-threshold",
        type=float,
        default=0.5,
        help="Threshold for |Cohen's d| to flag significant features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/feature_stats",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def aggregate_tokens(features: np.ndarray, method: str) -> np.ndarray:
    """
    Args:
        features: (n_images, n_tokens, d_lat)
        method: "max", "mean", or "p90"
    Returns:
        (n_images, d_lat)
    """
    if method == "max":
        return features.max(axis=1)
    if method == "mean":
        return features.mean(axis=1)
    if method == "p90":
        return np.percentile(features, 90, axis=1)
    raise ValueError(f"Unknown aggregation: {method}")


def compute_exploration_stats(arr: np.ndarray) -> Dict:
    """
    arr: (n_images, d_lat) — already aggregated over tokens.
    Computes distribution stats on the per-feature mean across images.
    """
    # Per-feature mean across images
    per_feat_mean = arr.mean(axis=0)      # (d_lat,)
    per_feat_max = arr.max(axis=0)        # (d_lat,)
    per_feat_min = arr.min(axis=0)        # (d_lat,)
    per_feat_std = arr.std(axis=0)        # (d_lat,)

    # Global stats on all activation values
    flat = arr.flatten()
    nonzero = flat[flat > 0]

    max_val = float(per_feat_max.max())
    min_val = float(per_feat_min.min())
    min_nonzero = float(nonzero.min()) if nonzero.size > 0 else 0.0
    mean_val = float(nonzero.mean()) if nonzero.size > 0 else 0.0
    median_val = float(np.median(nonzero)) if nonzero.size > 0 else 0.0

    # Percentiles on per-feature mean
    p10 = float(np.percentile(per_feat_mean, 10))
    p20 = float(np.percentile(per_feat_mean, 20))
    p50 = float(np.percentile(per_feat_mean, 50))
    p80 = float(np.percentile(per_feat_mean, 80))
    p90 = float(np.percentile(per_feat_mean, 90))
    p99 = float(np.percentile(per_feat_mean, 99))

    # Threshold-based counts (on per-feature max activation)
    th_10 = max_val / 10.0
    th_2 = max_val / 2.0
    n_above_10 = int((per_feat_max > th_10).sum())
    n_above_2 = int((per_feat_max > th_2).sum())

    # Active rate: fraction of images where feature is > 0
    active_rate = (arr > 0).mean(axis=0)
    mean_active_rate = float(active_rate.mean())

    return {
        "d_lat": int(arr.shape[1]),
        "n_images": int(arr.shape[0]),
        "total_values": int(arr.size),
        "nonzero_values": int(nonzero.size),
        "sparsity_rate": float(1.0 - nonzero.size / max(1, arr.size)),
        "global_max": max_val,
        "global_min": min_val,
        "global_min_nonzero": min_nonzero,
        "global_mean_nonzero": mean_val,
        "global_median_nonzero": median_val,
        "percentile_10": p10,
        "percentile_20": p20,
        "percentile_50": p50,
        "percentile_80": p80,
        "percentile_90": p90,
        "percentile_99": p99,
        "threshold_1_10_max": th_10,
        "threshold_1_2_max": th_2,
        "n_features_max_above_1_10": n_above_10,
        "n_features_max_above_1_2": n_above_2,
        "fraction_features_above_1_10": n_above_10 / max(1, arr.shape[1]),
        "fraction_features_above_1_2": n_above_2 / max(1, arr.shape[1]),
        "mean_active_rate_per_feature": mean_active_rate,
    }


def compute_paired_comparison(
    clean: np.ndarray, adv: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    clean, adv: (n_images, d_lat)
    Returns per-feature statistics.
    """
    diff = adv - clean  # (n_images, d_lat)
    mean_diff = diff.mean(axis=0)
    std_diff = diff.std(axis=0)
    cohens_d = mean_diff / (std_diff + 1e-12)
    mean_clean = clean.mean(axis=0)
    mean_adv = adv.mean(axis=0)
    return mean_diff, std_diff, cohens_d, mean_clean, mean_adv


def main():
    args = parse_args()

    # Load data
    print(f"Loading clean: {args.clean_npy}")
    clean_raw = np.load(args.clean_npy)  # (n_clean, n_tokens, d_lat)
    with open(args.clean_meta, "r", encoding="utf-8") as f:
        clean_meta = json.load(f)

    print(f"Loading adv:   {args.adv_npy}")
    adv_raw = np.load(args.adv_npy)
    with open(args.adv_meta, "r", encoding="utf-8") as f:
        adv_meta = json.load(f)

    print(f"Clean raw shape: {clean_raw.shape}")
    print(f"Adv raw shape:   {adv_raw.shape}")

    # Aggregate over tokens
    agg_methods = ["max", "mean", "p90"] if args.aggregation == "all" else [args.aggregation]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"channel_stat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for agg in agg_methods:
        print(f"\n{'='*60}")
        print(f"Aggregation: {agg}")
        print(f"{'='*60}")

        clean_agg = aggregate_tokens(clean_raw, agg)  # (n_images, d_lat)
        adv_agg = aggregate_tokens(adv_raw, agg)

        # ---- 1. Exploration stats ----
        clean_stats = compute_exploration_stats(clean_agg)
        adv_stats = compute_exploration_stats(adv_agg)

        print("\n--- Clean stats ---")
        for k, v in clean_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        print("\n--- Adv stats ---")
        for k, v in adv_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        # Save stats
        with open(run_dir / f"clean_stats_{agg}.json", "w", encoding="utf-8") as f:
            json.dump(clean_stats, f, indent=2, ensure_ascii=False)
        with open(run_dir / f"adv_stats_{agg}.json", "w", encoding="utf-8") as f:
            json.dump(adv_stats, f, indent=2, ensure_ascii=False)

        # ---- 2. Paired comparison ----
        mean_diff, std_diff, cohens_d, mean_clean, mean_adv = compute_paired_comparison(
            clean_agg, adv_agg
        )

        # Top features by |Cohen's d|
        topk = min(200, clean_agg.shape[1])
        top_idx = np.argsort(np.abs(cohens_d))[::-1][:topk]

        comparison_rows = []
        for i in top_idx:
            comparison_rows.append(
                {
                    "feature_idx": int(i),
                    "clean_mean": float(mean_clean[i]),
                    "adv_mean": float(mean_adv[i]),
                    "mean_diff": float(mean_diff[i]),
                    "std_diff": float(std_diff[i]),
                    "cohens_d": float(cohens_d[i]),
                    "direction": "up" if mean_diff[i] > 0 else "down",
                }
            )

        csv_path = run_dir / f"comparison_top200_{agg}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
            writer.writeheader()
            writer.writerows(comparison_rows)

        # ---- 3. Candidate feature lists ----
        sig_mask = np.abs(cohens_d) > args.cohen_threshold
        sig_up = np.where((mean_diff > 0) & sig_mask)[0].tolist()
        sig_down = np.where((mean_diff < 0) & sig_mask)[0].tolist()

        # Also include top by raw mean_diff magnitude
        top_raw_idx = np.argsort(np.abs(mean_diff))[::-1][:100]

        candidates = {
            "aggregation": agg,
            "cohen_threshold": args.cohen_threshold,
            "upregulated_in_adv": {
                "description": "Features significantly stronger in adversarial samples",
                "count": len(sig_up),
                "indices": sig_up,
            },
            "downregulated_in_adv": {
                "description": "Features significantly weaker in adversarial samples (source features suppressed)",
                "count": len(sig_down),
                "indices": sig_down,
            },
            "top50_by_cohens_d": {
                "count": 50,
                "indices": top_idx[:50].tolist(),
            },
            "top100_by_cohens_d": {
                "count": 100,
                "indices": top_idx[:100].tolist(),
            },
            "top100_by_raw_diff": {
                "count": 100,
                "indices": top_raw_idx.tolist(),
            },
        }

        with open(run_dir / f"candidates_{agg}.json", "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)

        # Summary
        n_up = int((mean_diff > 0).sum())
        n_down = int((mean_diff < 0).sum())
        print(f"\nPaired comparison ({agg}):")
        print(f"  Upregulated:   {n_up} ({n_up/clean_agg.shape[1]:.2%})")
        print(f"  Downregulated: {n_down} ({n_down/clean_agg.shape[1]:.2%})")
        print(f"  |Cohen's d| > {args.cohen_threshold}: {int(sig_mask.sum())}")
        print(f"  Significant up:   {len(sig_up)}")
        print(f"  Significant down: {len(sig_down)}")
        print(f"  Top 10 by |Cohen's d|: {top_idx[:10].tolist()}")

    # Save run metadata
    run_summary = {
        "run_name": run_name,
        "clean_npy": args.clean_npy,
        "clean_meta": args.clean_meta,
        "adv_npy": args.adv_npy,
        "adv_meta": args.adv_meta,
        "aggregation_methods": agg_methods,
        "cohen_threshold": args.cohen_threshold,
        "output_dir": str(run_dir),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll stats saved to: {run_dir}")


if __name__ == "__main__":
    main()
