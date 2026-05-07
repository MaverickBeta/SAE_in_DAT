#!/usr/bin/env python3
"""
Probe global entries from raw features.

Requirements for a global entry (token, channel):
  1. Clean activation freq >= 92% in at least MIN_CLASSES classes
  2. Among those classes, direction consistency >= 80%
     (i.e., >=80% are SUPPRESS or >=80% are ENHANCE, using |delta|>1.0)
  3. Delta values across those classes are approximately similar (low CV)

This script does NOT depend on per-class selected_entries.json.
It works directly from the raw clean/adversarial feature files.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_train_latent")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect_20_train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

CLASS_NAMES = [name for _, _, name in CLASSES]
N_CLASSES = len(CLASSES)
N_TOKENS = 49
N_CHANNELS = 12288


def main():
    parser = argparse.ArgumentParser(description="Probe global entries from raw SAE features")
    parser.add_argument("--freq", type=float, default=0.92,
                        help="Clean activation frequency threshold (default: 0.92)")
    parser.add_argument("--con", type=float, default=0.80,
                        help="Direction consistency threshold (default: 0.80)")
    parser.add_argument("--cv", type=float, default=0.50,
                        help="Coefficient of variation threshold (default: 0.50)")
    parser.add_argument("--min-classes", type=int, default=12,
                        help="Minimum classes with clean_freq >= threshold (default: 12)")
    parser.add_argument("--dir-thresh", type=float, default=1.0,
                        help="|delta| threshold to count as directional (default: 1.0)")
    args = parser.parse_args()

    MIN_CLASSES = args.min_classes
    FREQ_THRESH = args.freq
    DIR_THRESH = args.dir_thresh
    CONSISTENCY_THRESH = args.con
    CV_THRESH = args.cv

    print("=" * 80)
    print("PROBE GLOBAL ENTRIES (from raw features)")
    print("=" * 80)
    print(f"Requirements:")
    print(f"  1. clean_freq >= {FREQ_THRESH*100:.0f}% in >= {MIN_CLASSES} classes")
    print(f"  2. direction consistency >= {CONSISTENCY_THRESH*100:.0f}% (|delta| > {DIR_THRESH})")
    print(f"  3. delta CV < {CV_THRESH}")
    print()

    # ── Load all clean & adv features ──────────────────────────────────
    print("Loading features...")
    class_clean_feats = {}
    class_adv_feats = {}
    for wnid, cls_idx, name in CLASSES:
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        class_clean_feats[name] = np.load(clean_path)
        class_adv_feats[name] = np.load(adv_path)
        print(f"  {name:>15s}: clean={class_clean_feats[name].shape[0]}, adv={class_adv_feats[name].shape[0]}")

    # ── Precompute per-class per-entry statistics ──────────────────────
    print("\nPrecomputing statistics for all entries...")

    # Arrays: (n_classes, n_tokens, n_channels)
    clean_freq = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    clean_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    adv_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    delta = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)

    for i, name in enumerate(CLASS_NAMES):
        clean = class_clean_feats[name]
        adv = class_adv_feats[name]
        clean_freq[i] = (clean != 0).mean(axis=0)
        clean_mean[i] = clean.mean(axis=0)
        adv_mean[i] = adv.mean(axis=0)
        delta[i] = adv_mean[i] - clean_mean[i]

    # ── Filter step 1: freq >= 92% in >= MIN_CLASSES ───────────────────
    freq_mask = clean_freq >= FREQ_THRESH  # (N_CLASSES, N_TOKENS, N_CHANNELS)
    freq_count = freq_mask.sum(axis=0)     # (N_TOKENS, N_CHANNELS)

    candidate_mask = freq_count >= MIN_CLASSES  # (N_TOKENS, N_CHANNELS)
    candidate_indices = np.argwhere(candidate_mask)  # (N_candidates, 2)
    print(f"\nStep 1: {len(candidate_indices)} entries with clean_freq>={FREQ_THRESH*100:.0f}% in >= {MIN_CLASSES} classes")

    # ── Filter step 2 & 3: direction consistency + delta CV ────────────
    global_entries = []

    for tok, ch in candidate_indices:
        tok, ch = int(tok), int(ch)

        # Extract stats for this entry across all classes
        freqs = clean_freq[:, tok, ch]
        deltas = delta[:, tok, ch]

        # Only consider classes where clean_freq >= threshold
        valid_mask = freqs >= FREQ_THRESH
        valid_deltas = deltas[valid_mask]
        valid_classes = [CLASS_NAMES[i] for i in range(N_CLASSES) if valid_mask[i]]

        n_valid = len(valid_deltas)
        if n_valid == 0:
            continue

        # Direction counts using |delta| > DIR_THRESH
        n_suppress = int(np.sum(valid_deltas < -DIR_THRESH))
        n_enhance = int(np.sum(valid_deltas > DIR_THRESH))
        n_neutral = n_valid - n_suppress - n_enhance

        # Consistency: max(suppress, enhance) / total (including neutral)
        consistency = max(n_suppress, n_enhance) / n_valid if n_valid > 0 else 0.0

        if consistency < CONSISTENCY_THRESH:
            continue

        # Delta similarity (CV): only among directional classes
        directional_mask = np.abs(valid_deltas) > DIR_THRESH
        directional_deltas = valid_deltas[directional_mask]

        if len(directional_deltas) < 2:
            continue

        mean_delta = float(np.mean(directional_deltas))
        std_delta = float(np.std(directional_deltas))
        mean_abs_delta = float(np.mean(np.abs(directional_deltas)))
        cv = std_delta / mean_abs_delta if mean_abs_delta > 0 else float('inf')

        # Also compute min/max/range for reporting
        min_delta = float(np.min(directional_deltas))
        max_delta = float(np.max(directional_deltas))
        range_delta = max_delta - min_delta

        # Save this entry regardless of CV, but flag whether it passes CV threshold
        entry_info = {
            "token": tok,
            "channel": ch,
            "n_valid_classes": n_valid,
            "valid_classes": valid_classes,
            "n_suppress": n_suppress,
            "n_enhance": n_enhance,
            "n_neutral": n_neutral,
            "consistency": round(consistency, 4),
            "direction": "SUPPRESS" if n_suppress > n_enhance else "ENHANCE",
            "mean_delta": round(mean_delta, 6),
            "std_delta": round(std_delta, 6),
            "mean_abs_delta": round(mean_abs_delta, 6),
            "cv": round(cv, 4),
            "min_delta": round(min_delta, 6),
            "max_delta": round(max_delta, 6),
            "range_delta": round(range_delta, 6),
            "passes_cv": bool(cv < CV_THRESH),
            "per_class": [],
        }

        # Record per-class details
        for i, name in enumerate(CLASS_NAMES):
            if valid_mask[i]:
                entry_info["per_class"].append({
                    "class": name,
                    "clean_freq": round(float(freqs[i]), 4),
                    "clean_mean": round(float(clean_mean[i, tok, ch]), 6),
                    "adv_mean": round(float(adv_mean[i, tok, ch]), 6),
                    "delta": round(float(deltas[i]), 6),
                    "has_direction": bool(abs(deltas[i]) > DIR_THRESH),
                })

        global_entries.append(entry_info)

    # ── Sort by consistency, then by n_valid_classes ───────────────────
    global_entries.sort(key=lambda x: (-x["consistency"], -x["n_valid_classes"], x["cv"]))

    print(f"\nStep 2+3: {len(global_entries)} entries pass consistency >= {CONSISTENCY_THRESH}")
    print(f"  Of these, {sum(1 for e in global_entries if e['passes_cv'])} have CV < {CV_THRESH}")

    # ── Print top entries ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TOP GLOBAL ENTRIES")
    print("=" * 100)
    print(f"{'#':>3} {'Token':>6} {'Channel':>8} {'Freq':>5} {'Sup':>4} {'Enh':>4} {'Neu':>4} "
          f"{'Consist':>8} {'Dir':>10} {'MeanΔ':>10} {'StdΔ':>10} {'CV':>8} {'PassCV':>7}")
    print("-" * 100)

    for i, e in enumerate(global_entries[:30], 1):
        print(f"{i:>3} {e['token']:>6} {e['channel']:>8} {e['n_valid_classes']:>5} "
              f"{e['n_suppress']:>4} {e['n_enhance']:>4} {e['n_neutral']:>4} "
              f"{e['consistency']:>8.2f} {e['direction']:>10} "
              f"{e['mean_delta']:>+10.2f} {e['std_delta']:>10.2f} "
              f"{e['cv']:>8.3f} {'Yes' if e['passes_cv'] else 'No':>7}")

    # ── Save JSON ──────────────────────────────────────────────────────
    out_json = OUT_DIR / "probe_global_entries.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "min_classes": MIN_CLASSES,
                "freq_thresh": FREQ_THRESH,
                "dir_thresh": DIR_THRESH,
                "consistency_thresh": CONSISTENCY_THRESH,
                "cv_thresh": CV_THRESH,
            },
            "total_candidates_step1": int(candidate_mask.sum()),
            "total_global_entries": len(global_entries),
            "passes_cv": sum(1 for e in global_entries if e["passes_cv"]),
            "entries": global_entries,
        }, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # ── Plot: delta distribution of top global entries ─────────────────
    n_plot = min(20, len(global_entries))
    if n_plot > 0:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()

        for idx in range(n_plot):
            e = global_entries[idx]
            ax = axes[idx]

            classes = [d["class"] for d in e["per_class"]]
            deltas = [d["delta"] for d in e["per_class"]]
            colors = ["blue" if d < -DIR_THRESH else "red" if d > DIR_THRESH else "gray" for d in deltas]

            ax.barh(range(len(classes)), deltas, color=colors, alpha=0.7)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=6)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.axvline(-DIR_THRESH, color="blue", linestyle="--", alpha=0.3)
            ax.axvline(DIR_THRESH, color="red", linestyle="--", alpha=0.3)
            ax.set_title(f"t{e['token']}_c{e['channel']}\n"
                         f"consist={e['consistency']:.2f}, cv={e['cv']:.2f}", fontsize=8)
            ax.set_xlim(-15, 15)

        for idx in range(n_plot, 20):
            axes[idx].axis("off")

        plt.tight_layout()
        out_plot = OUT_DIR / "probe_global_entries_delta_distribution.png"
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_plot}")

    print("\nDone!")


if __name__ == "__main__":
    main()
