#!/usr/bin/env python3
"""
Batch selected_entry_inspect for 20 classes + cross-class global entry analysis.

Outputs:
  1. Per-class selected entries:   results_20classes/{wnid}_selected_entries.json
  2. Cross-class summary stats:    results_20classes/cross_class_summary.json
  3. Global candidate entries:     results_20classes/global_candidates.json

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_selected_entry_inspect_20classes.py
"""

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_latent")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20cls_train")

# ── GPU Setting ─────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

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

THRESH_COUNT = 92      # clean >= 92% (based on 100 images)
SEL_LOW = -1.0         # suppression
SEL_HIGH = 0.2         # enhancement
GLOBAL_THRESH_A = 8    # min classes for Level A global (out of 20)
GLOBAL_THRESH_B = 8    # min classes for Level B global (out of 20)


def select_entries(clean, adv):
    """Run selected_entry_inspect logic. Returns list of entry dicts."""
    n_img, n_tok, n_ch = clean.shape

    clean_count = np.sum(clean != 0, axis=0)
    adv_count = np.sum(adv != 0, axis=0)

    clean_sum = np.sum(clean, axis=0)
    adv_sum = np.sum(adv, axis=0)

    clean_mean = np.zeros_like(clean_sum, dtype=np.float32)
    adv_mean = np.zeros_like(adv_sum, dtype=np.float32)

    mask_c = clean_count > 0
    mask_a = adv_count > 0
    clean_mean[mask_c] = clean_sum[mask_c] / clean_count[mask_c]
    adv_mean[mask_a] = adv_sum[mask_a] / adv_count[mask_a]

    mask_high = clean_count >= THRESH_COUNT
    tok_idx, ch_idx = np.where(mask_high)

    entries = []
    for t, c in zip(tok_idx, ch_idx):
        cc = int(clean_count[t, c])
        ac = int(adv_count[t, c])
        cm = float(clean_mean[t, c])
        am = float(adv_mean[t, c]) if ac > 0 else 0.0
        delta = am - cm

        if delta < SEL_LOW or delta > SEL_HIGH:
            entries.append({
                "token": int(t),
                "channel": int(c),
                "clean_count": cc,
                "adv_count": ac,
                "clean_mean": round(cm, 6),
                "adv_mean": round(am, 6),
                "delta": round(delta, 6),
                "type": "SUPPRESS" if delta < 0 else "ENHANCE",
            })

    entries.sort(key=lambda x: x["delta"])
    return entries


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-class storage
    per_class_results = {}

    # Cross-class accumulators
    # Level A: strict (token, channel)
    level_a = defaultdict(lambda: {
        "count": 0,
        "suppress_count": 0,
        "enhance_count": 0,
        "clean_means": [],
        "adv_means": [],
        "deltas": [],
        "classes": [],
    })
    # Level B: channel-only
    level_b = defaultdict(lambda: {
        "count": 0,
        "suppress_count": 0,
        "enhance_count": 0,
        "clean_means": [],
        "adv_means": [],
        "deltas": [],
        "classes": [],
    })

    print("=" * 70)
    print("SELECTED ENTRY INSPECT: 20 CLASSES")
    print("=" * 70)
    print(f"Threshold: clean_count >= {THRESH_COUNT} ({THRESH_COUNT/100*100:.0f}%)")
    print(f"Selection: delta < {SEL_LOW}  or  delta > {SEL_HIGH}")
    print()

    # ── Step 1: Per-class analysis ──────────────────────────────────
    for wnid, cls_idx, name in CLASSES:
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"

        clean = np.load(clean_path)
        adv = np.load(adv_path)
        entries = select_entries(clean, adv)

        n_sup = sum(1 for e in entries if e["type"] == "SUPPRESS")
        n_enh = sum(1 for e in entries if e["type"] == "ENHANCE")
        print(f"{name:>15}  total={len(entries):>4}  SUPPRESS={n_sup:>4}  ENHANCE={n_enh:>4}")

        # Save per-class JSON
        per_class_results[name] = {
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_clean_images": int(clean.shape[0]),
            "n_adv_images": int(adv.shape[0]),
            "n_selected": len(entries),
            "n_suppress": n_sup,
            "n_enhance": n_enh,
            "selected_entries": entries,
        }

        json_path = OUT_DIR / f"{name}_selected_entries.json"
        with open(json_path, "w") as f:
            json.dump(per_class_results[name], f, indent=2)

        # Accumulate for cross-class analysis
        for e in entries:
            key_a = (e["token"], e["channel"])
            key_b = e["channel"]

            for level, key in [(level_a, key_a), (level_b, key_b)]:
                level[key]["count"] += 1
                level[key]["clean_means"].append(e["clean_mean"])
                level[key]["adv_means"].append(e["adv_mean"])
                level[key]["deltas"].append(e["delta"])
                level[key]["classes"].append(name)
                if e["type"] == "SUPPRESS":
                    level[key]["suppress_count"] += 1
                else:
                    level[key]["enhance_count"] += 1

    # ── Step 2: Cross-class summary ─────────────────────────────────
    print()
    print("=" * 70)
    print("CROSS-CLASS SUMMARY")
    print("=" * 70)

    # Level A frequency distribution
    print("\nLevel A (strict token+channel) frequency:")
    for f in range(1, 21):
        n = sum(1 for v in level_a.values() if v["count"] == f)
        if n > 0:
            bar = "█" * (n // 10)
            print(f"  {f:>2}/20 classes: {n:>4} entries  {bar}")

    # Level B frequency distribution
    print("\nLevel B (channel-only) frequency:")
    for f in range(1, 21):
        n = sum(1 for v in level_b.values() if v["count"] == f)
        if n > 0:
            bar = "█" * (n // 2)
            print(f"  {f:>2}/20 classes: {n:>4} channels  {bar}")

    # ── Step 3: Global candidates ───────────────────────────────────
    def get_global_candidates(level_dict, thresh, level_name):
        candidates = []
        for key, data in level_dict.items():
            if data["count"] < thresh:
                continue
            sup = data["suppress_count"]
            enh = data["enhance_count"]
            # Direction consistency: must be >80% one direction
            total_dir = sup + enh
            if total_dir == 0:
                continue
            consistency = max(sup, enh) / total_dir
            if consistency < 0.8:
                continue

            direction = "SUPPRESS" if sup > enh else "ENHANCE"
            candidates.append({
                "key": key,
                "freq": data["count"],
                "direction": direction,
                "consistency": round(consistency, 3),
                "suppress_count": sup,
                "enhance_count": enh,
                "mean_delta": round(float(np.mean(data["deltas"])), 4),
                "std_delta": round(float(np.std(data["deltas"])), 4),
                "mean_clean_mean": round(float(np.mean(data["clean_means"])), 4),
                "mean_adv_mean": round(float(np.mean(data["adv_means"])), 4),
                "classes": data["classes"],
            })

        candidates.sort(key=lambda x: (x["freq"], x["consistency"]), reverse=True)
        return candidates

    global_a = get_global_candidates(level_a, GLOBAL_THRESH_A, "A")
    global_b = get_global_candidates(level_b, GLOBAL_THRESH_B, "B")

    print(f"\nLevel A global candidates (freq>={GLOBAL_THRESH_A}, consistency>=0.8): {len(global_a)}")
    print(f"Level B global candidates (freq>={GLOBAL_THRESH_B}, consistency>=0.8): {len(global_b)}")

    # ── Step 4: Save results ────────────────────────────────────────
    summary = {
        "config": {
            "n_classes": len(CLASSES),
            "thresh_count": THRESH_COUNT,
            "sel_low": SEL_LOW,
            "sel_high": SEL_HIGH,
            "global_thresh_a": GLOBAL_THRESH_A,
            "global_thresh_b": GLOBAL_THRESH_B,
        },
        "per_class_summary": {
            name: {
                "wnid": data["wnid"],
                "class_idx": data["class_idx"],
                "n_clean_images": data["n_clean_images"],
                "n_adv_images": data["n_adv_images"],
                "n_selected": data["n_selected"],
                "n_suppress": data["n_suppress"],
                "n_enhance": data["n_enhance"],
            }
            for name, data in per_class_results.items()
        },
        "cross_class": {
            "level_a_total_unique_entries": len(level_a),
            "level_b_total_unique_channels": len(level_b),
        },
        "global_candidates": {
            "level_a": global_a,
            "level_b": global_b,
        },
    }

    summary_path = OUT_DIR / "cross_class_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved cross-class summary: {summary_path}")

    # Save a compact version for steering
    steering_targets = {}
    for c in global_a:
        token, channel = c["key"]
        steering_targets[f"{token}_{channel}"] = {
            "token": token,
            "channel": channel,
            "target_clean_mean": c["mean_clean_mean"],
            "direction": c["direction"],
            "freq": c["freq"],
        }

    global_path = OUT_DIR / "global_candidates.json"
    with open(global_path, "w") as f:
        json.dump({
            "n_candidates": len(global_a),
            "candidates": steering_targets,
        }, f, indent=2)
    print(f"Saved global candidates:   {global_path}")

    # ── Step 5: Print top candidates ────────────────────────────────
    if global_a:
        print("\n" + "=" * 70)
        print("TOP 20 LEVEL A GLOBAL CANDIDATES")
        print("=" * 70)
        print(f"{'#':>3} {'Token':>6} {'Channel':>8} {'Freq':>5} {'Dir':>10} {'Consist':>8} {'MeanΔ':>10}")
        print("-" * 70)
        for i, c in enumerate(global_a[:20], 1):
            t, ch = c["key"]
            print(f"{i:>3} {t:>6} {ch:>8} {c['freq']:>5} {c['direction']:>10} {c['consistency']:>8.2f} {c['mean_delta']:>+10.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
