#!/usr/bin/env python3
"""
Quick probe: load 5 representative classes, run selected_entry_inspect logic,
and print cross-class entry frequency statistics (Level A & Level B).

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python quick_probe_entries.py
"""

import os
import numpy as np
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ── Config ──────────────────────────────────────────────────────────
FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")

# 5 representative classes for quick probe
CLASSES = [
    ("n02077923", 150, "sea_lion"),
    ("n02129604", 292, "tiger"),
    ("n02123045", 281, "tabby_cat"),
    ("n04146614", 779, "school_bus"),
    ("n07747607", 950, "orange"),
]

THRESH_COUNT = 46      # clean >= 92%
SEL_LOW = -1.0         # suppression threshold
SEL_HIGH = 0.2         # enhancement threshold


def load_features(wnid, cls_idx):
    """Load clean and adv feature npy for a class."""
    clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
    adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"

    clean = np.load(clean_path)   # (50, 49, 12288)
    adv = np.load(adv_path)
    return clean, adv


def select_entries(clean, adv):
    """
    Run selected_entry_inspect logic.
    Returns list of dicts with keys: token, channel, clean_count, adv_count,
    clean_mean, adv_mean, delta, type ('SUPPRESS' or 'ENHANCE').
    """
    n_img, n_tok, n_ch = clean.shape

    # Per-entry nonzero counts
    clean_count = np.sum(clean != 0, axis=0)   # (49, 12288)
    adv_count = np.sum(adv != 0, axis=0)

    # Per-entry mean of nonzero values
    clean_sum = np.sum(clean, axis=0)
    adv_sum = np.sum(adv, axis=0)

    clean_mean = np.zeros_like(clean_sum, dtype=np.float32)
    adv_mean = np.zeros_like(adv_sum, dtype=np.float32)

    mask_c = clean_count > 0
    mask_a = adv_count > 0
    clean_mean[mask_c] = clean_sum[mask_c] / clean_count[mask_c]
    adv_mean[mask_a] = adv_sum[mask_a] / adv_count[mask_a]

    # Filter: clean >= 92%
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
                "clean_mean": cm,
                "adv_mean": am,
                "delta": delta,
                "type": "SUPPRESS" if delta < 0 else "ENHANCE",
            })

    # Sort by delta (most negative first)
    entries.sort(key=lambda x: x["delta"])
    return entries


def main():
    print("=" * 70)
    print("QUICK PROBE: Selected Entries for 5 Classes")
    print("=" * 70)
    print(f"Threshold: clean_count >= {THRESH_COUNT} ({THRESH_COUNT/50*100:.0f}%)")
    print(f"Selection: delta < {SEL_LOW}  or  delta > {SEL_HIGH}")
    print()

    all_entries = {}          # class_name -> list of entries
    level_a_counter = Counter()   # (token, channel) -> count
    level_b_counter = Counter()   # channel -> count
    suppress_counter = Counter()  # (token, channel) -> suppression count
    enhance_counter = Counter()   # (token, channel) -> enhancement count

    # ── Step 1: Per-class selection ─────────────────────────────────
    for wnid, cls_idx, name in CLASSES:
        print(f"Loading {name} ({wnid}) ...", end=" ")
        clean, adv = load_features(wnid, cls_idx)
        entries = select_entries(clean, adv)
        all_entries[name] = entries
        print(f"Selected: {len(entries)}  (SUPPRESS: {sum(1 for e in entries if e['type']=='SUPPRESS')}, ENHANCE: {sum(1 for e in entries if e['type']=='ENHANCE')})")

        for e in entries:
            key_a = (e["token"], e["channel"])
            key_b = e["channel"]
            level_a_counter[key_a] += 1
            level_b_counter[key_b] += 1
            if e["type"] == "SUPPRESS":
                suppress_counter[key_a] += 1
            else:
                enhance_counter[key_a] += 1

    print()

    # ── Step 2: Per-class detail table ──────────────────────────────
    print("=" * 70)
    print("PER-CLASS SUMMARY")
    print("=" * 70)
    print(f"{'Class':>15} {'Total':>8} {'Suppress':>10} {'Enhance':>10} {'% of 600K':>12}")
    print("-" * 70)
    total_entries_all = 49 * 12288
    for name in [c[2] for c in CLASSES]:
        entries = all_entries[name]
        n_sup = sum(1 for e in entries if e["type"] == "SUPPRESS")
        n_enh = sum(1 for e in entries if e["type"] == "ENHANCE")
        pct = len(entries) / total_entries_all * 100
        print(f"{name:>15} {len(entries):>8} {n_sup:>10} {n_enh:>10} {pct:>11.4f}%")
    print()

    # ── Step 3: Level A (strict token+channel) overlap ──────────────
    print("=" * 70)
    print("LEVEL A: STRICT (token, channel) OVERLAP")
    print("=" * 70)
    print(f"{'Rank':>5} {'Token':>6} {'Channel':>8} {'Freq':>6} {'Suppress':>10} {'Enhance':>10} {'Dominant':>10}")
    print("-" * 70)

    # Sort by frequency, then break ties by suppress+enhance consistency
    sorted_a = []
    for key, freq in level_a_counter.items():
        sup = suppress_counter.get(key, 0)
        enh = enhance_counter.get(key, 0)
        dom = "SUPPRESS" if sup > enh else "ENHANCE"
        sorted_a.append((freq, key, sup, enh, dom))

    sorted_a.sort(reverse=True, key=lambda x: (x[0], max(x[2], x[3])))

    for rank, (freq, key, sup, enh, dom) in enumerate(sorted_a[:10], 1):
        token, channel = key
        print(f"{rank:>5} {token:>6} {channel:>8} {freq:>6} {sup:>10} {enh:>10} {dom:>10}")

    print()

    # Frequency distribution histogram for Level A
    print("Level A frequency distribution:")
    for f in range(1, len(CLASSES) + 1):
        n = sum(1 for v in level_a_counter.values() if v == f)
        bar = "█" * n
        print(f"  {f:>2}/5 classes: {n:>4} entries  {bar}")
    print()

    # ── Step 4: Level B (channel-only) overlap ──────────────────────
    print("=" * 70)
    print("LEVEL B: CHANNEL-ONLY OVERLAP")
    print("=" * 70)
    print(f"{'Rank':>5} {'Channel':>8} {'Freq':>6} {'#Entries':>10}")
    print("-" * 70)

    sorted_b = sorted(level_b_counter.items(), key=lambda x: x[1], reverse=True)
    for rank, (ch, freq) in enumerate(sorted_b[:10], 1):
        # Count how many distinct (token, ch) pairs contributed to this channel
        n_pairs = sum(1 for key, f in level_a_counter.items() if key[1] == ch)
        print(f"{rank:>5} {ch:>8} {freq:>6} {n_pairs:>10}")

    print()

    # Frequency distribution for Level B
    print("Level B frequency distribution:")
    for f in range(1, len(CLASSES) + 1):
        n = sum(1 for v in level_b_counter.values() if v == f)
        bar = "█" * (n // 5)  # scale down for display
        print(f"  {f:>2}/5 classes: {n:>4} channels  {bar}")
    print()

    # ── Step 5: Consistent direction entries ────────────────────────
    print("=" * 70)
    print("CONSISTENT DIRECTION ENTRIES (Level A)")
    print("=" * 70)

    consistent = []
    for key, freq in level_a_counter.items():
        sup = suppress_counter.get(key, 0)
        enh = enhance_counter.get(key, 0)
        if sup > 0 and enh > 0:
            continue  # inconsistent
        if sup >= 2 or enh >= 2:  # appear in at least 2 classes with same direction
            consistent.append((freq, key, sup, enh, "SUPPRESS" if sup > 0 else "ENHANCE"))

    consistent.sort(reverse=True, key=lambda x: x[0])
    print(f"Total consistent entries (same dir in >=2 classes): {len(consistent)}")
    print()
    print(f"{'Rank':>5} {'Token':>6} {'Channel':>8} {'Freq':>6} {'Dir':>10}")
    print("-" * 70)
    for rank, (freq, key, sup, enh, d) in enumerate(consistent[:10], 1):
        token, channel = key
        print(f"{rank:>5} {token:>6} {channel:>8} {freq:>6} {d:>10}")

    print()
    print("=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
