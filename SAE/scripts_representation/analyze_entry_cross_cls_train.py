#!/usr/bin/env python3
"""
Analyze cross-class sharing of selected entries (training set).

Analysis pipeline (bottom-up):
  Phase 1: Per-class basic stats
    - How many selected entries per class?
    - Suppress vs Enhance breakdown
    - Delta distribution per class

  Phase 2: Pairwise entry overlap (all selected entries)
    - How many (token, channel) entries are shared between class A and B?
    - Of the shared entries, how many have consistent direction?
    - What is the delta gap for shared entries?

  Phase 3: Top-N cross-class probe
    - For each class A's top-N entries by |delta|:
      Probe on class B's clean/adv features for freq>=92% and |delta|>1.0

Inputs:
  - results_20cls_train/*_selected_entries.json
  - untar_samples_20cls_train_latent/*.npy

Outputs:
  - per_class_stats.json
  - pairwise_overlap_matrix.png + .json
  - pairwise_direction_consistency_matrix.png
  - topN_cross_class_share_matrix.png (N=30 default)
  - cross_class_shared_entries_detail.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ── Config ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20cls_train")
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

N_CLASSES = len(CLASSES)
CLASS_NAMES = [name for _, _, name in CLASSES]
FREQ_THRESH = 0.50   # 基于 training set 的计数阈值会在运行时计算
DELTA_THRESH = 0.1
TOP_N = 30


def load_selected_entries(name):
    """Load all selected entries for a class."""
    json_path = RESULTS_DIR / f"{name}_selected_entries.json"
    with open(json_path) as f:
        data = json.load(f)
    return data["selected_entries"]


def compute_entry_stats(clean, adv, token, channel):
    """Compute clean freq, clean mean, adv mean, delta for a single entry."""
    clean_vals = clean[:, token, channel]
    adv_vals = adv[:, token, channel]
    clean_freq = (clean_vals != 0).mean()
    clean_mean = clean_vals.mean()
    adv_mean = adv_vals.mean()
    delta = adv_mean - clean_mean
    return {
        "clean_freq": float(clean_freq),
        "clean_mean": float(clean_mean),
        "adv_mean": float(adv_mean),
        "delta": float(delta),
    }


def main():
    print("=" * 80)
    print("CROSS-CLASS ENTRY SHARING ANALYSIS (Training Set)")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Per-class basic stats
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 1: Per-Class Basic Statistics")
    print("=" * 80)

    class_entries = {}
    per_class_stats = []

    for wnid, cls_idx, name in CLASSES:
        entries = load_selected_entries(name)
        n_total = len(entries)
        n_sup = sum(1 for e in entries if e["type"] == "SUPPRESS")
        n_enh = sum(1 for e in entries if e["type"] == "ENHANCE")

        deltas = np.array([e["delta"] for e in entries])
        abs_deltas = np.abs(deltas)

        stats = {
            "name": name,
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_selected": n_total,
            "n_suppress": n_sup,
            "n_enhance": n_enh,
            "pct_suppress": round(100 * n_sup / n_total, 1) if n_total > 0 else 0,
            "pct_enhance": round(100 * n_enh / n_total, 1) if n_total > 0 else 0,
            "delta_min": round(float(deltas.min()), 4) if len(deltas) else None,
            "delta_max": round(float(deltas.max()), 4) if len(deltas) else None,
            "delta_mean": round(float(deltas.mean()), 4) if len(deltas) else None,
            "delta_median": round(float(np.median(deltas)), 4) if len(deltas) else None,
            "abs_delta_mean": round(float(abs_deltas.mean()), 4) if len(abs_deltas) else None,
            "abs_delta_median": round(float(np.median(abs_deltas)), 4) if len(abs_deltas) else None,
        }
        per_class_stats.append(stats)
        class_entries[name] = entries

    # Print table
    print(f"\n{'Class':>15} {'N_sel':>6} {'Sup':>5} {'Enh':>5} {'%Sup':>6} {'%Enh':>6} "
          f"{'|Δ|mean':>8} {'|Δ|med':>8} {'Δmin':>8} {'Δmax':>8}")
    print("-" * 90)
    for s in per_class_stats:
        print(f"{s['name']:>15} {s['n_selected']:>6} {s['n_suppress']:>5} {s['n_enhance']:>5} "
              f"{s['pct_suppress']:>5.1f}% {s['pct_enhance']:>5.1f}% "
              f"{s['abs_delta_mean']:>8.3f} {s['abs_delta_median']:>8.3f} "
              f"{s['delta_min']:>+8.2f} {s['delta_max']:>+8.2f}")
    print("-" * 90)

    # Plot 1: Per-class selected entry counts (stacked bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(N_CLASSES)
    sup_counts = [s["n_suppress"] for s in per_class_stats]
    enh_counts = [s["n_enhance"] for s in per_class_stats]
    ax.bar(x, sup_counts, label="SUPPRESS", color="#1f77b4", alpha=0.85)
    ax.bar(x, enh_counts, bottom=sup_counts, label="ENHANCE", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Selected Entries", fontsize=12)
    ax.set_title(f"Selected Entries per Class (Training Set)\nTotal: {sum(s['n_selected'] for s in per_class_stats)} entries across {N_CLASSES} classes", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_class_entry_counts.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'per_class_entry_counts.png'}")

    # Plot 2: |delta| distribution per class (boxplot)
    fig, ax = plt.subplots(figsize=(14, 6))
    abs_delta_lists = []
    labels = []
    for name in CLASS_NAMES:
        entries = class_entries[name]
        if entries:
            abs_delta_lists.append([abs(e["delta"]) for e in entries])
            labels.append(name)
    bp = ax.boxplot(abs_delta_lists, labels=labels, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("|Delta| (abs value)", fontsize=12)
    ax.set_title("Distribution of |Delta| per Class (Selected Entries)", fontsize=13)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_class_abs_delta_boxplot.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'per_class_abs_delta_boxplot.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Pairwise overlap of ALL selected entries
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 2: Pairwise Entry Overlap (All Selected Entries)")
    print("=" * 80)

    # Build sets: name -> set of (token, channel)
    entry_sets = {}
    entry_direction = {}  # (name, token, channel) -> "SUPPRESS"/"ENHANCE"
    entry_delta = {}      # (name, token, channel) -> delta
    for name in CLASS_NAMES:
        entry_sets[name] = set()
        for e in class_entries[name]:
            key = (e["token"], e["channel"])
            entry_sets[name].add(key)
            entry_direction[(name, e["token"], e["channel"])] = e["type"]
            entry_delta[(name, e["token"], e["channel"])] = e["delta"]

    # Pairwise overlap matrices
    overlap_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    consistent_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    incons_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    overlap_detail = []

    for i, name_a in enumerate(CLASS_NAMES):
        for j, name_b in enumerate(CLASS_NAMES):
            if i == j:
                overlap_matrix[i, j] = len(entry_sets[name_a])
                consistent_matrix[i, j] = len(entry_sets[name_a])
                continue

            shared = entry_sets[name_a] & entry_sets[name_b]
            overlap_matrix[i, j] = len(shared)

            n_consistent = 0
            n_inconsistent = 0
            for tok, ch in shared:
                dir_a = entry_direction[(name_a, tok, ch)]
                dir_b = entry_direction[(name_b, tok, ch)]
                if dir_a == dir_b:
                    n_consistent += 1
                else:
                    n_inconsistent += 1

                # Record detail for off-diagonal
                if i < j:  # only record once per pair
                    delta_a = entry_delta[(name_a, tok, ch)]
                    delta_b = entry_delta[(name_b, tok, ch)]
                    overlap_detail.append({
                        "entry_id": f"t{tok}_c{ch}",
                        "class_a": name_a,
                        "class_b": name_b,
                        "token": int(tok),
                        "channel": int(ch),
                        "dir_a": dir_a,
                        "dir_b": dir_b,
                        "consistent": dir_a == dir_b,
                        "delta_a": round(delta_a, 6),
                        "delta_b": round(delta_b, 6),
                        "abs_gap": round(abs(delta_a - delta_b), 6),
                    })

            consistent_matrix[i, j] = n_consistent
            incons_matrix[i, j] = n_inconsistent

    # Plot 2a: Overlap share ratio heatmap (row-normalized)
    ratio_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
    for i in range(N_CLASSES):
        total = overlap_matrix[i, i]
        for j in range(N_CLASSES):
            ratio_matrix[i, j] = overlap_matrix[i, j] / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(ratio_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Class B (shared with)", fontsize=12)
    ax.set_ylabel("Class A (owner)", fontsize=12)
    ax.set_title("Pairwise Entry Share Ratio\n(diagonal = 100%, off-diagonal = shared / owner's total)", fontsize=13)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j:
                text = f"{int(overlap_matrix[i, j])}\n(100%)"
                color = "white"
            else:
                text = f"{ratio_matrix[i, j]*100:.1f}%"
                color = "white" if ratio_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Share Ratio (%)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_overlap_matrix.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'pairwise_overlap_matrix.png'}")

    # Plot 2b: Direction consistency ratio heatmap
    consistency_ratio = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if overlap_matrix[i, j] > 0:
                consistency_ratio[i, j] = consistent_matrix[i, j] / overlap_matrix[i, j]
            else:
                consistency_ratio[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(consistency_ratio, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Class B", fontsize=12)
    ax.set_ylabel("Class A", fontsize=12)
    ax.set_title("Direction Consistency Ratio of Shared Entries\n(1.0 = all shared entries have same direction)", fontsize=13)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j:
                continue
            if overlap_matrix[i, j] > 0:
                text = f"{consistency_ratio[i,j]:.2f}"
                color = "white" if consistency_ratio[i, j] > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Consistency Ratio")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_direction_consistency_matrix.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'pairwise_direction_consistency_matrix.png'}")

    # Print summary: which pairs share the most
    print(f"\n{'Pair':>35} {'Shared':>7} {'Consistent':>11} {'Inconsistent':>13} {'Consist%':>9}")
    print("-" * 80)
    pair_summaries = []
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            pair_summaries.append({
                "pair": f"{CLASS_NAMES[i]} <-> {CLASS_NAMES[j]}",
                "shared": int(overlap_matrix[i, j]),
                "consistent": int(consistent_matrix[i, j]),
                "inconsistent": int(incons_matrix[i, j]),
                "ratio": consistency_ratio[i, j],
            })
    pair_summaries.sort(key=lambda x: -x["shared"])
    for p in pair_summaries[:15]:
        print(f"{p['pair']:>35} {p['shared']:>7} {p['consistent']:>11} {p['inconsistent']:>13} {p['ratio']*100:>8.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Top-N cross-class probe (on raw features)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"PHASE 3: Top-{TOP_N} Cross-Class Probe (on raw features)")
    print("=" * 80)

    # Load raw features
    print("\nLoading clean & adversarial features...")
    class_clean_feats = {}
    class_adv_feats = {}
    n_clean_list = {}
    n_adv_list = {}
    for wnid, cls_idx, name in CLASSES:
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        class_clean_feats[name] = np.load(clean_path)
        class_adv_feats[name] = np.load(adv_path)
        n_clean_list[name] = class_clean_feats[name].shape[0]
        n_adv_list[name] = class_adv_feats[name].shape[0]
        print(f"  {name:>15s}: clean={n_clean_list[name]:>4}, adv={n_adv_list[name]:>4}")

    # Determine per-class freq threshold (92% of clean samples)
    print("\nPer-class frequency thresholds (92% of clean samples):")
    freq_count_thresh = {}
    for name in CLASS_NAMES:
        thresh = int(np.ceil(n_clean_list[name] * FREQ_THRESH))
        freq_count_thresh[name] = thresh
        print(f"  {name:>15s}: {n_clean_list[name]} clean -> threshold = {thresh} activations")

    # Build top-N lookup
    topN_by_class = {}
    for name in CLASS_NAMES:
        entries = class_entries[name]
        # Sort by |delta| descending, take top N
        sorted_entries = sorted(entries, key=lambda x: abs(x["delta"]), reverse=True)
        topN_by_class[name] = sorted_entries[:TOP_N]
        print(f"  {name:>15s}: top-{TOP_N} by |delta| (had {len(entries)} total selected)")

    # 20x20 matrices
    share_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    dir_cons_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    shared_entries_detail = []

    print("\nAnalyzing cross-class sharing of top-N entries...")
    for i, (_, _, owner_name) in enumerate(CLASSES):
        owner_entries = topN_by_class[owner_name]
        owner_lookup = {}
        for e in owner_entries:
            owner_lookup[(e["token"], e["channel"])] = e

        for j, (_, _, probe_name) in enumerate(CLASSES):
            if owner_name == probe_name:
                share_matrix[i, j] = len(owner_entries)
                dir_cons_matrix[i, j] = len(owner_entries)
                continue

            probe_clean = class_clean_feats[probe_name]
            probe_adv = class_adv_feats[probe_name]
            probe_n_clean = n_clean_list[probe_name]
            probe_freq_thresh_count = freq_count_thresh[probe_name]

            shared_count = 0
            consistent_count = 0

            for e in owner_entries:
                tok, ch = e["token"], e["channel"]
                owner_delta = e["delta"]

                # Skip if owner delta doesn't meet threshold
                if abs(owner_delta) <= DELTA_THRESH:
                    continue

                # Compute probe stats
                probe_stats = compute_entry_stats(probe_clean, probe_adv, tok, ch)
                probe_delta = probe_stats["delta"]
                probe_clean_freq = probe_stats["clean_freq"]
                probe_clean_count = int(probe_clean_freq * probe_n_clean)

                # Check probe clean freq threshold
                if probe_clean_count < probe_freq_thresh_count:
                    continue

                # Check probe delta threshold
                if abs(probe_delta) <= DELTA_THRESH:
                    continue

                shared_count += 1

                # Check direction consistency
                owner_dir = "SUPPRESS" if owner_delta < 0 else "ENHANCE"
                probe_dir = "SUPPRESS" if probe_delta < 0 else "ENHANCE"
                direction_consistent = (owner_dir == probe_dir)

                if direction_consistent:
                    consistent_count += 1

                abs_gap = abs(owner_delta - probe_delta)
                max_delta = max(abs(owner_delta), abs(probe_delta))
                rel_gap = abs_gap / max_delta if max_delta > 0 else 0.0

                shared_entries_detail.append({
                    "entry_id": f"{owner_name}_t{tok}_c{ch}",
                    "owner_class": owner_name,
                    "probe_class": probe_name,
                    "token": int(tok),
                    "channel": int(ch),
                    "owner_delta": round(float(owner_delta), 6),
                    "probe_delta": round(float(probe_delta), 6),
                    "owner_direction": owner_dir,
                    "probe_direction": probe_dir,
                    "direction_consistent": direction_consistent,
                    "abs_gap": round(float(abs_gap), 6),
                    "rel_gap": round(float(rel_gap), 6),
                    "owner_clean_freq": round(float(e["clean_count"] / n_clean_list[owner_name]), 4),
                    "probe_clean_freq": round(float(probe_clean_freq), 4),
                })

            share_matrix[i, j] = shared_count
            dir_cons_matrix[i, j] = consistent_count

    # Plot 3a: Top-N sharing heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(share_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Probe Class (whose clean samples are tested)", fontsize=12)
    ax.set_ylabel(f"Owner Class (whose top-{TOP_N} entries)", fontsize=12)
    ax.set_title(
        f"Top-{TOP_N} Entry Sharing Matrix (Training Set)\n"
        f"(clean_freq>={FREQ_THRESH*100:.0f}%, |delta|>{DELTA_THRESH})",
        fontsize=13
    )
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if share_matrix[i, j] > TOP_N * 0.5 else "black"
            ax.text(j, i, int(share_matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="# Shared Entries")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"top{TOP_N}_cross_class_share_matrix.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / f'top{TOP_N}_cross_class_share_matrix.png'}")

    # Plot 3b: Direction-consistent sharing heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(dir_cons_matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Probe Class", fontsize=12)
    ax.set_ylabel(f"Owner Class (top-{TOP_N})", fontsize=12)
    ax.set_title(
        f"Direction-Consistent Top-{TOP_N} Shared Entries\n"
        f"(clean_freq>={FREQ_THRESH*100:.0f}%, |delta|>{DELTA_THRESH}, same direction)",
        fontsize=13
    )
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if dir_cons_matrix[i, j] > TOP_N * 0.4 else "black"
            ax.text(j, i, int(dir_cons_matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="# Consistent Entries")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"top{TOP_N}_cross_class_consistent_matrix.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / f'top{TOP_N}_cross_class_consistent_matrix.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    # Save per-class stats
    with open(OUT_DIR / "per_class_stats.json", "w") as f:
        json.dump({
            "config": {"n_classes": N_CLASSES, "freq_thresh": FREQ_THRESH, "delta_thresh": DELTA_THRESH},
            "per_class": per_class_stats,
        }, f, indent=2)
    print(f"\nSaved JSON: {OUT_DIR / 'per_class_stats.json'}")

    # Save pairwise overlap
    with open(OUT_DIR / "pairwise_overlap.json", "w") as f:
        json.dump({
            "overlap_matrix": overlap_matrix.tolist(),
            "consistent_matrix": consistent_matrix.tolist(),
            "inconsistent_matrix": incons_matrix.tolist(),
            "consistency_ratio": consistency_ratio.tolist(),
            "class_names": CLASS_NAMES,
            "overlap_detail": overlap_detail,
        }, f, indent=2)
    print(f"Saved JSON: {OUT_DIR / 'pairwise_overlap.json'}")

    # Save top-N cross-class detail
    with open(OUT_DIR / f"top{TOP_N}_cross_class_shared_detail.json", "w") as f:
        json.dump({
            "config": {
                "top_n": TOP_N,
                "freq_thresh": FREQ_THRESH,
                "delta_thresh": DELTA_THRESH,
            },
            "share_matrix": share_matrix.tolist(),
            "consistent_matrix": dir_cons_matrix.tolist(),
            "class_names": CLASS_NAMES,
            "shared_entries": shared_entries_detail,
        }, f, indent=2)
    print(f"Saved JSON: {OUT_DIR / f'top{TOP_N}_cross_class_shared_detail.json'}")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_selected = sum(s["n_selected"] for s in per_class_stats)
    total_sup = sum(s["n_suppress"] for s in per_class_stats)
    total_enh = sum(s["n_enhance"] for s in per_class_stats)
    print(f"\nTotal selected entries across {N_CLASSES} classes: {total_selected}")
    print(f"  SUPPRESS: {total_sup} ({100*total_sup/total_selected:.1f}%)")
    print(f"  ENHANCE:  {total_enh} ({100*total_enh/total_selected:.1f}%)")

    # Pairwise overlap summary
    total_shared_pairs = sum(1 for d in overlap_detail if d["consistent"])
    total_incons_pairs = sum(1 for d in overlap_detail if not d["consistent"])
    print(f"\nPairwise shared entries (all selected): {len(overlap_detail)}")
    print(f"  Direction-consistent:   {total_shared_pairs}")
    print(f"  Direction-inconsistent: {total_incons_pairs}")

    # Top-N sharing summary
    total_topN_shared = len(shared_entries_detail)
    total_topN_consistent = sum(1 for e in shared_entries_detail if e["direction_consistent"])
    print(f"\nTop-{TOP_N} cross-class shared entries: {total_topN_shared}")
    print(f"  Direction-consistent: {total_topN_consistent}")

    # Top probe classes per owner
    print(f"\n{'Owner Class':>15} {'Top-N':>6} {'Total Shared':>13} {'Consistent':>12} {'Top 3 Probe Classes':>40}")
    print("-" * 90)
    for i, name in enumerate(CLASS_NAMES):
        total_s = int(share_matrix[i].sum()) - share_matrix[i, i]
        total_c = int(dir_cons_matrix[i].sum()) - dir_cons_matrix[i, i]
        probe_counts = defaultdict(int)
        for e in shared_entries_detail:
            if e["owner_class"] == name:
                probe_counts[e["probe_class"]] += 1
        top3 = sorted(probe_counts.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join([f"{p}({c})" for p, c in top3]) if top3 else "None"
        print(f"{name:>15} {len(topN_by_class[name]):>6} {total_s:>13} {total_c:>12} {top3_str:>40}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
