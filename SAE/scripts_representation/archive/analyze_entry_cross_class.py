#!/usr/bin/env python3
"""
Analyze cross-class sharing of top-30 candidate entries.

For each class A's top-30 entries:
  For each other class B:
    1. Check if class B's CLEAN samples activate this entry with freq >= 92%
    2. Check if both class A and class B have |delta| > 1.0 for this entry
    3. If both pass, check direction consistency and compute delta gaps

Outputs:
  1. 20x20 heatmap of shared entry counts
  2. JSON with detailed sharing info per entry
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")
RESULTS_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20classes")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect_20")
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
FREQ_THRESH = 0.92   # 46/50 clean images
DELTA_THRESH = 1.0   # |delta| > 1.0


def load_top30_entries(name):
    """Load top-30 entries by |delta|."""
    json_path = RESULTS_DIR / f"{name}_selected_entries.json"
    with open(json_path) as f:
        data = json.load(f)
    entries = sorted(data["selected_entries"], key=lambda x: abs(x["delta"]), reverse=True)
    return entries[:30]


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
    print("Loading top-30 entries for all classes...")
    class_top30 = {}
    for wnid, cls_idx, name in CLASSES:
        class_top30[name] = load_top30_entries(name)
        print(f"  {name:>15s}: {len(class_top30[name])} entries")

    print("\nLoading clean & adversarial features...")
    class_clean_feats = {}
    class_adv_feats = {}
    for wnid, cls_idx, name in CLASSES:
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        class_clean_feats[name] = np.load(clean_path)
        class_adv_feats[name] = np.load(adv_path)
        print(f"  {name:>15s}: clean={class_clean_feats[name].shape[0]}, adv={class_adv_feats[name].shape[0]}")

    # Build lookup: (name, token, channel) -> delta from selected_entries.json
    # This gives us the "owner" delta directly
    owner_delta_lookup = {}
    for name in CLASS_NAMES:
        for e in class_top30[name]:
            key = (name, e["token"], e["channel"])
            owner_delta_lookup[key] = {
                "delta": e["delta"],
                "clean_mean": e["clean_mean"],
                "adv_mean": e["adv_mean"],
                "clean_count": e["clean_count"],
            }

    # 20x20 sharing matrix
    share_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    # 20x20 matrix for entries with consistent direction
    consistent_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

    shared_entries_detail = []

    print("\nAnalyzing cross-class sharing...")

    for i, (_, _, owner_name) in enumerate(CLASSES):
        owner_entries = class_top30[owner_name]

        for j, (_, _, probe_name) in enumerate(CLASSES):
            if owner_name == probe_name:
                # Diagonal: all top-30 entries are by definition from clean_freq>=92% pool
                share_matrix[i, j] = len(owner_entries)
                consistent_matrix[i, j] = len(owner_entries)
                continue

            probe_clean = class_clean_feats[probe_name]
            probe_adv = class_adv_feats[probe_name]

            shared_count = 0
            consistent_count = 0

            for e in owner_entries:
                tok, ch = e["token"], e["channel"]

                # Get owner delta from lookup
                owner_info = owner_delta_lookup[(owner_name, tok, ch)]
                owner_delta = owner_info["delta"]

                # Skip if owner delta doesn't meet threshold
                if abs(owner_delta) <= DELTA_THRESH:
                    continue

                # Compute probe stats for this entry
                probe_stats = compute_entry_stats(probe_clean, probe_adv, tok, ch)
                probe_delta = probe_stats["delta"]
                probe_clean_freq = probe_stats["clean_freq"]

                # Check clean freq threshold
                if probe_clean_freq < FREQ_THRESH:
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

                # Record detail
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
                    "owner_clean_freq": round(float(e["clean_count"] / 50.0), 4),  # approx
                    "probe_clean_freq": round(float(probe_clean_freq), 4),
                })

            share_matrix[i, j] = shared_count
            consistent_matrix[i, j] = consistent_count

    # ── Plot 1: Sharing Count Heatmap ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(share_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Probe Class (whose clean samples are tested)", fontsize=12)
    ax.set_ylabel("Owner Class (whose top-30 entries)", fontsize=12)
    ax.set_title(
        f"Shared Top-30 Entries Matrix\n"
        f"(clean_freq>={FREQ_THRESH*100:.0f}%, |delta|>{DELTA_THRESH})",
        fontsize=13
    )

    # Annotate each cell with count
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = ax.text(j, i, int(share_matrix[i, j]),
                           ha="center", va="center", color="black" if share_matrix[i, j] < 15 else "white",
                           fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("# Shared Entries", fontsize=11)

    plt.tight_layout()
    out_heatmap = OUT_DIR / "cross_class_share_matrix.png"
    fig.savefig(out_heatmap, dpi=200)
    plt.close(fig)
    print(f"\nSaved heatmap: {out_heatmap}")

    # ── Plot 2: Direction-Consistent Sharing Heatmap ───────────────────
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(consistent_matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Probe Class (whose clean samples are tested)", fontsize=12)
    ax.set_ylabel("Owner Class (whose top-30 entries)", fontsize=12)
    ax.set_title(
        f"Direction-Consistent Shared Entries Matrix\n"
        f"(clean_freq>={FREQ_THRESH*100:.0f}%, |delta|>{DELTA_THRESH}, same direction)",
        fontsize=13
    )

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = ax.text(j, i, int(consistent_matrix[i, j]),
                           ha="center", va="center", color="black" if consistent_matrix[i, j] < 10 else "white",
                           fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("# Consistent Entries", fontsize=11)

    plt.tight_layout()
    out_consistent = OUT_DIR / "cross_class_consistent_matrix.png"
    fig.savefig(out_consistent, dpi=200)
    plt.close(fig)
    print(f"Saved consistent heatmap: {out_consistent}")

    # ── Save JSON ──────────────────────────────────────────────────────
    out_json = OUT_DIR / "cross_class_shared_entries.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "freq_thresh": FREQ_THRESH,
                "delta_thresh": DELTA_THRESH,
                "n_classes": N_CLASSES,
            },
            "share_matrix": share_matrix.tolist(),
            "consistent_matrix": consistent_matrix.tolist(),
            "class_names": CLASS_NAMES,
            "shared_entries": shared_entries_detail,
        }, f, indent=2)
    print(f"Saved JSON: {out_json}")

    # ── Print Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Class Top-30 Entry Sharing")
    print("=" * 70)

    print(f"\nTotal shared (entry, probe_class) pairs: {len(shared_entries_detail)}")

    consistent_pairs = [e for e in shared_entries_detail if e["direction_consistent"]]
    print(f"Direction-consistent pairs: {len(consistent_pairs)}")

    # Per-owner summary
    print(f"\n{'Owner Class':>15s} {'Total Shared':>12s} {'Consistent':>12s} {'Top Probes (shared)':>30s}")
    print("-" * 75)
    for i, name in enumerate(CLASS_NAMES):
        total_shared = int(share_matrix[i].sum()) - share_matrix[i, i]  # exclude diagonal
        total_consistent = int(consistent_matrix[i].sum()) - consistent_matrix[i, i]

        # Find top probe classes by shared count
        probe_counts = defaultdict(int)
        for e in shared_entries_detail:
            if e["owner_class"] == name:
                probe_counts[e["probe_class"]] += 1
        top_probes = sorted(probe_counts.items(), key=lambda x: -x[1])[:3]
        top_probe_str = ", ".join([f"{p}({c})" for p, c in top_probes]) if top_probes else "None"

        print(f"{name:>15s} {total_shared:>12d} {total_consistent:>12d} {top_probe_str:>30s}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
