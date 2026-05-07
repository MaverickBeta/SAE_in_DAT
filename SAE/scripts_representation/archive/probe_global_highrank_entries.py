#!/usr/bin/env python3
"""
Probe: find entries that are BOTH high-freq across classes AND high-rank (high |delta|)
within each class.

This answers: do "global" entries also have strong causal impact per-class?

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python probe_global_highrank_entries.py
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20classes")

CLASSES = [
    "tench", "brambling", "bullfrog", "peacock", "tusker",
    "sea_lion", "tabby_cat", "leopard", "tiger", "ladybug",
    "coffee_mug", "computer_keyboard", "drum", "golf_ball", "miniskirt",
    "parachute", "school_bus", "sports_car", "artichoke", "orange",
]


def main():
    # ── Load all per-class selected entries ────────────────────────
    all_entries_by_class = {}
    for name in CLASSES:
        path = RESULTS_DIR / f"{name}_selected_entries.json"
        with open(path) as f:
            data = json.load(f)
        # Sort by |delta| descending to get rank
        entries = sorted(data["selected_entries"], key=lambda x: abs(x["delta"]), reverse=True)
        all_entries_by_class[name] = entries

    # ── Build per-entry cross-class stats ──────────────────────────
    # entry_stats[(token, channel)] = {
    #   "freq": int,
    #   "classes": [names],
    #   "ranks": [ranks],
    #   "deltas": [deltas],
    #   "top5_count": int,
    #   "top10_count": int,
    #   "top20_count": int,
    #   "top50_count": int,
    # }
    entry_stats = defaultdict(lambda: {
        "freq": 0, "classes": [], "ranks": [],
        "deltas": [], "abs_deltas": [],
        "top5_count": 0, "top10_count": 0,
        "top20_count": 0, "top50_count": 0,
    })

    for name in CLASSES:
        entries = all_entries_by_class[name]
        for rank, e in enumerate(entries, 1):
            key = (e["token"], e["channel"])
            st = entry_stats[key]
            st["freq"] += 1
            st["classes"].append(name)
            st["ranks"].append(rank)
            st["deltas"].append(e["delta"])
            st["abs_deltas"].append(abs(e["delta"]))
            if rank <= 5:
                st["top5_count"] += 1
            if rank <= 10:
                st["top10_count"] += 1
            if rank <= 20:
                st["top20_count"] += 1
            if rank <= 50:
                st["top50_count"] += 1

    # ── Define filters ─────────────────────────────────────────────
    # Filter A: high freq only (baseline)
    def filter_high_freq(st):
        return st["freq"] >= 8

    # Filter B: high freq + strong rank (in at least half of its classes, rank <= 50)
    def filter_freq_and_rank(st):
        if st["freq"] < 8:
            return False
        # require rank <= 50 in at least 50% of the classes it appears in
        return st["top50_count"] >= st["freq"] * 0.5

    # Filter C: even stricter (rank <= 20 in at least 50% of classes)
    def filter_freq_and_rank_strict(st):
        if st["freq"] < 8:
            return False
        return st["top20_count"] >= st["freq"] * 0.5

    # ── Evaluate each filter ───────────────────────────────────────
    filters = [
        ("High freq only (freq>=8)", filter_high_freq),
        ("Freq>=8 + top50 in >=50% classes", filter_freq_and_rank),
        ("Freq>=8 + top20 in >=50% classes", filter_freq_and_rank_strict),
    ]

    print("=" * 80)
    print("GLOBAL + HIGH-RANK ENTRY PROBE")
    print("=" * 80)
    print(f"Total unique entries across all classes: {len(entry_stats)}")
    print(f"Classes: {len(CLASSES)}")
    print()

    for label, filt in filters:
        matches = {k: v for k, v in entry_stats.items() if filt(v)}
        print(f"\n{'=' * 80}")
        print(f"FILTER: {label}")
        print(f"Matched entries: {len(matches)}")
        print("-" * 80)

        if not matches:
            print("  (no entries match)")
            continue

        # Sort by (freq, mean_abs_delta) descending
        sorted_matches = sorted(
            matches.items(),
            key=lambda x: (x[1]["freq"], sum(x[1]["abs_deltas"]) / len(x[1]["abs_deltas"])),
            reverse=True
        )

        print(f"{'Token':>6} {'Channel':>8} {'Freq':>5} {'Mean|Δ|':>8} {'MeanRank':>9} "
              f"{'Top5':>5} {'Top10':>6} {'Top20':>6} {'Top50':>6}")
        print("-" * 80)
        for (tok, ch), st in sorted_matches[:20]:
            mean_abs_d = sum(st["abs_deltas"]) / len(st["abs_deltas"])
            mean_rank = sum(st["ranks"]) / len(st["ranks"])
            print(f"{tok:>6} {ch:>8} {st['freq']:>5} {mean_abs_d:>8.3f} {mean_rank:>9.1f} "
                  f"{st['top5_count']:>5} {st['top10_count']:>6} {st['top20_count']:>6} {st['top50_count']:>6}")

        # Distribution of top50 ratio
        print(f"\nDistribution of top50 ratio (top50_count / freq):")
        ratios = []
        for st in matches.values():
            ratios.append(st["top50_count"] / st["freq"])
        import numpy as np
        print(f"  Min: {min(ratios):.2f}, Max: {max(ratios):.2f}, Mean: {np.mean(ratios):.2f}, Median: {np.median(ratios):.2f}")

    # ── Bonus: show top entries by cross-class mean |Δ| ────────────
    print(f"\n{'=' * 80}")
    print("BONUS: Top 20 entries by cross-class mean |Δ| (regardless of freq)")
    print(f"{'=' * 80}")

    all_sorted = sorted(
        entry_stats.items(),
        key=lambda x: sum(x[1]["abs_deltas"]) / len(x[1]["abs_deltas"]),
        reverse=True
    )

    print(f"{'Token':>6} {'Channel':>8} {'Freq':>5} {'Mean|Δ|':>8} {'MeanRank':>9} "
          f"{'Top5':>5} {'Top10':>6} {'Top20':>6} {'Top50':>6}")
    print("-" * 80)
    for (tok, ch), st in all_sorted[:20]:
        mean_abs_d = sum(st["abs_deltas"]) / len(st["abs_deltas"])
        mean_rank = sum(st["ranks"]) / len(st["ranks"])
        print(f"{tok:>6} {ch:>8} {st['freq']:>5} {mean_abs_d:>8.3f} {mean_rank:>9.1f} "
              f"{st['top5_count']:>5} {st['top10_count']:>6} {st['top20_count']:>6} {st['top50_count']:>6}")

    print("\nDone!")


if __name__ == "__main__":
    main()
