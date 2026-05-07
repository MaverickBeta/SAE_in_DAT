"""
Entry-level activation rate inspection for SAE features.

Key insight: with 50 images, activation rate can only take values in {0%, 2%, 4%, ..., 100%}
(1/50 increments). All histograms align to these discrete steps.

Outputs 4 figures + 1 JSON:
  fig1: clean vs adv grouped bar chart (2% bins, log y-axis)
  fig2: delta activation rate histogram (2% bins, log y-axis, colored by sign)
  fig3: activation value violin for 0%-100% entries (10% coarse bins)
  fig4: activation value violin for 90%-100% entries (2% fine bins)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect SAE feature activation at entry (token×channel) level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clean-npy", type=str, required=True)
    parser.add_argument("--adv-npy", type=str, required=True)
    parser.add_argument("--out-dir", type=str,
                        default="/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()

    print(f"Loading clean: {args.clean_npy}")
    clean = np.load(args.clean_npy)
    print(f"Loading adv:   {args.adv_npy}")
    adv = np.load(args.adv_npy)

    if args.max_images > 0:
        clean = clean[:args.max_images]
        adv = adv[:args.max_images]

    n_images, n_tokens, d_lat = clean.shape
    total_entries = n_tokens * d_lat
    step = 1.0 / n_images
    print(f"Shape: {clean.shape}, total entries: {total_entries:,}, rate step: {step:.3f} ({n_images} images)")

    # Exact integer counts (0..n_images)
    clean_flat = clean.reshape(n_images, -1)
    adv_flat = adv.reshape(n_images, -1)
    clean_count = (clean_flat > 0).sum(axis=0)
    adv_count = (adv_flat > 0).sum(axis=0)
    clean_rate = clean_count / n_images
    adv_rate = adv_count / n_images
    delta_rate = adv_rate - clean_rate

    dead_mask = (clean_count == 0) & (adv_count == 0)
    alive_count = int((~dead_mask).sum())
    print(f"Dead entries:  {int(dead_mask.sum()):,} ({dead_mask.sum()/total_entries*100:.2f}%)")
    print(f"Alive entries: {alive_count:,} ({alive_count/total_entries*100:.2f}%)")

    out_dir = ensure_dir(args.out_dir)
    print(f"Output: {out_dir}")

    # ========================================================================
    # Fig1: Grouped bar chart (clean vs adv) with LOG y-axis
    print("Generating Fig1 (grouped bar, log y)...")
    count_bins = np.arange(-0.5, n_images + 1.5, 1)
    clean_hist, _ = np.histogram(clean_count, bins=count_bins)
    adv_hist, _ = np.histogram(adv_count, bins=count_bins)

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_images + 1)
    bar_w = 0.4
    ax.bar(x - 0.2, clean_hist, width=bar_w, color="steelblue", alpha=0.8,
           edgecolor="white", linewidth=0.3, label="Clean")
    ax.bar(x + 0.2, adv_hist, width=bar_w, color="coral", alpha=0.8,
           edgecolor="white", linewidth=0.3, label="Adv")
    tick_pos = np.arange(0, n_images + 1, 5)
    tick_lab = [f"{int(c*100/n_images)}%" for c in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_title(f"Clean vs Adv Activation Rate Distribution (Grouped, Log Y)\n({alive_count:,} alive entries)",
                 fontsize=14)
    ax.set_xlabel("Activation rate", fontsize=12)
    ax.set_ylabel("Number of entries (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_grouped_bar_log.png"), dpi=300)
    plt.close(fig)
    print("Saved: fig1_grouped_bar_log.png")

    # ========================================================================
    # Fig2: Delta rate histogram with LOG y-axis
    print("Generating Fig2 (delta hist, log y)...")
    delta_count = adv_count.astype(int) - clean_count.astype(int)
    delta_min, delta_max = int(delta_count.min()), int(delta_count.max())
    delta_bins = np.arange(delta_min - 0.5, delta_max + 1.5, 1)
    delta_hist, _ = np.histogram(delta_count, bins=delta_bins)
    delta_x = np.arange(delta_min, delta_max + 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["steelblue" if v < 0 else "coral" if v > 0 else "gray" for v in delta_x]
    ax.bar(delta_x, delta_hist, color=colors, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Delta Activation Rate Distribution (Adv - Clean, Log Y)\n({alive_count:,} alive entries)",
                 fontsize=14)
    ax.set_xlabel(f"Delta count (±{n_images} = ±100%)", fontsize=12)
    ax.set_ylabel("Number of entries (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    delta_rate_ticks = np.arange(delta_min, delta_max + 1, 5)
    delta_rate_labels = [f"{int(v*100/n_images)}%" for v in delta_rate_ticks]
    ax.set_xticks(delta_rate_ticks)
    ax.set_xticklabels(delta_rate_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_delta_hist_log.png"), dpi=300)
    plt.close(fig)
    print("Saved: fig2_delta_hist_log.png")

    # ========================================================================
    # Helper: value collection for violin plots
    def collect_violin_data(start_count, end_count):
        """Collect non-zero values for entries with count in [start_count, end_count] (inclusive).
        Returns (clean_val_data, adv_val_data, clean_entry_counts, adv_entry_counts)"""
        n_bins = end_count - start_count + 1
        c_data = [[] for _ in range(n_bins)]
        a_data = [[] for _ in range(n_bins)]
        c_entries = [0] * n_bins
        a_entries = [0] * n_bins
        for entry_idx in range(total_entries):
            cc = int(clean_count[entry_idx])
            ac = int(adv_count[entry_idx])
            c_vals = clean_flat[:, entry_idx]
            c_nz = c_vals[c_vals > 0]
            a_vals = adv_flat[:, entry_idx]
            a_nz = a_vals[a_vals > 0]
            if start_count <= cc <= end_count:
                idx = cc - start_count
                c_data[idx].extend(c_nz.tolist())
                c_entries[idx] += 1
            if start_count <= ac <= end_count:
                idx = ac - start_count
                a_data[idx].extend(a_nz.tolist())
                a_entries[idx] += 1
        return c_data, a_data, c_entries, a_entries

    def plot_violin(c_data, a_data, labels, title, fname):
        n_bins = len(labels)
        positions = np.arange(n_bins)
        c_vdata, a_vdata = [], []
        c_pos, a_pos = [], []
        valid_labels = []
        for i in range(n_bins):
            has_c = len(c_data[i]) > 0
            has_a = len(a_data[i]) > 0
            if has_c or has_a:
                valid_labels.append(labels[i])
                if has_c:
                    c_vdata.append(c_data[i])
                    c_pos.append(positions[i] - 0.25)
                if has_a:
                    a_vdata.append(a_data[i])
                    a_pos.append(positions[i] + 0.25)

        fig, ax = plt.subplots(figsize=(max(12, n_bins * 2), 7))
        if c_vdata:
            parts_c = ax.violinplot(c_vdata, positions=c_pos, widths=0.45,
                                    showmeans=True, showmedians=False, showextrema=True)
            for pc in parts_c['bodies']:
                pc.set_facecolor('steelblue')
                pc.set_alpha(0.55)
            for pn in ('cmeans', 'cbars', 'cmins', 'cmaxes'):
                parts_c[pn].set_color('steelblue')
        if a_vdata:
            parts_a = ax.violinplot(a_vdata, positions=a_pos, widths=0.45,
                                    showmeans=True, showmedians=False, showextrema=True)
            for pc in parts_a['bodies']:
                pc.set_facecolor('coral')
                pc.set_alpha(0.55)
            for pn in ('cmeans', 'cbars', 'cmins', 'cmaxes'):
                parts_a[pn].set_color('coral')

        ax.set_xticks(positions[:len(valid_labels)])
        ax.set_xticklabels(valid_labels, rotation=0)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Activation rate bin", fontsize=12)
        ax.set_ylabel("Activation value (non-zero)", fontsize=12)
        ax.legend(handles=[Patch(facecolor='steelblue', alpha=0.55, label='Clean'),
                          Patch(facecolor='coral', alpha=0.55, label='Adv')], fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close(fig)
        print(f"Saved: {fname}")

    # ========================================================================
    # Fig3: Violin plot for 0%-100% with 10% coarse bins
    print("Generating Fig3 (value violin 0%-100%, 10% bins)...")
    n_coarse = 10
    coarse_labels = [f"{i*10}%\u2011{(i+1)*10}%" for i in range(n_coarse)]
    coarse_labels[8] = "82%\u201188%"
    coarse_labels[9] = "90%\u2011100%"

    # Use count ranges matching the labels:
    # bin 0: count 1-5, bin 1: 6-10, ..., bin 8: 41-44, bin 9: 45-50
    c_val_coarse, a_val_coarse, c_ent_coarse, a_ent_coarse = [[]]*4, [[]]*4, [[]]*4, [[]]*4
    c_val_coarse = [[] for _ in range(n_coarse)]
    a_val_coarse = [[] for _ in range(n_coarse)]
    c_ent_coarse = [0] * n_coarse
    a_ent_coarse = [0] * n_coarse

    for entry_idx in range(total_entries):
        cc = int(clean_count[entry_idx])
        ac = int(adv_count[entry_idx])
        c_vals = clean_flat[:, entry_idx]
        c_nz = c_vals[c_vals > 0]
        a_vals = adv_flat[:, entry_idx]
        a_nz = a_vals[a_vals > 0]

        def count_to_bin(c):
            if c == 0:
                return -1
            if c >= 45:
                return 9
            return (c - 1) // 5

        b_c = count_to_bin(cc)
        b_a = count_to_bin(ac)
        if b_c >= 0:
            c_val_coarse[b_c].extend(c_nz.tolist())
            c_ent_coarse[b_c] += 1
        if b_a >= 0:
            a_val_coarse[b_a].extend(a_nz.tolist())
            a_ent_coarse[b_a] += 1

    for i, lab in enumerate(coarse_labels):
        print(f"  Bin {lab}: Clean entries={c_ent_coarse[i]}, vals={len(c_val_coarse[i]):,}; "
              f"Adv entries={a_ent_coarse[i]}, vals={len(a_val_coarse[i]):,}")

    plot_violin(c_val_coarse, a_val_coarse, coarse_labels,
                "Activation Value Distribution by Activation Rate Bin\n(0%-100% range, 10% coarse bins, non-zero values only)",
                "fig3_value_violin_0_100.png")

    # ========================================================================
    # Fig4: Violin plot for 90%-100% with 2% fine bins
    print("Generating Fig4 (value violin 90%-100%, 2% bins)...")
    start_90 = 45  # count 45 = 90%
    end_90 = 50    # count 50 = 100%
    fine_labels_90 = [f"{int(c*100/n_images)}%\u2011{int((c+1)*100/n_images)}%" for c in range(start_90, end_90)]

    c_val_90, a_val_90, c_ent_90, a_ent_90 = collect_violin_data(start_90, end_90)

    for i, lab in enumerate(fine_labels_90):
        print(f"  Bin {lab}: Clean entries={c_ent_90[i]}, vals={len(c_val_90[i]):,}; "
              f"Adv entries={a_ent_90[i]}, vals={len(a_val_90[i]):,}")

    plot_violin(c_val_90, a_val_90, fine_labels_90,
                "Activation Value Distribution by Activation Rate Bin\n(90%-100% range, 2% fine bins, non-zero values only)",
                "fig4_value_violin_90_100.png")

    # ========================================================================
    # JSON stats
    print("Computing JSON stats...")
    stats = {
        "config": {
            "n_images": int(n_images),
            "n_tokens": int(n_tokens),
            "d_lat": int(d_lat),
            "total_entries": int(total_entries),
            "rate_step_pct": float(100 / n_images),
        },
        "alive_entries": {
            "count": alive_count,
            "rate": float(alive_count / total_entries),
            "dead_count": int(dead_mask.sum()),
            "dead_rate": float(dead_mask.sum() / total_entries),
        },
        "clean": {
            "mean_rate": float(clean_rate.mean()),
            "median_rate": float(np.median(clean_rate)),
            "max_rate": float(clean_rate.max()),
            "min_rate": float(clean_rate.min()),
            "gt_50pct": int((clean_count > n_images * 0.5).sum()),
            "gt_80pct": int((clean_count > n_images * 0.8).sum()),
            "eq_100pct": int((clean_count == n_images).sum()),
        },
        "adv": {
            "mean_rate": float(adv_rate.mean()),
            "median_rate": float(np.median(adv_rate)),
            "max_rate": float(adv_rate.max()),
            "min_rate": float(adv_rate.min()),
            "gt_50pct": int((adv_count > n_images * 0.5).sum()),
            "gt_80pct": int((adv_count > n_images * 0.8).sum()),
            "eq_100pct": int((adv_count == n_images).sum()),
        },
        "cross_condition": {
            "clean_only": int(((clean_count > 0) & (adv_count == 0)).sum()),
            "adv_only": int(((clean_count == 0) & (adv_count > 0)).sum()),
            "both_active": int(((clean_count > 0) & (adv_count > 0)).sum()),
            "neither_active": int(dead_mask.sum()),
        },
        "violin_bins_actual_ranges": {
            "0%-10%": "count 1-5 (2%-10%)",
            "10%-20%": "count 6-10 (12%-20%)",
            "20%-30%": "count 11-15 (22%-30%)",
            "30%-40%": "count 16-20 (32%-40%)",
            "40%-50%": "count 21-25 (42%-50%)",
            "50%-60%": "count 26-30 (52%-60%)",
            "60%-70%": "count 31-35 (62%-70%)",
            "70%-80%": "count 36-40 (72%-80%)",
            "82%-88%": "count 41-44 (82%-88%)",
            "90%-100%": "count 45-50 (90%-100%)",
        },
        "violin_0_100": {},
        "violin_90_100": {},
    }

    for i, lab in enumerate(coarse_labels):
        stats["violin_0_100"][lab] = {
            "clean_entries": c_ent_coarse[i],
            "adv_entries": a_ent_coarse[i],
            "clean_values": len(c_val_coarse[i]),
            "adv_values": len(a_val_coarse[i]),
            "clean_mean": float(np.mean(c_val_coarse[i])) if c_val_coarse[i] else None,
            "adv_mean": float(np.mean(a_val_coarse[i])) if a_val_coarse[i] else None,
        }

    for i, lab in enumerate(fine_labels_90):
        stats["violin_90_100"][lab] = {
            "clean_entries": c_ent_90[i],
            "adv_entries": a_ent_90[i],
            "clean_values": len(c_val_90[i]),
            "adv_values": len(a_val_90[i]),
            "clean_mean": float(np.mean(c_val_90[i])) if c_val_90[i] else None,
            "adv_mean": float(np.mean(a_val_90[i])) if a_val_90[i] else None,
        }

    json_path = os.path.join(out_dir, "entry_inspect_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== Summary ===")
    print(f"Clean mean rate: {clean_rate.mean():.4f}, Adv mean rate: {adv_rate.mean():.4f}")
    print(f"Clean >50%: {stats['clean']['gt_50pct']}, Adv >50%: {stats['adv']['gt_50pct']}")
    print(f"Clean >80%: {stats['clean']['gt_80pct']}, Adv >80%: {stats['adv']['gt_80pct']}")
    print(f"Clean-only: {stats['cross_condition']['clean_only']}, Adv-only: {stats['cross_condition']['adv_only']}")
    print(f"Output: {out_dir}")
    print("===============\n")


if __name__ == "__main__":
    main()
