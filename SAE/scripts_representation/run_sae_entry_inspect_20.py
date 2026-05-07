#!/usr/bin/env python3
"""
Inspect SAE entry activation patterns across all 20 classes.

Three plots:
  1. Full-range (0%–100%) activation rate histogram, log y-axis.
  2. Zoomed (50%–100%) activation rate histogram with smoothing.
  3. Per-bin mean & median activation value (80%–100%), line plot.

Entries = ALL (token, channel) combinations in the SAE latent space (49 x 12288).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")
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

N_TOKENS = 49
N_CHANNELS = 12288

# ── Accumulate statistics without loading everything into memory ────
clean_count = np.zeros((N_TOKENS, N_CHANNELS), dtype=np.int64)
clean_sum   = np.zeros((N_TOKENS, N_CHANNELS), dtype=np.float64)
adv_count   = np.zeros((N_TOKENS, N_CHANNELS), dtype=np.int64)
adv_sum     = np.zeros((N_TOKENS, N_CHANNELS), dtype=np.float64)

clean_total = 0
adv_total   = 0

print("Accumulating clean features...")
for wnid, cls_idx, name in CLASSES:
    path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
    if not path.exists():
        continue
    feat = np.load(path)
    clean_count += (feat != 0).sum(axis=0)
    clean_sum   += feat.sum(axis=0)
    clean_total += feat.shape[0]

print(f"Total clean samples: {clean_total}")

print("Accumulating adversarial features...")
for wnid, cls_idx, name in CLASSES:
    path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
    if not path.exists():
        continue
    feat = np.load(path)
    adv_count += (feat != 0).sum(axis=0)
    adv_sum   += feat.sum(axis=0)
    adv_total += feat.shape[0]

print(f"Total adv samples: {adv_total}\n")

# ── Compute per-entry rates and mean values ─────────────────────────
clean_rate = clean_count / clean_total
clean_mean = clean_sum / clean_total

adv_rate = adv_count / adv_total
adv_mean = adv_sum / adv_total

# Flatten for plotting
clean_rate_flat = clean_rate.ravel()
clean_mean_flat = clean_mean.ravel()

adv_rate_flat = adv_rate.ravel()
adv_mean_flat = adv_mean.ravel()

# ── Helper: smooth 1-D array with moving average ────────────────────
def smooth(y, window=7):
    if len(y) < window:
        return y
    pad = window // 2
    y_padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_padded, np.ones(window) / window, mode="valid")

def find_inflection(x, y_smoothed):
    d1 = np.gradient(y_smoothed, x)
    d2 = np.gradient(d1, x)
    idx = np.argmax(np.abs(d2))
    return idx, x[idx], y_smoothed[idx]

# ═══════════════════════════════════════════════════════════════════
# Plot 1: Full-range 0%–100% activation rate histogram
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(0, 1, 101)
bin_centers = (bins[:-1] + bins[1:]) / 2

hist_clean, _ = np.histogram(clean_rate_flat, bins=bins)
hist_adv,   _ = np.histogram(adv_rate_flat,   bins=bins)

ax.semilogy(bin_centers, hist_clean + 1, label="Clean", alpha=0.8, linewidth=1.5)
ax.semilogy(bin_centers, hist_adv + 1,   label="Adversarial", alpha=0.8, linewidth=1.5)

ax.set_xlabel("Activation Rate", fontsize=12)
ax.set_ylabel("Number of entries (log scale, +1 offset)", fontsize=12)
ax.set_title("Entry Activation Rate Distribution (0%–100%)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
out_path1 = OUT_DIR / "activation_rate_distribution_full.png"
fig.savefig(out_path1, dpi=200)
plt.close(fig)
print(f"Saved: {out_path1}")

# ═══════════════════════════════════════════════════════════════════
# Plot 2: Zoomed 50%–100% activation rate histogram with smoothing
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

bins2 = np.linspace(0.50, 1.0, 101)
bin_centers2 = (bins2[:-1] + bins2[1:]) / 2

hist_clean2, _ = np.histogram(clean_rate_flat, bins=bins2)
hist_adv2,   _ = np.histogram(adv_rate_flat,   bins=bins2)

window = 7
hist_clean2_s = smooth(hist_clean2, window=window)
hist_adv2_s   = smooth(hist_adv2,   window=window)

ax.semilogy(bin_centers2, hist_clean2 + 1, ":", color="steelblue", alpha=0.5, linewidth=1, label="Clean (raw)")
ax.semilogy(bin_centers2, hist_adv2 + 1, ":", color="coral", alpha=0.5, linewidth=1, label="Adversarial (raw)")
ax.semilogy(bin_centers2, hist_clean2_s + 1, "-", color="blue", linewidth=2, label="Clean (smoothed)")
ax.semilogy(bin_centers2, hist_adv2_s + 1,   "-", color="red",  linewidth=2, label="Adversarial (smoothed)")

# Inflection detection
idx_c, x_c, y_c = find_inflection(bin_centers2, hist_clean2_s)
idx_a, x_a, y_a = find_inflection(bin_centers2, hist_adv2_s)

ax.axvline(x_c, color="blue", linestyle="--", alpha=0.7)
ax.axvline(x_a, color="red",  linestyle="--", alpha=0.7)
ax.annotate(f"Clean inflection\n{x_c:.3f}", xy=(x_c, y_c + 1),
            xytext=(x_c - 0.05, (y_c + 1) * 3),
            fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6))
ax.annotate(f"Adv inflection\n{x_a:.3f}", xy=(x_a, y_a + 1),
            xytext=(x_a + 0.01, (y_a + 1) * 3),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))

ax.set_xlabel("Activation Rate", fontsize=12)
ax.set_ylabel("Number of entries (log scale, +1 offset)", fontsize=12)
ax.set_title("Entry Activation Rate Distribution (50%–100%) with Smoothing", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.set_xlim(0.50, 1.0)

plt.tight_layout()
out_path2 = OUT_DIR / "activation_rate_distribution_50_100.png"
fig.savefig(out_path2, dpi=200)
plt.close(fig)
print(f"Saved: {out_path2}")

# ═══════════════════════════════════════════════════════════════════
# Plot 3: Per-bin mean & median activation value (80%–100%)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

bin_edges = np.linspace(0.80, 1.0, 9)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_labels = [f"{bin_edges[i]:.3f}–{bin_edges[i+1]:.3f}" for i in range(8)]

clean_bin_mean   = []
clean_bin_median = []
adv_bin_mean     = []
adv_bin_median   = []

for i in range(8):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask_c = (clean_rate_flat >= lo) & (clean_rate_flat < hi)
    mask_a = (adv_rate_flat >= lo) & (adv_rate_flat < hi)
    if i == 7:  # last bin inclusive
        mask_c = (clean_rate_flat >= lo) & (clean_rate_flat <= hi)
        mask_a = (adv_rate_flat >= lo) & (adv_rate_flat <= hi)

    vals_c = clean_mean_flat[mask_c]
    vals_a = adv_mean_flat[mask_a]

    clean_bin_mean.append(np.mean(vals_c) if len(vals_c) > 0 else np.nan)
    clean_bin_median.append(np.median(vals_c) if len(vals_c) > 0 else np.nan)
    adv_bin_mean.append(np.mean(vals_a) if len(vals_a) > 0 else np.nan)
    adv_bin_median.append(np.median(vals_a) if len(vals_a) > 0 else np.nan)

x = np.arange(1, 9)
width = 0.18

# Bars: mean
ax.bar(x - 1.5 * width, clean_bin_mean,   width, label="Clean mean",   color="steelblue", alpha=0.8)
ax.bar(x - 0.5 * width, adv_bin_mean,     width, label="Adv mean",     color="coral",     alpha=0.8)
# Bars: median
ax.bar(x + 0.5 * width, clean_bin_median, width, label="Clean median", color="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=1)
ax.bar(x + 1.5 * width, adv_bin_median,   width, label="Adv median",   color="coral",     alpha=0.4, edgecolor="coral",     linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=10)
ax.set_xlabel("Activation Rate Bin", fontsize=12)
ax.set_ylabel("Mean Activation Value (global, includes zeros)", fontsize=12)
ax.set_title("Per-Bin Mean & Median Activation Value (80%–100%)", fontsize=13)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, axis="y", ls="--", alpha=0.3)

plt.tight_layout()
out_path3 = OUT_DIR / "rate_vs_value_mean_median.png"
fig.savefig(out_path3, dpi=200)
plt.close(fig)
print(f"Saved: {out_path3}")

# ── Save numeric summary ────────────────────────────────────────────
import json
summary = {
    "clean_total_samples": int(clean_total),
    "adv_total_samples": int(adv_total),
    "inflection_points_50_100": {
        "clean": float(x_c),
        "adv": float(x_a),
    },
    "per_bin_stats": []
}

for i in range(8):
    d = {
        "bin": bin_labels[i],
        "clean_mean": float(clean_bin_mean[i]) if not np.isnan(clean_bin_mean[i]) else None,
        "clean_median": float(clean_bin_median[i]) if not np.isnan(clean_bin_median[i]) else None,
        "adv_mean": float(adv_bin_mean[i]) if not np.isnan(adv_bin_mean[i]) else None,
        "adv_median": float(adv_bin_median[i]) if not np.isnan(adv_bin_median[i]) else None,
    }
    summary["per_bin_stats"].append(d)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"50%-100% inflection: clean={x_c:.4f}, adv={x_a:.4f}")
print(f"\nPer-bin mean & median values:")
for d in summary["per_bin_stats"]:
    print(f"  {d['bin']}: clean_mean={d['clean_mean']:.3f}, clean_median={d['clean_median']:.3f}, "
          f"adv_mean={d['adv_mean']:.3f}, adv_median={d['adv_median']:.3f}")
print(f"\nDone. Output saved to: {OUT_DIR}")
