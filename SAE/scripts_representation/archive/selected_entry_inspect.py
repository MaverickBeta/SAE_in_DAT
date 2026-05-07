"""
Selected Entry Inspection:
  Fig1: All entries with |Δ|>=0.1 only (remove dense middle), sorted by delta
  Fig2: Fine-grained histogram (100 bins)
  Fig3: Selected entries (Δ<-1 or Δ>0.2) for ablation candidates
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_FEAT = SCRIPT_DIR / "../results_representation/features/clean_stage3_k256_features.npy"
ADV_FEAT   = SCRIPT_DIR / "../results_representation/features/adv_stage3_k256_l2eps3_features.npy"
OUT_DIR    = SCRIPT_DIR / "../results_representation/entry_inspect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH_COUNT = 46   # 92% of 50
TAIL_THRESH  = 0.1  # for Fig1: only keep |Δ| >= 0.1 to remove dense middle
SEL_LOW  = -1.0     # for Fig3: strong suppression
SEL_HIGH = 0.2      # for Fig3: notable enhancement

# ── Load ────────────────────────────────────────────────────────────
print("Loading features...")
clean = np.load(CLEAN_FEAT)   # (50, 49, 12288)
adv   = np.load(ADV_FEAT)
N_IMG, N_TOK, N_CH = clean.shape

# ── Per-entry nonzero counts & mean values ──────────────────────────
clean_count = np.sum(clean != 0, axis=0)
adv_count   = np.sum(adv   != 0, axis=0)

clean_sum = np.sum(clean, axis=0)
adv_sum   = np.sum(adv,   axis=0)

clean_mean = np.zeros_like(clean_sum, dtype=np.float32)
adv_mean   = np.zeros_like(adv_sum,   dtype=np.float32)
mask_c = clean_count > 0
mask_a = adv_count   > 0
clean_mean[mask_c] = clean_sum[mask_c] / clean_count[mask_c]
adv_mean[mask_a]   = adv_sum[mask_a]   / adv_count[mask_a]

# ── Filter: clean >= 92% ────────────────────────────────────────────
mask_high = clean_count >= THRESH_COUNT
n_high = np.sum(mask_high)
print(f"Clean >= 92% entries: {n_high}")

tok_idx, ch_idx = np.where(mask_high)

entries = []
for t, c in zip(tok_idx, ch_idx):
    cc = int(clean_count[t, c])
    ac = int(adv_count[t, c])
    cm = float(clean_mean[t, c])
    am = float(adv_mean[t, c]) if ac > 0 else 0.0
    delta = am - cm
    entries.append({
        'token': int(t),
        'channel': int(c),
        'clean_count': cc,
        'adv_count': ac,
        'clean_rate': cc / N_IMG,
        'adv_rate': ac / N_IMG,
        'clean_mean': cm,
        'adv_mean': am,
        'delta': delta,
    })

# Sort by delta (most negative first)
entries.sort(key=lambda x: x['delta'])
deltas = np.array([e['delta'] for e in entries])

# ── Fig1: Tail entries only (|Δ| >= 0.1), remove dense middle ──────
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

tail_entries = [e for e in entries if abs(e['delta']) >= TAIL_THRESH]
tail_deltas = np.array([e['delta'] for e in tail_entries])
print(f"\nFig1: Entries with |Δ| >= {TAIL_THRESH}: {len(tail_entries)} (removed {len(entries)-len(tail_entries)} dense middle entries)")

ax = axes[0]
x = np.arange(len(tail_entries))
colors = ['#d62728' if d > 0 else '#1f77b4' for d in tail_deltas]
bars = ax.bar(x, tail_deltas, color=colors, width=0.9, edgecolor='none', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Annotate top suppression candidates (leftmost)
for i in range(min(8, len(tail_entries))):
    e = tail_entries[i]
    if e['delta'] < -0.5:
        ax.annotate(f"ch{e['channel']}\nt{e['token']}",
                    xy=(i, e['delta']), xytext=(i, e['delta'] - 0.25),
                    fontsize=5, ha='center', color='#1f77b4',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.4))

# Annotate top enhancement candidates (rightmost)
for i in range(1, min(9, len(tail_entries)+1)):
    e = tail_entries[-i]
    if e['delta'] > 0.2:
        ax.annotate(f"ch{e['channel']}\nt{e['token']}",
                    xy=(len(tail_entries)-i, e['delta']), xytext=(len(tail_entries)-i, e['delta'] + 0.12),
                    fontsize=5, ha='center', color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.4))

ax.set_xlabel(f'Entry Index (sorted by Δ, |Δ|>={TAIL_THRESH} only)')
ax.set_ylabel('Δ (adv_mean - clean_mean)')
ax.set_title(f'Fig 1: Tail Entries Only |Δ|>={TAIL_THRESH} (N={len(tail_entries)}), Dense Middle Removed')
ax.set_xlim(-1, len(tail_entries))

# ── Fig 2: Fine-grained histogram (100 bins) ────────────────────────
ax = axes[1]
bins_fine = np.linspace(deltas.min(), deltas.max(), 100)
n_hist, bins_hist, patches = ax.hist(deltas, bins=bins_fine, color='steelblue', edgecolor='white', alpha=0.8)

# Color the extreme tails differently
for i, (patch, left_edge) in enumerate(zip(patches, bins_hist[:-1])):
    if left_edge < SEL_LOW:
        patch.set_facecolor('#1f77b4')
    elif left_edge > SEL_HIGH:
        patch.set_facecolor('#d62728')

ax.axvline(0, color='black', linewidth=1.0, linestyle='-')
ax.axvline(np.median(deltas), color='green', linestyle='--', linewidth=1.2, label=f'median={np.median(deltas):.3f}')
ax.axvline(SEL_LOW, color='blue', linestyle='--', linewidth=1.0, label=f'suppression threshold={SEL_LOW}')
ax.axvline(SEL_HIGH, color='red', linestyle='--', linewidth=1.0, label=f'enhancement threshold={SEL_HIGH}')
ax.set_xlabel('Δ (adv_mean - clean_mean)')
ax.set_ylabel('Count')
ax.set_title('Fig 2: Fine-Grained Δ Distribution (100 bins)')
ax.legend(loc='upper left')

# Denser x-axis ticks
import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.tick_params(axis='x', which='both', labelsize=7)
ax.grid(axis='x', which='major', linestyle='--', alpha=0.3)

# ── Fig 3: Selected ablation candidates (Δ<-1 or Δ>0.2) ────────────
ax = axes[2]
sel_entries = [e for e in entries if e['delta'] < SEL_LOW or e['delta'] > SEL_HIGH]
sel_deltas = np.array([e['delta'] for e in sel_entries])

x_sel = np.arange(len(sel_entries))
colors_sel = ['#d62728' if d > 0 else '#1f77b4' for d in sel_deltas]

bars = ax.bar(x_sel, sel_deltas, color=colors_sel, width=0.85, edgecolor='black', linewidth=0.4, alpha=0.9)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax.axhline(SEL_LOW, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(SEL_HIGH, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xticks(x_sel)
labels = []
for e in sel_entries:
    if e['delta'] < SEL_LOW:
        labels.append(f"ch{e['channel']}\n▼")
    else:
        labels.append(f"ch{e['channel']}\n▲")
ax.set_xticklabels(labels, rotation=90, fontsize=6, ha='center')

ax.set_xlabel('Feature Index (channel)')
ax.set_ylabel('Δ (adv_mean - clean_mean)')
ax.set_title(f'Fig 3: Ablation Candidates (Δ<{SEL_LOW} or Δ>{SEL_HIGH}), N={len(sel_entries)}')
ax.set_xlim(-1, len(sel_entries))

for i, e in enumerate(sel_entries):
    if e['delta'] < -1.5 or e['delta'] > 0.35:
        ax.text(i, e['delta'] + (0.08 if e['delta'] > 0 else -0.08),
                f"t{e['token']}", ha='center', va='bottom' if e['delta'] > 0 else 'top',
                fontsize=5, color='black')

plt.tight_layout()
outfig = OUT_DIR / "selected_entry_inspect.png"
plt.savefig(outfig, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {outfig}")
plt.close()

# ── Save JSON ───────────────────────────────────────────────────────
json_data = {
    'threshold': 'clean_count >= 46 (92%)',
    'n_total_entries': len(entries),
    'tail_threshold': TAIL_THRESH,
    'n_tail_entries': len(tail_entries),
    'selection_criteria': {'low': SEL_LOW, 'high': SEL_HIGH},
    'n_selected': len(sel_entries),
    'delta_stats': {
        'min': float(deltas.min()), 'max': float(deltas.max()),
        'mean': float(deltas.mean()), 'median': float(np.median(deltas)),
        'std': float(np.std(deltas)),
    },
    'selected_entries': sel_entries,
}

json_path = OUT_DIR / "selected_entry_inspect.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"JSON saved: {json_path}")

print(f"\n{'='*70}")
print(f"ABLATION CANDIDATES (Δ < {SEL_LOW} or Δ > {SEL_HIGH})")
print(f"{'='*70}")
print(f"{'#':>3} {'Tok':>3} {'Ch':>6} {'Cleanμ':>8} {'Advμ':>8} {'Δ':>8} {'Adv%':>5}  {'Type':>12}")
print(f"{'-'*70}")
for i, e in enumerate(sel_entries):
    typ = 'SUPPRESS' if e['delta'] < 0 else 'ENHANCE'
    print(f"{i+1:>3} {e['token']:>3} {e['channel']:>6} {e['clean_mean']:>8.2f} {e['adv_mean']:>8.2f} {e['delta']:>+8.2f} {e['adv_rate']:>4.0%}  {typ:>12}")

print("\nDone!")
