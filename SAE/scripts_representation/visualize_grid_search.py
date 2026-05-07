#!/usr/bin/env python3
"""
Visualize grid search results: marginal effects of each parameter.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

JSON_PATH = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20classes_global_entry/grid_search_summary.json")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20classes_global_entry")


def load_data():
    with open(JSON_PATH) as f:
        data = json.load(f)
    return data["results"]


def plot_nent_vs_recovery(results):
    """Scatter: N_entries vs Recovery Rate."""
    nents = [r["n_entries"] for r in results]
    recovs = [r["recovery_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(nents, recovs, alpha=0.5, s=40, c="steelblue", edgecolors="white", linewidth=0.5)

    z = np.polyfit(nents, recovs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(nents), max(nents), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2,
            label=f"Trend (corr={np.corrcoef(nents, recovs)[0,1]:.3f})")

    ax.set_xlabel("Number of Entries (N_ent)", fontsize=12)
    ax.set_ylabel("Recovery Rate (%)", fontsize=12)
    ax.set_title("More Entries = Worse Recovery", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "viz_nent_vs_recovery.png", dpi=150)
    plt.close(fig)
    print("Saved: viz_nent_vs_recovery.png")


def plot_marginal_effects(results):
    """
    For each parameter, fix all others at their median/default,
    and plot how recovery changes as this parameter varies.
    """
    params = [
        ("cf", [0.88, 0.90, 0.92]),
        ("min_classes", [8, 10, 12]),
        ("con", [0.60, 0.70, 0.80]),
        ("dir_thresh", [0.5, 1.0, 2.0]),
        ("cv", [0.50, 0.80, 1.00]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (param_name, param_vals) in enumerate(params):
        ax = axes[idx]
        means = []
        stds = []

        for v in param_vals:
            # Filter results where this param == v, average recovery
            recovs = [r["recovery_rate"] for r in results if r[param_name] == v]
            means.append(np.mean(recovs))
            stds.append(np.std(recovs))

        ax.plot(param_vals, means, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.fill_between(param_vals,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="steelblue")

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("Mean Recovery Rate (%)", fontsize=12)
        ax.set_title(f"Marginal Effect: {param_name}", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Annotate values
        for x, y in zip(param_vals, means):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    # Hide extra subplot
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "viz_marginal_effects.png", dpi=150)
    plt.close(fig)
    print("Saved: viz_marginal_effects.png")


def plot_param_heatmap(results):
    """Heatmap: con vs dir_thresh, averaged over other params."""
    con_vals = sorted(set(r["con"] for r in results))
    dir_vals = sorted(set(r["dir_thresh"] for r in results))

    matrix = np.zeros((len(con_vals), len(dir_vals)))
    for i, con in enumerate(con_vals):
        for j, dir_t in enumerate(dir_vals):
            recovs = [r["recovery_rate"] for r in results if r["con"] == con and r["dir_thresh"] == dir_t]
            matrix[i, j] = np.mean(recovs)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=2.5, vmax=3.5)
    ax.set_xticks(range(len(dir_vals)))
    ax.set_yticks(range(len(con_vals)))
    ax.set_xticklabels(dir_vals)
    ax.set_yticklabels(con_vals)
    ax.set_xlabel("dir_thresh", fontsize=12)
    ax.set_ylabel("consistency (con)", fontsize=12)
    ax.set_title("Avg Recovery: con vs dir_thresh", fontsize=13)

    for i in range(len(con_vals)):
        for j in range(len(dir_vals)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Recovery Rate (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "viz_heatmap_con_dir.png", dpi=150)
    plt.close(fig)
    print("Saved: viz_heatmap_con_dir.png")


def main():
    results = load_data()
    print(f"Loaded {len(results)} results")
    print(f"Recovery range: {min(r['recovery_rate'] for r in results):.2f}% ~ {max(r['recovery_rate'] for r in results):.2f}%")
    print(f"N_ent range: {min(r['n_entries'] for r in results)} ~ {max(r['n_entries'] for r in results)}")
    print(f"Correlation(N_ent, recovery): {np.corrcoef([r['n_entries'] for r in results], [r['recovery_rate'] for r in results])[0,1]:.3f}")
    print()

    plot_nent_vs_recovery(results)
    plot_marginal_effects(results)
    plot_param_heatmap(results)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
