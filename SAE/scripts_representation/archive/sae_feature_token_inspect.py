#!/usr/bin/env python3
"""
Token-level inspection for selected SAE features.

Loads raw (n_images, n_tokens, d_lat) features from sae_feature_extract.py,
selects specified feature indices, reshapes tokens to 2D spatial maps (H, W),
and draws side-by-side heatmaps: clean vs adversarial.

No model inference is performed here.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw token-level heatmaps for selected SAE features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clean-npy",
        type=str,
        required=True,
        help="Path to clean features .npy from sae_feature_extract.py.",
    )
    parser.add_argument(
        "--clean-meta",
        type=str,
        required=True,
        help="Path to clean meta .json.",
    )
    parser.add_argument(
        "--adv-npy",
        type=str,
        required=True,
        help="Path to adversarial features .npy.",
    )
    parser.add_argument(
        "--adv-meta",
        type=str,
        required=True,
        help="Path to adversarial meta .json.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help='Comma-separated feature indices to inspect, e.g., "11318,4295,1929"',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If >0 and --features is empty, auto-select top-k features by |mean_diff|.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Max number of image pairs to plot. 0 = all.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for heatmaps.",
    )
    parser.add_argument(
        "--vmin-vmax-mode",
        type=str,
        choices=["separate", "shared"],
        default="shared",
        help=(
            "shared: use the same (min, max) color scale for clean and adv per feature; "
            "separate: each heatmap uses its own scale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/token_heatmaps",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def load_meta(meta_path: str) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_spatial_shape(n_tokens: int) -> Tuple[int, int]:
    """Infer H, W from n_tokens. Common: 49 -> 7x7, 196 -> 14x14, 784 -> 28x28."""
    h = int(np.sqrt(n_tokens))
    if h * h != n_tokens:
        raise ValueError(f"Cannot infer square spatial shape from n_tokens={n_tokens}")
    return h, h


def plot_feature_heatmap_pair(
    clean_map: np.ndarray,
    adv_map: np.ndarray,
    feature_idx: int,
    image_stem: str,
    out_path: str,
    cmap: str = "viridis",
    vmin_vmax_mode: str = "shared",
):
    """
    clean_map, adv_map: (H, W)
    Draws side-by-side heatmap with a shared colorbar.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.05]})

    if vmin_vmax_mode == "shared":
        vmin = min(clean_map.min(), adv_map.min())
        vmax = max(clean_map.max(), adv_map.max())
    else:
        vmin = vmax = None

    im0 = axes[0].imshow(clean_map, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"Clean\n{image_stem}", fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(adv_map, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title(f"Adversarial\n{image_stem}", fontsize=11)
    axes[1].axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im0, cax=axes[2])
    cbar.set_label("SAE Activation", rotation=270, labelpad=20)

    fig.suptitle(f"Feature {feature_idx} | Token-level Activation", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_summary_grid(
    clean_maps: np.ndarray,
    adv_maps: np.ndarray,
    feature_idx: int,
    image_stems: List[str],
    out_path: str,
    cmap: str = "viridis",
    max_cols: int = 5,
):
    """
    clean_maps, adv_maps: (n_images, H, W)
    Draws a grid: rows = image pairs, cols = clean | adv | diff.
    Only plots up to max_cols image pairs to keep readable.
    """
    n_images = min(len(image_stems), max_cols)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 3.5 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    vmin = min(clean_maps[:n_images].min(), adv_maps[:n_images].min())
    vmax = max(clean_maps[:n_images].max(), adv_maps[:n_images].max())

    for i in range(n_images):
        im0 = axes[i, 0].imshow(clean_maps[i], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[i, 0].set_title(f"Clean\n{image_stems[i]}", fontsize=9)
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(adv_maps[i], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[i, 1].set_title(f"Adv\n{image_stems[i]}", fontsize=9)
        axes[i, 1].axis("off")

        diff_map = adv_maps[i] - clean_maps[i]
        im2 = axes[i, 2].imshow(diff_map, cmap="RdBu_r", vmin=-max(abs(diff_map.min()), abs(diff_map.max())), vmax=max(abs(diff_map.min()), abs(diff_map.max())), interpolation="nearest")
        axes[i, 2].set_title(f"Diff (Adv - Clean)\n{image_stems[i]}", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(f"Feature {feature_idx} | Token-level Summary (Top {n_images} pairs)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    print(f"Loading clean: {args.clean_npy}")
    clean_raw = np.load(args.clean_npy)  # (N, n_tokens, d_lat)
    clean_meta = load_meta(args.clean_meta)

    print(f"Loading adv:   {args.adv_npy}")
    adv_raw = np.load(args.adv_npy)
    adv_meta = load_meta(args.adv_meta)

    print(f"Shapes: clean={clean_raw.shape}, adv={adv_raw.shape}")

    n_images, n_tokens, d_lat = clean_raw.shape
    h, w = infer_spatial_shape(n_tokens)
    print(f"Spatial map: {h}x{w} (n_tokens={n_tokens})")

    clean_stems = clean_meta.get("image_stems", [f"img_{i}" for i in range(n_images)])
    adv_stems = adv_meta.get("image_stems", [f"img_{i}" for i in range(n_images)])

    # Determine feature indices to inspect
    if args.features:
        feature_indices = [int(x.strip()) for x in args.features.split(",") if x.strip()]
    elif args.top_k > 0:
        # Auto-select by |mean_diff|
        clean_mean = clean_raw.mean(axis=1)  # (N, d_lat)
        adv_mean = adv_raw.mean(axis=1)
        mean_diff = (adv_mean - clean_mean).mean(axis=0)
        feature_indices = np.argsort(np.abs(mean_diff))[::-1][: args.top_k].tolist()
        print(f"Auto-selected top-{args.top_k} features by |mean_diff|: {feature_indices}")
    else:
        raise ValueError("Provide --features or --top-k")

    # Validate
    for fid in feature_indices:
        if not (0 <= fid < d_lat):
            raise ValueError(f"Feature index {fid} out of range [0, {d_lat})")

    # Output dir
    out_dir = Path(args.output_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    else:
        out_dir = out_dir / f"token_inspect_{'_'.join(map(str, feature_indices[:3]))}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Determine how many image pairs to plot
    n_plot = min(n_images, len(adv_raw)) if args.max_images == 0 else min(args.max_images, n_images, len(adv_raw))

    for fid in feature_indices:
        print(f"\nProcessing feature {fid}...")
        # Extract feature maps: (N, H, W)
        clean_maps = clean_raw[:n_plot, :, fid].reshape(-1, h, w)
        adv_maps = adv_raw[:n_plot, :, fid].reshape(-1, h, w)

        # 1. Per-image pair heatmaps
        feat_dir = out_dir / f"feature_{fid}"
        feat_dir.mkdir(exist_ok=True)

        for i in range(n_plot):
            plot_feature_heatmap_pair(
                clean_map=clean_maps[i],
                adv_map=adv_maps[i],
                feature_idx=fid,
                image_stem=clean_stems[i],
                out_path=feat_dir / f"{clean_stems[i]}_f{fid}.png",
                cmap=args.cmap,
                vmin_vmax_mode=args.vmin_vmax_mode,
            )

        # 2. Summary grid (first N images)
        plot_feature_summary_grid(
            clean_maps=clean_maps,
            adv_maps=adv_maps,
            feature_idx=fid,
            image_stems=clean_stems[:n_plot],
            out_path=out_dir / f"summary_feature_{fid}.png",
            cmap=args.cmap,
            max_cols=5,
        )

        print(f"  Saved {n_plot} per-image heatmaps + 1 summary grid.")

    print(f"\nAll heatmaps saved to: {out_dir}")


if __name__ == "__main__":
    main()
