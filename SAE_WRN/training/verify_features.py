#!/usr/bin/env python3
"""
Quick sanity check for extracted WRN34x10 CIFAR-10 features.
"""

import argparse
import os
import sys

import torch


def verify_dir(features_dir):
    pt_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".pt") and f != "merged.pt"])
    if not pt_files:
        print(f"No .pt files found in {features_dir}")
        return

    print(f"Found {len(pt_files)} class files in {features_dir}:\n")
    total_images = 0
    for fname in pt_files:
        fpath = os.path.join(features_dir, fname)
        tensor = torch.load(fpath, map_location="cpu", weights_only=True)
        if tensor.ndim == 3:
            # Could be [N, 640] already flattened
            n, d = tensor.shape
            print(f"  {fname:20s}: {tuple(tensor.shape)}  (flattened spatial, d={d})")
            total_images += n
        elif tensor.ndim == 4:
            n, c, h, w = tensor.shape
            print(f"  {fname:20s}: {tuple(tensor.shape)}  ({n} images, {c}ch, {h}x{w})")
            total_images += n
        else:
            print(f"  {fname:20s}: unexpected shape {tuple(tensor.shape)}")

    print(f"\nTotal images: {total_images}")

    # Compute global stats from all files
    print("\nComputing global feature stats...")
    all_means = []
    all_stds = []
    for fname in pt_files:
        fpath = os.path.join(features_dir, fname)
        tensor = torch.load(fpath, map_location="cpu", weights_only=True)
        if tensor.ndim == 4:
            # [N, C, H, W] -> reshape to [N*H*W, C] for stats
            n, c, h, w = tensor.shape
            flat = tensor.permute(0, 2, 3, 1).reshape(n * h * w, c)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])
        all_means.append(flat.mean(dim=0))
        all_stds.append(flat.std(dim=0))

    global_mean = torch.stack(all_means).mean(dim=0)
    global_std = torch.stack(all_stds).mean(dim=0)
    print(f"  Feature dim: {global_mean.shape[0]}")
    print(f"  Mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    print(f"  Std range:  [{global_std.min():.4f}, {global_std.max():.4f}]")


def verify_merged(merged_path):
    if not os.path.exists(merged_path):
        print(f"Merged file not found: {merged_path}")
        return

    data = torch.load(merged_path, map_location="cpu", weights_only=True)
    if "features" not in data:
        print("Merged file missing 'features' key")
        return

    feats = data["features"]
    print(f"Merged file: {merged_path}")
    print(f"  Shape: {tuple(feats.shape)}")
    if feats.ndim == 4:
        n, c, h, w = feats.shape
        print(f"  {n} images, {c} channels, {h}x{w} spatial")
        flat = feats.permute(0, 2, 3, 1).reshape(n * h * w, c)
        print(f"  Flattened: {tuple(flat.shape)}")
    else:
        print(f"  Feature shape: {tuple(feats.shape)}")

    if "norm_mean" in data and "norm_std" in data:
        print(f"  Contains precomputed norm_mean/norm_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="/Data_share/hongyi/DAT/SAE_WRN/features")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Verification")
    print("=" * 60)

    verify_dir(args.features_dir)

    merged_path = os.path.join(args.features_dir, "merged.pt")
    if os.path.exists(merged_path):
        print()
        verify_merged(merged_path)

    print("\n" + "=" * 60)
    print("Verification complete.")
