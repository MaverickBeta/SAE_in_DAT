#!/usr/bin/env python3
"""
Analyze SAE entry activation distributions for class 461 vs its top-1 misprediction target class 524.

Three distributions:
  1. Clean class 524 (100 images from ImageNet train)
  2. Clean class 461 (from adversarial_samples/train/clean)
  3. Adv class 461  (from adversarial_samples/train/adv)

For each non-zero entry (token, feature), compute mean activation across images.
Sort entries by: delta = mean(clean_461) - mean(clean_524), descending.
Entries only active in adv_461 (not in clean_461 or clean_524) are placed at delta=0.

Output: results/cls461_entry_inspect.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
TARGET_CLASS_IDX = 461
MISPRED_CLASS_IDX = 524


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_images=0):
        paths = sorted(Path(folder_path).glob("*"))
        self.image_paths = [
            str(p) for p in paths if p.is_file() and p.suffix in IMAGE_EXTS
        ]
        if 0 < max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(path).name


def build_model(device, ckpt_path):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, ckpt_path, weights_only=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_sae(device, sae_ckpt_path):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    return sae, norm_mean, norm_std


def get_wnid_for_class_idx(train_dir, class_idx):
    """Get WNID by alphabetical order (ImageFolder convention)."""
    all_dirs = sorted([d for d in Path(train_dir).iterdir() if d.is_dir()])
    if not (0 <= class_idx < len(all_dirs)):
        raise ValueError(f"Class index {class_idx} out of range (0-{len(all_dirs)-1})")
    return all_dirs[class_idx].name


def find_wnid_in_adv_root(adv_root, target_class_idx):
    """Find WNID folder in adv_root whose labels.txt contains target_class_idx."""
    for class_dir in Path(adv_root).iterdir():
        if not class_dir.is_dir() or class_dir.name == "sae_latent":
            continue
        labels_path = class_dir / "labels.txt"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            if int(parts[-1]) == target_class_idx:
                                return class_dir.name
                        except ValueError:
                            pass
    return None


@torch.no_grad()
def extract_nonzero_entries(loader, model, sae, norm_mean, norm_std, stage_idx, device):
    """Extract all non-zero SAE entries. Returns dict: (token, feature) -> [values]."""
    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)
    entries = {}

    for images, _ in tqdm(loader, desc="Extract", leave=False):
        images = images.to(device)
        _ = model(images)

        stage_out = captured["stage"]  # (bsz, c, h, w)
        bsz, c, h, w = stage_out.shape
        flat = stage_out.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)  # (bsz*h*w, d_lat)

        z_cpu = z.cpu().numpy()
        z_per_image = z_cpu.reshape(bsz, h * w, -1)  # (bsz, n_tokens, d_lat)

        for i in range(bsz):
            img_z = z_per_image[i]
            nonzero_mask = img_z > 0
            token_indices, feature_indices = np.where(nonzero_mask)
            values = img_z[nonzero_mask]
            for t, f, v in zip(token_indices, feature_indices, values):
                key = (int(t), int(f))
                entries.setdefault(key, []).append(float(v))

    handle.remove()
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="SAE entry inspection for class 461 vs 524"
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(
            REPO_ROOT
            / "SAE"
            / "project"
            / "checkpoints"
            / "stage3"
            / "k256_exp8"
            / "sae_stage3_din1536_exp8_k256_step_50000.pt"
        ),
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "ImageNet" / "train"),
    )
    parser.add_argument(
        "--adv-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument(
        "--max-entries-plot",
        type=int,
        default=0,
        help="Max entries to plot (0=all). If too many, auto-samples to ~4000.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device, args.base_ckpt)
    sae, norm_mean, norm_std = build_sae(device, args.sae_ckpt)

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    # ================================================================
    # 1. Locate directories
    # ================================================================
    wnid_524 = get_wnid_for_class_idx(args.train_dir, MISPRED_CLASS_IDX)
    wnid_461 = find_wnid_in_adv_root(args.adv_root, TARGET_CLASS_IDX)
    print(f"Class {MISPRED_CLASS_IDX} -> WNID: {wnid_524}")
    print(f"Class {TARGET_CLASS_IDX} -> WNID: {wnid_461}")

    clean_524_dir = Path(args.train_dir) / wnid_524
    clean_461_dir = Path(args.adv_root) / wnid_461 / "train" / "clean"
    adv_461_dir = Path(args.adv_root) / wnid_461 / "train" / "adv"

    # ================================================================
    # 2. Extract entries for three distributions
    # ================================================================
    print("\n[1/3] Extracting clean class 524...")
    ds_524 = ImageFolderDataset(str(clean_524_dir), transform=transform, max_images=args.n_images)
    ld_524 = DataLoader(ds_524, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    entries_524 = extract_nonzero_entries(ld_524, model, sae, norm_mean, norm_std, args.stage, device)
    print(f"  Images: {len(ds_524)}, Unique entries: {len(entries_524)}")

    print("\n[2/3] Extracting clean class 461...")
    ds_461c = ImageFolderDataset(str(clean_461_dir), transform=transform)
    ld_461c = DataLoader(ds_461c, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    entries_461c = extract_nonzero_entries(ld_461c, model, sae, norm_mean, norm_std, args.stage, device)
    print(f"  Images: {len(ds_461c)}, Unique entries: {len(entries_461c)}")

    print("\n[3/3] Extracting adv class 461...")
    ds_461a = ImageFolderDataset(str(adv_461_dir), transform=transform)
    ld_461a = DataLoader(ds_461a, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    entries_461a = extract_nonzero_entries(ld_461a, model, sae, norm_mean, norm_std, args.stage, device)
    print(f"  Images: {len(ds_461a)}, Unique entries: {len(entries_461a)}")

    del model, sae
    torch.cuda.empty_cache()

    # ================================================================
    # 3. Aggregate and compute delta
    # ================================================================
    print("\nAggregating entries...")
    all_entries = set(entries_524.keys()) | set(entries_461c.keys()) | set(entries_461a.keys())
    print(f"Total unique entries across 3 distributions: {len(all_entries)}")

    results = []
    for key in all_entries:
        t, f = key
        vals_524 = entries_524.get(key, [])
        vals_461c = entries_461c.get(key, [])
        vals_461a = entries_461a.get(key, [])

        mean_524 = float(np.mean(vals_524)) if vals_524 else 0.0
        mean_461c = float(np.mean(vals_461c)) if vals_461c else 0.0
        mean_461a = float(np.mean(vals_461a)) if vals_461a else 0.0

        in_524 = len(vals_524) > 0
        in_461c = len(vals_461c) > 0
        in_461a = len(vals_461a) > 0

        # Delta definition:
        #   If entry exists in clean_461 or clean_524: delta = mean(clean_461) - mean(clean_524)
        #   If entry only in adv_461: delta = 0
        if in_461c or in_524:
            delta = mean_461c - mean_524
        else:
            delta = 0.0

        results.append({
            "token": t,
            "feature": f,
            "mean_524": mean_524,
            "mean_461c": mean_461c,
            "mean_461a": mean_461a,
            "delta": delta,
            "in_524": in_524,
            "in_461c": in_461c,
            "in_461a": in_461a,
            "count_524": len(vals_524),
            "count_461c": len(vals_461c),
            "count_461a": len(vals_461a),
        })

    # Sort by delta descending
    results.sort(key=lambda x: x["delta"], reverse=True)

    # Optionally limit entries for plotting
    max_entries = args.max_entries_plot
    if max_entries <= 0 and len(results) > 5000:
        max_entries = 4000
        print(f"Auto-limiting to {max_entries} entries for plotting clarity")

    if 0 < max_entries < len(results):
        n_half = max_entries // 2
        # Keep top-N by delta + bottom-N by delta, avoiding duplicates
        top_part = results[:n_half]
        bottom_part = results[-n_half:]
        plot_data = top_part + bottom_part
        print(f"Plotting {len(plot_data)} entries (top {n_half} + bottom {n_half} by delta)")
    else:
        plot_data = results
        print(f"Plotting all {len(plot_data)} entries")

    # ================================================================
    # 4. Plot
    # ================================================================
    print("Generating plot...")
    x = np.arange(len(plot_data))
    y_524 = np.array([r["mean_524"] for r in plot_data])
    y_461c = np.array([r["mean_461c"] for r in plot_data])
    y_461a = np.array([r["mean_461a"] for r in plot_data])
    deltas = np.array([r["delta"] for r in plot_data])

    # Find approximate delta=0 boundary
    delta_zero_indices = np.where(np.isclose(deltas, 0.0, atol=1e-8))[0]

    fig, ax = plt.subplots(figsize=(22, 9))

    bar_width = 0.28
    offset_524 = -bar_width
    offset_461c = 0
    offset_461a = bar_width

    ax.bar(x + offset_524, y_524, width=bar_width, alpha=0.65, color='royalblue',
           label=f'Clean {MISPRED_CLASS_IDX} (top-1 mispred target)')
    ax.bar(x + offset_461c, y_461c, width=bar_width, alpha=0.65, color='forestgreen',
           label=f'Clean {TARGET_CLASS_IDX}')
    ax.bar(x + offset_461a, y_461a, width=bar_width, alpha=0.65, color='crimson',
           label=f'Adv {TARGET_CLASS_IDX}')

    # Mark delta=0 boundary
    if len(delta_zero_indices) > 0:
        first_zero = delta_zero_indices[0]
        last_zero = delta_zero_indices[-1]
        ax.axvline(x=first_zero - 0.5, color='black', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(x=last_zero + 0.5, color='black', linestyle='--', alpha=0.4, linewidth=1)
        # Shade delta=0 region
        ax.axvspan(first_zero - 0.5, last_zero + 0.5, alpha=0.08, color='gray',
                   label='Delta=0 region (adv-only)')

    ax.set_xlabel("Entries (sorted by δ = clean_461 − clean_524, descending)", fontsize=13)
    ax.set_ylabel("Mean Activation Value", fontsize=13)
    ax.set_title(
        f"SAE Entry Activation (Bar): Class {TARGET_CLASS_IDX} vs Top-1 Misprediction (Class {MISPRED_CLASS_IDX})\n"
        f"({len(plot_data)} entries shown)",
        fontsize=14,
    )
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    ax.text(0.02, 0.98, "← clean_461 stronger", transform=ax.transAxes, fontsize=10,
            va='top', ha='left', color='green')
    ax.text(0.98, 0.98, "clean_524 stronger →", transform=ax.transAxes, fontsize=10,
            va='top', ha='right', color='blue')

    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cls461_entry_inspect.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")

    # Also save JSON
    json_path = output_dir / "cls461_entry_inspect.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "target_class": TARGET_CLASS_IDX,
                    "mispred_class": MISPRED_CLASS_IDX,
                    "n_entries_total": len(results),
                    "n_entries_plotted": len(plot_data),
                },
                "entries": plot_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved data: {json_path}")


if __name__ == "__main__":
    main()
