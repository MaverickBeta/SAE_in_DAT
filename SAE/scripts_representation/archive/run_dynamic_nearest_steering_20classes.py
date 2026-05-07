#!/usr/bin/env python3
"""
Dynamic nearest-class steering for adversarial samples (Version B).
Supports two lookup table filtering strategies:
  v1: Layer 1 only (total clean activation >= 20 across 20 classes)
  v2: Layer 1 + Layer 2 (>=2 classes each with >=5 clean activations)

Conditions per class per version:
  1. Control:       No intervention
  2. SAE-only:      encode → decode without modification
  3. Dynamic Nearest: encode → per-entry nearest-class steering → decode

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_dynamic_nearest_steering_20classes.py --version v1
    python run_dynamic_nearest_steering_20classes.py --version v2
    python run_dynamic_nearest_steering_20classes.py --version all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint
from sae_core.model import TopKAutoencoder

# ── Config ──────────────────────────────────────────────────────────
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"

TRAIN_FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_train_latent")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_val")
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
OUT_DIR_BASE = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20cls_dynamic_nearest")

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

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


# ── Load model & SAE ────────────────────────────────────────────────
print("Loading base model...")
model = create_convnext_model(
    model_type="convnext_large",
    num_classes=1000,
    normalize_input=False,
    use_layernorm=True,
    use_convstem=True,
)
load_checkpoint(model, BASE_CKPT)
model = model.to(DEVICE)
model.eval()

print("Loading SAE...")
sae_ckpt = torch.load(SAE_CKPT, map_location=DEVICE)
config = sae_ckpt.get("config", {})
d_in = sae_ckpt.get("d_in", config.get("d_in", None))
d_lat = sae_ckpt.get("d_lat", config.get("d_lat", None))
k = config.get("k", sae_ckpt.get("k", None))
if d_lat is None:
    d_lat = int(d_in) * int(config.get("expansion_rate", 8))

sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
sae.load_state_dict(sae_ckpt["model_state_dict"])
sae = sae.to(DEVICE)
sae.eval()

norm_mean = sae_ckpt["norm_mean"].to(DEVICE)
norm_std = sae_ckpt["norm_std"].to(DEVICE)
print(f"  SAE: d_in={d_in}, d_lat={d_lat}, k={k}")

N_CLASSES = len(CLASSES)
N_TOKENS = 49
N_CHANNELS = int(d_lat)

# ── Image preprocessing ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_images_from_dir(img_dir, cls_idx):
    if not img_dir.is_dir():
        return []
    images = []
    for path in sorted(img_dir.glob("*.JPEG")):
        img = Image.open(path).convert("RGB")
        img = transform(img).to(DEVICE)
        images.append((img, cls_idx))
    return images


def load_adv_images(wnid, cls_idx):
    run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
    img_dir = ADV_ROOT / run_name / "adv_succ_real"
    return load_images_from_dir(img_dir, cls_idx)


def load_clean_images(wnid, cls_idx):
    img_dir = IMAGENET_VAL / wnid
    return load_images_from_dir(img_dir, cls_idx)


# ── Build Lookup Table ──────────────────────────────────────────────
def build_lookup_table(version):
    """
    Build lookup table from train clean features.

    version: "v1" or "v2"

    Returns:
      lut_tensor: (49, 12288, 20) torch tensor
      lut_mask:   (49, 12288) bool torch tensor
    """
    print(f"\nBuilding lookup table ({version})...")

    # Accumulate statistics across all 20 classes
    clean_counts = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.int32)
    clean_sums = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float64)

    for cls_idx, (wnid, imagenet_idx, name) in enumerate(CLASSES):
        npy_path = TRAIN_FEAT_DIR / f"clean_{wnid}_cls{imagenet_idx}_stage3_k256_features.npy"
        clean = np.load(npy_path)  # (100, 49, 12288)

        mask = (clean != 0)
        clean_counts[cls_idx] = mask.sum(axis=0)
        clean_sums[cls_idx] = clean.sum(axis=0)

    # Compute per-class clean mean
    clean_means = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        clean_means = np.where(clean_counts > 0, clean_sums / clean_counts, 0.0)

    # Layer 1: total clean activation >= 20 across all classes
    total_counts = clean_counts.sum(axis=0)  # (49, 12288)
    layer1_mask = total_counts >= 20

    if version == "v1":
        final_mask = layer1_mask
        print(f"  Layer 1 (total>=20): {layer1_mask.sum():,} entries")
    elif version == "v2":
        # Layer 2: >=2 classes each with >=5 clean activations
        class_active = (clean_counts >= 5).sum(axis=0)  # (49, 12288)
        layer2_mask = class_active >= 2
        final_mask = layer1_mask & layer2_mask
        print(f"  Layer 1 (total>=20): {layer1_mask.sum():,} entries")
        print(f"  Layer 2 (>=2 classes>=5): {layer2_mask.sum():,} entries")
    else:
        raise ValueError(f"Unknown version: {version}")

    # Build lut_tensor: (49, 12288, 20)
    # For each (token, channel), store 20 class means
    lut_tensor = np.zeros((N_TOKENS, N_CHANNELS, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        lut_tensor[:, :, c] = clean_means[c]

    lut_mask = final_mask

    n_total = N_TOKENS * N_CHANNELS
    print(f"  Final entries: {final_mask.sum():,} / {n_total:,} ({final_mask.sum() / n_total * 100:.2f}%)")

    return (
        torch.from_numpy(lut_tensor).to(DEVICE),
        torch.from_numpy(lut_mask).to(DEVICE),
    )


# ── Hook factories ──────────────────────────────────────────────────
def make_sae_only_hook():
    """SAE encode → decode without any modification."""
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        recon_norm = (z @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


def make_dynamic_nearest_hook(lut_tensor, lut_mask):
    """
    Dynamic nearest-class steering hook.

    For each activated entry (z != 0) that exists in lookup table:
      1. Find which of 20 classes has clean mean closest to current value
      2. Overwrite with that class's clean mean
    """
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)  # (bsz*49, 12288)
        z = z.reshape(bsz, N_TOKENS, N_CHANNELS)  # (bsz, 49, 12288)

        # Only steer entries that are both activated AND in lookup table
        active_mask = (z != 0)  # (bsz, 49, 12288)
        valid_mask = active_mask & lut_mask.unsqueeze(0)  # (bsz, 49, 12288)

        if valid_mask.any():
            b_idx, tok_idx, ch_idx = torch.where(valid_mask)
            current_vals = z[b_idx, tok_idx, ch_idx]  # (M,)

            # Lookup 20-class means for these entries: (M, 20)
            class_means = lut_tensor[tok_idx, ch_idx, :]

            # Find nearest class: argmin |current - class_mean|
            diffs = torch.abs(current_vals.unsqueeze(1) - class_means)  # (M, 20)
            nearest_cls = torch.argmin(diffs, dim=1)  # (M,)

            # Get target values
            M = b_idx.shape[0]
            targets = class_means[torch.arange(M, device=class_means.device), nearest_cls]

            # Overwrite
            z[b_idx, tok_idx, ch_idx] = targets

        # Decode
        z_flat = z.reshape(bsz * N_TOKENS, N_CHANNELS)
        recon_norm = (z_flat @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


# ── Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(images, hook_fn=None):
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    correct = 0
    total = 0
    try:
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Evaluating", leave=False):
            batch = images[i : i + BATCH_SIZE]
            imgs = torch.stack([img for img, _ in batch])
            labels = torch.tensor([label for _, label in batch], device=DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(batch)
    finally:
        if hook_fn is not None:
            handle.remove()
    return correct / total if total > 0 else 0.0


# ── Run single version ──────────────────────────────────────────────
def run_version(version):
    print("\n" + "=" * 70)
    print(f"DYNAMIC NEAREST STEERING — VERSION {version.upper()}")
    print("=" * 70)

    lut_tensor, lut_mask = build_lookup_table(version)
    out_dir = OUT_DIR_BASE / version
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")

        # Clean accuracy (ceiling)
        clean_images = load_clean_images(wnid, cls_idx)
        if clean_images:
            clean_acc = evaluate(clean_images)
            print(f"  Clean: {clean_acc * 100:.1f}% ({len(clean_images)} samples)")
        else:
            clean_acc = None

        # Adversarial samples
        images = load_adv_images(wnid, cls_idx)
        if not images:
            print(f"  [SKIP] No adv images found")
            continue
        print(f"  Adv samples: {len(images)}")

        # 1. Control
        print("  [1/3] Control...")
        control_acc = evaluate(images)

        # 2. SAE-only
        print("  [2/3] SAE-only...")
        sae_only_hook = make_sae_only_hook()
        sae_only_acc = evaluate(images, sae_only_hook)

        # 3. Dynamic Nearest
        print("  [3/3] Dynamic Nearest...")
        dyn_hook = make_dynamic_nearest_hook(lut_tensor, lut_mask)
        dyn_acc = evaluate(images, dyn_hook)

        clean_str = f"{clean_acc * 100:.1f}%" if clean_acc is not None else "N/A"
        print(
            f"  Clean: {clean_str}  Ctrl: {control_acc * 100:.1f}%  "
            f"SAE: {sae_only_acc * 100:.1f}%  Dyn: {dyn_acc * 100:.1f}%"
        )

        # Recovery rate
        recovery_rate = None
        if clean_acc is not None and clean_acc > control_acc:
            recovery_rate = (dyn_acc - control_acc) / (clean_acc - control_acc) * 100

        results.append(
            {
                "name": name,
                "wnid": wnid,
                "class_idx": cls_idx,
                "n_clean_samples": len(clean_images),
                "n_adv_samples": len(images),
                "clean_acc": clean_acc,
                "control_acc": control_acc,
                "sae_only_acc": sae_only_acc,
                "dynamic_acc": dyn_acc,
                "sae_only_vs_control": (sae_only_acc - control_acc) * 100,
                "dyn_vs_control": (dyn_acc - control_acc) * 100,
                "dyn_vs_sae_only": (dyn_acc - sae_only_acc) * 100,
                "recovery_rate": recovery_rate,
            }
        )

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "version": version,
        "n_classes_evaluated": len(results),
        "avg_clean_acc": avg("clean_acc"),
        "avg_control_acc": avg("control_acc"),
        "avg_sae_only_acc": avg("sae_only_acc"),
        "avg_dynamic_acc": avg("dynamic_acc"),
        "avg_sae_only_vs_control": avg("sae_only_vs_control"),
        "avg_dyn_vs_control": avg("dyn_vs_control"),
        "avg_dyn_vs_sae_only": avg("dyn_vs_sae_only"),
    }

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"RESULTS SUMMARY — VERSION {version.upper()}")
    print("=" * 95)
    print(
        f"{'Class':>15} {'N_clean':>7} {'N_adv':>5} {'Clean':>6} {'Ctrl':>6} "
        f"{'SAE':>6} {'Dyn':>6} {'D-C':>6} {'D-SAE':>6} {'Recov%':>7}"
    )
    print("-" * 95)

    for r in results:
        clean_str = f"{r['clean_acc'] * 100:>5.1f}%" if r["clean_acc"] is not None else f"{'N/A':>6}"
        rec_str = f"{r['recovery_rate']:>6.1f}%" if r["recovery_rate"] is not None else f"{'N/A':>7}"
        print(
            f"{r['name']:>15} {r['n_clean_samples']:>7} {r['n_adv_samples']:>5} "
            f"{clean_str} {r['control_acc'] * 100:>5.1f}% {r['sae_only_acc'] * 100:>5.1f}% "
            f"{r['dynamic_acc'] * 100:>5.1f}% "
            f"{r['dyn_vs_control']:>+5.1f} {r['dyn_vs_sae_only']:>+5.1f} {rec_str}"
        )

    print("-" * 95)
    print(
        f"{'OVERALL AVERAGE':>15} {'':>7} {'':>5} "
        f"{overall['avg_clean_acc'] * 100:>5.1f}% {overall['avg_control_acc'] * 100:>5.1f}% "
        f"{overall['avg_sae_only_acc'] * 100:>5.1f}% {overall['avg_dynamic_acc'] * 100:>5.1f}% "
        f"{overall['avg_dyn_vs_control']:>+5.1f} {overall['avg_dyn_vs_sae_only']:>+5.1f}"
    )
    print("=" * 95)

    # Save JSON
    out_path = out_dir / "dynamic_nearest_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "per_class": results,
                "overall_average": overall,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved: {out_path}")

    n_dyn_positive = sum(1 for r in results if r["dyn_vs_control"] > 0)
    n_dyn_better_sae = sum(1 for r in results if r["dynamic_acc"] > r["sae_only_acc"])
    print(f"\nDynamic > Control:      {n_dyn_positive}/{len(results)} classes")
    print(f"Dynamic > SAE-only:     {n_dyn_better_sae}/{len(results)} classes")
    print(f"\nKEY METRICS:")
    print(f"  Dynamic vs Control (raw gain):     {overall['avg_dyn_vs_control']:+.1f} pp")
    print(f"  Dynamic vs SAE-only (true effect): {overall['avg_dyn_vs_sae_only']:+.1f} pp")

    return results, overall

 
# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dynamic nearest-class steering")
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "all"],
        default="all",
        help="Which version to run: v1, v2, or all",
    )
    args = parser.parse_args()

    versions = ["v1", "v2"] if args.version == "all" else [args.version]
    all_results = {}

    for version in versions:
        results, overall = run_version(version)
        all_results[version] = {"per_class": results, "overall": overall}

    # If both versions run, print comparison
    if len(versions) == 2:
        print("\n" + "=" * 70)
        print("V1 vs V2 COMPARISON")
        print("=" * 70)
        v1 = all_results["v1"]["overall"]
        v2 = all_results["v2"]["overall"]
        print(f"{'Metric':>30} {'V1':>10} {'V2':>10} {'Diff':>10}")
        print("-" * 62)
        print(f"{'Avg Control Acc':>30} {v1['avg_control_acc'] * 100:>9.2f}% {v2['avg_control_acc'] * 100:>9.2f}% {'—':>10}")
        print(f"{'Avg SAE-only Acc':>30} {v1['avg_sae_only_acc'] * 100:>9.2f}% {v2['avg_sae_only_acc'] * 100:>9.2f}% {'—':>10}")
        print(
            f"{'Avg Dynamic Acc':>30} {v1['avg_dynamic_acc'] * 100:>9.2f}% "
            f"{v2['avg_dynamic_acc'] * 100:>9.2f}% "
            f"{(v2['avg_dynamic_acc'] - v1['avg_dynamic_acc']) * 100:>+9.2f}pp"
        )
        print(
            f"{'Dynamic vs Control (pp)':>30} {v1['avg_dyn_vs_control']:>+9.2f} "
            f"{v2['avg_dyn_vs_control']:>+9.2f} "
            f"{(v2['avg_dyn_vs_control'] - v1['avg_dyn_vs_control']):>+9.2f}"
        )
        print(
            f"{'Dynamic vs SAE-only (pp)':>30} {v1['avg_dyn_vs_sae_only']:>+9.2f} "
            f"{v2['avg_dyn_vs_sae_only']:>+9.2f} "
            f"{(v2['avg_dyn_vs_sae_only'] - v1['avg_dyn_vs_sae_only']):>+9.2f}"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
