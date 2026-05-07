#!/usr/bin/env python3
"""
Per-class steering: each class uses its OWN top 30 selected entries.

For each class:
  1. Control:   No intervention
  2. Experiment: Steer top 30 entries (by |delta|) to their clean_mean
  3. Validation: Steer 30 random entries (clean>=92%, non-top30) to their clean_mean

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_per_class_steering_20classes.py
"""

import os
import sys
import json
import random
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

FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/archive/features_20classes")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_val")
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
RESULTS_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20cls_train")
TRAIN_FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_latent")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20cls_per_class")

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)

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


# ── Read per-class selected entries ─────────────────────────────────
def get_top_and_random_entries(name, n_top=30, n_rand=30):
    """
    Returns:
      exp_targets: dict of top n_top entries -> clean_mean
      val_targets: dict of n_rand random entries -> clean_mean
    """
    json_path = RESULTS_DIR / f"{name}_selected_entries.json"
    with open(json_path) as f:
        data = json.load(f)

    # Sort by |delta| descending
    entries = sorted(data["selected_entries"], key=lambda x: abs(x["delta"]), reverse=True)

    # Top n_top for experiment
    top_entries = entries[:n_top]
    exp_targets = {}
    for e in top_entries:
        exp_targets[(e["token"], e["channel"])] = e["clean_mean"]

    # Random n_rand from remaining high-activation pool (val clean features for pool selection)
    val_clean_npy = FEAT_DIR / f"clean_{data['wnid']}_cls{data['class_idx']}_stage3_k256_features.npy"
    val_clean = np.load(val_clean_npy)
    val_clean_count = np.sum(val_clean != 0, axis=0)
    mask_high = val_clean_count >= 46

    top_keys = {(e["token"], e["channel"]) for e in top_entries}
    tok_idx, ch_idx = np.where(mask_high)
    candidates = []
    for t, c in zip(tok_idx, ch_idx):
        key = (int(t), int(c))
        if key not in top_keys:
            candidates.append(key)

    if len(candidates) < n_rand:
        n_rand = len(candidates)

    if n_rand == 0:
        return exp_targets, {}

    random_keys = random.sample(candidates, n_rand)

    # Random A target from TRAIN clean features (fair baseline)
    train_clean_npy = TRAIN_FEAT_DIR / f"clean_{data['wnid']}_cls{data['class_idx']}_stage3_k256_features.npy"
    train_clean = np.load(train_clean_npy)
    train_clean_count = np.sum(train_clean != 0, axis=0)
    train_clean_sum = np.sum(train_clean, axis=0)
    val_targets = {}
    for t, c in random_keys:
        cm = train_clean_sum[t, c] / train_clean_count[t, c] if train_clean_count[t, c] > 0 else 0.0
        val_targets[(t, c)] = float(cm)

    return exp_targets, val_targets


# ── Hook factory ────────────────────────────────────────────────────
def make_steering_hook(targets_dict):
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z_spatial = z.reshape(bsz, 7, 7, d_lat)
        for (token, channel), target_val in targets_dict.items():
            row = token // 7
            col = token % 7
            z_spatial[:, row, col, channel] = target_val
        z = z_spatial.reshape(bsz * 49, d_lat)
        recon_norm = (z @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


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


# ── Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(images, hook_fn=None):
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    correct = 0
    total = 0
    try:
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Evaluating", leave=False):
            batch = images[i:i+BATCH_SIZE]
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


# ── Main loop ───────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("PER-CLASS STEERING (Top 30 selected entries)")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")

        # Evaluate clean accuracy first (ceiling)
        clean_images = load_clean_images(wnid, cls_idx)
        if clean_images:
            print(f"  [0/5] Clean accuracy...")
            clean_acc = evaluate(clean_images)
        else:
            clean_acc = None

        images = load_adv_images(wnid, cls_idx)
        if not images:
            print(f"  [SKIP] No adv images found")
            continue
        print(f"  Adv samples: {len(images)}")

        exp_targets, val_targets = get_top_and_random_entries(name, n_top=30, n_rand=30)
        print(f"  Experiment entries: {len(exp_targets)}, Random entries: {len(val_targets)}")

        # 1. Control (no SAE)
        print("  [1/5] Control...")
        control_acc = evaluate(images)

        # 2. SAE-only (reconstruction control)
        print("  [2/5] SAE-only...")
        sae_only_hook = make_sae_only_hook()
        sae_only_acc = evaluate(images, sae_only_hook)

        # 3. Experiment
        print("  [3/5] Experiment (top 30 entries)...")
        exp_hook = make_steering_hook(exp_targets)
        exp_acc = evaluate(images, exp_hook)

        # 4. Validation
        val_acc = None
        if val_targets:
            print("  [4/5] Validation (30 random entries)...")
            val_hook = make_steering_hook(val_targets)
            val_acc = evaluate(images, val_hook)
        else:
            print("  [4/5] Validation skipped (not enough random entries)")

        val_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        clean_str = f"{clean_acc:.4f}" if clean_acc is not None else "N/A"
        print(f"  Clean: {clean_str}  Ctrl: {control_acc:.4f}  SAE: {sae_only_acc:.4f}  "
              f"Exp: {exp_acc:.4f}  Val: {val_str}")

        # Recovery rate
        recovery_rate = None
        if clean_acc is not None and clean_acc > control_acc:
            recovery_rate = (exp_acc - control_acc) / (clean_acc - control_acc) * 100

        results.append({
            "name": name,
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_clean_samples": len(clean_images),
            "n_adv_samples": len(images),
            "n_exp_entries": len(exp_targets),
            "n_val_entries": len(val_targets) if val_targets else 0,
            "clean_acc": clean_acc,
            "control_acc": control_acc,
            "sae_only_acc": sae_only_acc,
            "experiment_acc": exp_acc,
            "validation_acc": val_acc,
            "sae_only_vs_control": (sae_only_acc - control_acc) * 100,
            "exp_vs_control": (exp_acc - control_acc) * 100,
            "exp_vs_sae_only": (exp_acc - sae_only_acc) * 100,
            "exp_vs_val": (exp_acc - val_acc) * 100 if val_acc is not None else None,
            "recovery_rate": recovery_rate,
        })

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_clean_acc": avg("clean_acc"),
        "avg_control_acc": avg("control_acc"),
        "avg_sae_only_acc": avg("sae_only_acc"),
        "avg_experiment_acc": avg("experiment_acc"),
        "avg_validation_acc": avg("validation_acc"),
        "avg_sae_only_vs_control": avg("sae_only_vs_control"),
        "avg_exp_vs_control": avg("exp_vs_control"),
        "avg_exp_vs_sae_only": avg("exp_vs_sae_only"),
        "avg_exp_vs_val": avg("exp_vs_val"),
    }

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("RESULTS SUMMARY (per class)")
    print("=" * 95)
    print(f"{'Class':>15} {'N_clean':>7} {'N_adv':>5} {'Clean':>6} {'Ctrl':>6} {'SAE':>6} {'Exp':>6} {'Val':>6} {'E-C':>6} {'E-SAE':>6} {'E-V':>6} {'Recov%':>7}")
    print("-" * 95)

    for r in results:
        clean_str = f"{r['clean_acc']*100:>5.1f}%" if r['clean_acc'] is not None else f"{'N/A':>6}"
        val_str = f"{r['validation_acc']*100:>5.1f}%" if r['validation_acc'] is not None else f"{'N/A':>6}"
        ev_str = f"{r['exp_vs_val']:>+5.1f}" if r['exp_vs_val'] is not None else f"{'N/A':>6}"
        rec_str = f"{r['recovery_rate']:>6.1f}%" if r['recovery_rate'] is not None else f"{'N/A':>7}"
        print(f"{r['name']:>15} {r['n_clean_samples']:>7} {r['n_adv_samples']:>5} "
              f"{clean_str} {r['control_acc']*100:>5.1f}% {r['sae_only_acc']*100:>5.1f}% {r['experiment_acc']*100:>5.1f}% {val_str} "
              f"{r['exp_vs_control']:>+5.1f} {r['exp_vs_sae_only']:>+5.1f} {ev_str} {rec_str}")

    print("-" * 95)
    print(f"{'OVERALL AVERAGE':>15} {'':>7} {'':>5} "
          f"{overall['avg_clean_acc']*100:>5.1f}% {overall['avg_control_acc']*100:>5.1f}% {overall['avg_sae_only_acc']*100:>5.1f}% {overall['avg_experiment_acc']*100:>5.1f}% {overall['avg_validation_acc']*100:>5.1f}% "
          f"{overall['avg_exp_vs_control']:>+5.1f} {overall['avg_exp_vs_sae_only']:>+5.1f} {overall['avg_exp_vs_val']:>+5.1f}")
    print("=" * 95)

    # Save JSON
    out_path = OUT_DIR / "per_class_steering_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_class": results,
            "overall_average": overall,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Stats
    valid = [r for r in results if r['validation_acc'] is not None]
    if valid:
        n_exp_better = sum(1 for r in valid if r['experiment_acc'] > r['validation_acc'])
        n_exp_positive = sum(1 for r in valid if r['exp_vs_control'] > 0)
        n_sae_positive = sum(1 for r in results if r['sae_only_vs_control'] > 0)
        n_exp_better_sae = sum(1 for r in results if r['experiment_acc'] > r['sae_only_acc'])
        n_high_recovery = sum(1 for r in valid if r['recovery_rate'] is not None and r['recovery_rate'] >= 50)
        avg_recovery = sum(r['recovery_rate'] for r in valid if r['recovery_rate'] is not None) / \
                       sum(1 for r in valid if r['recovery_rate'] is not None)
        print(f"\nExperiment > Control:       {n_exp_positive}/{len(valid)} classes")
        print(f"SAE-only > Control:         {n_sae_positive}/{len(results)} classes")
        print(f"Experiment > SAE-only:      {n_exp_better_sae}/{len(results)} classes")
        print(f"Experiment > Validation:    {n_exp_better}/{len(valid)} classes")
        print(f"Recovery rate >= 50%:       {n_high_recovery}/{len(valid)} classes")
        print(f"Average recovery rate:      {avg_recovery:.1f}%")
        print(f"(Recovery rate = (Exp-Control)/(Clean-Control) × 100%)")
        print(f"\nKEY CAUSAL EVIDENCE:")
        print(f"  Exp vs Control (raw gain):     {overall['avg_exp_vs_control']:+.1f} pp")
        print(f"  Exp vs SAE-only (true effect): {overall['avg_exp_vs_sae_only']:+.1f} pp")
        print(f"  Exp vs Random (specificity):   {overall['avg_exp_vs_val']:+.1f} pp")

    print("=" * 95)


if __name__ == "__main__":
    main()
