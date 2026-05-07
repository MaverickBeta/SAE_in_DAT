#!/usr/bin/env python3
"""
Adaptive Attack on Dynamic Nearest Steering Defense.

Tests whether PGD, knowing the full defense pipeline, can bypass steering.
Uses BPDA (Backward Pass Differentiable Approximation) for the non-differentiable
nearest neighbor operation.

Experiment design:
  1. Load clean val images that are correctly classified by the model
  2. Undefended PGD: attack the raw model (baseline attack success rate)
  3. Adaptive PGD (identity BPDA): attack defended pipeline, backward=identity
  4. Adaptive PGD (soft BPDA): attack defended pipeline, backward=soft approx
  5. Evaluate ALL generated adversarial samples on:
     - Undefended model (should they fool the raw model?)
     - Defended model with actual hard steering (does defense hold?)

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    /Data_share/hongyi/conda_envs/rebm/bin/python adaptive_attack_20cls.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"
TRAIN_FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_latent")
CLEAN_VAL_ROOT = Path("/Data_share/hongyi/imagenet/val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/adaptive_attack_20cls")

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

N_CLASSES = len(CLASSES)
N_TOKENS = 49

# Attack parameters (match original APGD: L2, eps=3, 100 steps)
EPSILON_L2 = 3.0
NUM_STEPS = 100
ALPHA = 0.06  # 2 * eps / steps

# How many clean samples per class to attack (for speed; set to 50 for full eval)
N_SAMPLES_PER_CLASS = 10

# BPDA variants to test
BPDA_VARIANTS = [
    ("identity", None),        # Forward=hard, Backward=identity
    ("soft_T0.01", 0.01),      # Forward=hard, Backward=soft (T=0.01)
    ("soft_T0.05", 0.05),      # Forward=hard, Backward=soft (T=0.05)
    ("soft_T0.1", 0.1),        # Forward=hard, Backward=soft (T=0.1)
]

# ═══════════════════════════════════════════════════════════════════════
# Load model & SAE
# ═══════════════════════════════════════════════════════════════════════
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
N_CHANNELS = int(d_lat)
print(f"  SAE: d_in={d_in}, d_lat={d_lat}, k={k}")

# ═══════════════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ═══════════════════════════════════════════════════════════════════════
# Build Lookup Table (V1 only — best performer)
# ═══════════════════════════════════════════════════════════════════════
def build_lookup_table():
    print("\nBuilding lookup table (v1)...")
    clean_counts = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.int32)
    clean_sums = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float64)

    for cls_idx, (wnid, imagenet_idx, name) in enumerate(CLASSES):
        npy_path = TRAIN_FEAT_DIR / f"clean_{wnid}_cls{imagenet_idx}_stage3_k256_features.npy"
        clean = np.load(npy_path)  # (100, 49, 12288)
        mask = (clean != 0)
        clean_counts[cls_idx] = mask.sum(axis=0)
        clean_sums[cls_idx] = clean.sum(axis=0)

    clean_means = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        clean_means = np.where(clean_counts > 0, clean_sums / clean_counts, 0.0)

    total_counts = clean_counts.sum(axis=0)
    final_mask = total_counts >= 20

    lut_tensor = np.zeros((N_TOKENS, N_CHANNELS, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        lut_tensor[:, :, c] = clean_means[c]

    n_total = N_TOKENS * N_CHANNELS
    print(f"  Entries: {final_mask.sum():,} / {n_total:,} ({final_mask.sum() / n_total * 100:.2f}%)")

    return (
        torch.from_numpy(lut_tensor).to(DEVICE),
        torch.from_numpy(final_mask).to(DEVICE),
    )


LUT_TENSOR, LUT_MASK = build_lookup_table()


# ═══════════════════════════════════════════════════════════════════════
# Hard steering hook (actual defense, non-differentiable)
# ═══════════════════════════════════════════════════════════════════════
def make_hard_steering_hook():
    """The actual defense — used for evaluation, NOT during adaptive attack."""
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z = z.reshape(bsz, N_TOKENS, N_CHANNELS)

        active_mask = (z != 0)
        valid_mask = active_mask & LUT_MASK.unsqueeze(0)

        if valid_mask.any():
            b_idx, tok_idx, ch_idx = torch.where(valid_mask)
            current_vals = z[b_idx, tok_idx, ch_idx]
            class_means = LUT_TENSOR[tok_idx, ch_idx, :]
            diffs = torch.abs(current_vals.unsqueeze(1) - class_means)
            nearest_cls = torch.argmin(diffs, dim=1)
            M = b_idx.shape[0]
            targets = class_means[torch.arange(M, device=class_means.device), nearest_cls]
            z[b_idx, tok_idx, ch_idx] = targets

        z_flat = z.reshape(bsz * N_TOKENS, N_CHANNELS)
        recon_norm = (z_flat @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


# ═══════════════════════════════════════════════════════════════════════
# Adaptive Defended Model (BPDA)
# ═══════════════════════════════════════════════════════════════════════
class AdaptiveDefendedModel(nn.Module):
    """
    Wrapper that registers a BPDA forward hook for adaptive attack.

    Forward:  hard nearest steering (identical to actual defense)
    Backward: either identity or soft approximation
    """
    def __init__(self, model, sae, temperature=None, use_identity=False):
        super().__init__()
        self.model = model
        self.sae = sae
        self.temperature = temperature
        self.use_identity = use_identity
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.handle = None
        self._register_hook()

    def _bpda_hook(self, module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - self.norm_mean) / self.norm_std
        z = self.sae.encode(flat_norm)
        z = z.reshape(bsz, N_TOKENS, N_CHANNELS)

        active_mask = (z != 0)
        valid_mask = active_mask & LUT_MASK.unsqueeze(0)

        if valid_mask.any():
            b_idx, tok_idx, ch_idx = torch.where(valid_mask)
            current_vals = z[b_idx, tok_idx, ch_idx]
            class_means = LUT_TENSOR[tok_idx, ch_idx, :]  # (M, 20)

            diffs = torch.abs(current_vals.unsqueeze(1) - class_means)
            nearest_cls = torch.argmin(diffs, dim=1)
            M = b_idx.shape[0]
            hard_targets = class_means[torch.arange(M, device=class_means.device), nearest_cls]

            if self.use_identity:
                # Identity BPDA: forward=hard, backward=pass-through
                # Trick: hard_targets.detach() makes forward value = hard_targets
                #        (current_vals - current_vals.detach()) makes backward = identity
                steered = hard_targets.detach() + (current_vals - current_vals.detach())
            elif self.temperature is not None and self.temperature > 0:
                # Soft BPDA: forward=hard, backward=soft approximation
                soft_weights = F.softmax(-diffs / self.temperature, dim=1)
                soft_targets = (soft_weights * class_means).sum(dim=1)
                steered = hard_targets.detach() - soft_targets.detach() + soft_targets
            else:
                steered = hard_targets

            z = z.clone()
            z[b_idx, tok_idx, ch_idx] = steered

        z_flat = z.reshape(bsz * N_TOKENS, N_CHANNELS)
        recon_norm = (z_flat @ self.sae.W_dec) + self.sae.b_dec
        recon_flat = recon_norm * self.norm_std + self.norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon

    def _register_hook(self):
        self.handle = self.model.stages[3].register_forward_hook(self._bpda_hook)

    def forward(self, x):
        return self.model(x)

    def remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ═══════════════════════════════════════════════════════════════════════
# PGD L2 Attack
# ═══════════════════════════════════════════════════════════════════════
def pgd_l2_attack(images, labels, model, epsilon, steps, alpha):
    """
    PGD L2 untargeted attack.
    Returns adversarial images clamped to [0, 1].
    """
    x_adv = images.clone().detach()

    for step in range(steps):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # Normalize gradient to unit L2 norm
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = grad_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        grad_norm = grad_norm + 1e-10
        grad_normalized = grad / grad_norm

        # Gradient ascent step
        x_adv = x_adv.detach() + alpha * grad_normalized
        x_adv = x_adv.detach()

        # Project delta onto L2 ball
        delta = x_adv - images
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        delta_norm = delta_norm + 1e-10

        # Clamp norm to epsilon, preserve direction
        factor = torch.clamp(delta_norm, max=epsilon) / delta_norm
        delta = delta * factor

        x_adv = torch.clamp(images + delta, 0.0, 1.0)
        x_adv = x_adv.detach()

    return x_adv


# ═══════════════════════════════════════════════════════════════════════
# Load clean val images
# ═══════════════════════════════════════════════════════════════════════
def load_clean_val_images(max_per_class=N_SAMPLES_PER_CLASS):
    """Load clean val images that are correctly classified by the model."""
    all_images = []
    for wnid, cls_idx, name in CLASSES:
        class_dir = CLEAN_VAL_ROOT / wnid
        if not class_dir.is_dir():
            print(f"[WARN] {class_dir} not found, skipping")
            continue

        img_paths = sorted(class_dir.glob("*.JPEG"))
        loaded = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).to(DEVICE)
            loaded.append((img_t, cls_idx, str(path)))

        # Batch-check which are correctly classified
        correct_paths = []
        with torch.no_grad():
            for i in range(0, len(loaded), BATCH_SIZE):
                batch = loaded[i:i + BATCH_SIZE]
                imgs = torch.stack([x for x, _, _ in batch])
                labels = torch.tensor([y for _, y, _ in batch], device=DEVICE)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                for j, (_, label, path) in enumerate(batch):
                    if preds[j].item() == label:
                        correct_paths.append((imgs[j], label, path))

        # Take first max_per_class
        selected = correct_paths[:max_per_class]
        print(f"  [{name}] {len(selected)}/{len(loaded)} correctly classified (took first {len(selected)})")
        all_images.extend(selected)

    return all_images


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════
def evaluate_accuracy(images, labels, model_ref=None, use_defense=False):
    """Evaluate accuracy. If use_defense=True, uses hard steering hook."""
    if model_ref is None:
        model_ref = model

    hook = make_hard_steering_hook() if use_defense else None
    if hook is not None:
        handle = model_ref.stages[3].register_forward_hook(hook)

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]
            imgs = torch.stack(batch_imgs)
            lbls = torch.tensor(batch_labels, device=DEVICE)
            logits = model_ref(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += len(batch_imgs)

    if hook is not None:
        handle.remove()

    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 80)
    print("ADAPTIVE ATTACK ON DYNAMIC NEAREST STEERING")
    print("=" * 80)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load clean images
    print(f"\nLoading clean val images (max {N_SAMPLES_PER_CLASS}/class)...")
    clean_data = load_clean_val_images()
    if len(clean_data) == 0:
        print("No clean images loaded!")
        return

    clean_images = [x for x, _, _ in clean_data]
    clean_labels = [y for _, y, _ in clean_data]
    print(f"Total clean samples: {len(clean_images)}")

    # ── 1. Undefended PGD ──
    print("\n" + "-" * 80)
    print("[1/5] Undefended PGD (baseline)")
    print("-" * 80)
    adv_undefended = []
    for i in tqdm(range(0, len(clean_images), BATCH_SIZE), desc="Undefended PGD"):
        batch_imgs = torch.stack(clean_images[i:i + BATCH_SIZE])
        batch_labels = torch.tensor(clean_labels[i:i + BATCH_SIZE], device=DEVICE)
        adv_batch = pgd_l2_attack(batch_imgs, batch_labels, model, EPSILON_L2, NUM_STEPS, ALPHA)
        adv_undefended.extend([adv_batch[j] for j in range(adv_batch.size(0))])

    acc_undef_on_undef = evaluate_accuracy(adv_undefended, clean_labels, use_defense=False)
    acc_undef_on_def = evaluate_accuracy(adv_undefended, clean_labels, use_defense=True)
    print(f"  Undefended adv → Undefended model:  {acc_undef_on_undef * 100:.1f}% (attack success: {100 - acc_undef_on_undef * 100:.1f}%)")
    print(f"  Undefended adv → Defended model:    {acc_undef_on_def * 100:.1f}% (attack success: {100 - acc_undef_on_def * 100:.1f}%)")

    results = {
        "config": {
            "epsilon_l2": EPSILON_L2,
            "num_steps": NUM_STEPS,
            "alpha": ALPHA,
            "n_samples_per_class": N_SAMPLES_PER_CLASS,
            "total_clean_samples": len(clean_images),
        },
        "undefended_pgd": {
            "acc_on_undefended": acc_undef_on_undef,
            "acc_on_defended": acc_undef_on_def,
            "attack_success_undef": 1 - acc_undef_on_undef,
            "attack_success_def": 1 - acc_undef_on_def,
        },
        "adaptive": {},
    }

    # ── 2-5. Adaptive PGD with various BPDA approximations ──
    for variant_name, temp in BPDA_VARIANTS:
        print("\n" + "-" * 80)
        print(f"[Adaptive] {variant_name} (temp={temp})")
        print("-" * 80)

        use_id = (variant_name == "identity")
        adv_model = AdaptiveDefendedModel(model, sae, temperature=temp, use_identity=use_id)

        adv_batches = []
        for i in tqdm(range(0, len(clean_images), BATCH_SIZE), desc=f"Adaptive PGD ({variant_name})"):
            batch_imgs = torch.stack(clean_images[i:i + BATCH_SIZE])
            batch_labels = torch.tensor(clean_labels[i:i + BATCH_SIZE], device=DEVICE)
            adv_batch = pgd_l2_attack(batch_imgs, batch_labels, adv_model, EPSILON_L2, NUM_STEPS, ALPHA)
            adv_batches.extend([adv_batch[j] for j in range(adv_batch.size(0))])

        adv_model.remove_hook()
        del adv_model
        torch.cuda.empty_cache()

        # Evaluate generated adversarial samples
        acc_adv_on_undef = evaluate_accuracy(adv_batches, clean_labels, use_defense=False)
        acc_adv_on_def = evaluate_accuracy(adv_batches, clean_labels, use_defense=True)

        print(f"  Adaptive adv → Undefended model:    {acc_adv_on_undef * 100:.1f}% (attack success: {100 - acc_adv_on_undef * 100:.1f}%)")
        print(f"  Adaptive adv → Defended model:      {acc_adv_on_def * 100:.1f}% (attack success: {100 - acc_adv_on_def * 100:.1f}%)")

        results["adaptive"][variant_name] = {
            "temperature": temp,
            "use_identity": use_id,
            "acc_on_undefended": acc_adv_on_undef,
            "acc_on_defended": acc_adv_on_def,
            "attack_success_undef": 1 - acc_adv_on_undef,
            "attack_success_def": 1 - acc_adv_on_def,
        }

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY: Attack Success Rates")
    print("=" * 80)
    print(f"{'Attack Type':<25} {'vs Undefended':>15} {'vs Defended':>15} {'Defense Δ':>12}")
    print("-" * 80)

    base_undef = results["undefended_pgd"]["attack_success_undef"] * 100
    base_def = results["undefended_pgd"]["attack_success_def"] * 100
    print(f"{'Undefended PGD':<25} {base_undef:>14.1f}% {base_def:>14.1f}% {'—':>12}")

    for variant_name, data in results["adaptive"].items():
        su = data["attack_success_undef"] * 100
        sd = data["attack_success_def"] * 100
        delta = sd - base_def
        print(f"{variant_name:<25} {su:>14.1f}% {sd:>14.1f}% {delta:>+11.1f}%")

    print("=" * 80)
    print("\nInterpretation:")
    print("  • If 'vs Defended' is close to undefended PGD → defense BROKEN by adaptive attack")
    print("  • If 'vs Defended' stays low → defense ROBUST even against adaptive attack")
    print(f"  • Current non-adaptive baseline: attack success on defended = {base_def:.1f}%")

    # Save results
    out_path = OUT_DIR / "adaptive_attack_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
