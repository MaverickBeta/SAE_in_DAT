#!/usr/bin/env python3
"""
Dynamic nearest-class steering for CIFAR-10 WRN34x10 with on-the-fly APGD-CE L2 adversarial samples.
Multi-GPU accelerated (GPUs 4-5-6-7) with verbose progress reporting.

Usage:
    cd /Data_share/hongyi/DAT/SAE_WRN/inspection
    /Data_share/hongyi/conda_envs/rebm/bin/python dynamic_nearest_steering_cifar10.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.models.wide_resnet_innoutrobustness import WideResNet34x10
from rebm.attacks.attack_steps import L2Step
from sae_core.model import TopKAutoencoder

# ── Config ──────────────────────────────────────────────────────────
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE_WRN/checkpoints/k64_exp32_no_resample_ep10/epoch_10.pt"

TRAIN_FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE_WRN/features")
DATA_ROOT = "/Data_share/hongyi/DAT/data"
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE_WRN/inspection/results")

# Batch sizes tuned for 4x RTX 3090 (24GB)
BATCH_SIZE = 256          # eval single-GPU batch
ADV_BATCH_SIZE = 3072     # total across 4 GPUs (~768 per GPU)
LUT_BATCH_SIZE = 1024     # lookup table encoding per class
NUM_WORKERS = 16

DEVICE = torch.device("cuda:0")
N_CLASSES = 10
N_TOKENS = 64
D_IN = 640

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def print_gpu_stats(prefix=""):
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  [GPU {i+4}] {prefix} alloc={alloc:.2f}GB reserved={reserved:.2f}GB")


def time_elapsed(t0):
    return time.time() - t0


# ── Load model & SAE ────────────────────────────────────────────────
print("=" * 70)
print("PHASE 0: Loading model and SAE")
print("=" * 70)
t0 = time.time()

model = WideResNet34x10(
    num_classes=10,
    activation="relu",
    dropRate=0.0,
    return_feature_map=False,
    normalize_input=True,
    use_batchnorm=True,
)
ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt)
model = model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Base model loaded on {DEVICE} (physical GPU 4)")

# DataParallel model for attack generation (uses GPUs 4,5,6,7)
attack_model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
attack_model.eval()
print(f"  DataParallel attack model ready on GPUs 4-7")

sae_ckpt = torch.load(SAE_CKPT, map_location=DEVICE, weights_only=True)
config = sae_ckpt.get("config", {})
d_in = sae_ckpt.get("d_in", config.get("d_in", None))
d_lat = sae_ckpt.get("d_lat", config.get("d_lat", None))
k = config.get("k", sae_ckpt.get("k", None))
if d_lat is None:
    d_lat = int(d_in) * int(config.get("expansion_rate", 32))
if k is None:
    k = 64

sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
sae.load_state_dict(sae_ckpt["model_state_dict"])
sae = sae.to(DEVICE)
sae.eval()
for p in sae.parameters():
    p.requires_grad = False

norm_mean = sae_ckpt["norm_mean"].to(DEVICE)
norm_std = sae_ckpt["norm_std"].to(DEVICE)
print(f"  SAE: d_in={d_in}, d_lat={d_lat}, k={k}")
print(f"  Phase 0 done in {time_elapsed(t0):.1f}s")

N_CHANNELS = int(d_lat)

# ── CIFAR-10 dataset ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: Loading CIFAR-10 test set")
print("=" * 70)
t0 = time.time()

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
print(f"  CIFAR-10 test set: {len(test_dataset)} images")

all_clean_images = []
all_labels = []
for imgs, lbls in tqdm(test_loader, desc="  Loading to CPU tensors", ncols=70):
    all_clean_images.append(imgs)
    all_labels.append(lbls)
all_clean_images = torch.cat(all_clean_images, dim=0)  # [10000, 3, 32, 32]
all_labels = torch.cat(all_labels, dim=0)              # [10000]
print(f"  Collected {all_clean_images.shape[0]} clean images.")
print(f"  Phase 1 done in {time_elapsed(t0):.1f}s")


# ── Build Lookup Table ──────────────────────────────────────────────
def build_lookup_table(version):
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Building lookup table ({version})")
    print(f"{'=' * 70}")
    t0 = time.time()

    lut_cache_path = OUT_DIR / f"lookup_table_{version}.pt"
    if lut_cache_path.exists():
        print(f"  Found cached lookup table at {lut_cache_path}, loading...")
        cached = torch.load(lut_cache_path, map_location=DEVICE, weights_only=True)
        return cached["lut_tensor"], cached["lut_mask"]

    clean_counts = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.int32)
    clean_sums = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float64)

    for cls_idx, name in enumerate(CLASS_NAMES):
        feat_path = TRAIN_FEAT_DIR / f"{name}.pt"
        activations = torch.load(feat_path, map_location="cpu", weights_only=True)  # [5000, 640, 8, 8]
        N = activations.shape[0]
        n_batches = (N + LUT_BATCH_SIZE - 1) // LUT_BATCH_SIZE
        print(f"  [{cls_idx+1}/{N_CLASSES}] {name}: {N} samples, {n_batches} batches")

        for i in range(0, N, LUT_BATCH_SIZE):
            batch = activations[i:i + LUT_BATCH_SIZE].to(DEVICE)
            b = batch.size(0)
            flat = batch.permute(0, 2, 3, 1).reshape(-1, D_IN)
            flat_norm = (flat - norm_mean) / norm_std
            with torch.no_grad():
                z = sae.encode(flat_norm)
            z = z.reshape(b, N_TOKENS, N_CHANNELS).cpu().numpy()

            mask = (z != 0)
            clean_counts[cls_idx] += mask.sum(axis=0)
            clean_sums[cls_idx] += z.sum(axis=0)

    clean_means = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        clean_means = np.where(clean_counts > 0, clean_sums / clean_counts, 0.0)

    total_counts = clean_counts.sum(axis=0)
    layer1_mask = total_counts >= 20

    if version == "v1":
        final_mask = layer1_mask
        print(f"  Layer 1 (total>=20): {layer1_mask.sum():,} entries")
    elif version == "v2":
        class_active = (clean_counts >= 5).sum(axis=0)
        layer2_mask = class_active >= 2
        final_mask = layer1_mask & layer2_mask
        print(f"  Layer 1 (total>=20): {layer1_mask.sum():,} entries")
        print(f"  Layer 2 (>=2 classes>=5): {layer2_mask.sum():,} entries")
    else:
        raise ValueError(f"Unknown version: {version}")

    lut_tensor = np.zeros((N_TOKENS, N_CHANNELS, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        lut_tensor[:, :, c] = clean_means[c]

    lut_mask = final_mask
    n_total = N_TOKENS * N_CHANNELS
    print(f"  Final entries: {final_mask.sum():,} / {n_total:,} ({final_mask.sum() / n_total * 100:.2f}%)")

    lut_tensor_t = torch.from_numpy(lut_tensor).to(DEVICE)
    lut_mask_t = torch.from_numpy(lut_mask).to(DEVICE)

    # Cache for future runs
    torch.save({"lut_tensor": lut_tensor_t, "lut_mask": lut_mask_t}, lut_cache_path)
    print(f"  Cached to {lut_cache_path}")
    print(f"  Phase 2 ({version}) done in {time_elapsed(t0):.1f}s")

    return lut_tensor_t, lut_mask_t


# ── APGD-CE L2 Attack ───────────────────────────────────────────────
def apgd_ce_attack_l2(dp_model, x, labels, eps=1.0, steps=100, step_size=None, random_start=True):
    if step_size is None:
        step_size = eps / 4

    x0 = x.clone().detach()
    step = L2Step(orig_input=x0, eps=eps, step_size=step_size, use_grad=True)

    if random_start:
        x = step.random_perturb(x0)

    for step_idx in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = dp_model(x)
        loss = F.cross_entropy(logits, labels)
        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            retain_graph=False,
            create_graph=False,
        )
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x.clone().detach()


# ── Hook factories ──────────────────────────────────────────────────
def make_sae_only_hook():
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        recon_norm = (z @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


def make_dynamic_nearest_hook(lut_tensor, lut_mask):
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z = z.reshape(bsz, N_TOKENS, N_CHANNELS)

        active_mask = (z != 0)
        valid_mask = active_mask & lut_mask.unsqueeze(0)

        if valid_mask.any():
            b_idx, tok_idx, ch_idx = torch.where(valid_mask)
            current_vals = z[b_idx, tok_idx, ch_idx]
            class_means = lut_tensor[tok_idx, ch_idx, :]
            diffs = torch.abs(current_vals.unsqueeze(1) - class_means)
            nearest_cls = torch.argmin(diffs, dim=1)
            M = b_idx.shape[0]
            targets = class_means[torch.arange(M, device=class_means.device), nearest_cls]
            z[b_idx, tok_idx, ch_idx] = targets

        z_flat = z.reshape(bsz * N_TOKENS, N_CHANNELS)
        recon_norm = (z_flat @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn


# ── Evaluation helpers ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_detailed_tensor(images, labels, hook_fn=None):
    if hook_fn is not None:
        handle = model.activation.register_forward_hook(hook_fn)
    preds = []
    try:
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i : i + BATCH_SIZE].to(DEVICE)
            logits = model(batch)
            batch_preds = logits.argmax(dim=1).cpu()
            preds.append(batch_preds)
    finally:
        if hook_fn is not None:
            handle.remove()

    preds = torch.cat(preds, dim=0)
    acc = (preds == labels).float().mean().item() if len(images) > 0 else 0.0
    return acc, preds


def compute_quadrants(control_preds, dynamic_preds, labels):
    a = b = c = d = 0
    for cp, dp, lbl in zip(control_preds, dynamic_preds, labels):
        c_ok = (cp == lbl)
        d_ok = (dp == lbl)
        if c_ok and d_ok:
            a += 1
        elif c_ok and not d_ok:
            b += 1
        elif not c_ok and d_ok:
            c += 1
        else:
            d += 1
    return {
        "unchanged_correct": a,
        "regression": b,
        "recovery": c,
        "unchanged_wrong": d,
        "net_gain": c - b,
        "n_total": a + b + c + d,
    }


# ── Main ────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build lookup tables
    lut_v1, mask_v1 = build_lookup_table("v1")
    lut_v2, mask_v2 = build_lookup_table("v2")

    # ── Generate adversarial samples on-the-fly ─────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 3: Generating APGD-CE L2 adversarial samples (multi-GPU)")
    print(f"{'=' * 70}")
    print(f"  Attack config: L2, eps=1.0, steps=100, step_size=0.25")
    print(f"  DataParallel batch size: {ADV_BATCH_SIZE} total ({ADV_BATCH_SIZE // 4} per GPU)")
    print(f"  Samples: {len(all_clean_images)}")
    t0 = time.time()

    adv_save_path = OUT_DIR / "adv_samples_eps1.0.pt"
    if adv_save_path.exists():
        print(f"  Found cached adversarial samples at {adv_save_path}, loading...")
        adv_data = torch.load(adv_save_path, map_location="cpu", weights_only=True)
        all_adv_images = adv_data["adv_images"]
        all_clean_preds = adv_data["clean_preds"]
        all_adv_preds = adv_data["adv_preds"]
        print(f"  Loaded {len(all_adv_images)} adversarial samples.")
    else:
        all_adv_images = []
        all_clean_preds = []
        n_batches = (len(all_clean_images) + ADV_BATCH_SIZE - 1) // ADV_BATCH_SIZE

        for batch_idx, i in enumerate(range(0, len(all_clean_images), ADV_BATCH_SIZE)):
            batch_t0 = time.time()
            batch_imgs = all_clean_images[i : i + ADV_BATCH_SIZE].to(DEVICE)
            batch_labels = all_labels[i : i + ADV_BATCH_SIZE].to(DEVICE)

            with torch.no_grad():
                logits_clean = attack_model(batch_imgs)
                pred_clean = logits_clean.argmax(dim=1)

            adv_imgs = apgd_ce_attack_l2(
                dp_model=attack_model,
                x=batch_imgs,
                labels=batch_labels,
                eps=1.0,
                steps=100,
                step_size=None,
                random_start=True,
            )

            all_adv_images.append(adv_imgs.cpu())
            all_clean_preds.append(pred_clean.cpu())

            batch_dt = time.time() - batch_t0
            eta = batch_dt * (n_batches - batch_idx - 1)
            print(f"  [{batch_idx+1}/{n_batches}] batch_size={batch_imgs.size(0)}, time={batch_dt:.1f}s, ETA={eta/60:.1f}min")
            if batch_idx % 2 == 0:
                print_gpu_stats(prefix="mid-attack ")

        all_adv_images = torch.cat(all_adv_images, dim=0)
        all_clean_preds = torch.cat(all_clean_preds, dim=0)

        # Compute adv preds in one go
        print("  Computing adversarial predictions...")
        adv_pred_batches = []
        for i in range(0, len(all_adv_images), BATCH_SIZE):
            batch = all_adv_images[i:i+BATCH_SIZE].to(DEVICE)
            with torch.no_grad():
                adv_pred_batches.append(model(batch).argmax(dim=1).cpu())
        all_adv_preds = torch.cat(adv_pred_batches, dim=0)

        # Cache adversarial samples
        torch.save({
            "adv_images": all_adv_images,
            "clean_preds": all_clean_preds,
            "adv_preds": all_adv_preds,
            "labels": all_labels,
        }, adv_save_path)
        print(f"  Cached adversarial samples to {adv_save_path}")

    overall_clean_acc = (all_clean_preds == all_labels).float().mean().item()
    overall_adv_acc = (all_adv_preds == all_labels).float().mean().item()
    overall_asr = 1.0 - overall_adv_acc
    print(f"\n  Overall: Clean Acc={overall_clean_acc*100:.2f}%, Adv Acc={overall_adv_acc*100:.2f}%, ASR={overall_asr*100:.2f}%")
    print(f"  Phase 3 done in {time_elapsed(t0):.1f}s")

    # ── Per-class evaluation ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 4: Per-class evaluation with SAE hooks")
    print(f"{'=' * 70}")
    t0 = time.time()

    results = []

    for cls_idx, name in enumerate(CLASS_NAMES):
        cls_t0 = time.time()
        cls_mask = (all_labels == cls_idx)
        cls_clean = all_clean_images[cls_mask]
        cls_labels = all_labels[cls_mask]
        cls_adv = all_adv_images[cls_mask]

        cls_clean_preds = all_clean_preds[cls_mask]
        cls_adv_preds = all_adv_preds[cls_mask]
        succ_mask = (cls_clean_preds == cls_labels) & (cls_adv_preds != cls_labels)
        cls_succ_adv = cls_adv[succ_mask]
        cls_succ_labels = cls_labels[succ_mask]

        n_all = len(cls_adv)
        n_succ = len(cls_succ_adv)
        n_clean = len(cls_clean)

        if n_all == 0:
            continue

        print(f"\n  [{name}] cls={cls_idx} | all={n_all} succ={n_succ} clean={n_clean}")

        _, all_ctrl_preds = evaluate_detailed_tensor(cls_adv, cls_labels)
        _, succ_ctrl_preds = evaluate_detailed_tensor(cls_succ_adv, cls_succ_labels) if n_succ > 0 else (0.0, torch.tensor([], dtype=torch.long))

        all_sae_acc, all_sae_preds = evaluate_detailed_tensor(cls_adv, cls_labels, make_sae_only_hook())
        succ_sae_acc, succ_sae_preds = evaluate_detailed_tensor(cls_succ_adv, cls_succ_labels, make_sae_only_hook()) if n_succ > 0 else (None, torch.tensor([], dtype=torch.long))

        all_v1_acc, all_v1_preds = evaluate_detailed_tensor(cls_adv, cls_labels, make_dynamic_nearest_hook(lut_v1, mask_v1))
        succ_v1_acc, succ_v1_preds = evaluate_detailed_tensor(cls_succ_adv, cls_succ_labels, make_dynamic_nearest_hook(lut_v1, mask_v1)) if n_succ > 0 else (None, torch.tensor([], dtype=torch.long))

        all_v2_acc, all_v2_preds = evaluate_detailed_tensor(cls_adv, cls_labels, make_dynamic_nearest_hook(lut_v2, mask_v2))
        succ_v2_acc, succ_v2_preds = evaluate_detailed_tensor(cls_succ_adv, cls_succ_labels, make_dynamic_nearest_hook(lut_v2, mask_v2)) if n_succ > 0 else (None, torch.tensor([], dtype=torch.long))

        clean_ctrl_acc, _ = evaluate_detailed_tensor(cls_clean, cls_labels)
        clean_sae_acc, _ = evaluate_detailed_tensor(cls_clean, cls_labels, make_sae_only_hook())
        clean_v1_acc, _ = evaluate_detailed_tensor(cls_clean, cls_labels, make_dynamic_nearest_hook(lut_v1, mask_v1))
        clean_v2_acc, _ = evaluate_detailed_tensor(cls_clean, cls_labels, make_dynamic_nearest_hook(lut_v2, mask_v2))

        all_sae_q = compute_quadrants(all_ctrl_preds, all_sae_preds, cls_labels)
        succ_sae_q = compute_quadrants(succ_ctrl_preds, succ_sae_preds, cls_succ_labels) if n_succ > 0 else None
        all_v1_q = compute_quadrants(all_ctrl_preds, all_v1_preds, cls_labels)
        succ_v1_q = compute_quadrants(succ_ctrl_preds, succ_v1_preds, cls_succ_labels) if n_succ > 0 else None
        all_v2_q = compute_quadrants(all_ctrl_preds, all_v2_preds, cls_labels)
        succ_v2_q = compute_quadrants(succ_ctrl_preds, succ_v2_preds, cls_succ_labels) if n_succ > 0 else None

        ctrl_all = all_ctrl_preds.eq(cls_labels).float().mean().item()
        ctrl_succ = succ_ctrl_preds.eq(cls_succ_labels).float().mean().item() if n_succ > 0 else None

        print(f"    All:  Ctrl={ctrl_all*100:.1f}% SAE={all_sae_acc*100:.1f}% V1={all_v1_acc*100:.1f}% V2={all_v2_acc*100:.1f}%")
        print(f"    Clean: Ctrl={clean_ctrl_acc*100:.1f}% SAE={clean_sae_acc*100:.1f}% V1={clean_v1_acc*100:.1f}% V2={clean_v2_acc*100:.1f}%")
        if succ_v1_q:
            print(f"    Succ V1 net: {succ_v1_q['net_gain']:+d} (rec={succ_v1_q['recovery']}, reg={succ_v1_q['regression']})")
        print(f"    Class done in {time_elapsed(cls_t0):.1f}s")

        results.append(
            {
                "name": name,
                "class_idx": cls_idx,
                "n_all_adv": n_all,
                "n_succ_adv": n_succ,
                "n_clean": n_clean,
                "all_control_acc": ctrl_all,
                "all_sae_only_acc": all_sae_acc,
                "all_v1_acc": all_v1_acc,
                "all_v2_acc": all_v2_acc,
                "succ_control_acc": ctrl_succ,
                "succ_sae_only_acc": succ_sae_acc,
                "succ_v1_acc": succ_v1_acc,
                "succ_v2_acc": succ_v2_acc,
                "clean_control_acc": clean_ctrl_acc,
                "clean_sae_only_acc": clean_sae_acc,
                "clean_v1_acc": clean_v1_acc,
                "clean_v2_acc": clean_v2_acc,
                "all_sae_vs_control": (all_sae_acc - ctrl_all) * 100,
                "all_v1_vs_control": (all_v1_acc - ctrl_all) * 100,
                "all_v2_vs_control": (all_v2_acc - ctrl_all) * 100,
                "succ_sae_vs_control": (succ_sae_acc - ctrl_succ) * 100 if n_succ > 0 and succ_sae_acc is not None else None,
                "succ_v1_vs_control": (succ_v1_acc - ctrl_succ) * 100 if n_succ > 0 and succ_v1_acc is not None else None,
                "succ_v2_vs_control": (succ_v2_acc - ctrl_succ) * 100 if n_succ > 0 and succ_v2_acc is not None else None,
                "clean_sae_vs_control": (clean_sae_acc - clean_ctrl_acc) * 100,
                "clean_v1_vs_control": (clean_v1_acc - clean_ctrl_acc) * 100,
                "clean_v2_vs_control": (clean_v2_acc - clean_ctrl_acc) * 100,
                "all_sae_quadrants": all_sae_q,
                "succ_sae_quadrants": succ_sae_q,
                "all_v1_quadrants": all_v1_q,
                "succ_v1_quadrants": succ_v1_q,
                "all_v2_quadrants": all_v2_q,
                "succ_v2_quadrants": succ_v2_q,
            }
        )

    print(f"\n  Phase 4 done in {time_elapsed(t0):.1f}s")

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_all_control_acc": avg("all_control_acc"),
        "avg_all_sae_only_acc": avg("all_sae_only_acc"),
        "avg_all_v1_acc": avg("all_v1_acc"),
        "avg_all_v2_acc": avg("all_v2_acc"),
        "avg_succ_control_acc": avg("succ_control_acc"),
        "avg_succ_sae_only_acc": avg("succ_sae_only_acc"),
        "avg_succ_v1_acc": avg("succ_v1_acc"),
        "avg_succ_v2_acc": avg("succ_v2_acc"),
        "avg_clean_control_acc": avg("clean_control_acc"),
        "avg_clean_sae_only_acc": avg("clean_sae_only_acc"),
        "avg_clean_v1_acc": avg("clean_v1_acc"),
        "avg_clean_v2_acc": avg("clean_v2_acc"),
        "avg_all_sae_vs_control": avg("all_sae_vs_control"),
        "avg_all_v1_vs_control": avg("all_v1_vs_control"),
        "avg_all_v2_vs_control": avg("all_v2_vs_control"),
        "avg_succ_sae_vs_control": avg("succ_sae_vs_control"),
        "avg_succ_v1_vs_control": avg("succ_v1_vs_control"),
        "avg_succ_v2_vs_control": avg("succ_v2_vs_control"),
        "avg_clean_sae_vs_control": avg("clean_sae_vs_control"),
        "avg_clean_v1_vs_control": avg("clean_v1_vs_control"),
        "avg_clean_v2_vs_control": avg("clean_v2_vs_control"),
    }

    # ── Quadrant Summary ────────────────────────────────────────────
    def sum_quad(key):
        total = {"unchanged_correct": 0, "regression": 0, "recovery": 0, "unchanged_wrong": 0, "net_gain": 0, "n_total": 0}
        qkey = f"{key}_quadrants"
        for r in results:
            q = r.get(qkey)
            if q:
                for k_total in total:
                    total[k_total] += q.get(k_total, 0)
        return total

    all_sae_q_total = sum_quad("all_sae")
    succ_sae_q_total = sum_quad("succ_sae")
    all_v1_q_total = sum_quad("all_v1")
    succ_v1_q_total = sum_quad("succ_v1")
    all_v2_q_total = sum_quad("all_v2")
    succ_v2_q_total = sum_quad("succ_v2")

    def print_quad_table(title, q_total):
        if q_total["n_total"] == 0:
            return
        uc = q_total["unchanged_correct"]
        re = q_total["regression"]
        rc = q_total["recovery"]
        uw = q_total["unchanged_wrong"]
        nt = q_total["n_total"]
        print(f"\n{title}")
        print("-" * 70)
        print(f"  Unchanged correct (A):  {uc:>4} / {nt} ({uc/nt*100:>5.1f}%)")
        print(f"  Regression      (B):  {re:>4} / {nt} ({re/nt*100:>5.1f}%)")
        print(f"  Recovery        (C):  {rc:>4} / {nt} ({rc/nt*100:>5.1f}%)")
        print(f"  Unchanged wrong (D):  {uw:>4} / {nt} ({uw/nt*100:>5.1f}%)")
        print(f"  Net gain (C - B):     {q_total['net_gain']:+d}")
        print("-" * 70)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 160)
    print("RESULTS SUMMARY (Ctrl vs SAE-only vs Dynamic on succ / all / clean)")
    print("=" * 160)
    print(
        f"{'Class':>10} {'Ver':>4} {'N_succ':>6} {'N_all':>6} {'N_clean':>7} "
        f"{'SuccCtrl':>8} {'SuccSAE':>8} {'SuccV1':>8} {'SuccV2':>8} "
        f"{'AllCtrl':>8} {'AllSAE':>8} {'AllV1':>8} {'AllV2':>8} "
        f"{'CleanCtrl':>9} {'CleanSAE':>9} {'CleanV1':>9} {'CleanV2':>9}"
    )
    print("-" * 160)

    for r in results:
        sc = f"{r['succ_control_acc']*100:>7.1f}%" if r['succ_control_acc'] is not None else f"{'N/A':>8}"
        ac = f"{r['all_control_acc']*100:>7.1f}%"
        cc = f"{r['clean_control_acc']*100:>7.1f}%"

        ssae = f"{r['succ_sae_only_acc']*100:>7.1f}%" if r['succ_sae_only_acc'] is not None else f"{'N/A':>8}"
        sv1 = f"{r['succ_v1_acc']*100:>7.1f}%" if r['succ_v1_acc'] is not None else f"{'N/A':>8}"
        sv2 = f"{r['succ_v2_acc']*100:>7.1f}%" if r['succ_v2_acc'] is not None else f"{'N/A':>8}"

        asae = f"{r['all_sae_only_acc']*100:>7.1f}%"
        av1 = f"{r['all_v1_acc']*100:>7.1f}%"
        av2 = f"{r['all_v2_acc']*100:>7.1f}%"

        csae = f"{r['clean_sae_only_acc']*100:>7.1f}%"
        cv1 = f"{r['clean_v1_acc']*100:>7.1f}%"
        cv2 = f"{r['clean_v2_acc']*100:>7.1f}%"

        print(
            f"{r['name']:>10} {'SAE':>4} {r['n_succ_adv']:>6} {r['n_all_adv']:>6} {r['n_clean']:>7} "
            f"{sc} {ssae} {sv1} {sv2} "
            f"{ac} {asae} {av1} {av2} "
            f"{cc} {csae} {cv1} {cv2}"
        )
        print(
            f"{'':>10} {'V1':>4} {r['n_succ_adv']:>6} {r['n_all_adv']:>6} {r['n_clean']:>7} "
            f"{sc} {ssae} {sv1} {sv2} "
            f"{ac} {asae} {av1} {av2} "
            f"{cc} {csae} {cv1} {cv2}"
        )
        print(
            f"{'':>10} {'V2':>4} {r['n_succ_adv']:>6} {r['n_all_adv']:>6} {r['n_clean']:>7} "
            f"{sc} {ssae} {sv1} {sv2} "
            f"{ac} {asae} {av1} {av2} "
            f"{cc} {csae} {cv1} {cv2}"
        )

    print("-" * 160)
    sc_ov = f"{overall['avg_succ_control_acc']*100:>7.1f}%"
    ac_ov = f"{overall['avg_all_control_acc']*100:>7.1f}%"
    cc_ov = f"{overall['avg_clean_control_acc']*100:>7.1f}%"

    print(
        f"{'OVERALL':>10} {'SAE':>4} {'':>6} {'':>6} {'':>7} "
        f"{sc_ov} {overall['avg_succ_sae_only_acc']*100:>7.1f}% {overall['avg_succ_v1_acc']*100:>7.1f}% {overall['avg_succ_v2_acc']*100:>7.1f}% "
        f"{ac_ov} {overall['avg_all_sae_only_acc']*100:>7.1f}% {overall['avg_all_v1_acc']*100:>7.1f}% {overall['avg_all_v2_acc']*100:>7.1f}% "
        f"{cc_ov} {overall['avg_clean_sae_only_acc']*100:>7.1f}% {overall['avg_clean_v1_acc']*100:>7.1f}% {overall['avg_clean_v2_acc']*100:>7.1f}%"
    )
    print("=" * 160)

    print_quad_table("QUADRANT ANALYSIS — All samples + SAE-only", all_sae_q_total)
    print_quad_table("QUADRANT ANALYSIS — All samples + V1", all_v1_q_total)
    print_quad_table("QUADRANT ANALYSIS — All samples + V2", all_v2_q_total)
    print_quad_table("QUADRANT ANALYSIS — Succ samples + SAE-only", succ_sae_q_total)
    print_quad_table("QUADRANT ANALYSIS — Succ samples + V1", succ_v1_q_total)
    print_quad_table("QUADRANT ANALYSIS — Succ samples + V2", succ_v2_q_total)

    # Save JSON
    out_path = OUT_DIR / "dynamic_nearest_results_cifar10.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "per_class": results,
                "overall_average": overall,
                "meta": {
                    "base_ckpt": BASE_CKPT,
                    "sae_ckpt": SAE_CKPT,
                    "attack": "apgd_ce_l2_eps1_steps100",
                    "n_classes": N_CLASSES,
                    "n_tokens": N_TOKENS,
                    "n_channels": N_CHANNELS,
                    "d_in": D_IN,
                    "adv_batch_size": ADV_BATCH_SIZE,
                    "gpus": "4,5,6,7",
                },
            },
            f,
            indent=2,
            default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x,
        )
    print(f"\nResults saved: {out_path}")

    print(f"\nKEY METRICS (vs Control):")
    print(f"  All  SAE: {overall['avg_all_sae_vs_control']:+.1f} pp")
    print(f"  All  V1:  {overall['avg_all_v1_vs_control']:+.1f} pp")
    print(f"  All  V2:  {overall['avg_all_v2_vs_control']:+.1f} pp")
    print(f"  Succ SAE: {overall['avg_succ_sae_vs_control']:+.1f} pp")
    print(f"  Succ V1:  {overall['avg_succ_v1_vs_control']:+.1f} pp")
    print(f"  Succ V2:  {overall['avg_succ_v2_vs_control']:+.1f} pp")
    print(f"  Clean SAE: {overall['avg_clean_sae_vs_control']:+.1f} pp")
    print(f"  Clean V1:  {overall['avg_clean_v1_vs_control']:+.1f} pp")
    print(f"  Clean V2:  {overall['avg_clean_v2_vs_control']:+.1f} pp")


if __name__ == "__main__":
    main()
