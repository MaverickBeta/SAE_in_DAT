#!/usr/bin/env python3
"""
Grid search over lookup table thresholds for dynamic nearest steering.

Precomputes:
  1. latent_stats.npz  (clean_counts, clean_sums from train features)
  2. adv_samples_eps1.0.pt  (adversarial samples, if not exists)
  3. base_predictions.pt  (Control & SAE-only preds for clean/all/succ)

Then grid-searches V1/V2 threshold combinations and saves results.

Usage:
    cd /Data_share/hongyi/DAT/SAE_WRN/inspection
    CUDA_VISIBLE_DEVICES=4,5,6,7 /Data_share/hongyi/conda_envs/rebm/bin/python grid_search_thresholds.py
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

BATCH_SIZE = 512
ADV_BATCH_SIZE = 3072
LUT_BATCH_SIZE = 1024
NUM_WORKERS = 16

DEVICE = torch.device("cuda:0")
N_CLASSES = 10
N_TOKENS = 64
N_CHANNELS = 20480
D_IN = 640

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── Search space ────────────────────────────────────────────────────
# Tuned for CIFAR-10 (50k train images, k=64, d_lat=20480)
# Based on original ImageNet design: threshold ≈ 1% of expected activations
# Our expected activation per channel ≈ 10,000 → threshold range: 100~5000
V1_LAYER1_THRESHOLDS = [50, 100, 200, 500, 1000, 2000, 5000]

V2_LAYER1_THRESHOLDS = [100, 500]      # covered by layer2, minimal impact
V2_CLASS_THRESHOLDS = [50, 100, 200, 500]
V2_MIN_CLASSES = [3, 5, 8]


def time_elapsed(t0):
    return time.time() - t0


# ── Load model & SAE ────────────────────────────────────────────────
print("=" * 70)
print("Loading model and SAE")
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

attack_model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
attack_model.eval()
print(f"  DataParallel attack model on GPUs 4-7")

# eval stays single-GPU (hook + SAE cross-device issues with DataParallel)
print(f"  Eval: single GPU (cuda:0)")

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
print(f"  Loaded in {time_elapsed(t0):.1f}s")


# ── CIFAR-10 test set ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading CIFAR-10 test set")
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

all_clean_images = []
all_labels = []
for imgs, lbls in tqdm(test_loader, desc="  Loading", ncols=70):
    all_clean_images.append(imgs)
    all_labels.append(lbls)
all_clean_images = torch.cat(all_clean_images, dim=0)  # [10000, 3, 32, 32]
all_labels = torch.cat(all_labels, dim=0)              # [10000]
print(f"  {len(all_clean_images)} images loaded in {time_elapsed(t0):.1f}s")


# ── Helpers (defined before use) ────────────────────────────────────
def make_sae_hook():
    def hook_fn(module, input, output):
        bsz, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        recon_norm = (z @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return recon
    return hook_fn


def make_dynamic_hook(lut_tensor, lut_mask):
    def hook_fn(module, input, output):
        bsz, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
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
        recon = recon_flat.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return recon
    return hook_fn


def build_lut_from_stats(clean_counts, clean_sums, version, params):
    clean_means = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        clean_means = np.where(clean_counts > 0, clean_sums / clean_counts, 0.0)

    total_counts = clean_counts.sum(axis=0)

    if version == "v1":
        t1 = params["layer1_threshold"]
        layer1_mask = total_counts >= t1
        final_mask = layer1_mask
    elif version == "v2":
        t1 = params["layer1_threshold"]
        t_cls = params["layer2_class_threshold"]
        min_cls = params["layer2_min_classes"]
        layer1_mask = total_counts >= t1
        class_active = (clean_counts >= t_cls).sum(axis=0)
        layer2_mask = class_active >= min_cls
        final_mask = layer1_mask & layer2_mask
    else:
        raise ValueError(version)

    lut_tensor = np.zeros((N_TOKENS, N_CHANNELS, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        lut_tensor[:, :, c] = clean_means[c]

    return (
        torch.from_numpy(lut_tensor).to(DEVICE),
        torch.from_numpy(final_mask).to(DEVICE),
        final_mask.sum(),
    )


def evaluate_config(version, params, lut_tensor, lut_mask, all_clean_images, all_adv_images,
                    all_labels, succ_mask, clean_ctrl_preds, clean_sae_preds,
                    all_ctrl_preds, all_sae_preds):
    """Evaluate one threshold config. Returns dict with metrics."""
    hook = make_dynamic_hook(lut_tensor, lut_mask)

    def eval_preds(images):
        handle = model.activation.register_forward_hook(hook)
        preds = []
        try:
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE].to(DEVICE)
                logits = model(batch)
                preds.append(logits.argmax(dim=1).cpu())
        finally:
            handle.remove()
        return torch.cat(preds, dim=0)

    clean_dyn_preds = eval_preds(all_clean_images)
    all_dyn_preds = eval_preds(all_adv_images)

    results = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        cls_mask = (all_labels == cls_idx)
        cls_succ_mask = cls_mask & succ_mask

        n_all = cls_mask.sum().item()
        n_succ = cls_succ_mask.sum().item()
        n_clean = n_all

        cc = (clean_ctrl_preds[cls_mask] == cls_idx).float().mean().item()
        cs = (clean_sae_preds[cls_mask] == cls_idx).float().mean().item()
        cd = (clean_dyn_preds[cls_mask] == cls_idx).float().mean().item()

        ac = (all_ctrl_preds[cls_mask] == cls_idx).float().mean().item()
        ad = (all_dyn_preds[cls_mask] == cls_idx).float().mean().item()

        if n_succ > 0:
            sc = (all_ctrl_preds[cls_succ_mask] == cls_idx).float().mean().item()
            sd = (all_dyn_preds[cls_succ_mask] == cls_idx).float().mean().item()
        else:
            sc = sd = None

        results.append({
            "name": name, "class_idx": cls_idx,
            "n_all_adv": n_all, "n_succ_adv": n_succ, "n_clean": n_clean,
            "all_control_acc": ac, "all_dynamic_acc": ad,
            "succ_control_acc": sc, "succ_dynamic_acc": sd,
            "clean_control_acc": cc, "clean_sae_acc": cs, "clean_dynamic_acc": cd,
        })

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "avg_all_control_acc": avg("all_control_acc"),
        "avg_all_dynamic_acc": avg("all_dynamic_acc"),
        "avg_all_vs_control": (avg("all_dynamic_acc") - avg("all_control_acc")) * 100,
        "avg_succ_control_acc": avg("succ_control_acc"),
        "avg_succ_dynamic_acc": avg("succ_dynamic_acc"),
        "avg_succ_vs_control": (avg("succ_dynamic_acc") - avg("succ_control_acc")) * 100,
        "avg_clean_control_acc": avg("clean_control_acc"),
        "avg_clean_sae_acc": avg("clean_sae_acc"),
        "avg_clean_dynamic_acc": avg("clean_dynamic_acc"),
        "avg_clean_dynamic_vs_control": (avg("clean_dynamic_acc") - avg("clean_control_acc")) * 100,
    }

    return {"per_class": results, "overall": overall}


# ── Phase 1: Precompute latent stats ────────────────────────────────
print("\n" + "=" * 70)
print("Phase 1: Precomputing latent stats from train features")
print("=" * 70)
t0 = time.time()

stats_path = OUT_DIR / "latent_stats.npz"
if stats_path.exists():
    print(f"  Loading cached stats from {stats_path}")
    data = np.load(stats_path)
    clean_counts = data["clean_counts"]
    clean_sums = data["clean_sums"]
    print(f"  Loaded in {time_elapsed(t0):.1f}s")
else:
    clean_counts = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.int32)
    clean_sums = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float64)

    for cls_idx, name in enumerate(CLASS_NAMES):
        feat_path = TRAIN_FEAT_DIR / f"{name}.pt"
        activations = torch.load(feat_path, map_location="cpu", weights_only=True)
        N = activations.shape[0]
        print(f"  [{cls_idx+1}/{N_CLASSES}] {name}: {N} samples")

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

    np.savez(stats_path, clean_counts=clean_counts, clean_sums=clean_sums)
    print(f"  Saved to {stats_path}")
    print(f"  Phase 1 done in {time_elapsed(t0):.1f}s")


# ── Phase 2: Generate adversarial samples ───────────────────────────
print("\n" + "=" * 70)
print("Phase 2: Generating adversarial samples (eps=1.0)")
print("=" * 70)
t0 = time.time()

def apgd_ce_attack_l2(dp_model, x, labels, eps=1.0, steps=100, step_size=None, random_start=True):
    if step_size is None:
        step_size = eps / 4
    x0 = x.clone().detach()
    step = L2Step(orig_input=x0, eps=eps, step_size=step_size, use_grad=True)
    if random_start:
        x = step.random_perturb(x0)
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = dp_model(x)
        loss = F.cross_entropy(logits, labels)
        (grad,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x.clone().detach()

adv_save_path = OUT_DIR / "adv_samples_eps1.0.pt"
if adv_save_path.exists():
    print(f"  Loading cached adversarial samples from {adv_save_path}")
    adv_data = torch.load(adv_save_path, map_location="cpu", weights_only=True)
    all_adv_images = adv_data["adv_images"]
    all_clean_preds = adv_data["clean_preds"]
    all_adv_preds = adv_data["adv_preds"]
    print(f"  Loaded {len(all_adv_images)} samples in {time_elapsed(t0):.1f}s")
else:
    all_adv_images = []
    all_clean_preds = []
    n_batches = (len(all_clean_images) + ADV_BATCH_SIZE - 1) // ADV_BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(all_clean_images), ADV_BATCH_SIZE)):
        batch_t0 = time.time()
        batch_imgs = all_clean_images[i:i + ADV_BATCH_SIZE].to(DEVICE)
        batch_labels = all_labels[i:i + ADV_BATCH_SIZE].to(DEVICE)

        with torch.no_grad():
            logits_clean = attack_model(batch_imgs)
            pred_clean = logits_clean.argmax(dim=1)

        adv_imgs = apgd_ce_attack_l2(
            dp_model=attack_model, x=batch_imgs, labels=batch_labels,
            eps=1.0, steps=100, step_size=None, random_start=True,
        )

        all_adv_images.append(adv_imgs.cpu())
        all_clean_preds.append(pred_clean.cpu())

        dt = time.time() - batch_t0
        eta = dt * (n_batches - batch_idx - 1)
        print(f"  [{batch_idx+1}/{n_batches}] bs={batch_imgs.size(0)}, {dt:.1f}s, ETA={eta/60:.1f}min")

    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_clean_preds = torch.cat(all_clean_preds, dim=0)

    adv_pred_batches = []
    for i in range(0, len(all_adv_images), BATCH_SIZE):
        batch = all_adv_images[i:i + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            adv_pred_batches.append(model(batch).argmax(dim=1).cpu())
    all_adv_preds = torch.cat(adv_pred_batches, dim=0)

    torch.save({
        "adv_images": all_adv_images,
        "clean_preds": all_clean_preds,
        "adv_preds": all_adv_preds,
        "labels": all_labels,
    }, adv_save_path)
    print(f"  Saved to {adv_save_path}")
    print(f"  Phase 2 done in {time_elapsed(t0):.1f}s")

overall_clean_acc = (all_clean_preds == all_labels).float().mean().item()
overall_adv_acc = (all_adv_preds == all_labels).float().mean().item()
print(f"  Overall: Clean Acc={overall_clean_acc*100:.2f}%, Adv Acc={overall_adv_acc*100:.2f}%")


# ── Phase 3: Precompute base predictions ────────────────────────────
print("\n" + "=" * 70)
print("Phase 3: Precomputing Control & SAE-only predictions")
print("=" * 70)
t0 = time.time()

base_pred_path = OUT_DIR / "base_predictions.pt"
if base_pred_path.exists():
    print(f"  Loading cached base predictions from {base_pred_path}")
    bp = torch.load(base_pred_path, map_location="cpu", weights_only=True)
    clean_ctrl_preds = bp["clean_ctrl"]
    clean_sae_preds = bp["clean_sae"]
    all_ctrl_preds = bp["all_ctrl"]
    all_sae_preds = bp["all_sae"]
    print(f"  Loaded in {time_elapsed(t0):.1f}s")
else:
    def eval_preds(images, hook_fn=None):
        if hook_fn is not None:
            handle = model.activation.register_forward_hook(hook_fn)
        preds = []
        try:
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE].to(DEVICE)
                logits = model(batch)
                preds.append(logits.argmax(dim=1).cpu())
        finally:
            if hook_fn is not None:
                handle.remove()
        return torch.cat(preds, dim=0)

    clean_ctrl_preds = eval_preds(all_clean_images)
    clean_sae_preds = eval_preds(all_clean_images, make_sae_hook())
    all_ctrl_preds = all_adv_preds
    all_sae_preds = eval_preds(all_adv_images, make_sae_hook())

    torch.save({
        "clean_ctrl": clean_ctrl_preds,
        "clean_sae": clean_sae_preds,
        "all_ctrl": all_ctrl_preds,
        "all_sae": all_sae_preds,
    }, base_pred_path)
    print(f"  Saved to {base_pred_path}")
    print(f"  Phase 3 done in {time_elapsed(t0):.1f}s")

succ_mask = (all_clean_preds == all_labels) & (all_adv_preds != all_labels)
print(f"  Succ adv samples: {succ_mask.sum().item()} / {len(all_labels)}")


# ── Phase 4: Grid search ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase 4: Grid search over thresholds")
print("=" * 70)

all_results = []

# V1 configs
v1_configs = [{"layer1_threshold": t} for t in V1_LAYER1_THRESHOLDS]
print(f"\nV1 configs: {len(v1_configs)}")
for idx, params in enumerate(v1_configs):
    t0_cfg = time.time()
    config_id = f"v1_t{params['layer1_threshold']}"
    print(f"\n[{idx+1}/{len(v1_configs)}] {config_id}")

    lut, mask, n_entries = build_lut_from_stats(clean_counts, clean_sums, "v1", params)
    print(f"  Entries: {n_entries:,} / {N_TOKENS * N_CHANNELS:,} ({n_entries / (N_TOKENS * N_CHANNELS) * 100:.2f}%)")

    res = evaluate_config("v1", params, lut, mask, all_clean_images, all_adv_images,
                          all_labels, succ_mask, clean_ctrl_preds, clean_sae_preds,
                          all_ctrl_preds, all_sae_preds)
    res["config_id"] = config_id
    res["version"] = "v1"
    res["params"] = params
    res["n_lut_entries"] = int(n_entries)
    all_results.append(res)

    print(f"  All dyn vs ctrl: {res['overall']['avg_all_vs_control']:+.2f}pp")
    print(f"  Succ dyn vs ctrl: {res['overall']['avg_succ_vs_control']:+.2f}pp")
    print(f"  Clean dyn vs ctrl: {res['overall']['avg_clean_dynamic_vs_control']:+.2f}pp")
    print(f"  Done in {time_elapsed(t0_cfg):.1f}s")

# V2 configs
v2_configs = [
    {
        "layer1_threshold": t1,
        "layer2_class_threshold": t_cls,
        "layer2_min_classes": min_cls,
    }
    for t1 in V2_LAYER1_THRESHOLDS
    for t_cls in V2_CLASS_THRESHOLDS
    for min_cls in V2_MIN_CLASSES
]
print(f"\nV2 configs: {len(v2_configs)}")
for idx, params in enumerate(v2_configs):
    t0_cfg = time.time()
    config_id = f"v2_t{params['layer1_threshold']}_c{params['layer2_class_threshold']}_m{params['layer2_min_classes']}"
    print(f"\n[{idx+1}/{len(v2_configs)}] {config_id}")

    lut, mask, n_entries = build_lut_from_stats(clean_counts, clean_sums, "v2", params)
    print(f"  Entries: {n_entries:,} / {N_TOKENS * N_CHANNELS:,} ({n_entries / (N_TOKENS * N_CHANNELS) * 100:.2f}%)")

    res = evaluate_config("v2", params, lut, mask, all_clean_images, all_adv_images,
                          all_labels, succ_mask, clean_ctrl_preds, clean_sae_preds,
                          all_ctrl_preds, all_sae_preds)
    res["config_id"] = config_id
    res["version"] = "v2"
    res["params"] = params
    res["n_lut_entries"] = int(n_entries)
    all_results.append(res)

    print(f"  All dyn vs ctrl: {res['overall']['avg_all_vs_control']:+.2f}pp")
    print(f"  Succ dyn vs ctrl: {res['overall']['avg_succ_vs_control']:+.2f}pp")
    print(f"  Clean dyn vs ctrl: {res['overall']['avg_clean_dynamic_vs_control']:+.2f}pp")
    print(f"  Done in {time_elapsed(t0_cfg):.1f}s")

# ── Save results ────────────────────────────────────────────────────
out_json = OUT_DIR / "grid_search_results.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"\nAll results saved to {out_json}")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP CONFIGS SUMMARY")
print("=" * 70)

v1_res = [r for r in all_results if r["version"] == "v1"]
v2_res = [r for r in all_results if r["version"] == "v2"]

print("\n--- V1: Top 5 by Succ Improvement ---")
v1_sorted = sorted(v1_res, key=lambda x: x["overall"]["avg_succ_vs_control"], reverse=True)
for r in v1_sorted[:5]:
    o = r["overall"]
    print(f"  {r['config_id']:15s} | entries={r['n_lut_entries']:6d} | "
          f"all={o['avg_all_vs_control']:+.2f}pp | succ={o['avg_succ_vs_control']:+.2f}pp | clean={o['avg_clean_dynamic_vs_control']:+.2f}pp")

print("\n--- V2: Top 5 by Succ Improvement ---")
v2_sorted = sorted(v2_res, key=lambda x: x["overall"]["avg_succ_vs_control"], reverse=True)
for r in v2_sorted[:5]:
    o = r["overall"]
    print(f"  {r['config_id']:25s} | entries={r['n_lut_entries']:6d} | "
          f"all={o['avg_all_vs_control']:+.2f}pp | succ={o['avg_succ_vs_control']:+.2f}pp | clean={o['avg_clean_dynamic_vs_control']:+.2f}pp")

print("\n--- V1: Top 5 by All Improvement ---")
v1_sorted_all = sorted(v1_res, key=lambda x: x["overall"]["avg_all_vs_control"], reverse=True)
for r in v1_sorted_all[:5]:
    o = r["overall"]
    print(f"  {r['config_id']:15s} | entries={r['n_lut_entries']:6d} | "
          f"all={o['avg_all_vs_control']:+.2f}pp | succ={o['avg_succ_vs_control']:+.2f}pp | clean={o['avg_clean_dynamic_vs_control']:+.2f}pp")

print("\n--- V2: Top 5 by All Improvement ---")
v2_sorted_all = sorted(v2_res, key=lambda x: x["overall"]["avg_all_vs_control"], reverse=True)
for r in v2_sorted_all[:5]:
    o = r["overall"]
    print(f"  {r['config_id']:25s} | entries={r['n_lut_entries']:6d} | "
          f"all={o['avg_all_vs_control']:+.2f}pp | succ={o['avg_succ_vs_control']:+.2f}pp | clean={o['avg_clean_dynamic_vs_control']:+.2f}pp")

print("\n" + "=" * 70)
print("Grid search complete!")
print("=" * 70)
