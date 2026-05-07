#!/usr/bin/env python3
"""
Cross-class steering validation for 20 representative classes.

Four conditions per class:
  1. Control:   No intervention
  2. Global:    Steer 26 global entries to cross-class clean_mean
  3. Random A:  Steer 26 random entries (non-candidate, clean>=92%) to their clean_mean
  4. Random B:  Steer same 26 global entries but with shuffled targets

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python run_sae_entry_steering_20cls.py
"""

import os
import sys
import json
import numpy as np
import random
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
GLOBAL_JSON = Path("/Data_share/hongyi/DAT/SAE/results_representation/results_20cls_train/global_candidates.json")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20cls_train")

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


def load_adv_images(wnid, cls_idx):
    """Load adversarial images from adv_succ_real/. Returns list of (image_tensor, true_label)."""
    run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
    img_dir = ADV_ROOT / run_name / "adv_succ_real"
    if not img_dir.is_dir():
        return []

    images = []
    for path in sorted(img_dir.glob("*.JPEG")):
        img = Image.open(path).convert("RGB")
        img = transform(img).to(DEVICE)
        images.append((img, cls_idx))
    return images


# ── Read global candidates ──────────────────────────────────────────
print(f"Loading global candidates from {GLOBAL_JSON}")
with open(GLOBAL_JSON) as f:
    global_data = json.load(f)

global_targets = {}
for key, info in global_data["candidates"].items():
    token = info["token"]
    channel = info["channel"]
    global_targets[(token, channel)] = info["target_clean_mean"]

print(f"  Total global entries: {len(global_targets)}")

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
    """SAE encode → decode without any modification. Controls for reconstruction error."""
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


# ── Build Random A & Random B targets ───────────────────────────────
def build_random_a_targets(clean_npy_path, n=26):
    """Random entries from clean>=92% pool, excluding global candidates."""
    clean = np.load(clean_npy_path)
    clean_count = np.sum(clean != 0, axis=0)
    mask_high = clean_count >= 46

    global_keys = set(global_targets.keys())
    tok_idx, ch_idx = np.where(mask_high)
    candidates = []
    for t, c in zip(tok_idx, ch_idx):
        if (int(t), int(c)) not in global_keys:
            candidates.append((int(t), int(c)))

    if len(candidates) < n:
        return None

    selected = random.sample(candidates, n)
    clean_sum = np.sum(clean, axis=0)
    targets = {}
    for t, c in selected:
        cm = clean_sum[t, c] / clean_count[t, c] if clean_count[t, c] > 0 else 0.0
        targets[(t, c)] = float(cm)
    return targets


def build_random_b_targets():
    """Same global entries, but shuffled targets."""
    items = list(global_targets.items())
    keys = [k for k, v in items]
    vals = [v for k, v in items]
    random.shuffle(vals)
    return {k: v for k, v in zip(keys, vals)}


# ── Main loop ───────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("CROSS-CLASS STEERING VALIDATION (5 conditions)")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")
        images = load_adv_images(wnid, cls_idx)
        if not images:
            print(f"  [SKIP] No adv images found")
            continue
        print(f"  Adv samples: {len(images)}")

        # 1. Control (no SAE)
        print("  [1/5] Control (no SAE)...")
        control_acc = evaluate(images)

        # 2. SAE-only (encode → decode, no modification)
        print("  [2/5] SAE-only (reconstruction control)...")
        sae_only_hook = make_sae_only_hook()
        sae_only_acc = evaluate(images, sae_only_hook)

        # 3. Global
        print(f"  [3/5] Global {len(global_targets)} entries...")
        global_hook = make_steering_hook(global_targets)
        global_acc = evaluate(images, global_hook)

        # 4. Random A (target from TRAIN clean features, NOT val)
        train_clean_npy = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_latent") / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        random_a_targets = build_random_a_targets(train_clean_npy, n=26)
        if random_a_targets is None:
            print(f"  [SKIP] Not enough random entries")
            random_a_acc = None
        else:
            print("  [4/5] Random A...")
            random_a_hook = make_steering_hook(random_a_targets)
            random_a_acc = evaluate(images, random_a_hook)

        # 5. Random B
        print("  [5/5] Random B (shuffled targets)...")
        random_b_targets = build_random_b_targets()
        random_b_hook = make_steering_hook(random_b_targets)
        random_b_acc = evaluate(images, random_b_hook)

        ra_str = f"{random_a_acc:.4f}" if random_a_acc is not None else "N/A"
        print(f"  Control: {control_acc:.4f}  SAE-only: {sae_only_acc:.4f}  Global: {global_acc:.4f}  "
              f"RandomA: {ra_str}  "
              f"RandomB: {random_b_acc:.4f}")

        results.append({
            "name": name,
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_adv_samples": len(images),
            "control_acc": control_acc,
            "sae_only_acc": sae_only_acc,
            "global_acc": global_acc,
            "random_a_acc": random_a_acc,
            "random_b_acc": random_b_acc,
            "sae_only_vs_control": (sae_only_acc - control_acc) * 100,
            "global_vs_control": (global_acc - control_acc) * 100,
            "global_vs_sae_only": (global_acc - sae_only_acc) * 100,
            "global_vs_random_a": (global_acc - random_a_acc) * 100 if random_a_acc is not None else None,
            "global_vs_random_b": (global_acc - random_b_acc) * 100,
        })

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_control_acc": avg("control_acc"),
        "avg_sae_only_acc": avg("sae_only_acc"),
        "avg_global_acc": avg("global_acc"),
        "avg_random_a_acc": avg("random_a_acc"),
        "avg_random_b_acc": avg("random_b_acc"),
        "avg_global_vs_control": avg("global_vs_control"),
        "avg_global_vs_sae_only": avg("global_vs_sae_only"),
        "avg_global_vs_random_a": avg("global_vs_random_a"),
        "avg_global_vs_random_b": avg("global_vs_random_b"),
    }

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("RESULTS SUMMARY (per class)")
    print("=" * 85)
    print(f"{'Class':>15} {'N':>3} {'Ctrl':>6} {'SAE':>6} {'Global':>7} {'RandA':>7} {'RandB':>7} {'G-C':>6} {'G-SAE':>6} {'G-RA':>6} {'G-RB':>6}")
    print("-" * 85)

    for r in results:
        ra_str = f"{r['random_a_acc']*100:>5.1f}%" if r['random_a_acc'] is not None else f"{'N/A':>6}"
        ra_diff = f"{r['global_vs_random_a']:>+5.1f}" if r['global_vs_random_a'] is not None else f"{'N/A':>6}"
        print(f"{r['name']:>15} {r['n_adv_samples']:>3} "
              f"{r['control_acc']*100:>5.1f}% {r['sae_only_acc']*100:>5.1f}% {r['global_acc']*100:>6.1f}% "
              f"{ra_str} "
              f"{r['random_b_acc']*100:>5.1f}% "
              f"{r['global_vs_control']:>+5.1f} "
              f"{r['global_vs_sae_only']:>+5.1f} "
              f"{ra_diff} "
              f"{r['global_vs_random_b']:>+5.1f}")

    print("-" * 85)
    print(f"{'OVERALL AVERAGE':>15} {'':>3} "
          f"{overall['avg_control_acc']*100:>5.1f}% {overall['avg_sae_only_acc']*100:>5.1f}% {overall['avg_global_acc']*100:>6.1f}% "
          f"{overall['avg_random_a_acc']*100:>5.1f}% {overall['avg_random_b_acc']*100:>5.1f}% "
          f"{overall['avg_global_vs_control']:>+5.1f} "
          f"{overall['avg_global_vs_sae_only']:>+5.1f} "
          f"{overall['avg_global_vs_random_a']:>+5.1f} "
          f"{overall['avg_global_vs_random_b']:>+5.1f}")
    print("=" * 85)

    # Save JSON
    out_path = OUT_DIR / "steering_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_class": results,
            "overall_average": overall,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Overall stats
    valid = [r for r in results if r['random_a_acc'] is not None]
    if valid:
        n_global_better_a = sum(1 for r in valid if r['global_acc'] > r['random_a_acc'])
        n_global_better_b = sum(1 for r in valid if r['global_acc'] > r['random_b_acc'])
        n_global_better_sae = sum(1 for r in results if r['global_acc'] > r['sae_only_acc'])
        print(f"\nGlobal > Random A:  {n_global_better_a}/{len(valid)} classes")
        print(f"Global > Random B:  {n_global_better_b}/{len(valid)} classes")
        print(f"Global > SAE-only:  {n_global_better_sae}/{len(results)} classes")
        print(f"\nKEY CAUSAL EVIDENCE:")
        print(f"  Global vs Control (raw gain):     {overall['avg_global_vs_control']:+.1f} pp")
        print(f"  Global vs SAE-only (true effect): {overall['avg_global_vs_sae_only']:+.1f} pp")
        print(f"  Global vs Random A (specificity): {overall['avg_global_vs_random_a']:+.1f} pp")
        print(f"  Global vs Random B (target specificity): {overall['avg_global_vs_random_b']:+.1f} pp")


if __name__ == "__main__":
    main()
