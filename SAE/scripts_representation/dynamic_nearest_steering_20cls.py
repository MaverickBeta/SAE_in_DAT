#!/usr/bin/env python3
"""
Dynamic nearest-class steering for adversarial samples (Version B).
Supports two lookup table filtering strategies:
  v1: Layer 1 only (total clean activation >= 20 across 20 classes)
  v2: Layer 1 + Layer 2 (>=2 classes each with >=5 clean activations)

Evaluates on succ adversarial samples only:
  1. Control:       No intervention
  2. SAE-only:      encode → decode without modification
  3. Dynamic v1:    encode → per-entry nearest-class steering → decode
  4. Dynamic v2:    encode → per-entry nearest-class steering → decode (stricter lookup)

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    /Data_share/hongyi/conda_envs/rebm/bin/python dynamic_nearest_steering_20cls.py
"""

import os
import sys
import json
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
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20cls_dynamic_nearest")

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


def load_all_adv_images(wnid, cls_idx):
    """Load ALL adversarial images (both succ and fail) from the run directory."""
    run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
    img_dir = ADV_ROOT / run_name
    if not img_dir.is_dir():
        return []
    images = []
    for path in sorted(img_dir.glob("*.JPEG")):
        # Only load files directly in the run dir, not in subdirs like adv_succ_real
        if path.parent.name == run_name:
            img = Image.open(path).convert("RGB")
            img = transform(img).to(DEVICE)
            images.append((img, cls_idx))
    return images


def load_succ_adv_images(wnid, cls_idx):
    """Load only successful adversarial images from adv_succ_real/."""
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


def load_clean_images(wnid, cls_idx):
    """Load clean validation images from ImageNet val."""
    img_dir = IMAGENET_VAL / wnid
    if not img_dir.is_dir():
        return []
    images = []
    for path in sorted(img_dir.glob("*.JPEG")):
        img = Image.open(path).convert("RGB")
        img = transform(img).to(DEVICE)
        images.append((img, cls_idx))
    return images


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


@torch.no_grad()
def evaluate_detailed(images, hook_fn=None):
    """Evaluate and return per-sample predictions for quadrant analysis."""
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    preds = []
    try:
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Evaluating", leave=False):
            batch = images[i : i + BATCH_SIZE]
            imgs = torch.stack([img for img, _ in batch])
            logits = model(imgs)
            batch_preds = logits.argmax(dim=1).cpu().tolist()
            preds.extend(batch_preds)
    finally:
        if hook_fn is not None:
            handle.remove()

    labels = [label for _, label in images]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    acc = correct / len(images) if images else 0.0
    return acc, preds


def compute_quadrants(control_preds, dynamic_preds, labels):
    """
    Compute confusion-quadrant counts:
      A: Control correct → Dynamic correct   (unchanged correct)
      B: Control correct → Dynamic wrong     (regression)
      C: Control wrong   → Dynamic correct   (recovery)
      D: Control wrong   → Dynamic wrong     (unchanged wrong)
    Returns dict with counts and net gain.
    """
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
    print("\n" + "=" * 70)
    print("DYNAMIC NEAREST STEERING — V1 & V2 (Succ + All Adv + Clean)")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline results from eval_20cls.py
    eval_path = Path("/Data_share/hongyi/DAT/SAE/results_representation/eval_20cls/eval_results.json")
    baseline = {}
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        for r in eval_data.get("per_class", []):
            baseline[r["name"]] = {
                "all_control_acc": r.get("all_control_acc"),
                "all_sae_only_acc": r.get("all_sae_only_acc"),
                "succ_control_acc": r.get("succ_control_acc"),
                "succ_sae_only_acc": r.get("succ_sae_only_acc"),
            }
        print(f"Loaded baseline from {eval_path}")
    else:
        print(f"[WARNING] Baseline not found at {eval_path}, run eval_20cls.py first.")

    # Build both lookup tables upfront
    lut_v1, mask_v1 = build_lookup_table("v1")
    lut_v2, mask_v2 = build_lookup_table("v2")

    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")

        all_images = load_all_adv_images(wnid, cls_idx)
        succ_images = load_succ_adv_images(wnid, cls_idx)
        clean_images = load_clean_images(wnid, cls_idx)

        if not all_images:
            print(f"  [SKIP] No adv images found")
            continue
        print(f"  All adv samples: {len(all_images)}")
        print(f"  Succ adv samples: {len(succ_images)}")
        print(f"  Clean samples: {len(clean_images)}")

        # Get control predictions for detailed analysis
        print("  [0/7] Control predictions (for quadrant analysis)...")
        _, all_ctrl_preds = evaluate_detailed(all_images)
        _, succ_ctrl_preds = evaluate_detailed(succ_images) if succ_images else (0.0, [])

        all_labels = [label for _, label in all_images]
        succ_labels = [label for _, label in succ_images]

        # 1. Dynamic v1 on succ
        print("  [1/7] Dynamic v1 (succ)...")
        succ_v1_acc, succ_v1_preds = evaluate_detailed(succ_images, make_dynamic_nearest_hook(lut_v1, mask_v1)) if succ_images else (None, [])

        # 2. Dynamic v2 on succ
        print("  [2/7] Dynamic v2 (succ)...")
        succ_v2_acc, succ_v2_preds = evaluate_detailed(succ_images, make_dynamic_nearest_hook(lut_v2, mask_v2)) if succ_images else (None, [])

        # 3. Dynamic v1 on all
        print("  [3/7] Dynamic v1 (all)...")
        all_v1_acc, all_v1_preds = evaluate_detailed(all_images, make_dynamic_nearest_hook(lut_v1, mask_v1))

        # 4. Dynamic v2 on all
        print("  [4/7] Dynamic v2 (all)...")
        all_v2_acc, all_v2_preds = evaluate_detailed(all_images, make_dynamic_nearest_hook(lut_v2, mask_v2))

        # 5-7. Clean evaluations
        print("  [5/7] Clean control...")
        clean_ctrl_acc = evaluate(clean_images) if clean_images else None
        print("  [6/7] Clean v1...")
        clean_v1_acc = evaluate(clean_images, make_dynamic_nearest_hook(lut_v1, mask_v1)) if clean_images else None
        print("  [7/7] Clean v2...")
        clean_v2_acc = evaluate(clean_images, make_dynamic_nearest_hook(lut_v2, mask_v2)) if clean_images else None

        # Quadrant analysis
        succ_v1_q = compute_quadrants(succ_ctrl_preds, succ_v1_preds, succ_labels) if succ_images else None
        succ_v2_q = compute_quadrants(succ_ctrl_preds, succ_v2_preds, succ_labels) if succ_images else None
        all_v1_q = compute_quadrants(all_ctrl_preds, all_v1_preds, all_labels)
        all_v2_q = compute_quadrants(all_ctrl_preds, all_v2_preds, all_labels)

        base = baseline.get(name, {})
        all_ctrl = base.get("all_control_acc")
        all_sae = base.get("all_sae_only_acc")
        succ_ctrl = base.get("succ_control_acc")
        succ_sae = base.get("succ_sae_only_acc")

        s1 = f"{succ_v1_acc*100:.1f}%" if succ_v1_acc is not None else "N/A"
        s2 = f"{succ_v2_acc*100:.1f}%" if succ_v2_acc is not None else "N/A"
        c1 = f"{clean_v1_acc*100:.1f}%" if clean_v1_acc is not None else "N/A"
        c2 = f"{clean_v2_acc*100:.1f}%" if clean_v2_acc is not None else "N/A"
        print(
            f"  Succ: V1={s1} V2={s2} | "
            f"All: V1={all_v1_acc*100:.1f}% V2={all_v2_acc*100:.1f}% | "
            f"Clean: V1={c1} V2={c2}"
        )
        if succ_v1_q:
            print(f"    Succ V1 quad: +{succ_v1_q['recovery']} / -{succ_v1_q['regression']} (net {succ_v1_q['net_gain']:+d})")
        if all_v1_q:
            print(f"    All  V1 quad: +{all_v1_q['recovery']} / -{all_v1_q['regression']} (net {all_v1_q['net_gain']:+d})")

        results.append(
            {
                "name": name,
                "wnid": wnid,
                "class_idx": cls_idx,
                "n_all_adv": len(all_images),
                "n_succ_adv": len(succ_images),
                "n_clean": len(clean_images),
                "succ_control_acc": succ_ctrl,
                "succ_sae_only_acc": succ_sae,
                "all_control_acc": all_ctrl,
                "all_sae_only_acc": all_sae,
                "succ_v1_acc": succ_v1_acc,
                "succ_v2_acc": succ_v2_acc,
                "all_v1_acc": all_v1_acc,
                "all_v2_acc": all_v2_acc,
                "clean_control_acc": clean_ctrl_acc,
                "clean_v1_acc": clean_v1_acc,
                "clean_v2_acc": clean_v2_acc,
                "succ_v1_vs_control": (succ_v1_acc - succ_ctrl) * 100 if succ_v1_acc is not None and succ_ctrl is not None else None,
                "succ_v2_vs_control": (succ_v2_acc - succ_ctrl) * 100 if succ_v2_acc is not None and succ_ctrl is not None else None,
                "all_v1_vs_control": (all_v1_acc - all_ctrl) * 100 if all_ctrl is not None else None,
                "all_v2_vs_control": (all_v2_acc - all_ctrl) * 100 if all_ctrl is not None else None,
                "clean_v1_vs_control": (clean_v1_acc - clean_ctrl_acc) * 100 if clean_v1_acc is not None and clean_ctrl_acc is not None else None,
                "clean_v2_vs_control": (clean_v2_acc - clean_ctrl_acc) * 100 if clean_v2_acc is not None and clean_ctrl_acc is not None else None,
                "succ_v1_quadrants": succ_v1_q,
                "succ_v2_quadrants": succ_v2_q,
                "all_v1_quadrants": all_v1_q,
                "all_v2_quadrants": all_v2_q,
            }
        )

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_succ_v1_acc": avg("succ_v1_acc"),
        "avg_succ_v2_acc": avg("succ_v2_acc"),
        "avg_all_v1_acc": avg("all_v1_acc"),
        "avg_all_v2_acc": avg("all_v2_acc"),
        "avg_clean_control_acc": avg("clean_control_acc"),
        "avg_clean_v1_acc": avg("clean_v1_acc"),
        "avg_clean_v2_acc": avg("clean_v2_acc"),
        "avg_succ_v1_vs_control": avg("succ_v1_vs_control"),
        "avg_succ_v2_vs_control": avg("succ_v2_vs_control"),
        "avg_all_v1_vs_control": avg("all_v1_vs_control"),
        "avg_all_v2_vs_control": avg("all_v2_vs_control"),
        "avg_clean_v1_vs_control": avg("clean_v1_vs_control"),
        "avg_clean_v2_vs_control": avg("clean_v2_vs_control"),
    }

    # ── Quadrant Summary ────────────────────────────────────────────
    def sum_quad(key, version):
        total = {"unchanged_correct": 0, "regression": 0, "recovery": 0, "unchanged_wrong": 0, "net_gain": 0, "n_total": 0}
        qkey = f"{key}_quadrants"
        for r in results:
            q = r.get(qkey)
            if q:
                for k in total:
                    total[k] += q.get(k, 0)
        return total

    all_v1_q_total = sum_quad("all_v1", "v1")
    all_v2_q_total = sum_quad("all_v2", "v2")
    succ_v1_q_total = sum_quad("succ_v1", "v1")
    succ_v2_q_total = sum_quad("succ_v2", "v2")

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
        print(f"  Regression      (B):  {re:>4} / {nt} ({re/nt*100:>5.1f}%)  ← steering 把对的变错")
        print(f"  Recovery        (C):  {rc:>4} / {nt} ({rc/nt*100:>5.1f}%)  ← steering 把错的变对")
        print(f"  Unchanged wrong (D):  {uw:>4} / {nt} ({uw/nt*100:>5.1f}%)")
        print(f"  Net gain (C - B):     {q_total['net_gain']:+d}")
        print("-" * 70)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 145)
    print("RESULTS SUMMARY (per version: Ctrl vs Dynamic on succ / all / clean)")
    print("=" * 145)
    print(
        f"{'Class':>15} {'Ver':>4} {'N_succ':>6} {'N_all':>6} {'N_clean':>7} "
        f"{'SuccCtrl':>8} {'SuccDyn':>8} {'AllCtrl':>8} {'AllDyn':>8} "
        f"{'CleanCtrl':>9} {'CleanDyn':>9} {'ΔSucc':>7} {'ΔAll':>7} {'ΔClean':>7}"
    )
    print("-" * 145)

    for r in results:
        sc = f"{r['succ_control_acc']*100:>7.1f}%" if r['succ_control_acc'] is not None else f"{'N/A':>8}"
        ac = f"{r['all_control_acc']*100:>7.1f}%" if r['all_control_acc'] is not None else f"{'N/A':>8}"
        cc = f"{r['clean_control_acc']*100:>7.1f}%" if r['clean_control_acc'] is not None else f"{'N/A':>8}"
        # v1
        sv1 = f"{r['succ_v1_acc']*100:>7.1f}%" if r['succ_v1_acc'] is not None else f"{'N/A':>8}"
        cv1 = f"{r['clean_v1_acc']*100:>7.1f}%" if r['clean_v1_acc'] is not None else f"{'N/A':>8}"
        ds1 = f"{r['succ_v1_vs_control']:>+6.1f}" if r['succ_v1_vs_control'] is not None else f"{'N/A':>7}"
        da1 = f"{r['all_v1_vs_control']:>+6.1f}" if r['all_v1_vs_control'] is not None else f"{'N/A':>7}"
        dc1 = f"{r['clean_v1_vs_control']:>+6.1f}" if r['clean_v1_vs_control'] is not None else f"{'N/A':>7}"
        print(
            f"{r['name']:>15} {'v1':>4} {r['n_succ_adv']:>6} {r['n_all_adv']:>6} {r['n_clean']:>7} "
            f"{sc} {sv1} {ac} {r['all_v1_acc']*100:>7.1f}% "
            f"{cc} {cv1} {ds1} {da1} {dc1}"
        )
        # v2
        sv2 = f"{r['succ_v2_acc']*100:>7.1f}%" if r['succ_v2_acc'] is not None else f"{'N/A':>8}"
        cv2 = f"{r['clean_v2_acc']*100:>7.1f}%" if r['clean_v2_acc'] is not None else f"{'N/A':>8}"
        ds2 = f"{r['succ_v2_vs_control']:>+6.1f}" if r['succ_v2_vs_control'] is not None else f"{'N/A':>7}"
        da2 = f"{r['all_v2_vs_control']:>+6.1f}" if r['all_v2_vs_control'] is not None else f"{'N/A':>7}"
        dc2 = f"{r['clean_v2_vs_control']:>+6.1f}" if r['clean_v2_vs_control'] is not None else f"{'N/A':>7}"
        print(
            f"{'':>15} {'v2':>4} {r['n_succ_adv']:>6} {r['n_all_adv']:>6} {r['n_clean']:>7} "
            f"{sc} {sv2} {ac} {r['all_v2_acc']*100:>7.1f}% "
            f"{cc} {cv2} {ds2} {da2} {dc2}"
        )

    print("-" * 145)
    sc_ov = f"{avg('succ_control_acc')*100:>7.1f}%"
    ac_ov = f"{avg('all_control_acc')*100:>7.1f}%"
    cc_ov = f"{avg('clean_control_acc')*100:>7.1f}%"
    print(
        f"{'OVERALL AVERAGE':>15} {'v1':>4} {'':>6} {'':>6} {'':>7} "
        f"{sc_ov} {overall['avg_succ_v1_acc']*100:>7.1f}% {ac_ov} {overall['avg_all_v1_acc']*100:>7.1f}% "
        f"{cc_ov} {overall['avg_clean_v1_acc']*100:>7.1f}% "
        f"{overall['avg_succ_v1_vs_control']:>+6.1f} {overall['avg_all_v1_vs_control']:>+6.1f} {overall['avg_clean_v1_vs_control']:>+6.1f}"
    )
    print(
        f"{'':>15} {'v2':>4} {'':>6} {'':>6} {'':>7} "
        f"{sc_ov} {overall['avg_succ_v2_acc']*100:>7.1f}% {ac_ov} {overall['avg_all_v2_acc']*100:>7.1f}% "
        f"{cc_ov} {overall['avg_clean_v2_acc']*100:>7.1f}% "
        f"{overall['avg_succ_v2_vs_control']:>+6.1f} {overall['avg_all_v2_vs_control']:>+6.1f} {overall['avg_clean_v2_vs_control']:>+6.1f}"
    )
    print("=" * 145)

    print_quad_table("QUADRANT ANALYSIS — All samples + V1", all_v1_q_total)
    print_quad_table("QUADRANT ANALYSIS — All samples + V2", all_v2_q_total)
    print_quad_table("QUADRANT ANALYSIS — Succ samples + V1", succ_v1_q_total)
    print_quad_table("QUADRANT ANALYSIS — Succ samples + V2", succ_v2_q_total)

    # Save JSON
    out_path = OUT_DIR / "dynamic_nearest_results.json"
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

    n_sv1_pos = sum(1 for r in results if r["succ_v1_vs_control"] is not None and r["succ_v1_vs_control"] > 0)
    n_sv2_pos = sum(1 for r in results if r["succ_v2_vs_control"] is not None and r["succ_v2_vs_control"] > 0)
    n_av1_pos = sum(1 for r in results if r["all_v1_vs_control"] is not None and r["all_v1_vs_control"] > 0)
    n_av2_pos = sum(1 for r in results if r["all_v2_vs_control"] is not None and r["all_v2_vs_control"] > 0)
    print(f"\nV1-succ > Control: {n_sv1_pos}/{len(results)} classes")
    print(f"V2-succ > Control: {n_sv2_pos}/{len(results)} classes")
    print(f"V1-all  > Control: {n_av1_pos}/{len(results)} classes")
    print(f"V2-all  > Control: {n_av2_pos}/{len(results)} classes")
    n_cv1_pos = sum(1 for r in results if r["clean_v1_vs_control"] is not None and r["clean_v1_vs_control"] > 0)
    n_cv2_pos = sum(1 for r in results if r["clean_v2_vs_control"] is not None and r["clean_v2_vs_control"] > 0)
    print(f"V1-clean > Control: {n_cv1_pos}/{len(results)} classes")
    print(f"V2-clean > Control: {n_cv2_pos}/{len(results)} classes")
    print(f"\nKEY METRICS:")
    print(f"  V1-succ vs Control: {overall['avg_succ_v1_vs_control']:+.1f} pp")
    print(f"  V2-succ vs Control: {overall['avg_succ_v2_vs_control']:+.1f} pp")
    print(f"  V1-all  vs Control: {overall['avg_all_v1_vs_control']:+.1f} pp")
    print(f"  V2-all  vs Control: {overall['avg_all_v2_vs_control']:+.1f} pp")
    print(f"  V1-clean vs Control: {overall['avg_clean_v1_vs_control']:+.1f} pp")
    print(f"  V2-clean vs Control: {overall['avg_clean_v2_vs_control']:+.1f} pp")


if __name__ == "__main__":
    main()
