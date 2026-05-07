#!/usr/bin/env python3
"""
Dynamic nearest-class steering DIRECTLY on ConvNeXT stage3 raw embeddings.
No SAE involved. Operates on the dense (49, 1536) feature map.

For each spatial location and channel, replaces the value with the closest
class mean (out of 20 classes) computed from ImageNet training samples.

Evaluates on:
  - All adversarial samples (succ + fail)
  - Successful adversarial samples only (adv_succ_real/)

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    /Data_share/hongyi/conda_envs/rebm/bin/python raw_embed_dynamic_nearest_steering_20cls.py
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

# ── Config ──────────────────────────────────────────────────────────
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"

TRAIN_ROOT = Path("/Data_share/hongyi/DAT/data/ImageNet/train")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/raw_embed_dynamic_nearest_steering_20cls")

BATCH_SIZE = 64          # larger batch for clean feature extraction
EVAL_BATCH_SIZE = 8      # smaller for adv eval (fewer samples per class)
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
N_CHANNELS = 1536        # ConvNeXt-Large stage3 output channels

# ── Load model ──────────────────────────────────────────────────────
print("Loading ConvNeXT model...")
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

# ── Image preprocessing ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_clean_train_images(wnid, cls_idx):
    """Load all clean training images for a class from ImageNet train."""
    img_dir = TRAIN_ROOT / wnid
    if not img_dir.is_dir():
        return []
    images = []
    for path in sorted(img_dir.glob("*.JPEG")):
        img = Image.open(path).convert("RGB")
        img = transform(img).to(DEVICE)
        images.append((img, cls_idx))
    return images


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


# ── Build Lookup Table from raw stage3 embeddings ───────────────────
@torch.no_grad()
def build_lookup_table():
    """
    Extract ConvNeXT stage3 raw embeddings for all clean train images
    and compute per-class, per-token, per-channel mean.

    Returns:
      lut_tensor: (N_CLASSES, N_TOKENS, N_CHANNELS) torch tensor on DEVICE
    """
    print("\nBuilding lookup table from raw ConvNeXT stage3 embeddings...")
    print(f"  Target shape per class: ({N_TOKENS}, {N_CHANNELS})")

    # Accumulators: sum and count for each (class, token, channel)
    clean_sums = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float64)
    clean_counts = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.int64)

    captured = {}

    def hook_fn(module, input, output):
        captured["stage"] = output.detach()

    handle = model.stages[3].register_forward_hook(hook_fn)

    try:
        for cls_idx, (wnid, imagenet_idx, name) in enumerate(CLASSES):
            images = load_clean_train_images(wnid, cls_idx)
            n_imgs = len(images)
            print(f"  [{name}] {n_imgs} clean train images")
            if n_imgs == 0:
                continue

            for i in tqdm(range(0, n_imgs, BATCH_SIZE), desc=f"Extract {name}", leave=False):
                batch = images[i : i + BATCH_SIZE]
                imgs = torch.stack([img for img, _ in batch])
                _ = model(imgs)

                stage_out = captured["stage"]  # (bsz, 1536, 7, 7)
                bsz, c, h, w = stage_out.shape
                # (bsz, 49, 1536)
                feats = stage_out.permute(0, 2, 3, 1).reshape(bsz, h * w, c)
                feats_np = feats.cpu().numpy().astype(np.float64)

                clean_sums[cls_idx] += feats_np.sum(axis=0)
                clean_counts[cls_idx] += bsz
    finally:
        handle.remove()

    with np.errstate(divide="ignore", invalid="ignore"):
        lut = np.where(clean_counts > 0, clean_sums / clean_counts, 0.0)

    lut_tensor = torch.from_numpy(lut.astype(np.float32)).to(DEVICE)
    print(f"  Lookup table ready: {lut_tensor.shape}")
    return lut_tensor


# ── Hook factory ────────────────────────────────────────────────────
def make_dynamic_nearest_hook(lut_tensor):
    """
    Dynamic nearest-class steering hook on raw ConvNeXT stage3 embeddings.

    For EACH entry (token, channel) in the dense feature map:
      1. Find which of 20 classes has clean mean closest to current value
      2. Overwrite with that class's clean mean
    """
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        # (bsz, 49, 1536)
        h = output.permute(0, 2, 3, 1).reshape(bsz, N_TOKENS, N_CHANNELS)

        # Memory-efficient argmin over classes: process one class at a time
        # to avoid allocating (bsz, 20, 49, 1536) intermediate tensor.
        best_diff = None
        nearest_cls = None
        for c in range(N_CLASSES):
            diff = torch.abs(h - lut_tensor[c])          # (bsz, 49, 1536)
            if best_diff is None:
                best_diff = diff
                nearest_cls = torch.zeros_like(diff, dtype=torch.long)
            else:
                mask = diff < best_diff
                best_diff = torch.where(mask, diff, best_diff)
                nearest_cls = torch.where(mask, c, nearest_cls)

        # Gather target values via advanced indexing
        b_idx = torch.arange(bsz, device=DEVICE).view(bsz, 1, 1).expand(bsz, N_TOKENS, N_CHANNELS)
        t_idx = torch.arange(N_TOKENS, device=DEVICE).view(1, N_TOKENS, 1).expand(bsz, N_TOKENS, N_CHANNELS)
        c_idx = torch.arange(N_CHANNELS, device=DEVICE).view(1, 1, N_CHANNELS).expand(bsz, N_TOKENS, N_CHANNELS)
        targets = lut_tensor[nearest_cls, t_idx, c_idx]  # (bsz, 49, 1536)

        # Replace and reshape back
        h_replaced = targets.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return h_replaced
    return hook_fn


# ── Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(images, hook_fn=None, batch_size=EVAL_BATCH_SIZE):
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    correct = 0
    total = 0
    try:
        for i in tqdm(range(0, len(images), batch_size), desc="Evaluating", leave=False):
            batch = images[i : i + batch_size]
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
def evaluate_detailed(images, hook_fn=None, batch_size=EVAL_BATCH_SIZE):
    """Evaluate and return per-sample predictions for quadrant analysis."""
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    preds = []
    try:
        for i in tqdm(range(0, len(images), batch_size), desc="Evaluating", leave=False):
            batch = images[i : i + batch_size]
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
    print("RAW EMBEDDING DYNAMIC NEAREST STEERING — 20 Classes")
    print("Operates directly on ConvNeXT stage3 (49, 1536) dense features")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build lookup table once
    lut_tensor = build_lookup_table()

    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")

        all_images = load_all_adv_images(wnid, cls_idx)
        succ_images = load_succ_adv_images(wnid, cls_idx)

        if not all_images:
            print(f"  [SKIP] No adv images found")
            continue
        print(f"  All adv samples: {len(all_images)}")
        print(f"  Succ adv samples: {len(succ_images)}")

        # Get control predictions for quadrant analysis
        print("  [1/4] Control predictions (all adv)...")
        _, all_ctrl_preds = evaluate_detailed(all_images)
        _, succ_ctrl_preds = evaluate_detailed(succ_images) if succ_images else (0.0, [])

        all_labels = [label for _, label in all_images]
        succ_labels = [label for _, label in succ_images]

        # Dynamic steering on all adv
        print("  [2/4] Dynamic steering (all adv)...")
        all_dyn_acc, all_dyn_preds = evaluate_detailed(all_images, make_dynamic_nearest_hook(lut_tensor))

        # Dynamic steering on succ adv
        print("  [3/4] Dynamic steering (succ adv)...")
        succ_dyn_acc, succ_dyn_preds = evaluate_detailed(succ_images, make_dynamic_nearest_hook(lut_tensor)) if succ_images else (None, [])

        # Quadrant analysis
        all_q = compute_quadrants(all_ctrl_preds, all_dyn_preds, all_labels)
        succ_q = compute_quadrants(succ_ctrl_preds, succ_dyn_preds, succ_labels) if succ_images else None

        all_ctrl_acc = sum(1 for p, l in zip(all_ctrl_preds, all_labels) if p == l) / len(all_labels)
        succ_ctrl_acc = sum(1 for p, l in zip(succ_ctrl_preds, succ_labels) if p == l) / len(succ_labels) if succ_images else None
        succ_dyn_str = f"{succ_dyn_acc*100:.1f}%" if succ_dyn_acc is not None else "N/A"
        print(
            f"  All:  Ctrl={all_ctrl_acc*100:.1f}% Dyn={all_dyn_acc*100:.1f}% | "
            f"Succ: Ctrl={succ_ctrl_acc*100:.1f}% Dyn={succ_dyn_str}"
        )
        print(f"    All  quad: +{all_q['recovery']} / -{all_q['regression']} (net {all_q['net_gain']:+d})")
        if succ_q:
            print(f"    Succ quad: +{succ_q['recovery']} / -{succ_q['regression']} (net {succ_q['net_gain']:+d})")

        results.append({
            "name": name,
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_all_adv": len(all_images),
            "n_succ_adv": len(succ_images),
            "all_control_acc": sum(1 for p, l in zip(all_ctrl_preds, all_labels) if p == l) / len(all_labels),
            "all_dynamic_acc": all_dyn_acc,
            "succ_control_acc": sum(1 for p, l in zip(succ_ctrl_preds, succ_labels) if p == l) / len(succ_labels) if succ_images else None,
            "succ_dynamic_acc": succ_dyn_acc,
            "all_quadrants": all_q,
            "succ_quadrants": succ_q,
        })

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_all_control_acc": avg("all_control_acc"),
        "avg_all_dynamic_acc": avg("all_dynamic_acc"),
        "avg_succ_control_acc": avg("succ_control_acc"),
        "avg_succ_dynamic_acc": avg("succ_dynamic_acc"),
    }

    # ── Quadrant Summary ────────────────────────────────────────────
    def sum_quad(key):
        total = {"unchanged_correct": 0, "regression": 0, "recovery": 0, "unchanged_wrong": 0, "net_gain": 0, "n_total": 0}
        for r in results:
            q = r.get(key)
            if q:
                for k in total:
                    total[k] += q.get(k, 0)
        return total

    all_q_total = sum_quad("all_quadrants")
    succ_q_total = sum_quad("succ_quadrants")

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
    print("\n" + "=" * 95)
    print("RESULTS SUMMARY (Control vs Dynamic on raw embeddings)")
    print("=" * 95)
    print(
        f"{'Class':>15} {'N_all':>6} {'N_succ':>6} {'AllCtrl':>8} {'AllDyn':>8} "
        f"{'SuccCtrl':>8} {'SuccDyn':>8} {'ΔAll':>7} {'ΔSucc':>7}"
    )
    print("-" * 95)

    for r in results:
        sc = f"{r['succ_control_acc']*100:>7.1f}%" if r['succ_control_acc'] is not None else f"{'N/A':>8}"
        sd = f"{r['succ_dynamic_acc']*100:>7.1f}%" if r['succ_dynamic_acc'] is not None else f"{'N/A':>8}"
        ds = f"{(r['succ_dynamic_acc'] - r['succ_control_acc'])*100:>+6.1f}" if r['succ_dynamic_acc'] is not None and r['succ_control_acc'] is not None else f"{'N/A':>7}"
        da = f"{(r['all_dynamic_acc'] - r['all_control_acc'])*100:>+6.1f}"
        print(
            f"{r['name']:>15} {r['n_all_adv']:>6} {r['n_succ_adv']:>6} "
            f"{r['all_control_acc']*100:>7.1f}% {r['all_dynamic_acc']*100:>7.1f}% "
            f"{sc} {sd} {da} {ds}"
        )

    print("-" * 95)
    print(
        f"{'OVERALL AVERAGE':>15} {'':>6} {'':>6} "
        f"{overall['avg_all_control_acc']*100:>7.1f}% {overall['avg_all_dynamic_acc']*100:>7.1f}% "
        f"{overall['avg_succ_control_acc']*100:>7.1f}% {overall['avg_succ_dynamic_acc']*100:>7.1f}% "
        f"{(overall['avg_all_dynamic_acc'] - overall['avg_all_control_acc'])*100:>+6.1f} "
        f"{(overall['avg_succ_dynamic_acc'] - overall['avg_succ_control_acc'])*100:>+6.1f}"
    )
    print("=" * 95)

    print_quad_table("QUADRANT ANALYSIS — All adversarial samples", all_q_total)
    print_quad_table("QUADRANT ANALYSIS — Successful adversarial samples", succ_q_total)

    # Save JSON
    out_path = OUT_DIR / "raw_embed_dynamic_nearest_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_class": results,
            "overall_average": overall,
            "all_quadrants_total": all_q_total,
            "succ_quadrants_total": succ_q_total,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")

    n_all_pos = sum(1 for r in results if (r["all_dynamic_acc"] - r["all_control_acc"]) > 0)
    n_succ_pos = sum(1 for r in results if r["succ_dynamic_acc"] is not None and (r["succ_dynamic_acc"] - r["succ_control_acc"]) > 0)
    print(f"\nAll  adv: Dynamic > Control: {n_all_pos}/{len(results)} classes")
    print(f"Succ adv: Dynamic > Control: {n_succ_pos}/{len(results)} classes")
    print(f"\nKEY METRICS:")
    print(f"  All  adv  vs Control: {(overall['avg_all_dynamic_acc'] - overall['avg_all_control_acc'])*100:+.1f} pp")
    print(f"  Succ adv vs Control: {(overall['avg_succ_dynamic_acc'] - overall['avg_succ_control_acc'])*100:+.1f} pp")


if __name__ == "__main__":
    main()
