#!/usr/bin/env python3
"""
Baseline evaluation for 20 representative classes.

Tests 4 conditions per class:
  1. All adv samples  → original model (no hook)
  2. All adv samples  → SAE-only (encode→decode, no modification)
  3. Succ adv samples → original model (no hook)
  4. Succ adv samples → SAE-only (encode→decode, no modification)

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    /Data_share/hongyi/conda_envs/rebm/bin/python eval_20cls.py
"""

import os
import sys
import json
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

ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/eval_20cls")

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


# ── Hook factory ────────────────────────────────────────────────────
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


# ── Main loop ───────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (4 conditions per class)")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")

        all_adv_images = load_all_adv_images(wnid, cls_idx)
        succ_adv_images = load_succ_adv_images(wnid, cls_idx)

        if not all_adv_images:
            print(f"  [SKIP] No adv images found")
            continue

        print(f"  All adv samples: {len(all_adv_images)}")
        print(f"  Succ adv samples: {len(succ_adv_images)}")

        # 1. All adv → Control (no SAE)
        print("  [1/4] All adv → Control...")
        all_control_acc = evaluate(all_adv_images)

        # 2. All adv → SAE-only
        print("  [2/4] All adv → SAE-only...")
        all_sae_only_acc = evaluate(all_adv_images, make_sae_only_hook())

        # 3. Succ adv → Control (no SAE)
        succ_control_acc = None
        if succ_adv_images:
            print("  [3/4] Succ adv → Control...")
            succ_control_acc = evaluate(succ_adv_images)

        # 4. Succ adv → SAE-only
        succ_sae_only_acc = None
        if succ_adv_images:
            print("  [4/4] Succ adv → SAE-only...")
            succ_sae_only_acc = evaluate(succ_adv_images, make_sae_only_hook())

        sc_str = f"{succ_control_acc*100:.1f}%" if succ_control_acc is not None else "N/A"
        ss_str = f"{succ_sae_only_acc*100:.1f}%" if succ_sae_only_acc is not None else "N/A"
        print(
            f"  All:  Ctrl={all_control_acc*100:.1f}%  SAE={all_sae_only_acc*100:.1f}%\n"
            f"  Succ: Ctrl={sc_str}  SAE={ss_str}"
        )

        results.append(
            {
                "name": name,
                "wnid": wnid,
                "class_idx": cls_idx,
                "n_all_adv": len(all_adv_images),
                "n_succ_adv": len(succ_adv_images),
                "all_control_acc": all_control_acc,
                "all_sae_only_acc": all_sae_only_acc,
                "succ_control_acc": succ_control_acc,
                "succ_sae_only_acc": succ_sae_only_acc,
                "all_sae_vs_control": (all_sae_only_acc - all_control_acc) * 100,
                "succ_sae_vs_control": (succ_sae_only_acc - succ_control_acc) * 100
                if succ_control_acc is not None and succ_sae_only_acc is not None
                else None,
            }
        )

    # ── Compute overall averages ────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    overall = {
        "n_classes_evaluated": len(results),
        "avg_all_control_acc": avg("all_control_acc"),
        "avg_all_sae_only_acc": avg("all_sae_only_acc"),
        "avg_succ_control_acc": avg("succ_control_acc"),
        "avg_succ_sae_only_acc": avg("succ_sae_only_acc"),
        "avg_all_sae_vs_control": avg("all_sae_vs_control"),
        "avg_succ_sae_vs_control": avg("succ_sae_vs_control"),
    }

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("RESULTS SUMMARY")
    print("=" * 95)
    print(
        f"{'Class':>15} {'N_all':>5} {'N_succ':>6} {'All_Ctrl':>8} {'All_SAE':>8} "
        f"{'Succ_Ctrl':>9} {'Succ_SAE':>8} {'AllΔ':>6} {'SuccΔ':>6}"
    )
    print("-" * 95)

    for r in results:
        sc_str = f"{r['succ_control_acc']*100:>7.1f}%" if r["succ_control_acc"] is not None else f"{'N/A':>8}"
        ss_str = f"{r['succ_sae_only_acc']*100:>7.1f}%" if r["succ_sae_only_acc"] is not None else f"{'N/A':>8}"
        svd_str = f"{r['succ_sae_vs_control']:>+5.1f}" if r["succ_sae_vs_control"] is not None else f"{'N/A':>6}"
        print(
            f"{r['name']:>15} {r['n_all_adv']:>5} {r['n_succ_adv']:>6} "
            f"{r['all_control_acc']*100:>7.1f}% {r['all_sae_only_acc']*100:>7.1f}% "
            f"{sc_str} {ss_str} "
            f"{r['all_sae_vs_control']:>+5.1f} {svd_str}"
        )

    print("-" * 95)
    print(
        f"{'OVERALL AVERAGE':>15} {'':>5} {'':>6} "
        f"{overall['avg_all_control_acc']*100:>7.1f}% {overall['avg_all_sae_only_acc']*100:>7.1f}% "
        f"{overall['avg_succ_control_acc']*100:>7.1f}% {overall['avg_succ_sae_only_acc']*100:>7.1f}% "
        f"{overall['avg_all_sae_vs_control']:>+5.1f} {overall['avg_succ_sae_vs_control']:>+5.1f}"
    )
    print("=" * 95)

    # Save JSON
    out_path = OUT_DIR / "eval_results.json"
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

    # Stats
    n_all_sae_better = sum(1 for r in results if r["all_sae_vs_control"] > 0)
    n_succ_sae_better = sum(
        1
        for r in results
        if r["succ_sae_vs_control"] is not None and r["succ_sae_vs_control"] > 0
    )
    print(f"\nAll adv: SAE > Control:  {n_all_sae_better}/{len(results)} classes")
    print(f"Succ adv: SAE > Control:  {n_succ_sae_better}/{len(results)} classes")
    print(f"\nKEY METRICS:")
    print(f"  All adv  - Control (robust acc):  {overall['avg_all_control_acc']*100:.1f}%")
    print(f"  All adv  - SAE-only:              {overall['avg_all_sae_only_acc']*100:.1f}%")
    print(f"  Succ adv - Control:               {overall['avg_succ_control_acc']*100:.1f}%")
    print(f"  Succ adv - SAE-only:              {overall['avg_succ_sae_only_acc']*100:.1f}%")
    print(f"  All adv  - SAE effect:            {overall['avg_all_sae_vs_control']:+.1f} pp")
    print(f"  Succ adv - SAE effect:            {overall['avg_succ_sae_vs_control']:+.1f} pp")


if __name__ == "__main__":
    main()
