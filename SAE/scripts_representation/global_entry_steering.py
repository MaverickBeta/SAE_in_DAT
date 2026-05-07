#!/usr/bin/env python3
"""
Universal steering using global entries across all 20 classes.

Steers the same global entries for ALL classes' adversarial samples.
Each entry is adjusted by subtracting its cross-class mean delta.

Hyperparameters (via argparse):
  --cf / --clean-freq:   min per-class clean frequency threshold
  --con / --consistency: min direction consistency threshold
  --cv:                  max coefficient of variation threshold
  --min-classes:         minimum number of valid classes per entry
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
GLOBAL_ENTRIES_JSON = "/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect_20_train/probe_global_entries.json"
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20cls_val")
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20classes_global_entry")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Load global entries ─────────────────────────────────────────────
print(f"\nLoading global entries from {GLOBAL_ENTRIES_JSON}...")
with open(GLOBAL_ENTRIES_JSON) as f:
    global_data = json.load(f)


def filter_entries(raw_entries, cf_thresh, con_thresh, cv_thresh, min_classes=12):
    """
    Dynamically filter entries based on per-class clean_freq and recomputed consistency/CV.
    """
    filtered = []
    for e in raw_entries:
        # 1. Keep only classes with clean_freq >= cf_thresh
        valid_classes = [
            pc for pc in e["per_class"]
            if pc["clean_freq"] >= cf_thresh
        ]
        n_valid = len(valid_classes)
        if n_valid < min_classes:
            continue

        # 2. Recompute consistency based on remaining classes
        n_has_dir = sum(1 for pc in valid_classes if pc.get("has_direction", True))
        consistency = n_has_dir / n_valid if n_valid > 0 else 0.0
        if consistency < con_thresh:
            continue

        # 3. Recompute mean_delta, std_delta, CV based on remaining classes
        deltas = np.array([pc["delta"] for pc in valid_classes], dtype=np.float64)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=0))
        if abs(mean_delta) < 1e-12:
            continue
        cv = std_delta / abs(mean_delta)
        if cv >= cv_thresh:
            continue

        # Build filtered entry (keep original token/channel but use recomputed stats)
        filtered.append({
            "token": e["token"],
            "channel": e["channel"],
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "cv": cv,
            "consistency": consistency,
            "n_valid": n_valid,
            "direction": e.get("direction", "SUPPRESS"),
        })
    return filtered


# ── Image preprocessing ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def load_all_adv_images(wnid, cls_idx):
    """Load ALL adversarial samples (succ + fail) from the run directory."""
    run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
    img_dir = ADV_ROOT / run_name
    all_images = []
    succ_images = []
    if img_dir.is_dir():
        for path in sorted(img_dir.glob("*.JPEG")):
            # Skip files inside subdirectories (e.g., adv_succ_real)
            if path.parent != img_dir:
                continue
            img = Image.open(path).convert("RGB")
            img = transform(img).to(DEVICE)
            all_images.append((img, cls_idx))
            if "_succ" in path.name:
                succ_images.append((img, cls_idx))
    return all_images, succ_images


def load_clean_images(wnid, cls_idx):
    """Load clean validation images for this class."""
    img_dir = IMAGENET_VAL / wnid
    images = []
    if img_dir.is_dir():
        for path in sorted(img_dir.glob("*.JPEG")):
            img = Image.open(path).convert("RGB")
            img = transform(img).to(DEVICE)
            images.append((img, cls_idx))
    return images

# ── Steering hook ───────────────────────────────────────────────────
def make_universal_steering_hook(entries):
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z_spatial = z.reshape(bsz, 7, 7, d_lat)
        for e in entries:
            token, channel = e["token"], e["channel"]
            row = token // 7
            col = token % 7
            mean_delta = e["mean_delta"]
            # Subtract mean_delta (mean_delta is negative, so this adds back)
            z_spatial[:, row, col, channel] = z_spatial[:, row, col, channel] - mean_delta
        z = z_spatial.reshape(bsz * 49, d_lat)
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
    parser = argparse.ArgumentParser(description="Universal global entry steering")
    parser.add_argument("--cf", "--clean-freq", type=float, default=0.92,
                        help="Minimum per-class clean frequency threshold (default: 0.92)")
    parser.add_argument("--con", "--consistency", type=float, default=0.80,
                        help="Minimum direction consistency threshold (default: 0.80)")
    parser.add_argument("--cv", type=float, default=0.80,
                        help="Maximum coefficient of variation threshold (default: 0.80)")
    parser.add_argument("--min-classes", type=int, default=12,
                        help="Minimum number of valid classes per entry (default: 12)")
    args = parser.parse_args()

    cf_str = f"{int(args.cf * 100):d}"
    con_str = f"{int(args.con * 100):d}"
    cv_str = f"{int(args.cv * 100):d}"
    suffix = f"cf_{cf_str}_con_{con_str}_cv_{cv_str}"

    # Dynamically filter entries
    entries_info = filter_entries(
        global_data["entries"],
        cf_thresh=args.cf,
        con_thresh=args.con,
        cv_thresh=args.cv,
        min_classes=args.min_classes,
    )
    print(f"\nUsing {len(entries_info)} global entries (cf>={args.cf}, con>={args.con}, cv<{args.cv}, min_classes>={args.min_classes}):")
    for e in entries_info:
        print(f"  t{e['token']}_c{e['channel']}: mean_delta={e['mean_delta']:+.3f}, "
              f"consistency={e['consistency']:.2f}, CV={e['cv']:.3f}, n_valid={e['n_valid']}")

    print("\n" + "=" * 80)
    print("UNIVERSAL GLOBAL ENTRY STEERING")
    print(f"  Params: {suffix}")
    print("  Robust Accuracy: ALL adversarial samples (succ + fail)")
    print("  Recovery Rate:   ONLY succ samples / relative to clean ceiling")
    print("=" * 80)

    results = []
    # Robust (all adv) accumulators
    total_all_adv = 0
    total_all_control_correct = 0
    total_all_steering_correct = 0
    # Succ-only accumulators
    total_succ = 0
    total_succ_control_correct = 0
    total_succ_steering_correct = 0
    # Clean accumulators
    total_clean = 0
    total_clean_correct = 0

    for wnid, cls_idx, name in CLASSES:
        print(f"\n[{name}] wnid={wnid} cls={cls_idx}")
        
        # ── Load images ───────────────────────────────────────────────
        all_adv_images, succ_adv_images = load_all_adv_images(wnid, cls_idx)
        clean_images = load_clean_images(wnid, cls_idx)
        
        if not all_adv_images:
            print(f"  [SKIP] No adversarial images found")
            continue
        print(f"  Clean: {len(clean_images)} | All adv: {len(all_adv_images)} | Succ: {len(succ_adv_images)}")

        # ── Clean accuracy (ceiling) ──────────────────────────────────
        print("  [Clean] ...")
        clean_acc = evaluate(clean_images)

        # ── Robust: all adversarial samples ───────────────────────────
        print("  [Robust Control] All adv, no steering...")
        robust_control_acc = evaluate(all_adv_images)
        
        print("  [Robust Steering] All adv, universal steering...")
        hook = make_universal_steering_hook(entries_info)
        robust_steering_acc = evaluate(all_adv_images, hook)

        # ── Recovery: only succ samples ───────────────────────────────
        if succ_adv_images:
            print("  [Succ Control] Succ only, no steering...")
            succ_control_acc = evaluate(succ_adv_images)
            
            print("  [Succ Steering] Succ only, universal steering...")
            succ_steering_acc = evaluate(succ_adv_images, hook)
        else:
            succ_control_acc = 0.0
            succ_steering_acc = 0.0

        # ── Recovery rate ─────────────────────────────────────────────
        recovery_rate = None
        if clean_acc > succ_control_acc:
            recovery_rate = (succ_steering_acc - succ_control_acc) / (clean_acc - succ_control_acc) * 100

        print(f"  Clean: {clean_acc:.4f}")
        print(f"  Robust Control: {robust_control_acc:.4f}  Robust Steering: {robust_steering_acc:.4f}  "
              f"Improvement: {robust_steering_acc - robust_control_acc:+.4f}")
        if recovery_rate is not None:
            print(f"  Succ Control: {succ_control_acc:.4f}  Succ Steering: {succ_steering_acc:.4f}  "
                  f"Recovery: {recovery_rate:.1f}%")

        # ── Accumulate ────────────────────────────────────────────────
        total_all_adv += len(all_adv_images)
        total_all_control_correct += int(robust_control_acc * len(all_adv_images))
        total_all_steering_correct += int(robust_steering_acc * len(all_adv_images))
        
        total_succ += len(succ_adv_images)
        total_succ_control_correct += int(succ_control_acc * len(succ_adv_images))
        total_succ_steering_correct += int(succ_steering_acc * len(succ_adv_images))
        
        total_clean += len(clean_images)
        total_clean_correct += int(clean_acc * len(clean_images))

        results.append({
            "name": name,
            "wnid": wnid,
            "class_idx": cls_idx,
            "n_clean": len(clean_images),
            "n_all_adv": len(all_adv_images),
            "n_succ": len(succ_adv_images),
            "clean_acc": clean_acc,
            "robust_control_acc": robust_control_acc,
            "robust_steering_acc": robust_steering_acc,
            "succ_control_acc": succ_control_acc,
            "succ_steering_acc": succ_steering_acc,
            "recovery_rate": recovery_rate,
        })

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Class':>15} {'Clean':>6} {'N_all':>5} {'N_suc':>5} {'Robust_C':>8} {'Robust_S':>9} {'Succ_C':>7} {'Succ_S':>8} {'Recov%':>7}")
    print("-" * 80)

    for r in results:
        rec_str = f"{r['recovery_rate']:.1f}" if r['recovery_rate'] is not None else "N/A"
        print(f"{r['name']:>15} {r['clean_acc']*100:>5.1f}% {r['n_all_adv']:>5} {r['n_succ']:>5} "
              f"{r['robust_control_acc']*100:>7.1f}% {r['robust_steering_acc']*100:>8.1f}% "
              f"{r['succ_control_acc']*100:>6.1f}% {r['succ_steering_acc']*100:>7.1f}% "
              f"{rec_str:>7}")

    # Overall
    overall_clean = total_clean_correct / total_clean if total_clean > 0 else 0
    overall_robust_control = total_all_control_correct / total_all_adv if total_all_adv > 0 else 0
    overall_robust_steering = total_all_steering_correct / total_all_adv if total_all_adv > 0 else 0
    overall_succ_control = total_succ_control_correct / total_succ if total_succ > 0 else 0
    overall_succ_steering = total_succ_steering_correct / total_succ if total_succ > 0 else 0
    overall_recovery = None
    if overall_clean > overall_succ_control:
        overall_recovery = (overall_succ_steering - overall_succ_control) / (overall_clean - overall_succ_control) * 100

    print("-" * 80)
    overall_recov_str = f"{overall_recovery:.1f}%" if overall_recovery is not None else "N/A"
    print(f"{'OVERALL':>15} {overall_clean*100:>5.1f}% {total_all_adv:>5} {total_succ:>5} "
          f"{overall_robust_control*100:>7.1f}% {overall_robust_steering*100:>8.1f}% "
          f"{overall_succ_control*100:>6.1f}% {overall_succ_steering*100:>7.1f}% "
          f"{overall_recov_str:>7}")

    # Save JSON
    out_path = OUT_DIR / f"global_steering_results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "cf": args.cf,
                "consistency": args.con,
                "cv": args.cv,
                "min_classes": args.min_classes,
                "global_entries": [{"token": e["token"], "channel": e["channel"], 
                                   "mean_delta": e["mean_delta"], "consistency": e["consistency"],
                                   "cv": e["cv"], "n_valid": e["n_valid"]} for e in entries_info],
            },
            "overall": {
                "n_clean": total_clean,
                "n_all_adv": total_all_adv,
                "n_succ": total_succ,
                "clean_acc": overall_clean,
                "robust_control_acc": overall_robust_control,
                "robust_steering_acc": overall_robust_steering,
                "succ_control_acc": overall_succ_control,
                "succ_steering_acc": overall_succ_steering,
                "recovery_rate": overall_recovery,
            },
            "per_class": results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
