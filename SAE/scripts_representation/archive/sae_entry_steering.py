"""
Entry-Level Steering Ablation:
  - Control: Adv samples, no intervention
  - Experiment: Steering 30 selected entries back to clean mean
  - Validation: Steering 30 random entries (clean>=92%, non-candidate) back to clean mean
"""

import os
import sys
import json
import csv
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# ── Paths ───────────────────────────────────────────────────────────
sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint
from sae_core.model import TopKAutoencoder

# ── Config ──────────────────────────────────────────────────────────
BASE_CKPT   = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT    = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"
ADV_ROOT    = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples/apgd_ce_l2_eps3.0000_steps100_20260424_140904")
ADV_IMG_DIR = ADV_ROOT / "adv_succ_real"
ADV_CSV     = ADV_ROOT / "eval_results.csv"
JSON_PATH   = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect/selected_entry_inspect.json")
CLEAN_FEAT  = "/Data_share/hongyi/DAT/SAE/results_representation/features/clean_stage3_k256_features.npy"
OUT_DIR     = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect")

CLASS_IDX   = 150   # sea lion
BATCH_SIZE  = 8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load model ──────────────────────────────────────────────────────
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
    expansion_rate = config.get("expansion_rate", 8)
    d_lat = int(d_in) * int(expansion_rate)

sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
sae.load_state_dict(sae_ckpt["model_state_dict"])
sae = sae.to(DEVICE)
sae.eval()

norm_mean = sae_ckpt["norm_mean"].to(DEVICE)
norm_std = sae_ckpt["norm_std"].to(DEVICE)

print(f"  SAE: d_in={d_in}, d_lat={d_lat}, k={k}")

# ── Image preprocessing ─────────────────────────────────────────────
def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

preprocess = get_preprocess()

# ── Load adversarial samples ────────────────────────────────────────
print("Loading adversarial samples...")

# Read eval_results.csv to identify class-150 samples that successfully escaped
adv_samples = []
with open(ADV_CSV, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pred_clean = int(row["pred_clean"])
        escaped = int(row["escaped"])
        if pred_clean == CLASS_IDX and escaped == 1:
            img_path = ADV_IMG_DIR / row["filename"]
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
                image = preprocess(image).to(DEVICE)
                adv_samples.append({
                    "image": image,
                    "true_label": CLASS_IDX,
                    "predicted_label": int(row["pred_adv"]),
                })
            else:
                print(f"    Warning: missing image {img_path}")

print(f"  Found {len(adv_samples)} adv samples for class {CLASS_IDX}")

# ── Load JSON candidates ────────────────────────────────────────────
print(f"Loading candidates from {JSON_PATH}")
with open(JSON_PATH) as f:
    json_data = json.load(f)

selected_entries = json_data["selected_entries"]
print(f"  Selected candidates: {len(selected_entries)}")

# ── Build steering targets ──────────────────────────────────────────
print("Building steering targets...")

# 1. Experiment group: 30 selected entries
exp_targets = {}
for e in selected_entries:
    key = (e["token"], e["channel"])
    exp_targets[key] = e["clean_mean"]

# 2. Validation group: 30 random entries from clean>=92% pool
clean = np.load(CLEAN_FEAT)  # (50, 49, d_lat)
N_IMG, N_TOK, N_CH = clean.shape

clean_count = np.sum(clean != 0, axis=0)
mask_high = clean_count >= 46  # 92%

selected_keys = {(e["token"], e["channel"]) for e in selected_entries}
all_high_entries = []
tok_idx, ch_idx = np.where(mask_high)
for t, c in zip(tok_idx, ch_idx):
    if (int(t), int(c)) not in selected_keys:
        all_high_entries.append((int(t), int(c)))

print(f"  Pool for random sampling: {len(all_high_entries)} entries")
random_entries = random.sample(all_high_entries, 30)

# Compute clean_mean for random entries
clean_sum = np.sum(clean, axis=0)
clean_mean_map = np.zeros_like(clean_sum, dtype=np.float32)
mask_c = clean_count > 0
clean_mean_map[mask_c] = clean_sum[mask_c] / clean_count[mask_c]

val_targets = {}
for t, c in random_entries:
    val_targets[(t, c)] = float(clean_mean_map[t, c])

print(f"  Validation (random) entries: {len(val_targets)}")

# ── Hook factory ────────────────────────────────────────────────────
def make_steering_hook(targets_dict):
    """
    Hook registered on model.stages[3].
    Receives ConvNeXt feature (bsz, 1536, 7, 7), normalizes, encodes to SAE latents,
    steers specified entries, then decodes back to feature space.
    """
    def hook_fn(module, input, output):
        bsz, channels, height, width = output.shape
        # Flatten spatial: (bsz, 1536, 7, 7) -> (bsz*49, 1536)
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        
        # Encode to SAE latents: (bsz*49, d_lat)
        z = sae.encode(flat_norm)
        z_spatial = z.reshape(bsz, 7, 7, d_lat)
        
        # Steering: set specified (token, channel) to target clean_mean
        for (token, channel), target_val in targets_dict.items():
            row = token // 7
            col = token % 7
            z_spatial[:, row, col, channel] = target_val
        
        # Decode back
        z = z_spatial.reshape(bsz * 49, d_lat)
        recon_norm = (z @ sae.W_dec) + sae.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon
    return hook_fn

# ── Evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(samples, hook_fn=None):
    if hook_fn is not None:
        handle = model.stages[3].register_forward_hook(hook_fn)
    
    correct = 0
    total = 0
    
    try:
        for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Evaluating"):
            batch = samples[i:i+BATCH_SIZE]
            imgs = torch.stack([s["image"] for s in batch])
            labels = torch.tensor([s["true_label"] for s in batch], device=DEVICE)
            
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(batch)
    finally:
        if hook_fn is not None:
            handle.remove()
    
    return correct / total if total > 0 else 0.0

# ── Run experiments ─────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING STEERING EXPERIMENTS")
print("="*60)

print("\n[1/3] Control: Adv samples, no intervention")
control_acc = evaluate(adv_samples)
print(f"  Accuracy: {control_acc:.4f} ({control_acc*100:.1f}%)")

print(f"\n[2/3] Experiment: Steering {len(exp_targets)} selected entries to clean mean")
exp_hook = make_steering_hook(exp_targets)
exp_acc = evaluate(adv_samples, exp_hook)
print(f"  Accuracy: {exp_acc:.4f} ({exp_acc*100:.1f}%)")

print(f"\n[3/3] Validation: Steering {len(val_targets)} random entries to clean mean")
val_hook = make_steering_hook(val_targets)
val_acc = evaluate(adv_samples, val_hook)
print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Condition':<40} {'Accuracy':>10}")
print("-"*60)
print(f"{'Control (no steering)':<40} {control_acc*100:>9.1f}%")
print(f"{'Experiment (30 selected entries)':<40} {exp_acc*100:>9.1f}%")
print(f"{'Validation (30 random entries)':<40} {val_acc*100:>9.1f}%")
print("="*60)

improvement = (exp_acc - control_acc) * 100
print(f"\nImprovement over control: {improvement:+.1f} percentage points")

if exp_acc > val_acc:
    print("Selected entries have CAUSAL effect (better than random)")
else:
    print("Selected entries do NOT outperform random (no specific causal effect)")

# ── Save results ────────────────────────────────────────────────────
results = {
    "control_acc": control_acc,
    "experiment_acc": exp_acc,
    "validation_acc": val_acc,
    "improvement_pct": improvement,
    "n_adv_samples": len(adv_samples),
    "n_exp_entries": len(exp_targets),
    "n_val_entries": len(val_targets),
    "experiment_entries": [
        {"token": k[0], "channel": k[1], "clean_mean": v}
        for k, v in exp_targets.items()
    ],
    "validation_entries": [
        {"token": k[0], "channel": k[1], "clean_mean": v}
        for k, v in val_targets.items()
    ],
}

out_path = OUT_DIR / "steering_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
