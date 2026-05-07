"""
Quick test: PGD purification on a single n01484850 image.

NOTE: The 30 selected entries in steering_results.json were computed for
class 150 (sea lion).  n01484850 is class 2.  Running with sea-lion entries
is therefore a cross-category sanity check — it verifies the pipeline works,
but the SAE-constraint may not be semantically meaningful for this class.
For a proper class-2 experiment you would need to re-run entry selection
on n01484850 adversarial samples.
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint
from sae_core.model import TopKAutoencoder
from rebm.attacks.attack_steps import L2Step

# ── Config ──────────────────────────────────────────────────────────
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"
STEERING_JSON = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect/steering_results.json")

ADV_PATH = Path("/Data_share/hongyi/DAT/SAE/adversarial_samples/n01484850/train/adv/n01484850_10085_adv.JPEG")
CLEAN_PATH = Path("/Data_share/hongyi/DAT/SAE/adversarial_samples/n01484850/train/clean/n01484850_10085.JPEG")

TRUE_LABEL = 2   # n01484850
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/sae_gen")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PGD hyper-params
EPS = 2.0
STEP_SIZE = 0.2
STEPS = 100
LAMBDA_SAE = 10.0

# ── Preprocess ──────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ── Load model & SAE ────────────────────────────────────────────────
print("Loading model...")
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

# ── Load steering entries (class-150 entries reused as sanity check) ─
print(f"Loading entries from {STEERING_JSON}")
with open(STEERING_JSON) as f:
    steering_data = json.load(f)
exp_entries = steering_data["experiment_entries"]
exp_targets = {}
for e in exp_entries:
    exp_targets[(e["token"], e["channel"])] = e["clean_mean"]
print(f"  Loaded {len(exp_targets)} entries (WARNING: these are for class 150, not class {TRUE_LABEL})")

# ── Load images ─────────────────────────────────────────────────────
print(f"\nLoading images...")
adv_img = preprocess(Image.open(ADV_PATH).convert("RGB")).to(DEVICE)
clean_img = preprocess(Image.open(CLEAN_PATH).convert("RGB")).to(DEVICE)
print(f"  Adv:   {ADV_PATH.name}")
print(f"  Clean: {CLEAN_PATH.name}")

# ── Quick evaluation of clean / adv ─────────────────────────────────
with torch.no_grad():
    clean_logits = model(clean_img.unsqueeze(0))
    adv_logits = model(adv_img.unsqueeze(0))
    clean_pred = clean_logits.argmax(dim=1).item()
    adv_pred = adv_logits.argmax(dim=1).item()
    clean_conf = F.softmax(clean_logits, dim=1)[0, TRUE_LABEL].item()
    adv_conf = F.softmax(adv_logits, dim=1)[0, TRUE_LABEL].item()

print(f"\nBefore purification:")
print(f"  Clean -> pred={clean_pred}, true_conf={clean_conf:.4f}")
print(f"  Adv   -> pred={adv_pred}, true_conf={adv_conf:.4f}")

# ── Purification core ───────────────────────────────────────────────
@torch.enable_grad()
def purify(x_adv, true_label, eps, step_size, steps, lambda_sae):
    x = x_adv.unsqueeze(0).clone().detach()
    x0 = x.clone().detach()
    label_tensor = torch.tensor([true_label], device=DEVICE)
    step = L2Step(eps=eps, orig_input=x0, step_size=step_size)
    history = []

    for i in range(steps):
        x = x.clone().detach().requires_grad_(True)

        # forward + capture stage3
        stage3_output = None
        def hook(m, inp, out):
            nonlocal stage3_output
            stage3_output = out
            return out
        handle = model.stages[3].register_forward_hook(hook)
        logits = model(x)
        handle.remove()

        # losses
        clf_loss = F.cross_entropy(logits, label_tensor)

        bsz, c, h, w = stage3_output.shape
        flat = stage3_output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z_spatial = z.reshape(bsz, h, w, d_lat)

        sae_loss = 0.0
        for (token, ch), target_val in exp_targets.items():
            row = token // 7
            col = token % 7
            sae_loss += ((z_spatial[:, row, col, ch] - target_val) ** 2).mean()

        total_loss = clf_loss + lambda_sae * sae_loss
        (grad,) = torch.autograd.grad(total_loss, x, retain_graph=False, create_graph=False)

        with torch.no_grad():
            x = step.step(x, -grad)
            x = step.project(x)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            true_conf = probs[0, true_label].item()

        history.append({
            "step": i,
            "clf_loss": clf_loss.item(),
            "sae_loss": sae_loss.item(),
            "total_loss": total_loss.item(),
            "pred": pred,
            "true_conf": true_conf,
        })

        if i % 20 == 0:
            print(f"  Step {i:03d}: pred={pred}, true_conf={true_conf:.4f}, "
                  f"clf={clf_loss.item():.4f}, sae={sae_loss.item():.4f}")

    return x.detach().squeeze(0), history


print(f"\nRunning PGD purification (eps={EPS}, step_size={STEP_SIZE}, steps={STEPS}, lambda_sae={LAMBDA_SAE})...")
purified_img, history = purify(adv_img, TRUE_LABEL, EPS, STEP_SIZE, STEPS, LAMBDA_SAE)

# ── Final eval ──────────────────────────────────────────────────────
with torch.no_grad():
    pur_logits = model(purified_img.unsqueeze(0))
    pur_pred = pur_logits.argmax(dim=1).item()
    pur_conf = F.softmax(pur_logits, dim=1)[0, TRUE_LABEL].item()

print(f"\nAfter purification:")
print(f"  Purified -> pred={pur_pred}, true_conf={pur_conf:.4f}")
print(f"  Restored: {pur_pred == TRUE_LABEL}")

# ── Save outputs ────────────────────────────────────────────────────
(OUT_DIR / "purified").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

# 1. purified image
pur_path = OUT_DIR / "purified" / "n01484850_10085_purified.png"
vutils.save_image(purified_img, pur_path)
print(f"\nSaved purified image: {pur_path}")

# 2. comparison grid
grid = vutils.make_grid(
    torch.stack([clean_img, adv_img, purified_img]),
    nrow=3, padding=4, pad_value=1.0
)
fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
ax.set_xticks([])
ax.set_yticks([])
labels = [
    f"Clean\n pred={clean_pred}  conf={clean_conf:.2f}",
    f"Adv\n pred={adv_pred}  conf={adv_conf:.2f}",
    f"Purified\n pred={pur_pred}  conf={pur_conf:.2f}",
]
width = grid.shape[2]
for pos, label in zip([width/6, width/2, width*5/6], labels):
    ax.text(pos, grid.shape[1]+15, label, ha="center", va="top", fontsize=11)
fig.savefig(OUT_DIR / "plots" / "n01484850_10085_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved comparison plot: {OUT_DIR / 'plots' / 'n01484850_10085_compare.png'}")

# 3. loss curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
steps = [h["step"] for h in history]
axes[0].plot(steps, [h["clf_loss"] for h in history], color="C0")
axes[0].set_title("Classification Loss (CE)")
axes[0].set_xlabel("Step")
axes[0].grid(True, alpha=0.3)
axes[1].plot(steps, [h["sae_loss"] for h in history], color="C1")
axes[1].set_title("SAE Constraint Loss")
axes[1].set_xlabel("Step")
axes[1].grid(True, alpha=0.3)
axes[2].plot(steps, [h["total_loss"] for h in history], color="C2")
axes[2].set_title("Total Loss")
axes[2].set_xlabel("Step")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "plots" / "n01484850_10085_loss.png", dpi=150)
plt.close(fig)
print(f"Saved loss curve: {OUT_DIR / 'plots' / 'n01484850_10085_loss.png'}")

# 4. confidence curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, [h["true_conf"] for h in history], color="C3", linewidth=2)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_title(f"Confidence on True Class ({TRUE_LABEL}) during Purification")
ax.set_xlabel("Step")
ax.set_ylabel("Confidence")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
fig.savefig(OUT_DIR / "plots" / "n01484850_10085_conf.png", dpi=150)
plt.close(fig)
print(f"Saved confidence curve: {OUT_DIR / 'plots' / 'n01484850_10085_conf.png'}")

# 5. JSON summary
summary = {
    "image": "n01484850_10085",
    "true_label": TRUE_LABEL,
    "clean_pred": clean_pred,
    "clean_conf": clean_conf,
    "adv_pred": adv_pred,
    "adv_conf": adv_conf,
    "purified_pred": pur_pred,
    "purified_conf": pur_conf,
    "restored": int(pur_pred == TRUE_LABEL),
    "history": history,
    "config": {"eps": EPS, "step_size": STEP_SIZE, "steps": STEPS, "lambda_sae": LAMBDA_SAE},
}
with open(OUT_DIR / "n01484850_10085_result.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved JSON: {OUT_DIR / 'n01484850_10085_result.json'}")

print("\n" + "="*60)
print("DONE")
print("="*60)
