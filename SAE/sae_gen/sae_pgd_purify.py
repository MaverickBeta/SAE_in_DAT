"""
SAE-Constrained PGD Purification (方案 B)
==========================================
从对抗样本出发，使用带 L2 约束的 PGD 优化像素，同时：
1. 最小化交叉熵损失（让模型正确分类为目标类别）
2. 最小化 SAE 特定 entries 与 clean_mean 的偏差

输出：
  - purified/      : 净化后的图像
  - plots/loss/    : 每步损失曲线
  - plots/compare/ : Clean vs Adv vs Purified 对比图
  - results.json   : 最终指标汇总
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# matplotlib 使用无后端模式（适合服务器）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────
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
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples/apgd_ce_l2_eps3.0000_steps100_20260424_140904")
ADV_IMG_DIR = ADV_ROOT / "adv_succ_real"
ADV_CSV = ADV_ROOT / "eval_results.csv"
STEERING_JSON = Path("/Data_share/hongyi/DAT/SAE/results_representation/entry_inspect/steering_results.json")
CLEAN_ROOT = Path("/Data_share/hongyi/imagenet/val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/sae_gen")

CLASS_IDX = 150       # sea lion
BATCH_SIZE = 1        # 单张处理，方便记录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PGD 超参数（可调）
EPS = 2.0             # L2 扰动预算（相对较大，给修复留空间）
STEP_SIZE = 0.2       # 每步走的距离
STEPS = 100           # 总步数
LAMBDA_SAE = 10.0     # SAE 约束损失的权重

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

# ── Preprocessing ───────────────────────────────────────────────────
def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

preprocess = get_preprocess()

# ── Load steering entries ───────────────────────────────────────────
print(f"Loading steering entries from {STEERING_JSON}")
with open(STEERING_JSON) as f:
    steering_data = json.load(f)

exp_entries = steering_data["experiment_entries"]  # 30 selected entries
print(f"  Experiment entries: {len(exp_entries)}")

# 构建字典加速索引
exp_targets = {}
for e in exp_entries:
    exp_targets[(e["token"], e["channel"])] = e["clean_mean"]

# ── Load adversarial samples ────────────────────────────────────────
print("Loading adversarial samples...")
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
                    "filename": row["filename"],
                    "source_filename": row["source_filename"],
                    "true_label": CLASS_IDX,
                    "pred_adv": int(row["pred_adv"]),
                })
            else:
                print(f"    Warning: missing image {img_path}")

print(f"  Found {len(adv_samples)} adv samples for class {CLASS_IDX}")

# ── Helper: find clean image ────────────────────────────────────────
def find_clean_image(source_filename: str) -> Path | None:
    """在 imagenet/val/ 子目录中搜索原始 clean 图像."""
    matches = list(CLEAN_ROOT.glob(f"**/{source_filename}"))
    return matches[0] if matches else None


# ── Core: SAE-constrained PGD purification ──────────────────────────
@torch.enable_grad()
def purify_with_sae_pgd(
    adv_img: torch.Tensor,
    true_label: int,
    eps: float = EPS,
    step_size: float = STEP_SIZE,
    steps: int = STEPS,
    lambda_sae: float = LAMBDA_SAE,
) -> tuple[torch.Tensor, list[dict]]:
    """
    从对抗样本出发，做 SAE 约束的 PGD 净化。

    Returns:
        purified_img: (3, 224, 224)
        loss_history: 每步的指标列表
    """
    x = adv_img.unsqueeze(0).clone().detach().to(DEVICE)  # (1, 3, 224, 224)
    x0 = x.clone().detach()
    label_tensor = torch.tensor([true_label], device=DEVICE)

    step = L2Step(eps=eps, orig_input=x0, step_size=step_size)
    loss_history = []

    for i in range(steps):
        x = x.clone().detach().requires_grad_(True)

        # ---- 一次前向传播同时获取 logits 和 stage3 特征 ----
        stage3_output = None

        def capture_hook(module, input, output):
            nonlocal stage3_output
            stage3_output = output
            return output

        handle = model.stages[3].register_forward_hook(capture_hook)
        logits = model(x)
        handle.remove()

        # ---- 1. 分类损失（最小化 CE = 让分类正确）----
        clf_loss = F.cross_entropy(logits, label_tensor)

        # ---- 2. SAE 约束损失 ----
        bsz, channels, height, width = stage3_output.shape
        flat = stage3_output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae.encode(flat_norm)
        z_spatial = z.reshape(bsz, height, width, d_lat)

        sae_loss = 0.0
        for (token, channel), target_val in exp_targets.items():
            row = token // 7
            col = token % 7
            diff = z_spatial[:, row, col, channel] - target_val
            sae_loss += (diff * diff).mean()

        # ---- 总损失 ----
        total_loss = clf_loss + lambda_sae * sae_loss

        # ---- 反向传播 ----
        (grad,) = torch.autograd.grad(
            outputs=total_loss,
            inputs=[x],
            retain_graph=False,
            create_graph=False,
        )

        # ---- PGD 更新：沿负梯度走（最小化），然后投影回 L2 ball ----
        with torch.no_grad():
            # step.step 默认沿 g 方向走；传入 -grad 即沿负梯度（下降）
            x = step.step(x, -grad)
            x = step.project(x)

        # ---- 记录指标 ----
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()
            true_conf = probs[0, true_label].item()

        loss_history.append({
            "step": i,
            "clf_loss": round(clf_loss.item(), 6),
            "sae_loss": round(sae_loss.item(), 6),
            "total_loss": round(total_loss.item(), 6),
            "pred": pred,
            "pred_conf": round(conf, 4),
            "true_conf": round(true_conf, 4),
            "is_correct": int(pred == true_label),
        })

    return x.detach().squeeze(0), loss_history


# ── Visualization helpers ───────────────────────────────────────────
def plot_loss_curve(loss_history: list[dict], save_path: Path):
    """绘制损失曲线."""
    steps = [d["step"] for d in loss_history]
    clf_losses = [d["clf_loss"] for d in loss_history]
    sae_losses = [d["sae_loss"] for d in loss_history]
    total_losses = [d["total_loss"] for d in loss_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, clf_losses, color="C0")
    axes[0].set_title("Classification Loss (CE)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, sae_losses, color="C1")
    axes[1].set_title("SAE Constraint Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, total_losses, color="C2")
    axes[2].set_title("Total Loss")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confidence_curve(loss_history: list[dict], save_path: Path, true_label: int):
    """绘制目标类别置信度变化."""
    steps = [d["step"] for d in loss_history]
    true_confs = [d["true_conf"] for d in loss_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, true_confs, color="C3", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")
    ax.set_title(f"Confidence on True Class ({true_label})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_comparison_grid(
    clean_img: torch.Tensor,
    adv_img: torch.Tensor,
    purified_img: torch.Tensor,
    labels: list[str],
    save_path: Path,
):
    """保存 1×3 对比图: Clean | Adv | Purified."""
    grid = torch.stack([clean_img, adv_img, purified_img])  # (3, 3, H, W)
    grid_img = vutils.make_grid(grid, nrow=3, padding=4, pad_value=1.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    ax.set_xticks([])
    ax.set_yticks([])

    # 在底部添加文字标签
    width = grid_img.shape[2]
    positions = [width / 6, width / 2, width * 5 / 6]
    for pos, label in zip(positions, labels):
        ax.text(pos, grid_img.shape[1] + 15, label, ha="center", va="top", fontsize=12)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main loop ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STARTING SAE-CONSTRAINED PGD PURIFICATION")
print(f"  eps={EPS}, step_size={STEP_SIZE}, steps={STEPS}, lambda_sae={LAMBDA_SAE}")
print("=" * 60)

# 创建输出目录
(OUT_DIR / "purified").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots" / "loss").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots" / "confidence").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots" / "compare").mkdir(parents=True, exist_ok=True)

all_results = []

for idx, sample in enumerate(tqdm(adv_samples, desc="Purifying")):
    filename = sample["filename"]
    source_filename = sample["source_filename"]
    adv_img = sample["image"]
    true_label = sample["true_label"]

    print(f"\n[{idx+1}/{len(adv_samples)}] {filename}")

    # ---- 加载对应的 clean 图像 ----
    clean_path = find_clean_image(source_filename)
    if clean_path is None:
        print(f"  [SKIP] Clean image not found: {source_filename}")
        continue

    clean_img = preprocess(Image.open(clean_path).convert("RGB")).to(DEVICE)

    # ---- 验证 clean 和 adv 的初始预测（用于记录）----
    with torch.no_grad():
        clean_logits = model(clean_img.unsqueeze(0))
        adv_logits = model(adv_img.unsqueeze(0))
        clean_pred = clean_logits.argmax(dim=1).item()
        adv_pred = adv_logits.argmax(dim=1).item()
        clean_conf = F.softmax(clean_logits, dim=1)[0, true_label].item()
        adv_conf = F.softmax(adv_logits, dim=1)[0, true_label].item()

    print(f"  Clean: pred={clean_pred}, true_conf={clean_conf:.3f}")
    print(f"  Adv:   pred={adv_pred}, true_conf={adv_conf:.3f}")

    # ---- 运行 PGD 净化 ----
    purified_img, loss_history = purify_with_sae_pgd(
        adv_img=adv_img,
        true_label=true_label,
        eps=EPS,
        step_size=STEP_SIZE,
        steps=STEPS,
        lambda_sae=LAMBDA_SAE,
    )

    # ---- 最终验证 ----
    with torch.no_grad():
        pur_logits = model(purified_img.unsqueeze(0))
        pur_pred = pur_logits.argmax(dim=1).item()
        pur_conf = F.softmax(pur_logits, dim=1)[0, true_label].item()

    print(f"  Pur:   pred={pur_pred}, true_conf={pur_conf:.3f}")

    # ---- 保存净化图像 ----
    pur_filename = filename.replace(".JPEG", "_purified.png")
    pur_path = OUT_DIR / "purified" / pur_filename
    vutils.save_image(purified_img, pur_path)

    # ---- 保存可视化 ----
    base_name = filename.replace(".JPEG", "")
    plot_loss_curve(loss_history, OUT_DIR / "plots" / "loss" / f"{base_name}_loss.png")
    plot_confidence_curve(loss_history, OUT_DIR / "plots" / "confidence" / f"{base_name}_conf.png", true_label)

    labels = [
        f"Clean\npred={clean_pred}, conf={clean_conf:.2f}",
        f"Adv\npred={adv_pred}, conf={adv_conf:.2f}",
        f"Purified\npred={pur_pred}, conf={pur_conf:.2f}",
    ]
    save_comparison_grid(
        clean_img, adv_img, purified_img,
        labels, OUT_DIR / "plots" / "compare" / f"{base_name}_compare.png"
    )

    # ---- 汇总记录 ----
    all_results.append({
        "filename": filename,
        "source_filename": source_filename,
        "true_label": true_label,
        "clean_pred": clean_pred,
        "clean_conf": round(clean_conf, 4),
        "adv_pred": adv_pred,
        "adv_conf": round(adv_conf, 4),
        "purified_pred": pur_pred,
        "purified_conf": round(pur_conf, 4),
        "restored": int(pur_pred == true_label),
        "final_clf_loss": loss_history[-1]["clf_loss"],
        "final_sae_loss": loss_history[-1]["sae_loss"],
        "loss_history": loss_history,
    })

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total = len(all_results)
restored = sum(r["restored"] for r in all_results)
print(f"Total samples:     {total}")
print(f"Restored (correct): {restored} / {total} = {restored/total*100:.1f}%")

# 保存 JSON 结果
results_path = OUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump({
        "config": {
            "eps": EPS,
            "step_size": STEP_SIZE,
            "steps": STEPS,
            "lambda_sae": LAMBDA_SAE,
            "class_idx": CLASS_IDX,
        },
        "summary": {
            "total": total,
            "restored": restored,
            "restoration_rate": restored / total if total > 0 else 0.0,
        },
        "per_sample": all_results,
    }, f, indent=2)

print(f"\nAll results saved to: {results_path}")
