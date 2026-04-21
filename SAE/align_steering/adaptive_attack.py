#!/usr/bin/env python3
"""
Adaptive Attack on SAE Cosine NCM classifier.

攻击者知道防御系统使用 SAE Cosine NCM 做分类，直接优化：
  loss = -mean_p cos(z_p(x_adv), mu_{y_true,p})

其中 z_p = SAE.encode(stage3_output_p)，mu_{c,p} 是 train set 统计的类均值。

评估时同时报告：
  1. Model Head 在对抗样本上的准确率（挂载 SAE hook）
  2. SAE Cosine NCM 在对抗样本上的 Top1/Top3/Top5
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: T.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        class_name = path.parent.name
        return x, class_name, str(path.name)


def build_model(device: torch.device, checkpoint: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, checkpoint)
    model = model.to(device)
    model.eval()
    return model


def build_sae(device: torch.device, sae_ckpt_path: str):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))
    if d_in is None:
        raise KeyError("Cannot resolve d_in")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat")
        d_lat = int(d_in) * int(expansion_rate)
    if k is None:
        raise KeyError("Cannot resolve k")

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    return sae, norm_mean, norm_std, {"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)}


def make_sae_hook(sae_model, norm_mean, norm_std):
    def hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        x_reconstruct, _, _ = sae_model(flat_norm)
        x_out = x_reconstruct * norm_std + norm_mean
        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return hook


def get_class_dirs(val_dir: Path) -> List[Path]:
    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def sample_images(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        return []
    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


# ============================================================================
# Adaptive Attack: minimize SAE cosine to true class
# ============================================================================

def adaptive_sae_attack(
    model: torch.nn.Module,
    sae_model: TopKAutoencoder,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    x: torch.Tensor,
    labels: torch.Tensor,
    cls_idx_t: torch.Tensor,    # [C, P, k]
    cls_val_t: torch.Tensor,    # [C, P, k]
    eps: float,
    steps: int,
    step_size: float = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    自适应攻击：直接最小化正确类在 SAE latent space 中的平均 cosine similarity。

    攻击时不挂载 SAE hook，直接 forward 到 stage 3 获取原始特征 f，
    encode 得到 z，与正确类统计量 mu 计算 cosine loss。
    """
    x0 = x.clone().detach()
    if step_size is None:
        step_size = eps / 4

    if random_start:
        x = x + torch.empty_like(x).uniform_(-eps, eps)
        x = torch.clamp(x, 0, 1)

    device = x.device
    B = x.size(0)
    P = 49
    k = 64

    for step in range(steps):
        x = x.clone().detach().requires_grad_(True)

        # Forward 到 stage 3（不挂载 hook，获取原始特征）
        out = model.stem(x)
        out = model.stages[0](out)
        out = model.stages[1](out)
        out = model.stages[2](out)
        f = model.stages[3](out)          # [B, 1536, 7, 7]

        # SAE encode
        Bf, C, H, W = f.shape
        flat = f.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae_model.encode(flat_norm)   # [B*49, 65536]
        z = z.reshape(Bf, H * W, -1)      # [B, 49, 65536]

        # 构建正确类的 target_mean [B, P, d_lat]
        target_mean = torch.zeros(B, P, z.shape[-1], device=device)
        for i in range(B):
            y = labels[i].item()
            for p in range(P):
                idx = cls_idx_t[y, p]     # [64]
                val = cls_val_t[y, p]     # [64]
                target_mean[i, p].scatter_(0, idx, val)

        # 计算 cosine similarity: [B, P]
        z_norm = torch.norm(z, dim=-1) + 1e-8
        mu_norm = torch.norm(target_mean, dim=-1) + 1e-8
        dot = (z * target_mean).sum(dim=-1)
        cosine = dot / (z_norm * mu_norm)     # [B, P]

        # 攻击目标：最小化正确类的平均 cosine → loss = -mean(cosine)
        loss = -cosine.mean(dim=1).sum()

        # Backward
        grad, = torch.autograd.grad(loss, x)

        # PGD step
        with torch.no_grad():
            x = x + step_size * grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = torch.clamp(x, 0, 1)

    return x.clone().detach()


# ============================================================================
# Sparse Cosine NCM (reused from eval_dis_sparse.py)
# ============================================================================

def sparse_cosine_ncm(
    adv_idx: torch.Tensor,      # [N, P, k]
    adv_val: torch.Tensor,      # [N, P, k]
    cls_idx: torch.Tensor,      # [C, P, k]
    cls_val: torch.Tensor,      # [C, P, k]
    device: torch.device,
) -> torch.Tensor:
    """TopK-sparse cosine similarity for NCM classification."""
    N, P, k = adv_idx.shape
    C = cls_idx.shape[0]
    d_lat = 65536

    cls_norms = torch.norm(cls_val, dim=-1).to(device)
    cosine_sum = torch.zeros(N, C, device=device, dtype=torch.float32)

    adv_idx_d = adv_idx.long().to(device)
    adv_val_d = adv_val.to(device)
    cls_idx_d = cls_idx.long().to(device)
    cls_val_d = cls_val.to(device)

    for p in range(P):
        cls_dense = torch.zeros(C, d_lat, device=device, dtype=torch.float32)
        cls_dense.scatter_(1, cls_idx_d[:, p, :], cls_val_d[:, p, :])

        dot = torch.zeros(N, C, device=device, dtype=torch.float32)
        for j in range(k):
            idx_j = adv_idx_d[:, p, j]
            val_j = adv_val_d[:, p, j]
            gathered = cls_dense[:, idx_j]
            dot += gathered.t() * val_j.unsqueeze(1)

        adv_norm = torch.norm(adv_val_d[:, p, :], dim=1)
        cosine_sum += dot / (adv_norm.unsqueeze(1) * cls_norms[:, p].unsqueeze(0) + 1e-8)

    return cosine_sum / P


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_adversarial(
    model: torch.nn.Module,
    sae_model: TopKAutoencoder,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    x_adv: torch.Tensor,
    labels: torch.Tensor,
    cls_idx_np: np.ndarray,
    cls_val_np: np.ndarray,
    class_names: List[str],
    true_class: str,
    device: torch.device,
) -> Dict:
    """
    对对抗样本同时评估：
      1. Model Head 准确率（挂载 SAE hook）
      2. SAE Cosine NCM Top1/Top3/Top5（capture z，sparse cosine）
    """
    N = x_adv.size(0)

    # ---------- 1. Model Head (with SAE hook) ----------
    hook = make_sae_hook(sae_model, norm_mean, norm_std)
    handle = model.stages[3].register_forward_hook(hook)

    with torch.no_grad():
        logits = model(x_adv)
        preds_head = logits.argmax(dim=1)
        head_correct = (preds_head == labels).sum().item()

    handle.remove()

    # ---------- 2. SAE Cosine NCM (capture z) ----------
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae_model.encode(flat_norm)
        captured["z"] = z.reshape(B, H * W, -1)

    handle_cap = model.stages[3].register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(x_adv)
        z_adv = captured["z"]          # [N, 49, 65536]

    handle_cap.remove()

    # 将 z_adv 转成 TopK 稀疏格式 [N, 49, 64]
    z_reshaped = z_adv.reshape(-1, z_adv.shape[-1])   # [N*49, 65536]
    z_topk_val, z_topk_idx = torch.topk(z_reshaped, k=64, dim=-1)
    z_idx = z_topk_idx.reshape(N, 49, 64).cpu()
    z_val = z_topk_val.reshape(N, 49, 64).cpu()

    # Sparse cosine NCM
    cls_idx_t = torch.from_numpy(cls_idx_np).long()
    cls_val_t = torch.from_numpy(cls_val_np).float()

    sim_cosine = sparse_cosine_ncm(z_idx, z_val, cls_idx_t, cls_val_t, device)

    # TopK predictions
    top5_idx = torch.topk(sim_cosine, k=5, dim=-1).indices.cpu().numpy()

    t1 = 0
    t3 = 0
    t5 = 0
    for i in range(N):
        pred_classes = [class_names[idx] for idx in top5_idx[i]]
        if pred_classes[0] == true_class:
            t1 += 1
        if true_class in pred_classes[:3]:
            t3 += 1
        if true_class in pred_classes[:5]:
            t5 += 1

    return {
        "head_accuracy": 100.0 * head_correct / N,
        "ncm_top1": 100.0 * t1 / N,
        "ncm_top3": 100.0 * t3 / N,
        "ncm_top5": 100.0 * t5 / N,
        "total": N,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Attack on SAE Cosine NCM classifier"
    )
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-ckpt", type=str,
                        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"))
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--class-npz", type=str,
                        default=str(Path(__file__).resolve().parent / "sae_stat_results_v2.npz"))
    parser.add_argument("--save-adversarial", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples" / "adaptive"))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + SAE
    print(f"Loading model from: {args.checkpoint}")
    model = build_model(device, args.checkpoint)

    print(f"Loading SAE from: {args.sae_ckpt}")
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)
    print(f"SAE: d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")

    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if sae_cfg["d_in"] != expected_din:
        raise ValueError(f"Stage mismatch: stage{args.sae_stage} expects d_in={expected_din}")

    # Load class statistics
    cls_data = np.load(args.class_npz, allow_pickle=True)
    cls_names = list(cls_data["class_names"])
    cls_indices = cls_data["spatial_indices"]        # [1000, 49, 64]
    cls_activations = cls_data["spatial_activations"] # [1000, 49, 64]
    print(f"Loaded class statistics: {len(cls_names)} classes")

    # Preload class stats to GPU tensor (for attack loop)
    cls_idx_t = torch.from_numpy(cls_indices).long().to(device)
    cls_val_t = torch.from_numpy(cls_activations).float().to(device)

    # Select class
    val_dir = Path(args.val_dir).resolve()
    class_dirs = get_class_dirs(val_dir)
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    if args.class_name:
        selected_class = args.class_name
        if selected_class not in class_to_idx:
            raise ValueError(f"Class {selected_class} not found")
    else:
        selected_class = random.choice([d.name for d in class_dirs])

    selected_class_idx = class_to_idx[selected_class]
    print(f"\nSelected class: {selected_class} (index: {selected_class_idx})")

    image_paths = sample_images(val_dir / selected_class, args.n_samples)
    print(f"Sampled {len(image_paths)} images")

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ====== Step 1: Clean accuracy ======
    print(f"\n{'='*60}")
    print("Clean Accuracy Evaluation")
    print(f"{'='*60}")

    clean_correct_head = 0
    clean_correct_ncm = 0
    total = 0

    for images, class_names, _ in tqdm(dataloader, desc="Clean eval"):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)
        B = images.size(0)

        # Head (with SAE hook)
        hook = make_sae_hook(sae_model, norm_mean, norm_std)
        handle = model.stages[args.sae_stage].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
            clean_correct_head += (preds == labels).sum().item()
        handle.remove()

        # NCM (capture z)
        captured: Dict[str, torch.Tensor] = {}
        def cap_hook(m, inp, out):
            Bc, C, H, W = out.shape
            flat = out.permute(0, 2, 3, 1).reshape(-1, C)
            flat_norm = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat_norm)
            captured["z"] = z.reshape(Bc, H * W, -1)

        handle_cap = model.stages[args.sae_stage].register_forward_hook(cap_hook)
        with torch.no_grad():
            _ = model(images)
            z_clean = captured["z"]      # [B, 49, 65536]
        handle_cap.remove()

        # z -> TopK
        z_r = z_clean.reshape(-1, z_clean.shape[-1])
        z_topk_val, z_topk_idx = torch.topk(z_r, k=64, dim=-1)
        z_idx = z_topk_idx.reshape(B, 49, 64).cpu()
        z_val = z_topk_val.reshape(B, 49, 64).cpu()

        # Sparse cosine NCM
        sim = sparse_cosine_ncm(z_idx, z_val,
                                torch.from_numpy(cls_indices).long(),
                                torch.from_numpy(cls_activations).float(),
                                device)
        pred_ncm = sim.argmax(dim=-1).cpu().numpy()
        for i in range(B):
            if cls_names[pred_ncm[i]] == selected_class:
                clean_correct_ncm += 1

        total += B

    clean_head_acc = 100.0 * clean_correct_head / total
    clean_ncm_top1 = 100.0 * clean_correct_ncm / total

    print(f"  Clean Head Acc:     {clean_head_acc:.2f}%")
    print(f"  Clean NCM Top1:     {clean_ncm_top1:.2f}%")

    # ====== Step 2: Generate adaptive attack ======
    print(f"\n{'='*60}")
    print(f"Adaptive Attack")
    print(f"  Epsilon: {args.eps:.4f} ({args.eps*255:.1f}/255)")
    print(f"  Steps: {args.steps}")
    print(f"  Step size: {args.step_size or args.eps/4:.4f}")
    print(f"  Loss: -mean_p cos(z_p, mu_true)")
    print(f"{'='*60}")

    all_adv_images = []
    all_labels = []

    for images, class_names, _ in tqdm(dataloader, desc="Adaptive attack"):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)

        x_adv = adaptive_sae_attack(
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            x=images,
            labels=labels,
            cls_idx_t=cls_idx_t,
            cls_val_t=cls_val_t,
            eps=args.eps,
            steps=args.steps,
            step_size=args.step_size,
        )

        all_adv_images.append(x_adv.cpu())
        all_labels.append(labels.cpu())

    all_adv_images = torch.cat(all_adv_images, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    # ====== Step 3: Evaluate adversarial ======
    print(f"\n{'='*60}")
    print("Adversarial Evaluation")
    print(f"{'='*60}")

    results = evaluate_adversarial(
        model=model,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        x_adv=all_adv_images,
        labels=all_labels,
        cls_idx_np=cls_indices,
        cls_val_np=cls_activations,
        class_names=cls_names,
        true_class=selected_class,
        device=device,
    )

    adv_head_acc = results["head_accuracy"]
    adv_ncm_top1 = results["ncm_top1"]
    adv_ncm_top3 = results["ncm_top3"]
    adv_ncm_top5 = results["ncm_top5"]

    print(f"  Adv Head Acc:       {adv_head_acc:.2f}%")
    print(f"  Adv NCM Top1:       {adv_ncm_top1:.2f}%")
    print(f"  Adv NCM Top3:       {adv_ncm_top3:.2f}%")
    print(f"  Adv NCM Top5:       {adv_ncm_top5:.2f}%")
    print(f"{'='*60}")
    print(f"  Head ASR:           {100 - adv_head_acc:.2f}%")
    print(f"  NCM ASR:            {100 - adv_ncm_top1:.2f}%")
    print(f"  NCM vs Head gap:    {adv_ncm_top1 - adv_head_acc:+.2f}%")
    print(f"{'='*60}")

    # ====== Save results ======
    output_data = {
        "meta": {
            "class_name": selected_class,
            "class_index": int(selected_class_idx),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sae_ckpt": str(Path(args.sae_ckpt).resolve()),
            "eps": float(args.eps),
            "steps": args.steps,
            "n_samples": total,
            "seed": args.seed,
        },
        "clean": {
            "head_accuracy": round(clean_head_acc, 2),
            "ncm_top1": round(clean_ncm_top1, 2),
        },
        "adversarial": {
            "head_accuracy": round(adv_head_acc, 2),
            "ncm_top1": round(adv_ncm_top1, 2),
            "ncm_top3": round(adv_ncm_top3, 2),
            "ncm_top5": round(adv_ncm_top5, 2),
            "ncm_vs_head_gap": round(adv_ncm_top1 - adv_head_acc, 2),
        }
    }

    if args.output_json is None:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        output_json_path = out_root / f"adaptive_attack_results_{selected_class}.json"
    else:
        output_json_path = Path(args.output_json)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_json_path}")

    # Save adversarial images
    if args.save_adversarial:
        adv_dir = Path(args.output_dir) / selected_class
        adv_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving adversarial images to: {adv_dir}")
        for i in range(all_adv_images.size(0)):
            img = torch.clamp(all_adv_images[i], 0, 1).cpu()
            T.ToPILImage()(img).save(adv_dir / f"adv_{i:04d}.png")
        print(f"Saved {all_adv_images.size(0)} images")

    print("\nDone!")


if __name__ == "__main__":
    main()
