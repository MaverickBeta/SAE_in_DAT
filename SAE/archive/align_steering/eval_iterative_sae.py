#!/usr/bin/env python3
"""
评估迭代 SAE hook 对干净样本和对抗样本的影响。
支持 T=1,2,5 自动对比，批量处理多个类别。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        class_name = path.parent.name
        return x, class_name


def build_model(device, checkpoint):
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


def build_sae(device, sae_ckpt_path):
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


def make_iterative_sae_hook(sae_model, norm_mean, norm_std, iterations=1):
    def hook(module, input, output):
        f = output  # [B, C, H, W]
        for _ in range(iterations):
            B, C, H, W = f.shape
            flat = f.permute(0, 2, 3, 1).reshape(-1, C)  # [B, C, H, W] -> [B*H*W, C]
            flat_norm = (flat - norm_mean) / norm_std
            x_reconstruct, _, _ = sae_model(flat_norm)
            f = (x_reconstruct * norm_std + norm_mean).reshape(B, H, W, C).permute(0, 3, 1, 2)  # back to [B, C, H, W]
        return f
    return hook


def evaluate_loader(model, dataloader, class_to_idx, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, class_names in tqdm(dataloader, desc="Eval", leave=False):
            labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate iterative SAE hook")
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--adv-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sae-ckpt", type=str, required=True)
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--iterations", type=str, default="1,2,5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-list", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading base model...")
    model = build_model(device, args.checkpoint)
    print("Loading SAE...")
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(f"Stage mismatch: stage{args.sae_stage} expects d_in={expected_din}")
    print(f"SAE: stage={args.sae_stage}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")

    val_dir = Path(args.val_dir).resolve()
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    adv_root = Path(args.adv_root)
    if args.class_list:
        with open(args.class_list, "r") as f:
            selected_classes = [line.strip() for line in f if line.strip()]
    else:
        selected_classes = sorted([d.name for d in adv_root.iterdir() if d.is_dir()])
    print(f"Classes to evaluate: {len(selected_classes)}")

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

    clean_paths_by_class = {}
    adv_paths_by_class = {}

    for cls in selected_classes:
        if cls not in class_to_idx:
            print(f"[Skip] {cls} not in val set")
            continue
        clean_dir = val_dir / cls
        clean_imgs = sorted([p for p in clean_dir.iterdir() if p.is_file() and p.suffix in exts])
        if clean_imgs:
            clean_paths_by_class[cls] = clean_imgs
        adv_dir = adv_root / cls
        if adv_dir.exists():
            adv_imgs = sorted([p for p in adv_dir.iterdir() if p.is_file() and p.suffix in exts])
            if adv_imgs:
                adv_paths_by_class[cls] = adv_imgs

    iteration_list = [int(x.strip()) for x in args.iterations.split(",")]
    print(f"Testing iterations: {iteration_list}")

    results = {}

    for n_iter in iteration_list:
        print(f"\n{'='*60}")
        print(f"Iterations T={n_iter}")
        print(f"{'='*60}")

        hook = make_iterative_sae_hook(sae_model, norm_mean, norm_std, iterations=n_iter)
        handle = model.stages[args.sae_stage].register_forward_hook(hook)

        clean_accs = []
        print(f"[T={n_iter}] Evaluating clean samples...")
        for cls in tqdm(selected_classes, desc="Clean", unit="class", leave=False):
            if cls not in clean_paths_by_class:
                continue
            dataset = ImagePathDataset(clean_paths_by_class[cls], transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            acc = evaluate_loader(model, loader, class_to_idx, device)
            clean_accs.append(acc)

        adv_accs = []
        print(f"[T={n_iter}] Evaluating adversarial samples...")
        for cls in tqdm(selected_classes, desc="Adv", unit="class", leave=False):
            if cls not in adv_paths_by_class:
                continue
            dataset = ImagePathDataset(adv_paths_by_class[cls], transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            acc = evaluate_loader(model, loader, class_to_idx, device)
            adv_accs.append(acc)

        handle.remove()

        avg_clean = np.mean(clean_accs) if clean_accs else 0.0
        avg_adv = np.mean(adv_accs) if adv_accs else 0.0
        std_clean = np.std(clean_accs) if clean_accs else 0.0
        std_adv = np.std(adv_accs) if adv_accs else 0.0

        results[f"T{n_iter}"] = {
            "clean_accuracy": round(avg_clean, 2),
            "clean_std": round(std_clean, 2),
            "adv_accuracy": round(avg_adv, 2),
            "adv_std": round(std_adv, 2),
            "asr": round(100.0 - avg_adv, 2),
            "num_classes_clean": len(clean_accs),
            "num_classes_adv": len(adv_accs),
        }

        print(f"  Clean Acc: {avg_clean:.2f}% ± {std_clean:.2f}%")
        print(f"  Adv Acc:   {avg_adv:.2f}% ± {std_adv:.2f}%")
        print(f"  ASR:       {100.0 - avg_adv:.2f}%")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'T':<5} {'Clean Acc':>18} {'Adv Acc':>18} {'ASR':>18}")
    print("-"*60)
    for n_iter in iteration_list:
        key = f"T{n_iter}"
        r = results[key]
        print(f"{key:<5} {r['clean_accuracy']:>17.2f}% {r['adv_accuracy']:>17.2f}% {r['asr']:>17.2f}%")
    print("="*60)

    if args.output_json:
        output_data = {
            "meta": {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "sae_ckpt": str(Path(args.sae_ckpt).resolve()),
                "sae_stage": args.sae_stage,
                "iterations": iteration_list,
                "adv_root": str(adv_root.resolve()),
            },
            "results": results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
