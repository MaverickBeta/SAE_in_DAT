#!/usr/bin/env python3
"""
标准 APGD-CE 攻击 SAE-NCM 模型。

攻击者使用标准 CrossEntropy 攻击，不知道模型已被替换为 NCM。
模型输出 [B, 1000] 的 cosine scores 而不是 logits。
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ncm_model import build_ncm_model

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def get_class_dirs(val_dir: Path) -> List[Path]:
    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def sample_images(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        return []
    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


def apgd_ce_attack(model, x, labels, eps=8 / 255, steps=100, step_size=None, random_start=True):
    """标准 APGD-CE：模型输出 scores，攻击代码不变。"""
    x0 = x.clone().detach()
    if step_size is None:
        step_size = eps / 4

    if random_start:
        x = x + torch.empty_like(x).uniform_(-eps, eps)
        x = torch.clamp(x, 0, 1)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)

        scores = model(x)                # [B, 1000]  NCM cosine scores
        loss = F.cross_entropy(scores, labels)

        grad, = torch.autograd.grad(loss, x)

        with torch.no_grad():
            x = x + step_size * grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = torch.clamp(x, 0, 1)

    return x.clone().detach()


def evaluate(model, dataloader, device, eps, steps, step_size, class_to_idx):
    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, class_names, _ in tqdm(dataloader, desc="Evaluating"):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)
        B = images.size(0)
        total += B

        # Clean
        with torch.no_grad():
            scores_clean = model(images)
            preds_clean = scores_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()

        # Attack
        x_adv = apgd_ce_attack(model, images, labels, eps, steps, step_size)

        # Adv
        with torch.no_grad():
            scores_adv = model(x_adv)
            preds_adv = scores_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()

    return {
        "clean_accuracy": 100.0 * clean_correct / total,
        "adversarial_accuracy": 100.0 * adv_correct / total,
        "total_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(description="APGD-CE on SAE-NCM model")
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=0)
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
    parser.add_argument("--class-npz", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "align_steering" / "sae_stat_results_v2.npz"))
    parser.add_argument("--save-adversarial", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples"))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build NCM model
    print("Building SAE-NCM model...")
    model, class_names = build_ncm_model(device, args.checkpoint, args.sae_ckpt, args.class_npz)

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

    # Evaluate
    print(f"\n{'='*60}")
    print(f"APGD-CE on SAE-NCM model")
    print(f"  Epsilon: {args.eps:.4f} ({args.eps*255:.1f}/255)")
    print(f"  Steps: {args.steps}")
    print(f"{'='*60}")

    results = evaluate(model, dataloader, device, args.eps, args.steps, args.step_size, class_to_idx)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
    print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
    print(f"  Attack Success Rate: {100 - results['adversarial_accuracy']:.2f}%")
    print(f"{'='*60}")

    # Save
    output_data = {
        "meta": {
            "class_name": selected_class,
            "class_index": int(selected_class_idx),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "eps": float(args.eps),
            "steps": args.steps,
            "n_samples": results["total_samples"],
            "seed": args.seed,
        },
        "accuracy": {
            "clean_accuracy": float(results["clean_accuracy"]),
            "adversarial_accuracy": float(results["adversarial_accuracy"]),
            "attack_success_rate": float(100 - results["adversarial_accuracy"]),
            "total_samples": int(results["total_samples"]),
        }
    }

    if args.output_json is None:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        output_json_path = out_root / f"ncm_attack_results_{selected_class}.json"
    else:
        output_json_path = Path(args.output_json)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_json_path}")

    if args.save_adversarial:
        adv_dir = Path(args.output_dir) / selected_class
        adv_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving adversarial images to: {adv_dir}")

        img_idx = 0
        for images, class_names, _ in dataloader:
            labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
            images = images.to(device)
            x_adv = apgd_ce_attack(model, images, labels, args.eps, args.steps, args.step_size)

            for i in range(x_adv.size(0)):
                img = torch.clamp(x_adv[i], 0, 1).cpu()
                T.ToPILImage()(img).save(adv_dir / f"adv_{img_idx:04d}.png")
                img_idx += 1

        print(f"Saved {img_idx} images")

    print("\nDone!")


if __name__ == "__main__":
    main()
