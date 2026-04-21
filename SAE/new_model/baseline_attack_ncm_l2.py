#!/usr/bin/env python3
"""
L2 PGD-CE 攻击 SAE-NCM 模型（与训练配置一致）。

使用 L2 范数约束：eps=3.0, step_size=3.0, steps=110。
与模型训练时的对抗配置相同。
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

sys.path.insert(0, str(REPO_ROOT))
from rebm.attacks.attack_steps import L2Step


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


def l2_pgd_ce_attack(model, x, labels, eps=3.0, steps=110, step_size=3.0, random_start=True):
    """L2 PGD-CE 攻击（与模型训练配置一致）。"""
    x0 = x.clone().detach()

    step = L2Step(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)

        scores = model(x)                # [B, 1000]  NCM cosine scores
        loss = F.cross_entropy(scores, labels)

        grad, = torch.autograd.grad(loss, x)

        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

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
        x_adv = l2_pgd_ce_attack(model, images, labels, eps, steps, step_size)

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
    parser = argparse.ArgumentParser(description="L2 PGD-CE on SAE-NCM model")
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--eps", type=float, default=3.0,
                        help="L2 perturbation budget (default 3.0, matching training config)")
    parser.add_argument("--steps", type=int, default=110,
                        help="PGD steps (default 110, matching training config)")
    parser.add_argument("--step-size", type=float, default=3.0,
                        help="L2 step size (default 3.0, matching training config)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-ckpt", type=str,
                        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"))
    parser.add_argument("--class-npz", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "align_steering" / "sae_stat_results_v2.npz"))
    parser.add_argument("--save-adversarial", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples_l2"))
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip class if output JSON already exists")
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
    print(f"L2 PGD-CE on SAE-NCM model")
    print(f"  L2 Epsilon: {args.eps}")
    print(f"  Steps: {args.steps}")
    print(f"  Step size: {args.step_size}")
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
            "sae_ckpt": str(Path(args.sae_ckpt).resolve()) if args.sae_ckpt else None,
            "class_npz": str(Path(args.class_npz).resolve()) if args.class_npz else None,
            "eps": float(args.eps),
            "steps": args.steps,
            "step_size": args.step_size,
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
        output_json_path = out_root / f"ncm_l2_attack_results_{selected_class}.json"
    else:
        output_json_path = Path(args.output_json)

    if args.skip_existing and output_json_path.exists():
        print(f"\n[SKIP] {output_json_path} already exists, skipping.")
        return

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
