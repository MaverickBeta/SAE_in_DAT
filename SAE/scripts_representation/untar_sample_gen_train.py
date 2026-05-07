#!/usr/bin/env python3
"""
Untargeted adversarial sample generator for SAE representation analysis.

Key design choices:
- No SAE is loaded (attack the base model only).
- Supports both L2 and Linf threat models (DAT was trained with L2).
- Outputs paired clean/adv filenames for downstream SAE inspection.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model

try:
    import torchattacks
except ImportError as e:
    raise ImportError("Please install torchattacks: pip install torchattacks") from e

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        all_files = sorted(Path(folder_path).glob("*"))
        self.image_paths = [str(p) for p in all_files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(path).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate untargeted adversarial samples for SAE analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/Data_share/hongyi/DAT/data/ImageNet/val/n02077923",
        help="Directory containing clean source-class images.",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth",
        help="ConvNeXt checkpoint path (NO SAE).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/untar_samples",
        help="Root directory to save adversarial images and metadata.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name suffix. If empty, auto-generated from params.",
    )

    # Attack parameters
    parser.add_argument(
        "--norm",
        type=str,
        choices=["L2", "Linf"],
        default="L2",
        help=(
            "Threat model norm. "
            "DAT was trained with L2 (eps=3.0, steps=3); use L2 for in-distribution attacks. "
            "Linf is common in RobustBench but out-of-distribution for this model."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3.0,
        help=(
            "Perturbation budget. "
            "For L2: DAT training used 3.0 (on [0,1] images). "
            "For Linf: common values are 4/255 (~0.016) or 8/255 (~0.031)."
        ),
    )
    parser.add_argument("--steps", type=int, default=100, help="APGD attack steps.")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "dlr"],
        default="ce",
        help="APGD loss function.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help="APGD restarts. Increase for stronger attacks at the cost of speed.",
    )

    # Data / runtime
    parser.add_argument("--source-cls", type=int, default=150, help="Source class label.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = use all images in source-dir.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_transform():
    # Match the model's expected input: [0,1] range, 224x224.
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Clamp tensor to [0,1] for saving as image."""
    return torch.clamp(t, 0.0, 1.0)


def build_model(device: torch.device, ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def build_attack(model, norm: str, eps: float, steps: int, loss: str, n_restarts: int):
    if norm == "L2":
        # torchattacks APGD uses eps directly as the L2 radius.
        atk = torchattacks.APGD(
            model,
            norm="L2",
            eps=eps,
            steps=steps,
            loss=loss,
            n_restarts=n_restarts,
            eot_iter=1,
        )
    else:
        atk = torchattacks.APGD(
            model,
            norm="Linf",
            eps=eps,
            steps=steps,
            loss=loss,
            n_restarts=n_restarts,
            eot_iter=1,
        )
    # APGD defaults to untargeted mode; do not call set_mode_targeted_by_label
    # because APGD does not support targeted attacks in this torchattacks version.
    atk.set_normalization_used(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # [0,1] input
    return atk


def save_image_tensor(tensor: torch.Tensor, path: str):
    """Save a CHW tensor in [0,1] as a JPEG."""
    denorm = denormalize_tensor(tensor)
    pil = transforms.ToPILImage()(denorm.cpu())
    pil.save(path, quality=95)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Norm: {args.norm}, eps: {args.eps}, steps: {args.steps}, loss: {args.loss}")

    # Paths
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    run_tag = args.run_name or (
        f"apgd_{args.loss}_{args.norm.lower()}_eps{args.eps:.4f}_steps{args.steps}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Model
    model = build_model(device, args.base_ckpt)

    # Data
    dataset = ImageFolderDataset(str(source_dir), transform=get_transform())
    image_paths = dataset.image_paths
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
        # rebuild dataset with truncated list
        dataset = ImageFolderDataset(str(source_dir), transform=get_transform())
        dataset.image_paths = image_paths

    if len(dataset) == 0:
        raise RuntimeError("No images found in source directory.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Attack
    attack = build_attack(
        model,
        norm=args.norm,
        eps=args.eps,
        steps=args.steps,
        loss=args.loss,
        n_restarts=args.n_restarts,
    )

    # Run attack
    rows: List[Dict] = []
    total = 0
    clean_correct = 0
    adv_escape = 0

    for images, filenames in tqdm(loader, desc="Attack"):
        images = images.to(device)
        labels = torch.full((images.size(0),), args.source_cls, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_clean = model(images)
            preds_clean = logits_clean.argmax(dim=1)

        # Generate adversarial examples
        adv_images = attack(images, labels)
        adv_images = denormalize_tensor(adv_images)

        with torch.no_grad():
            logits_adv = model(adv_images)
            preds_adv = logits_adv.argmax(dim=1)
            confs_adv = torch.softmax(logits_adv, dim=1).max(dim=1).values

        for i in range(images.size(0)):
            fname = filenames[i]
            stem = Path(fname).stem
            pred_clean = int(preds_clean[i].item())
            pred_adv = int(preds_adv[i].item())
            conf_adv = float(confs_adv[i].item())
            escaped = int(pred_adv != args.source_cls)

            clean_correct += int(pred_clean == args.source_cls)
            adv_escape += escaped
            total += 1

            # Save adversarial image
            status = "succ" if escaped else "fail"
            out_name = f"{stem}_{status}.JPEG"
            out_path = output_dir / out_name
            save_image_tensor(adv_images[i], str(out_path))

            rows.append(
                {
                    "filename": out_name,
                    "source_filename": fname,
                    "pred_clean": pred_clean,
                    "pred_adv": pred_adv,
                    "conf_adv": round(conf_adv, 6),
                    "escaped": escaped,
                }
            )

    # Save metadata
    summary = {
        "source_dir": str(source_dir),
        "base_ckpt": args.base_ckpt,
        "output_dir": str(output_dir),
        "norm": args.norm,
        "eps": args.eps,
        "steps": args.steps,
        "loss": args.loss,
        "n_restarts": args.n_restarts,
        "source_cls": args.source_cls,
        "total_images": total,
        "clean_source_acc": float(clean_correct / max(1, total)),
        "escape_rate": float(adv_escape / max(1, total)),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"Total images: {total}")
    print(f"Clean source accuracy: {summary['clean_source_acc']:.2%}")
    print(f"Escape rate: {summary['escape_rate']:.2%}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
