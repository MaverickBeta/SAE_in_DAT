#!/usr/bin/env python3
"""
Analyze adversarial attack statistics on train/adv samples.

For each of the 100 classes, load the model (DAT, no SAE) and evaluate
how its own train/adv images are misclassified.

Outputs:
  - Per-class error rate (sorted descending)
  - Most common misprediction targets for each class
  - Global attack success rate

Output: DAT/SAE/steering/results/ad_attack_stat.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        paths = sorted(Path(folder_path).glob("*"))
        self.image_paths = [
            str(p) for p in paths if p.is_file() and p.suffix in IMAGE_EXTS
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(path).name


def build_model(device: torch.device, ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, ckpt_path, weights_only=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def read_class_idx(class_dir: Path) -> int:
    labels_path = class_dir / "labels.txt"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            pass
    return 0


@torch.no_grad()
def evaluate_and_collect(model, dataloader, true_class_idx, device):
    """
    Returns:
        total: int
        correct: int
        predicted_classes: list of int (all predictions)
    """
    total = 0
    correct = 0
    predicted_classes = []

    for images, _ in dataloader:
        images = images.to(device)
        labels = torch.full(
            (images.size(0),), true_class_idx, dtype=torch.long, device=device
        )
        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += images.size(0)
        predicted_classes.extend(preds.cpu().tolist())

    return total, correct, predicted_classes


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial attack statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adv-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples"),
        help="Root directory containing {wnid}/train/adv",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--top-mispredictions",
        type=int,
        default=5,
        help="Number of top misprediction targets to record per class",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Scan classes
    adv_root = Path(args.adv_root)
    class_dirs = sorted(
        [d for d in adv_root.iterdir() if d.is_dir() and d.name != "sae_latent"]
    )
    print(f"Found {len(class_dirs)} classes")

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    model = build_model(device, args.base_ckpt)

    per_class_stats = []
    total_samples = 0
    total_correct = 0

    for class_dir in tqdm(class_dirs, desc="Evaluating"):
        wnid = class_dir.name
        train_adv_dir = class_dir / "train" / "adv"
        if not train_adv_dir.is_dir():
            continue

        class_idx = read_class_idx(class_dir)
        dataset = ImageFolderDataset(str(train_adv_dir), transform=transform)
        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        total, correct, preds = evaluate_and_collect(
            model, loader, class_idx, device
        )

        # Count mispredictions
        incorrect_preds = [p for p in preds if p != class_idx]
        pred_counter = Counter(incorrect_preds)
        top_mispredictions = [
            {"predicted_class": int(cls_id), "count": int(cnt)}
            for cls_id, cnt in pred_counter.most_common(args.top_mispredictions)
        ]

        error_rate = (total - correct) / total if total > 0 else 0.0

        per_class_stats.append(
            {
                "wnid": wnid,
                "class_idx": class_idx,
                "total_samples": total,
                "correct": correct,
                "incorrect": total - correct,
                "error_rate": float(error_rate),
                "top_mispredictions": top_mispredictions,
            }
        )

        total_samples += total
        total_correct += correct

    # Sort by error_rate descending (most vulnerable first)
    per_class_stats.sort(key=lambda x: x["error_rate"], reverse=True)

    overall_error_rate = (
        (total_samples - total_correct) / total_samples if total_samples > 0 else 0.0
    )

    result = {
        "config": {
            "base_ckpt": args.base_ckpt,
            "adv_root": str(adv_root),
            "total_classes_evaluated": len(per_class_stats),
        },
        "summary": {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "total_incorrect": total_samples - total_correct,
            "overall_error_rate": float(overall_error_rate),
            "overall_attack_success_rate": float(overall_error_rate),
        },
        "per_class": per_class_stats,
        "ranking": [
            {
                "rank": i + 1,
                "wnid": s["wnid"],
                "class_idx": s["class_idx"],
                "error_rate": s["error_rate"],
                "incorrect": s["incorrect"],
                "total": s["total_samples"],
            }
            for i, s in enumerate(per_class_stats)
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ad_attack_stat.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nOverall attack success rate: {overall_error_rate*100:.2f}%")
    print(f"Results saved to {output_path}")

    # Print top 20 most vulnerable classes
    print("\n" + "=" * 70)
    print("TOP 20 MOST VULNERABLE CLASSES (highest error rate)")
    print("=" * 70)
    print(f"{'Rank':>6} {'WNID':>12} {'Cls':>5} {'ErrRate':>10} {'Wrong':>6} {'Total':>6} {'Top Misprediction':>20}")
    print("-" * 70)
    for s in per_class_stats[:20]:
        top_pred = s["top_mispredictions"][0] if s["top_mispredictions"] else {"predicted_class": -1, "count": 0}
        print(
            f"{per_class_stats.index(s)+1:>6} {s['wnid']:>12} {s['class_idx']:>5} "
            f"{s['error_rate']*100:>9.1f}% {s['incorrect']:>6} {s['total_samples']:>6} "
            f"cls{top_pred['predicted_class']} ({top_pred['count']}x)"
        )

    # Print bottom 10 (most robust)
    print("\n" + "=" * 70)
    print("BOTTOM 10 MOST ROBUST CLASSES (lowest error rate)")
    print("=" * 70)
    for s in per_class_stats[-10:]:
        top_pred = s["top_mispredictions"][0] if s["top_mispredictions"] else {"predicted_class": -1, "count": 0}
        print(
            f"{per_class_stats.index(s)+1:>6} {s['wnid']:>12} {s['class_idx']:>5} "
            f"{s['error_rate']*100:>9.1f}% {s['incorrect']:>6} {s['total_samples']:>6} "
            f"cls{top_pred['predicted_class']} ({top_pred['count']}x)"
        )


if __name__ == "__main__":
    main()
