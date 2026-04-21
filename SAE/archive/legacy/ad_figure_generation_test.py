#!/usr/bin/env python3
"""
Quick targetability probe for attacks used in ad_figure_generation.py.

What it checks:
1) Whether each attack object accepts targeted mode via set_mode_targeted_by_function.
2) Whether a tiny targeted run executes and how often predictions hit the target class.

This is a capability smoke test, not a full robustness benchmark.
"""

import os
import sys
import json
import glob
import traceback
from typing import Dict, Any, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

try:
    import torchattacks
except ImportError:
    print("Please install torchattacks first: pip install torchattacks")
    sys.exit(1)

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint

# Keep the same classes as ad_figure_generation.py
TARGET_CLS = 62
SOURCE_CLS = 150
SOURCE_WNID = "n02077923"

IMAGENET_VAL_DIR = "/Data_share/hongyi/DAT/data/ImageNet/val"
CHECKPOINT_PATH = "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth"


class SimpleImageDataset(Dataset):
    def __init__(self, folder_path: str, max_images: int = 16):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG")
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        self.image_paths = sorted(image_paths)[:max_images]
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), os.path.basename(path)


def build_model(device: torch.device) -> torch.nn.Module:
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    return model


def build_attacks(model: torch.nn.Module) -> Dict[str, Any]:
    # Mild params for quick probing only.
    return {
        "pgd": torchattacks.PGD(model, eps=12 / 255, alpha=2 / 255, steps=20),
        "cw": torchattacks.CW(model, c=1.0, kappa=0, steps=50),
        "apgd_ce": torchattacks.APGD(model, eps=24 / 255, steps=50, loss="ce"),
        "apgd_t": torchattacks.APGDT(model, eps=24 / 255, steps=50),
        "fab_t": torchattacks.FAB(model, eps=16 / 255, steps=50),
        "square": torchattacks.Square(model, eps=24 / 255, n_queries=1000),
    }


def try_enable_targeted(atk: Any, device: torch.device) -> (bool, str):
    try:
        atk.set_mode_targeted_by_function(
            target_map_function=lambda images, labels: torch.full(
                (images.size(0),), TARGET_CLS, dtype=torch.long, device=device
            )
        )
        return True, "targeted API accepted"
    except Exception as e:
        return False, f"targeted API rejected: {type(e).__name__}: {e}"


def evaluate_preds(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    total = max(1, x.size(0))
    target_hits = (preds == TARGET_CLS).sum().item()
    escaped = (preds != SOURCE_CLS).sum().item()
    return {
        "target_hit_rate": target_hits / total,
        "escape_rate": escaped / total,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source_dir = os.path.join(IMAGENET_VAL_DIR, SOURCE_WNID)
    ds = SimpleImageDataset(source_dir, max_images=16)
    if len(ds) == 0:
        print(f"No images found in {source_dir}")
        return

    loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=False, num_workers=0)
    images, _ = next(iter(loader))
    images = images.to(device)

    model = build_model(device)

    clean_stats = evaluate_preds(model, images)
    print(
        "Clean probe => "
        f"target_hit_rate={clean_stats['target_hit_rate']:.3f}, "
        f"escape_rate={clean_stats['escape_rate']:.3f}"
    )

    results = []
    attacks = build_attacks(model)

    for name, atk in attacks.items():
        row = {
            "attack": name,
            "targeted_api_supported": False,
            "run_ok": False,
            "target_hit_rate": None,
            "escape_rate": None,
            "note": "",
        }

        supported, note = try_enable_targeted(atk, device)
        row["targeted_api_supported"] = supported
        row["note"] = note

        try:
            y_src = torch.full((images.size(0),), SOURCE_CLS, dtype=torch.long, device=device)
            x_adv = atk(images, y_src)
            adv_stats = evaluate_preds(model, x_adv)
            row["run_ok"] = True
            row["target_hit_rate"] = adv_stats["target_hit_rate"]
            row["escape_rate"] = adv_stats["escape_rate"]
        except Exception as e:
            row["note"] += f" | run failed: {type(e).__name__}: {e}"

        print(
            f"[{name:8s}] targeted_supported={row['targeted_api_supported']} "
            f"run_ok={row['run_ok']} "
            f"target_hit_rate={row['target_hit_rate']} escape_rate={row['escape_rate']}"
        )
        results.append(row)

    out = {
        "device": str(device),
        "num_probe_images": int(images.size(0)),
        "source_class": SOURCE_CLS,
        "target_class": TARGET_CLS,
        "clean_probe": clean_stats,
        "results": results,
    }

    out_dir = "/Data_share/hongyi/DAT/SAE/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "attack_targetability_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved report:", out_path)
    print("\nInterpretation tips:")
    print("- targeted_api_supported=True means the attack object accepts targeted mode setup.")
    print("- target_hit_rate>0 on this tiny probe suggests targeted behavior is working in practice.")
    print("- escape_rate can be high even when true targeted hits are low.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
