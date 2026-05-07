#!/usr/bin/env python3
"""
Abnormal-entry steering for class 461.

Loads abnormal entries from cls461_entry_inspect.json,
evaluates class-461 val/adv samples under three conditions:
  1. Baseline (no SAE)
  2. SAE + steer abnormal entries back to clean_461 mean  (Method 1)
  3. SAE + zero-out abnormal entries                       (Method 2)

Supports steering left-half, right-half, or all abnormal entries.

Usage:
    python sae_steering_abnormal.py --abnormal-mode left  --threshold 2.0
    python sae_steering_abnormal.py --abnormal-mode right --threshold 2.0
    python sae_steering_abnormal.py --abnormal-mode all   --threshold 2.0
"""

import argparse
import json
import sys
from pathlib import Path

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
TARGET_CLASS_IDX = 461


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
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


def build_model(device, ckpt_path):
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


def build_sae(device, sae_ckpt_path):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    return sae, norm_mean, norm_std


def make_steering_hook(sae_model, norm_mean, norm_std, steering_entries):
    """
    steering_entries: list of dicts with keys: token, feature, target_val
    """

    def hook_fn(module, input, output):
        bsz, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std

        z = sae_model.encode(flat_norm)
        z_spatial = z.reshape(bsz, h, w, -1)

        for entry in steering_entries:
            feat_idx = entry["feature"]
            tok_idx = entry["token"]
            target_val = entry["target_val"]
            row = tok_idx // w
            col = tok_idx % w
            z_spatial[:, row, col, feat_idx] = target_val

        z = z_spatial.reshape(bsz * h * w, -1)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return recon

    return hook_fn


@torch.no_grad()
def evaluate(model, loader, class_idx, device):
    correct = 0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        labels = torch.full(
            (images.size(0),), class_idx, dtype=torch.long, device=device
        )
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total if total > 0 else 0.0


def find_wnid_for_class_idx(adv_root, target_class_idx):
    for class_dir in Path(adv_root).iterdir():
        if not class_dir.is_dir() or class_dir.name == "sae_latent":
            continue
        labels_path = class_dir / "labels.txt"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            if int(parts[-1]) == target_class_idx:
                                return class_dir.name
                        except ValueError:
                            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Abnormal-entry steering for class 461 val/adv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(
            REPO_ROOT
            / "SAE"
            / "project"
            / "checkpoints"
            / "stage3"
            / "k256_exp8"
            / "sae_stage3_din1536_exp8_k256_step_50000.pt"
        ),
    )
    parser.add_argument(
        "--adv-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples"),
    )
    parser.add_argument(
        "--inspect-json",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "results"
            / "cls461_entry_inspect.json"
        ),
    )
    parser.add_argument(
        "--abnormal-mode",
        type=str,
        choices=["left", "right", "all"],
        default="left",
        help="left: delta>0 abnormal; right: delta<0 abnormal; all: both",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Abnormal threshold: adv > max(clean_461,clean_524) * threshold",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ================================================================
    # 1. Load abnormal entries from inspect JSON
    # ================================================================
    with open(args.inspect_json, "r", encoding="utf-8") as f:
        inspect_data = json.load(f)

    entries = inspect_data["entries"]

    abnormal_entries = []
    for e in entries:
        max_clean = max(e["mean_461c"], e["mean_524"])
        if max_clean <= 0:
            continue
        if e["mean_461a"] <= max_clean * args.threshold:
            continue

        is_left = e["delta"] > 0
        is_right = e["delta"] < 0

        if args.abnormal_mode == "left" and not is_left:
            continue
        if args.abnormal_mode == "right" and not is_right:
            continue
        if args.abnormal_mode == "all" and not (is_left or is_right):
            continue

        abnormal_entries.append(e)

    print(f"\nAbnormal mode: {args.abnormal_mode}, threshold: {args.threshold}x")
    print(f"Selected abnormal entries: {len(abnormal_entries)}")

    # Method 1: steer back to clean_461 mean
    method1_entries = [
        {"token": e["token"], "feature": e["feature"], "target_val": e["mean_461c"]}
        for e in abnormal_entries
    ]
    # Method 2: zero out
    method2_entries = [
        {"token": e["token"], "feature": e["feature"], "target_val": 0.0}
        for e in abnormal_entries
    ]

    # ================================================================
    # 2. Locate val/adv directory for class 461
    # ================================================================
    wnid_461 = find_wnid_for_class_idx(args.adv_root, TARGET_CLASS_IDX)
    if wnid_461 is None:
        raise ValueError(f"Could not find WNID for class {TARGET_CLASS_IDX}")
    print(f"Class {TARGET_CLASS_IDX} -> WNID: {wnid_461}")

    val_adv_dir = Path(args.adv_root) / wnid_461 / "val" / "adv"
    if not val_adv_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {val_adv_dir}")

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    dataset = ImageFolderDataset(str(val_adv_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Val adv samples: {len(dataset)}")

    # ================================================================
    # 3. Run three evaluations
    # ================================================================
    results = {}

    # --- 1. Baseline (no SAE) ---
    print("\n" + "=" * 60)
    print("[1/3] Baseline (no SAE)")
    print("=" * 60)
    model = build_model(device, args.base_ckpt)
    acc = evaluate(model, loader, TARGET_CLASS_IDX, device)
    results["baseline"] = acc
    print(f"Accuracy: {acc*100:.2f}%")
    del model
    torch.cuda.empty_cache()

    # --- 2. Method 1: steer to clean_461 mean ---
    print("\n" + "=" * 60)
    print("[2/3] Method 1: steer abnormal entries -> clean_461 mean")
    print("=" * 60)
    model = build_model(device, args.base_ckpt)
    sae, norm_mean, norm_std = build_sae(device, args.sae_ckpt)
    hook_fn = make_steering_hook(sae, norm_mean, norm_std, method1_entries)
    handle = model.stages[args.stage].register_forward_hook(hook_fn)
    acc = evaluate(model, loader, TARGET_CLASS_IDX, device)
    results["method1"] = acc
    print(f"Accuracy: {acc*100:.2f}%")
    handle.remove()
    del model, sae
    torch.cuda.empty_cache()

    # --- 3. Method 2: zero out ---
    print("\n" + "=" * 60)
    print("[3/3] Method 2: zero-out abnormal entries")
    print("=" * 60)
    model = build_model(device, args.base_ckpt)
    sae, norm_mean, norm_std = build_sae(device, args.sae_ckpt)
    hook_fn = make_steering_hook(sae, norm_mean, norm_std, method2_entries)
    handle = model.stages[args.stage].register_forward_hook(hook_fn)
    acc = evaluate(model, loader, TARGET_CLASS_IDX, device)
    results["method2"] = acc
    print(f"Accuracy: {acc*100:.2f}%")
    handle.remove()
    del model, sae
    torch.cuda.empty_cache()

    # ================================================================
    # 4. Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Abnormal entries steered: {len(abnormal_entries)}")
    print(f"  Mode: {args.abnormal_mode}, Threshold: {args.threshold}x")
    print()
    bl = results["baseline"] * 100
    m1 = results["method1"] * 100
    m2 = results["method2"] * 100
    print(f"  Baseline (no SAE):        {bl:>7.2f}%")
    print(f"  Method 1 (clean_mean):    {m1:>7.2f}%  (Δ={m1-bl:+.2f}pp)")
    print(f"  Method 2 (zero):          {m2:>7.2f}%  (Δ={m2-bl:+.2f}pp)")


if __name__ == "__main__":
    main()
