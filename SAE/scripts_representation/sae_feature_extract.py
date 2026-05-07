#!/usr/bin/env python3
"""
Extract raw SAE latent features for a batch of images.

Output shape: (n_images, n_tokens, d_lat)
- n_images: number of input images
- n_tokens: H*W spatial tokens at the hooked stage (e.g., 7*7=49 for stage3)
- d_lat: SAE latent dimension (e.g., 12288 for k256_exp8)

No aggregation is performed here. Aggregation (max/mean/p90 over tokens) is
left to downstream stat scripts.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None, max_images: int = 0):
        paths = sorted(Path(folder_path).glob("*"))
        self.image_paths = [str(p) for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if max_images > 0:
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(path).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw SAE latent features (n_images, n_tokens, d_lat).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing images to extract features from.",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth",
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        required=True,
        help="Path to SAE checkpoint.",
    )
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/features",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--max-images", type=int, default=0, help="0 = use all images.")
    return parser.parse_args()


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


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


def build_sae(device: torch.device, sae_ckpt_path: str):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = ckpt.get("k", config.get("k", None))
    if d_in is None:
        raise KeyError("Cannot resolve d_in")
    if k is None:
        raise KeyError("Cannot resolve k")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg.update({"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)})
    return sae, norm_mean, norm_std, resolved_cfg


def extract_raw_features(
    loader: DataLoader,
    model,
    sae,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Returns:
        features: (n_images, n_tokens, d_lat) float32 array
        stems: list of image stem names
        meta: dict with shape info
    """
    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    all_features: List[np.ndarray] = []
    all_stems: List[str] = []
    d_lat = None
    n_tokens = None

    with torch.no_grad():
        for images, stems in tqdm(loader, desc="Extract", leave=False):
            images = images.to(device)
            _ = model(images)

            stage_out = captured["stage"]  # (bsz, c, h, w)
            bsz, c, h, w = stage_out.shape
            flat = stage_out.permute(0, 2, 3, 1).reshape(-1, c)
            flat_norm = (flat - norm_mean) / norm_std
            z = sae.encode(flat_norm)  # (bsz*h*w, d_lat)

            if d_lat is None:
                d_lat = int(z.shape[1])
                n_tokens = h * w

            # Split back to per-image and move to CPU
            z_cpu = z.detach().cpu().numpy().astype(np.float32)
            z_per_image = z_cpu.reshape(bsz, n_tokens, d_lat)

            all_features.append(z_per_image)
            all_stems.extend(list(stems))

    handle.remove()

    features = np.concatenate(all_features, axis=0)  # (n_images, n_tokens, d_lat)
    meta = {
        "shape": features.shape,
        "n_images": int(features.shape[0]),
        "n_tokens": int(features.shape[1]),
        "d_lat": int(features.shape[2]),
        "stage": stage_idx,
    }
    return features, all_stems, meta


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Output setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        out_prefix = args.run_name
    else:
        out_prefix = f"stage{args.stage}_{Path(args.sae_ckpt).stem}"

    out_npy = out_dir / f"{out_prefix}_features.npy"
    out_meta = out_dir / f"{out_prefix}_meta.json"

    # Model + SAE
    model = build_model(device, args.base_ckpt)
    sae, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

    # Verify stage
    expected_din = {0: 192, 1: 384, 2: 768, 3: 1536}[args.stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.stage} expects d_in={expected_din}, "
            f"SAE has d_in={sae_cfg['d_in']}"
        )
    print(f"SAE config: d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")

    # Data
    dataset = ImageFolderDataset(str(input_dir), transform=get_transform(), max_images=args.max_images)
    print(f"Images found: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("No images found")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Extract
    features, stems, meta = extract_raw_features(
        loader, model, sae, norm_mean, norm_std, args.stage, device
    )

    # Save
    np.save(out_npy, features)
    meta.update(
        {
            "input_dir": str(input_dir),
            "base_ckpt": args.base_ckpt,
            "sae_ckpt": args.sae_ckpt,
            "sae_config": sae_cfg,
            "image_stems": stems,
            "feature_npy": str(out_npy),
        }
    )
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved features: {out_npy}  shape={features.shape}")
    print(f"Saved meta:     {out_meta}")


if __name__ == "__main__":
    main()
