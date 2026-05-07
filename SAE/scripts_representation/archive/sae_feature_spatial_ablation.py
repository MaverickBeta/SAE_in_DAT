"""
Spatial-selective SAE feature ablation on adversarial samples.

Unlike channel-level global ablation (all tokens), this script supports
ablation on specific spatial positions (7x7 grid) to test whether
adversarial effects are localized.

Key masks:
  - global:     all 49 tokens
  - boundary:   outer ring (24 tokens)
  - center:     inner 5x5 (25 tokens), negative control
  - topdiff_k:  per-image top-k tokens by adv-clean diff
  - random_k:   random k tokens per image

Example (11318 on adv samples with boundary mask):
    python sae_feature_spatial_ablation.py \
        --sae-ckpt /path/to/stage3_k256.pt \
        --stage 3 --source-cls 150 --source-wnid n02077923 \
        --image-dir /path/to/adv_samples/succ \
        --ablate-features 11318 \
        --spatial-mode boundary \
        --run-name "adv_11318_boundary"
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


def get_spatial_masks():
    """Return named 7x7 boolean masks. True = ablate this position."""
    masks = {}

    # Global: all positions
    masks["global"] = np.ones((7, 7), dtype=bool)

    # Boundary: outer ring
    boundary = np.zeros((7, 7), dtype=bool)
    boundary[0, :] = True
    boundary[6, :] = True
    boundary[:, 0] = True
    boundary[:, 6] = True
    masks["boundary"] = boundary

    # Center: inner 5x5
    center = np.zeros((7, 7), dtype=bool)
    center[1:6, 1:6] = True
    masks["center"] = center

    # Corner: four corners only
    corner = np.zeros((7, 7), dtype=bool)
    corner[0, 0] = True
    corner[0, 6] = True
    corner[6, 0] = True
    corner[6, 6] = True
    masks["corner"] = corner

    # Edge-non-corner: boundary minus corners (20 tokens)
    edge_nc = boundary.copy()
    edge_nc[0, 0] = False
    edge_nc[0, 6] = False
    edge_nc[6, 0] = False
    edge_nc[6, 6] = False
    masks["edge_noncorner"] = edge_nc

    return masks


def compute_topdiff_mask(diff_7x7: np.ndarray, k: int) -> np.ndarray:
    """
    diff_7x7: (7, 7) array of adv - clean feature values.
    Returns boolean mask with top-k absolute diff positions set to True.
    """
    flat_idx = np.argsort(np.abs(diff_7x7).ravel())[::-1][:k]
    mask = np.zeros((7, 7), dtype=bool)
    mask.ravel()[flat_idx] = True
    return mask


def compute_random_mask(rng: np.random.Generator, k: int) -> np.ndarray:
    """Random k positions out of 49."""
    mask = np.zeros((7, 7), dtype=bool)
    flat_idx = rng.choice(49, size=k, replace=False)
    mask.ravel()[flat_idx] = True
    return mask


class FileListDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None, cache_images: bool = False):
        self.image_paths = image_paths
        self.transform = transform
        self.cache_images = cache_images
        self.cached = None

        if self.cache_images:
            self.cached = []
            for path in self.image_paths:
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                self.cached.append((image, os.path.basename(path)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cached is not None:
            return self.cached[idx]
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatial-selective SAE feature ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-ckpt", type=str,
                        default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
    parser.add_argument("--sae-ckpt", type=str, required=True)
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--source-cls", type=int, default=150)
    parser.add_argument("--source-wnid", type=str, default="n02077923")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images to evaluate (clean or adv).")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ablate-features", type=int, nargs="+", required=True,
                        help="List of feature channel indices to ablate (e.g., 11318 1929).")
    parser.add_argument("--spatial-mode", type=str, default="global",
                        choices=["global", "boundary", "center", "corner",
                                 "edge_noncorner", "topdiff", "random"],
                        help="Spatial mask mode.")
    parser.add_argument("--topdiff-k", type=int, default=5,
                        help="For topdiff mode: ablate top-k tokens by abs(diff).")
    parser.add_argument("--random-k", type=int, default=5,
                        help="For random mode: ablate k random tokens.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--recon-mode", type=str, choices=["raw", "normalized"],
                        default="raw")
    parser.add_argument("--cache-images", action="store_true", default=False)
    parser.add_argument("--results-dir", type=str,
                        default="/Data_share/hongyi/DAT/SAE/results_representation/spatial_ablation")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--clean-features-npy", type=str, default="",
                        help="Path to clean features .npy for computing topdiff mask.")
    parser.add_argument("--adv-features-npy", type=str, default="",
                        help="Path to adv features .npy for computing topdiff mask.")
    return parser.parse_args()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def list_images(folder: str) -> List[str]:
    all_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)])
    return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def build_model(device: torch.device, base_ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, base_ckpt_path)
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
        raise KeyError("Cannot resolve d_in from checkpoint")
    if k is None:
        raise KeyError("Cannot resolve k from checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from checkpoint")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg["d_in"] = int(d_in)
    resolved_cfg["d_lat"] = int(d_lat)
    resolved_cfg["k"] = int(k)
    return sae, norm_mean, norm_std, resolved_cfg


def load_precomputed_features(clean_npy: str, adv_npy: str, max_images: int
                              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load (n_images, 49, d_lat) feature arrays."""
    clean_feat = None
    adv_feat = None
    if clean_npy and os.path.exists(clean_npy):
        clean_feat = np.load(clean_npy)
        if max_images > 0:
            clean_feat = clean_feat[:max_images]
        print(f"Loaded clean features: {clean_feat.shape} from {clean_npy}")
    if adv_npy and os.path.exists(adv_npy):
        adv_feat = np.load(adv_npy)
        if max_images > 0:
            adv_feat = adv_feat[:max_images]
        print(f"Loaded adv features: {adv_feat.shape} from {adv_npy}")
    return clean_feat, adv_feat


def evaluate_spatial_ablation(
    loader: DataLoader,
    model,
    source_cls: int,
    device: torch.device,
    stage_idx: int,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    recon_mode: str,
    ablate_channels: List[int],
    spatial_mask_2d: Optional[np.ndarray] = None,
    per_image_masks: Optional[np.ndarray] = None,  # (N, 7, 7) bool
) -> Dict:
    """
    spatial_mask_2d: (7, 7) bool, same mask for all images in batch.
    per_image_masks: (N, 7, 7) bool, individual mask per image.
    Only one should be provided.
    """
    handle = None
    ablate_tensor = torch.tensor(ablate_channels, dtype=torch.long, device=device)

    # Precompute fixed mask tensor if using global spatial mask
    fixed_mask_tensor = None
    if spatial_mask_2d is not None:
        fixed_mask_tensor = torch.from_numpy(spatial_mask_2d).to(device)  # (7, 7)

    def hook_fn(_, __, output):
        bsz, channels, height, width = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat = (flat - norm_mean) / norm_std
        z = sae_model.encode(flat)  # (bsz*49, d_lat)

        # Reshape to spatial for selective ablation
        d_lat = z.shape[1]
        z_spatial = z.reshape(bsz, 7, 7, d_lat)

        if per_image_masks is not None:
            # per_image_masks should be aligned with current batch indices
            # This is handled by precomputing all masks and slicing in the loop
            pass  # Will be handled differently below
        elif fixed_mask_tensor is not None:
            for ch in ablate_channels:
                z_spatial[:, :, :, ch] = z_spatial[:, :, :, ch] * (~fixed_mask_tensor)

        z = z_spatial.reshape(bsz * 49, d_lat)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
        if recon_mode == "raw":
            recon_flat = recon_norm * norm_std + norm_mean
        else:
            recon_flat = recon_norm
        recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
        return recon

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    total = 0
    source_correct = 0
    source_escape = 0
    source_logit_sum = 0.0
    source_prob_sum = 0.0
    source_margin_sum = 0.0
    all_preds = []
    all_margins = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            source_logits = logits[:, source_cls]
            other_logits = logits.clone()
            other_logits[:, source_cls] = float("-inf")
            best_other_logits = other_logits.max(dim=1).values
            margins = source_logits - best_other_logits

            source_correct += int((preds == source_cls).sum().item())
            source_escape += int((preds != source_cls).sum().item())
            source_logit_sum += float(source_logits.sum().item())
            source_prob_sum += float(probs[:, source_cls].sum().item())
            source_margin_sum += float(margins.sum().item())
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_margins.extend(margins.cpu().numpy().tolist())

    if handle is not None:
        handle.remove()

    if total == 0:
        raise RuntimeError("No images evaluated")

    return {
        "num_images": int(total),
        "source_top1_acc": float(source_correct / total),
        "escape_rate": float(source_escape / total),
        "avg_source_logit": float(source_logit_sum / total),
        "avg_source_prob": float(source_prob_sum / total),
        "avg_source_margin": float(source_margin_sum / total),
        "predictions": all_preds,
        "margins": all_margins,
    }


def evaluate_with_per_image_masks(
    image_paths: List[str],
    model,
    source_cls: int,
    device: torch.device,
    stage_idx: int,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    recon_mode: str,
    ablate_channels: List[int],
    per_image_masks: np.ndarray,  # (N, 7, 7) bool
    batch_size: int,
) -> Dict:
    """
    Evaluate with per-image spatial masks. Processes one image at a time
    to ensure mask alignment (simpler than batch alignment).
    """
    transform = get_transform()
    ablate_tensor = torch.tensor(ablate_channels, dtype=torch.long, device=device)

    total = 0
    source_correct = 0
    source_escape = 0
    source_logit_sum = 0.0
    source_prob_sum = 0.0
    source_margin_sum = 0.0
    all_preds = []
    all_margins = []

    for idx, path in enumerate(tqdm(image_paths, desc="Eval per-image mask")):
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        mask_2d = torch.from_numpy(per_image_masks[idx]).to(device)  # (7, 7)

        def hook_fn(_, __, output):
            bsz, channels, height, width = output.shape
            flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)
            d_lat = z.shape[1]
            z_spatial = z.reshape(bsz, 7, 7, d_lat)
            for ch in ablate_channels:
                z_spatial[:, :, :, ch] = z_spatial[:, :, :, ch] * (~mask_2d)
            z = z_spatial.reshape(bsz * 49, d_lat)
            recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
            if recon_mode == "raw":
                recon_flat = recon_norm * norm_std + norm_mean
            else:
                recon_flat = recon_norm
            recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
            return recon

        handle = model.stages[stage_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            source_logits = logits[:, source_cls]
            other_logits = logits.clone()
            other_logits[:, source_cls] = float("-inf")
            best_other_logits = other_logits.max(dim=1).values
            margins = source_logits - best_other_logits

            source_correct += int((preds == source_cls).sum().item())
            source_escape += int((preds != source_cls).sum().item())
            source_logit_sum += float(source_logits.sum().item())
            source_prob_sum += float(probs[:, source_cls].sum().item())
            source_margin_sum += float(margins.sum().item())
            total += 1

            all_preds.append(int(preds.item()))
            all_margins.append(float(margins.item()))

        handle.remove()

    return {
        "num_images": int(total),
        "source_top1_acc": float(source_correct / total),
        "escape_rate": float(source_escape / total),
        "avg_source_logit": float(source_logit_sum / total),
        "avg_source_prob": float(source_prob_sum / total),
        "avg_source_margin": float(source_margin_sum / total),
        "predictions": all_preds,
        "margins": all_margins,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    image_paths = list_images(args.image_dir)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)} from {args.image_dir}")
    print(f"Ablate channels: {args.ablate_features}")
    print(f"Spatial mode: {args.spatial_mode}")

    model = build_model(device, args.base_ckpt)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

    # Validate stage alignment
    if STAGE_DIN_MAP[args.stage] != int(sae_cfg["d_in"]):
        raise ValueError(
            f"Stage mismatch: stage{args.stage} expects d_in={STAGE_DIN_MAP[args.stage]}, "
            f"SAE has d_in={sae_cfg['d_in']}"
        )

    # Build output directory
    feat_tag = "_".join(str(f) for f in args.ablate_features)
    k_tag = f"k{sae_cfg['k']}"
    leaf = args.run_name if args.run_name else (
        f"{args.source_wnid}_stage{args.stage}_{k_tag}_"
        f"f{feat_tag}_spatial_{args.spatial_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = os.path.join(args.results_dir, f"stage{args.stage}", k_tag, leaf)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load precomputed features for topdiff mask
    clean_feat, adv_feat = load_precomputed_features(
        args.clean_features_npy, args.adv_features_npy, args.max_images
    )

    # Baseline: original model (no SAE)
    dataset = FileListDataset(image_paths, transform=get_transform(), cache_images=args.cache_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    def eval_baseline_original():
        total, correct, escape = 0, 0, 0
        logit_sum, prob_sum, margin_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                source_logits = logits[:, args.source_cls]
                other = logits.clone()
                other[:, args.source_cls] = float("-inf")
                margins = source_logits - other.max(dim=1).values
                correct += int((preds == args.source_cls).sum().item())
                escape += int((preds != args.source_cls).sum().item())
                logit_sum += float(source_logits.sum().item())
                prob_sum += float(probs[:, args.source_cls].sum().item())
                margin_sum += float(margins.sum().item())
                total += images.size(0)
        return {
            "num_images": total,
            "source_top1_acc": correct / total,
            "escape_rate": escape / total,
            "avg_source_logit": logit_sum / total,
            "avg_source_prob": prob_sum / total,
            "avg_source_margin": margin_sum / total,
        }

    baseline_original = eval_baseline_original()
    print(f"\nBaseline (original): acc={baseline_original['source_top1_acc']:.3f}, "
          f"escape={baseline_original['escape_rate']:.3f}, "
          f"margin={baseline_original['avg_source_margin']:.3f}")

    # Baseline: SAE recon without ablation
    # Need a simple function for this
    def eval_sae_no_ablate():
        handle = None
        def hook_fn(_, __, output):
            bsz, channels, height, width = output.shape
            flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)
            recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
            if args.recon_mode == "raw":
                recon_flat = recon_norm * norm_std + norm_mean
            else:
                recon_flat = recon_norm
            recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
            return recon
        handle = model.stages[args.stage].register_forward_hook(hook_fn)
        total, correct, escape = 0, 0, 0
        logit_sum, prob_sum, margin_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                source_logits = logits[:, args.source_cls]
                other = logits.clone()
                other[:, args.source_cls] = float("-inf")
                margins = source_logits - other.max(dim=1).values
                correct += int((preds == args.source_cls).sum().item())
                escape += int((preds != args.source_cls).sum().item())
                logit_sum += float(source_logits.sum().item())
                prob_sum += float(probs[:, args.source_cls].sum().item())
                margin_sum += float(margins.sum().item())
                total += images.size(0)
        handle.remove()
        return {
            "num_images": total,
            "source_top1_acc": correct / total,
            "escape_rate": escape / total,
            "avg_source_logit": logit_sum / total,
            "avg_source_prob": prob_sum / total,
            "avg_source_margin": margin_sum / total,
        }

    baseline_sae = eval_sae_no_ablate()
    print(f"Baseline (SAE recon no ablation): acc={baseline_sae['source_top1_acc']:.3f}, "
          f"escape={baseline_sae['escape_rate']:.3f}, "
          f"margin={baseline_sae['avg_source_margin']:.3f}")

    # Run spatial ablation experiments
    rows = []
    base_keys = ["num_images", "source_top1_acc", "escape_rate", "avg_source_logit", "avg_source_prob", "avg_source_margin"]
    rows.append({"condition": "original_passthrough", "n_ablated": 0,
                 "spatial_mode": "none", "n_tokens_ablated": 0,
                 **{k: baseline_original[k] for k in base_keys}})
    rows.append({"condition": "sae_recon_no_ablation", "n_ablated": 0,
                 "spatial_mode": "none", "n_tokens_ablated": 0,
                 **{k: baseline_sae[k] for k in base_keys}})

    spatial_masks = get_spatial_masks()

    # Case 1: Fixed spatial mask (global, boundary, center, corner, edge_noncorner)
    if args.spatial_mode in spatial_masks:
        mask_2d = spatial_masks[args.spatial_mode]
        n_tokens = int(mask_2d.sum())
        print(f"\nRunning spatial ablation: {args.spatial_mode} ({n_tokens} tokens)")

        metrics = evaluate_spatial_ablation(
            loader=loader,
            model=model,
            source_cls=args.source_cls,
            device=device,
            stage_idx=args.stage,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            recon_mode=args.recon_mode,
            ablate_channels=args.ablate_features,
            spatial_mask_2d=mask_2d,
            per_image_masks=None,
        )
        rows.append({
            "condition": f"ablate_{args.spatial_mode}",
            "n_ablated": len(args.ablate_features),
            "spatial_mode": args.spatial_mode,
            "n_tokens_ablated": n_tokens,
            "source_top1_acc": metrics["source_top1_acc"],
            "escape_rate": metrics["escape_rate"],
            "avg_source_logit": metrics["avg_source_logit"],
            "avg_source_prob": metrics["avg_source_prob"],
            "avg_source_margin": metrics["avg_source_margin"],
        })
        print(f"  Result: acc={metrics['source_top1_acc']:.3f}, "
              f"escape={metrics['escape_rate']:.3f}, "
              f"margin={metrics['avg_source_margin']:.3f}")

    # Case 2: topdiff mode
    elif args.spatial_mode == "topdiff":
        if clean_feat is None or adv_feat is None:
            raise ValueError("--clean-features-npy and --adv-features-npy required for topdiff mode")
        if len(clean_feat) < len(image_paths) or len(adv_feat) < len(image_paths):
            raise ValueError(f"Feature arrays too small: clean={clean_feat.shape}, adv={adv_feat.shape}")

        k = args.topdiff_k
        print(f"\nRunning topdiff ablation: top-{k} tokens by abs(diff) per image")

        # Compute per-image masks
        n_imgs = len(image_paths)
        per_image_masks = np.zeros((n_imgs, 7, 7), dtype=bool)
        for i in range(n_imgs):
            diff = adv_feat[i].mean(axis=1) - clean_feat[i].mean(axis=1)  # (49,)
            diff_7x7 = diff.reshape(7, 7)
            # For specific channels, compute diff on those channels
            ch_diff = adv_feat[i][:, args.ablate_features].mean(axis=1) - \
                      clean_feat[i][:, args.ablate_features].mean(axis=1)
            diff_7x7 = ch_diff.reshape(7, 7)
            per_image_masks[i] = compute_topdiff_mask(diff_7x7, k)

        metrics = evaluate_with_per_image_masks(
            image_paths=image_paths,
            model=model,
            source_cls=args.source_cls,
            device=device,
            stage_idx=args.stage,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            recon_mode=args.recon_mode,
            ablate_channels=args.ablate_features,
            per_image_masks=per_image_masks,
            batch_size=1,
        )
        rows.append({
            "condition": f"ablate_topdiff_{k}",
            "n_ablated": len(args.ablate_features),
            "spatial_mode": f"topdiff_{k}",
            "n_tokens_ablated": k,
            "source_top1_acc": metrics["source_top1_acc"],
            "escape_rate": metrics["escape_rate"],
            "avg_source_logit": metrics["avg_source_logit"],
            "avg_source_prob": metrics["avg_source_prob"],
            "avg_source_margin": metrics["avg_source_margin"],
        })
        print(f"  Result: acc={metrics['source_top1_acc']:.3f}, "
              f"escape={metrics['escape_rate']:.3f}, "
              f"margin={metrics['avg_source_margin']:.3f}")

    # Case 3: random mode
    elif args.spatial_mode == "random":
        k = args.random_k
        print(f"\nRunning random ablation: {k} random tokens per image")

        n_imgs = len(image_paths)
        per_image_masks = np.zeros((n_imgs, 7, 7), dtype=bool)
        for i in range(n_imgs):
            per_image_masks[i] = compute_random_mask(rng, k)

        metrics = evaluate_with_per_image_masks(
            image_paths=image_paths,
            model=model,
            source_cls=args.source_cls,
            device=device,
            stage_idx=args.stage,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            recon_mode=args.recon_mode,
            ablate_channels=args.ablate_features,
            per_image_masks=per_image_masks,
            batch_size=1,
        )
        rows.append({
            "condition": f"ablate_random_{k}",
            "n_ablated": len(args.ablate_features),
            "spatial_mode": f"random_{k}",
            "n_tokens_ablated": k,
            "source_top1_acc": metrics["source_top1_acc"],
            "escape_rate": metrics["escape_rate"],
            "avg_source_logit": metrics["avg_source_logit"],
            "avg_source_prob": metrics["avg_source_prob"],
            "avg_source_margin": metrics["avg_source_margin"],
        })
        print(f"  Result: acc={metrics['source_top1_acc']:.3f}, "
              f"escape={metrics['escape_rate']:.3f}, "
              f"margin={metrics['avg_source_margin']:.3f}")

    # Save CSV
    csv_path = os.path.join(out_dir, "ablation_results.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Save summary JSON
    summary = {
        "config": {
            "base_ckpt": args.base_ckpt,
            "sae_ckpt": args.sae_ckpt,
            "stage": args.stage,
            "source_cls": args.source_cls,
            "source_wnid": args.source_wnid,
            "image_dir": args.image_dir,
            "num_images": len(image_paths),
            "ablate_features": args.ablate_features,
            "spatial_mode": args.spatial_mode,
            "topdiff_k": args.topdiff_k,
            "random_k": args.random_k,
            "recon_mode": args.recon_mode,
            "sae_config": {k: int(v) if isinstance(v, (int, np.integer)) else v
                          for k, v in sae_cfg.items()},
        },
        "baselines": {
            "original_passthrough": baseline_original,
            "sae_recon_no_ablation": baseline_sae,
        },
        "results": rows[2:] if len(rows) > 2 else [],
    }
    json_path = os.path.join(out_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
