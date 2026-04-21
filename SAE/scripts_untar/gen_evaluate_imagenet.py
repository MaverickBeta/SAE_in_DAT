#!/usr/bin/env python3
"""
ImageNet robust evaluation + adversarial sample generation.

This script keeps the same core protocol as evaluate_imagenet_robustbench.py:
- same model construction / checkpoint loading
- same preprocessing (Resize(floor(img_size/0.875)) -> CenterCrop -> ToTensor)
- same threat model + eps semantics

It additionally saves generated adversarial examples and metadata for downstream use.
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Resolve DAT repository root from current script location: DAT/SAE/scripts_untar/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Use local forks/modules from the repository.
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

import numpy as np
import robustbench
import torch
from autoattack import AutoAttack
from PIL import Image
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import clean_accuracy
from timm.models import create_model
from timm.models.resnet import Bottleneck, _create_resnet
from torchvision import transforms

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


def get_preprocessing_function(img_size: int = 224, crop_pct: float = 0.875):
    """Create preprocessing function for arbitrary image size."""
    scale_size = int(math.floor(img_size / crop_pct))

    def preprocess(x):
        return transforms.Compose(
            [
                transforms.Resize(
                    scale_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )(x)

    return preprocess


def load_custom_model(checkpoint_path: str, architecture: str = "resnet50"):
    """Load custom model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if architecture == "resnet50":
        model = create_model("resnet50", pretrained=False, num_classes=1000)
    elif architecture == "wide_resnet50_4":
        model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), base_width=256)
        model = _create_resnet(
            "wide_resnet50_4", pretrained=False, num_classes=1000, **model_args
        )
    elif architecture == "convnext_large":
        model = create_convnext_model(
            model_type="convnext_large",
            num_classes=1000,
            normalize_input=False,
            use_layernorm=True,
            use_convstem=True,
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported yet")

    return load_checkpoint(model, checkpoint_path, weights_only=False)


def build_sae(device: torch.device, sae_ckpt_path: str):
    """Load SAE checkpoint and resolve config fields robustly."""
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))

    if d_in is None:
        raise KeyError("Cannot resolve d_in from SAE checkpoint")
    if k is None:
        raise KeyError("Cannot resolve k from SAE checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from SAE checkpoint")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    # SAE is used as a fixed transform during evaluation/attack generation.
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg["d_in"] = int(d_in)
    resolved_cfg["d_lat"] = int(d_lat)
    resolved_cfg["k"] = int(k)
    return sae, norm_mean, norm_std, resolved_cfg


def resolve_sae_stage(sae_cfg: dict, requested_stage: Optional[int]) -> int:
    """Resolve stage from explicit argument or SAE d_in."""
    if requested_stage is not None:
        if requested_stage not in STAGE_DIN_MAP:
            raise ValueError(f"Invalid sae_stage={requested_stage}, expected 0..3")
        expected_din = STAGE_DIN_MAP[requested_stage]
        if int(sae_cfg["d_in"]) != expected_din:
            raise ValueError(
                f"Stage/SAE mismatch: stage{requested_stage} expects d_in={expected_din}, "
                f"but SAE has d_in={sae_cfg['d_in']}"
            )
        return requested_stage

    inferred = DIN_STAGE_MAP.get(int(sae_cfg["d_in"]), None)
    if inferred is None:
        raise ValueError(
            f"Cannot infer stage from SAE d_in={sae_cfg['d_in']}. "
            f"Please pass --sae_stage explicitly."
        )
    return inferred


def get_stage_host_model(model):
    """Return the model object that owns .stages (unwrap DataParallel if needed)."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def attach_sae_hook(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    recon_mode: str,
):
    """Attach SAE reconstruction hook to one ConvNeXt stage."""
    stage_host = get_stage_host_model(model)

    def hook_fn(_, __, output):
        bsz, channels, height, width = output.shape
        flat_raw = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat_raw - norm_mean) / norm_std

        z = sae_model.encode(flat_norm)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec

        if recon_mode == "raw":
            recon_flat = recon_norm * norm_std + norm_mean
        else:
            recon_flat = recon_norm

        return recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)

    return stage_host.stages[stage_idx].register_forward_hook(hook_fn)


def parse_subset_indices(
    subset_indices_file: Optional[str], n_loaded: int
) -> Optional[torch.Tensor]:
    """Load subset indices from a text/json file and validate bounds."""
    if subset_indices_file is None:
        return None

    path = Path(subset_indices_file)
    if not path.exists():
        raise FileNotFoundError(f"subset_indices_file not found: {subset_indices_file}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON subset file must be a list of integer indices")
        idx = [int(x) for x in data]
    else:
        idx = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx.append(int(line))

    if len(idx) == 0:
        raise ValueError("subset_indices_file is empty")

    idx_tensor = torch.tensor(idx, dtype=torch.long)
    if (idx_tensor < 0).any() or (idx_tensor >= n_loaded).any():
        raise ValueError(
            f"subset index out of range. valid: [0, {n_loaded - 1}], "
            f"received min={int(idx_tensor.min())}, max={int(idx_tensor.max())}"
        )

    return idx_tensor


def get_imagenet_class_index_from_val(data_dir: str, source_wnid: str) -> int:
    """Resolve ImageNet class index based on sorted val class folders (ImageFolder convention)."""
    val_dir = Path(data_dir) / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")

    class_dirs = sorted(
        [p.name for p in val_dir.iterdir() if p.is_dir()]
    )
    if source_wnid not in class_dirs:
        raise ValueError(
            f"source_wnid '{source_wnid}' not found under {val_dir}"
        )

    return class_dirs.index(source_wnid)


def load_imagenet_val_class(
    data_dir: str,
    source_wnid: str,
    preprocess,
    n_examples: int,
):
    """Load samples from a single ImageNet val class folder with the shared preprocessing."""
    class_dir = Path(data_dir) / "val" / source_wnid
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    image_paths = sorted(glob.glob(str(class_dir / "*")))
    image_paths = [
        p
        for p in image_paths
        if os.path.isfile(p)
        and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in class directory: {class_dir}")

    if n_examples > 0:
        image_paths = image_paths[:n_examples]

    class_idx = get_imagenet_class_index_from_val(data_dir, source_wnid)
    x_list = []
    for p in image_paths:
        with Image.open(p).convert("RGB") as img:
            x_list.append(preprocess(img))

    x = torch.stack(x_list, dim=0)
    y = torch.full((x.shape[0],), class_idx, dtype=torch.long)
    sample_ids = torch.arange(x.shape[0], dtype=torch.long)
    return x, y, sample_ids


def save_adv_outputs(
    save_dir: Path,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    y_true: torch.Tensor,
    y_pred_clean: torch.Tensor,
    y_pred_adv: torch.Tensor,
    sample_ids: torch.Tensor,
    shard_size: int = 256,
    save_format: str = "pt",
):
    """Save adversarial outputs in shards for easier downstream processing."""
    save_dir.mkdir(parents=True, exist_ok=True)
    n = x_adv.shape[0]

    manifest = {
        "num_samples": int(n),
        "save_format": save_format,
        "shard_size": int(shard_size),
        "num_shards": int(math.ceil(n / shard_size)),
    }

    with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        shard_id = start // shard_size

        clean_chunk = x_clean[start:end].detach().cpu()
        adv_chunk = x_adv[start:end].detach().cpu()
        y_chunk = y_true[start:end].detach().cpu()
        pred_clean_chunk = y_pred_clean[start:end].detach().cpu()
        pred_adv_chunk = y_pred_adv[start:end].detach().cpu()
        sample_id_chunk = sample_ids[start:end].detach().cpu()

        if save_format == "pt":
            torch.save(
                {
                    "x_clean": clean_chunk,
                    "x_adv": adv_chunk,
                    "y_true": y_chunk,
                    "y_pred_clean": pred_clean_chunk,
                    "y_pred_adv": pred_adv_chunk,
                    "sample_ids": sample_id_chunk,
                },
                save_dir / f"shard_{shard_id:04d}.pt",
            )
        elif save_format == "npz":
            clean_np = np.clip(clean_chunk.numpy(), 0.0, 1.0)
            adv_np = np.clip(adv_chunk.numpy(), 0.0, 1.0)
            np.savez_compressed(
                save_dir / f"shard_{shard_id:04d}.npz",
                x_clean=(clean_np * 255.0).round().astype(np.uint8),
                x_adv=(adv_np * 255.0).round().astype(np.uint8),
                y_true=y_chunk.numpy().astype(np.int64),
                y_pred_clean=pred_clean_chunk.numpy().astype(np.int64),
                y_pred_adv=pred_adv_chunk.numpy().astype(np.int64),
                sample_ids=sample_id_chunk.numpy().astype(np.int64),
            )
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial examples while evaluating ImageNet robustness"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--threat_model",
        type=str,
        default="L2",
        choices=["L2", "Linf"],
    )
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--n_examples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--architecture", type=str, default="convnext_large")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--save_adv_dir",
        type=str,
        default="data/gen_adv_imagenet",
        help="Output root for saved adversarial samples (default: data/gen_adv_imagenet)",
    )
    parser.add_argument(
        "--source_wnid",
        type=str,
        default="n02077923",
        help="Optional ImageNet val class WNID to load directly (default: n02077923, sea lion)",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="npz",
        choices=["pt", "npz"],
    )
    parser.add_argument(
        "--save_shard_size",
        type=int,
        default=256,
        help="Number of samples per saved shard",
    )
    parser.add_argument(
        "--subset_indices_file",
        type=str,
        default=None,
        help="Optional txt/json file with subset indices over the loaded set",
    )
    parser.add_argument(
        "--sae_ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
        help="SAE checkpoint path. Set empty string to disable SAE.",
    )
    parser.add_argument(
        "--sae_stage",
        type=int,
        default=None,
        help="ConvNeXt stage index to attach SAE hook (0..3). Default: infer from SAE d_in.",
    )
    parser.add_argument(
        "--sae_recon_mode",
        type=str,
        default="raw",
        choices=["raw", "norm"],
        help="SAE reconstruction mode: raw (denormalized) or norm.",
    )

    args = parser.parse_args()

    # Device mapping for torchrun-style launches.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}/{world_size}] Using device: {device}")

    model = load_custom_model(args.checkpoint, args.architecture)
    model = model.eval()

    # Keep consistency checks aligned with the existing DAT evaluation script.
    if args.architecture in ["resnet50", "wide_resnet50_4"]:
        assert hasattr(model, "normalize_input") and model.normalize_input is True
    elif args.architecture == "convnext_large":
        assert hasattr(model, "normalize_input") and model.normalize_input is False

    model = model.to(device)

    # Optional fallback DataParallel for single-process multi-GPU runs.
    if world_size == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    sae_hook_handle = None
    sae_enabled = bool(args.sae_ckpt and args.sae_ckpt.strip())
    if sae_enabled:
        if not os.path.exists(args.sae_ckpt):
            raise FileNotFoundError(f"SAE checkpoint not found: {args.sae_ckpt}")
        sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)
        sae_stage = resolve_sae_stage(sae_cfg, args.sae_stage)
        sae_hook_handle = attach_sae_hook(
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            stage_idx=sae_stage,
            recon_mode=args.sae_recon_mode,
        )
        print(
            f"SAE enabled: ckpt={args.sae_ckpt}, stage={sae_stage}, "
            f"d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}, mode={args.sae_recon_mode}"
        )
    else:
        print("SAE disabled (empty --sae_ckpt)")

    try:
        preprocessing = get_preprocessing_function(args.img_size)
        scale_size = int(math.floor(args.img_size / 0.875))
        print(f"Preprocessing: Resize{scale_size}Crop{args.img_size}")

        if args.source_wnid:
            x_test, y_test, sample_ids = load_imagenet_val_class(
                data_dir=args.data_dir,
                source_wnid=args.source_wnid,
                preprocess=preprocessing,
                n_examples=args.n_examples,
            )
            class_idx = int(y_test[0].item())
            print(
                f"Loaded class-only set: wnid={args.source_wnid}, class_idx={class_idx}, samples={len(x_test)}"
            )
        else:
            x_test, y_test = robustbench.data.load_imagenet(
                n_examples=args.n_examples,
                data_dir=args.data_dir,
                prepr=preprocessing,
            )
            sample_ids = torch.arange(len(x_test), dtype=torch.long)
            print(f"Loaded ImageNet examples: {len(x_test)}")

        subset_idx = parse_subset_indices(args.subset_indices_file, len(x_test))
        if subset_idx is not None:
            x_test = x_test[subset_idx]
            y_test = y_test[subset_idx]
            sample_ids = subset_idx.clone()
            print(f"Applied subset file: {args.subset_indices_file}, kept {len(x_test)} samples")

        # For multi-process runs, shard after optional subset filtering.
        if world_size > 1:
            chunk_size = math.ceil(len(x_test) / world_size)
            start = rank * chunk_size
            end = min(start + chunk_size, len(x_test))
            x_test = x_test[start:end]
            y_test = y_test[start:end]
            sample_ids = sample_ids[start:end]
            print(
                f"[Rank {rank}/{world_size}] Evaluating slice [{start}:{end}] ({len(x_test)} samples)"
            )

        if len(x_test) == 0:
            print(f"[Rank {rank}] Empty shard after slicing, nothing to do.")
            return

        threat_model = ThreatModel(args.threat_model)

        clean_acc = clean_accuracy(
            model,
            x_test,
            y_test,
            batch_size=args.batch_size,
            device=device,
        )

        adversary = AutoAttack(
            model,
            norm=threat_model.value,
            eps=args.eps,
            version="standard",
            device=device,
        )
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

        robust_acc = clean_accuracy(
            model,
            x_adv,
            y_test,
            batch_size=args.batch_size,
            device=device,
        )

        with torch.no_grad():
            clean_logits = []
            adv_logits = []
            for start in range(0, len(x_test), args.batch_size):
                end = min(start + args.batch_size, len(x_test))
                clean_batch = x_test[start:end].to(device)
                adv_batch = x_adv[start:end].to(device)
                clean_logits.append(model(clean_batch).detach().cpu())
                adv_logits.append(model(adv_batch).detach().cpu())
            y_pred_clean = torch.cat(clean_logits).argmax(dim=1)
            y_pred_adv = torch.cat(adv_logits).argmax(dim=1)

        rank_save_dir = Path(args.save_adv_dir) / f"rank{rank:03d}"
        save_adv_outputs(
            save_dir=rank_save_dir,
            x_clean=x_test,
            x_adv=x_adv,
            y_true=y_test,
            y_pred_clean=y_pred_clean,
            y_pred_adv=y_pred_adv,
            sample_ids=sample_ids,
            shard_size=args.save_shard_size,
            save_format=args.save_format,
        )

        result = {
            "rank": rank,
            "world_size": world_size,
            "num_samples": int(len(x_test)),
            "clean_accuracy": float(clean_acc),
            "robust_accuracy": float(robust_acc),
            "threat_model": args.threat_model,
            "eps": args.eps,
            "checkpoint": args.checkpoint,
            "architecture": args.architecture,
            "img_size": args.img_size,
        }

        with open(rank_save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("=" * 60)
        print(f"[Rank {rank}] RESULTS")
        print(f"  Samples: {len(x_test)}")
        print(f"  Clean accuracy:  {clean_acc:.4f} ({clean_acc:.2%})")
        print(f"  Robust accuracy: {robust_acc:.4f} ({robust_acc:.2%})")
        print(f"  Saved outputs to: {rank_save_dir}")
        print("=" * 60)
    finally:
        if sae_hook_handle is not None:
            sae_hook_handle.remove()


if __name__ == "__main__":
    main()
