#!/usr/bin/env python3
"""Compute class-wise SAE activation statistics on ImageNet train set.

For each class folder under ImageNet train directory:
1) Randomly sample up to N images (default: 500).
2) Extract ConvNeXT stage features (default: stage 3).
3) Encode with SAE.
4) Compute class mean latent activation and report Top-K feature indices/values.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Resolve DAT root from DAT/SAE/align_steering/sae_stat.py
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str], class_indices: List[int], transform: T.Compose):
        self.image_paths = image_paths
        self.class_indices = class_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        y = self.class_indices[idx]
        return x, y


def build_base_model(device: torch.device, checkpoint: str):
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


def build_sae(device: torch.device, sae_ckpt_path: str):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))

    if d_in is None:
        raise KeyError("Cannot resolve d_in from SAE checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from SAE checkpoint")
        d_lat = int(d_in) * int(expansion_rate)
    if k is None:
        raise KeyError("Cannot resolve k from SAE checkpoint")

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg["d_in"] = int(d_in)
    resolved_cfg["d_lat"] = int(d_lat)
    resolved_cfg["k"] = int(k)
    return sae, norm_mean, norm_std, resolved_cfg


def list_class_dirs(imagenet_train_dir: Path) -> List[Path]:
    if not imagenet_train_dir.exists():
        raise FileNotFoundError(f"Directory not found: {imagenet_train_dir}")
    class_dirs = sorted([p for p in imagenet_train_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {imagenet_train_dir}")
    return class_dirs


def list_images_in_class(class_dir: Path) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def build_sample_manifest(
    class_dirs: List[Path], images_per_class: int
) -> Tuple[List[str], List[int], List[str], Dict[str, int], Dict[str, str]]:
    class_names: List[str] = []
    per_class_image_count: Dict[str, int] = {}
    skipped_classes: Dict[str, str] = {}
    all_image_paths: List[str] = []
    all_class_indices: List[int] = []

    for class_dir in class_dirs:
        image_paths = list_images_in_class(class_dir)
        if not image_paths:
            skipped_classes[class_dir.name] = "no_images"
            continue

        take_n = min(len(image_paths), images_per_class)
        if len(image_paths) > take_n:
            image_paths = random.sample(image_paths, k=take_n)

        cls_idx = len(class_names)
        class_names.append(class_dir.name)
        per_class_image_count[class_dir.name] = int(len(image_paths))

        all_image_paths.extend([str(p) for p in image_paths])
        all_class_indices.extend([cls_idx] * len(image_paths))

    return class_names, all_class_indices, all_image_paths, per_class_image_count, skipped_classes


def chunked_list(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def compute_all_class_mean_latent(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    class_names: List[str],
    class_image_counts: Dict[str, int],
    image_paths: List[str],
    class_indices: List[int],
    batch_size: int,
    device: torch.device,
    transform: T.Compose,
    num_workers: int,
    prefetch_factor: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    def worker_init_fn(worker_id):
        print(f"[DataLoader worker {worker_id}] started", flush=True)

    dataset = ImagePathDataset(
        image_paths=image_paths,
        class_indices=class_indices,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": False,
        "timeout": 120,
        "worker_init_fn": worker_init_fn,
    }
    # Disable persistent_workers for debug and compatibility
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    num_classes = len(class_names)
    d_lat = int(sae_model.W_dec.shape[0])
    class_sums = torch.zeros((num_classes, d_lat), dtype=torch.float64)
    class_token_counts = torch.zeros(num_classes, dtype=torch.long)

    class_spatial_sums: Optional[torch.Tensor] = None
    spatial_image_counts = torch.zeros(num_classes, dtype=torch.long)
    spatial_size = 0

    import time
    total_tokens = 0
    batch_times = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(tqdm(loader, desc="Batches", unit="batch", leave=False)):
            t0 = time.time()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            _ = model(xb)   # xb: [B, 3, 224, 224]

            feat = captured["feat"]     # feat: [B, C, H, W], 例如 stage3: [B, 1536, 7, 7]
            _, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)

            labels_per_token = yb.repeat_interleave(h * w)
            uniq, inv = torch.unique(labels_per_token, sorted=False, return_inverse=True)

            sums_by_uniq = torch.zeros((uniq.shape[0], z.shape[1]), device=device, dtype=z.dtype)
            sums_by_uniq.index_add_(0, inv, z)
            counts_by_uniq = torch.bincount(inv, minlength=uniq.shape[0])

            uniq_cpu = uniq.cpu()
            class_sums[uniq_cpu] += sums_by_uniq.to(dtype=torch.float64).cpu()
            class_token_counts[uniq_cpu] += counts_by_uniq.cpu()

            B = xb.shape[0]
            if class_spatial_sums is None:
                spatial_size = h * w
                class_spatial_sums = torch.zeros((num_classes, spatial_size, d_lat), dtype=torch.float64)
            z_spatial = z.reshape(B, spatial_size, d_lat)
            for i in range(B):
                cls_idx = int(yb[i].item())
                class_spatial_sums[cls_idx] += z_spatial[i].cpu().to(torch.float64)
                spatial_image_counts[cls_idx] += 1

            t1 = time.time()
            batch_time = t1 - t0
            batch_tokens = int(z.shape[0])
            total_tokens += batch_tokens
            batch_times.append(batch_time)
            print(f"[Batch {batch_idx}] time: {batch_time:.3f}s, batch_size: {xb.shape[0]}, tokens: {batch_tokens}, throughput: {batch_tokens/batch_time:.1f} tokens/s", flush=True)
    if batch_times:
        print(f"[Summary] {len(batch_times)} batches, avg time: {np.mean(batch_times):.3f}s, avg throughput: {total_tokens/sum(batch_times):.1f} tokens/s", flush=True)

    handle.remove()

    mean_by_class: Dict[str, np.ndarray] = {}
    token_count_by_class: Dict[str, int] = {}
    spatial_mean_by_class: Dict[str, np.ndarray] = {}

    for cls_idx, cls_name in enumerate(class_names):
        tok = int(class_token_counts[cls_idx].item())
        if tok <= 0:
            raise RuntimeError(f"No SAE activations were extracted for class {cls_name}")
        mean_vec = (class_sums[cls_idx] / float(tok)).to(dtype=torch.float32).numpy()
        mean_by_class[cls_name] = mean_vec
        token_count_by_class[cls_name] = tok

        expected_tokens = int(class_image_counts[cls_name]) * (h * w)
        if tok != expected_tokens:
            raise RuntimeError(
                f"Token count mismatch for {cls_name}: got {tok}, expected {expected_tokens}"
            )

        img_cnt = int(spatial_image_counts[cls_idx].item())
        if img_cnt <= 0:
            raise RuntimeError(f"No images were processed for class {cls_name}")
        spatial_mean = (class_spatial_sums[cls_idx] / float(img_cnt)).to(dtype=torch.float32).numpy()
        spatial_mean_by_class[cls_name] = spatial_mean

    return mean_by_class, token_count_by_class, spatial_mean_by_class, spatial_size


def topk_from_mean(mean_vec: np.ndarray, topk: int) -> Tuple[List[int], List[float]]:
    k = min(topk, int(mean_vec.shape[0]))
    order = np.argsort(mean_vec)[::-1][:k]
    idx = [int(x) for x in order]
    val = [float(mean_vec[x]) for x in order]
    return idx, val


def topk_from_spatial_mean(spatial_mean: np.ndarray, topk: int) -> Tuple[List[List[int]], List[List[float]]]:
    k = min(topk, int(spatial_mean.shape[1]))
    indices: List[List[int]] = []
    values: List[List[float]] = []
    for row in spatial_mean:
        order = np.argsort(row)[::-1][:k]
        indices.append([int(x) for x in order])
        values.append([float(row[x]) for x in order])
    return indices, values


def main():
    parser = argparse.ArgumentParser(description="Class-wise SAE activation statistics (supports in-memory class batching)")
    parser.add_argument("--class-batch-size", type=int, default=0, help="Number of classes to process per batch (0=all at once, recommended: 10~50 for HDD)")
    parser.add_argument(
        "--imagenet-train-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "ImageNet" / "train"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
    )
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=max(2, min(16, os.cpu_count() or 8)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-limit", type=int, default=0, help="0 means use all classes")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "sae_stat_results.json"),
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Path for compressed NPZ with spatial arrays. Defaults to same stem as --output-json.",
    )
    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="Auto-name output files based on SAE checkpoint stem to avoid overwriting.",
    )

    args = parser.parse_args()

    if args.auto_name:
        ckpt_stem = Path(args.sae_ckpt).stem
        out_dir = Path(args.output_json).resolve().parent
        args.output_json = str(out_dir / f"{ckpt_stem}_sae_stat_results.json")
        if args.output_npz is None:
            args.output_npz = str(out_dir / f"{ckpt_stem}_sae_stat_results.npz")

    if args.images_per_class <= 0:
        raise ValueError("images-per-class must be > 0")
    if args.topk <= 0:
        raise ValueError("topk must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("num-workers must be >= 0")
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be > 0 when num-workers > 0")
    if args.torch_num_threads <= 0:
        raise ValueError("torch-num-threads must be > 0")
    if args.sae_stage not in STAGE_DIN_MAP:
        raise ValueError("sae-stage must be one of {0,1,2,3}")
    if args.class_limit < 0:
        raise ValueError("class-limit must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_num_threads)

    imagenet_train_dir = Path(args.imagenet_train_dir).resolve()
    class_dirs = list_class_dirs(imagenet_train_dir)

    if args.class_limit > 0:
        if args.class_limit > len(class_dirs):
            raise ValueError(
                f"class-limit={args.class_limit} exceeds available classes ({len(class_dirs)})"
            )
        class_dirs = random.sample(class_dirs, k=args.class_limit)
        class_dirs = sorted(class_dirs, key=lambda p: p.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    model = build_base_model(device=device, checkpoint=args.checkpoint)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)

    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.sae_stage} expects d_in={expected_din}, SAE has d_in={sae_cfg['d_in']}"
        )

    print(
        f"SAE loaded: stage={args.sae_stage}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )


    # Split classes into batches for in-memory processing
    all_class_dirs = class_dirs
    class_batch_size = args.class_batch_size
    if class_batch_size <= 0 or class_batch_size > len(all_class_dirs):
        class_batches = [all_class_dirs]
    else:
        class_batches = list(chunked_list(all_class_dirs, class_batch_size))

    merged_class_stats: Dict[str, Dict[str, object]] = {}
    merged_skipped_classes: Dict[str, str] = {}
    total_classes = 0

    spatial_indices_list: List[np.ndarray] = []
    spatial_activations_list: List[np.ndarray] = []
    spatial_class_names: List[str] = []

    for batch_idx, batch_class_dirs in enumerate(class_batches):
        print(f"\n[Batch {batch_idx+1}/{len(class_batches)}] Processing {len(batch_class_dirs)} classes...")
        class_names, all_class_indices, all_image_paths, class_image_counts, skipped_classes = build_sample_manifest(
            class_dirs=batch_class_dirs,
            images_per_class=args.images_per_class,
        )
        if not class_names:
            print("No valid images in this batch, skipping.")
            merged_skipped_classes.update(skipped_classes)
            continue

        print(
            f"Prepared {len(class_names)} classes, {len(all_image_paths)} images, "
            f"workers={args.num_workers}, batch_size={args.batch_size}, "
            f"torch_threads={args.torch_num_threads}"
        )

        mean_by_class, token_count_by_class, spatial_mean_by_class, spatial_size = compute_all_class_mean_latent(
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            stage_idx=args.sae_stage,
            class_names=class_names,
            class_image_counts=class_image_counts,
            image_paths=all_image_paths,
            class_indices=all_class_indices,
            batch_size=args.batch_size,
            device=device,
            transform=transform,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

        for cls_name in tqdm(class_names, desc="Finalize", unit="class"):
            mean_vec = mean_by_class[cls_name]
            token_count = token_count_by_class[cls_name]
            top_idx, top_val = topk_from_mean(mean_vec, topk=args.topk)

            spatial_mean = spatial_mean_by_class[cls_name]
            spatial_top_idx, spatial_top_val = topk_from_spatial_mean(spatial_mean, topk=args.topk)

            merged_class_stats[cls_name] = {
                "image_count": int(class_image_counts[cls_name]),
                "token_count": int(token_count),
                "top_features": {
                    "indices": top_idx,
                    "mean_activation": top_val,
                },
            }
            spatial_class_names.append(cls_name)
            spatial_indices_list.append(np.array(spatial_top_idx, dtype=np.int32))
            spatial_activations_list.append(np.array(spatial_top_val, dtype=np.float32))
        merged_skipped_classes.update(skipped_classes)
        total_classes += len(class_names)
        print(f"[Batch {batch_idx+1}] Done. Total processed so far: {total_classes}")

    if not merged_class_stats:
        raise RuntimeError("No class statistics computed. Check dataset path and image files.")

    out = {
        "meta": {
            "imagenet_train_dir": str(imagenet_train_dir),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "sae_ckpt": str(Path(args.sae_ckpt).resolve()),
            "sae_stage": int(args.sae_stage),
            "sae_d_in": int(sae_cfg["d_in"]),
            "sae_d_lat": int(sae_cfg["d_lat"]),
            "sae_k": int(sae_cfg["k"]),
            "seed": int(args.seed),
            "images_per_class": int(args.images_per_class),
            "topk": int(args.topk),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "torch_num_threads": int(args.torch_num_threads),
            "class_limit": int(args.class_limit),
            "class_batch_size": int(args.class_batch_size),
            "spatial_size": int(spatial_size),
            "num_classes_used": int(len(merged_class_stats)),
            "num_classes_skipped": int(len(merged_skipped_classes)),
        },
        "class_stats": merged_class_stats,
        "skipped_classes": merged_skipped_classes,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if spatial_indices_list:
        npz_path = Path(args.output_npz).resolve() if args.output_npz else out_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            class_names=np.array(spatial_class_names, dtype=object),
            spatial_indices=np.stack(spatial_indices_list, axis=0),           # [N, H*W, topk]
            spatial_activations=np.stack(spatial_activations_list, axis=0),   # [N, H*W, topk]
        )
        print(f"Saved NPZ: {npz_path}")

    print(f"Saved JSON: {out_path}")
    print(f"Classes processed: {len(merged_class_stats)}")
    print(f"Classes skipped: {len(merged_skipped_classes)}")


if __name__ == "__main__":
    main()
