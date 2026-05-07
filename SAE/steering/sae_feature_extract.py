#!/usr/bin/env python3
"""
Extract SAE latent features for 100-class adversarial samples.

Input:
    DAT/SAE/adversarial_samples/{wnid}/
        ├── train/clean/     (100 images)
        ├── train/adv/       (100 images)
        ├── val/clean/       (50 images)
        └── val/adv/         (50 images)

Output (400 independent files + index JSON):
    DAT/SAE/adversarial_samples/sae_latent/
        ├── n01440764_train_clean_features.npy
        ├── n01440764_train_clean_meta.json
        ├── n01440764_train_adv_features.npy
        ├── n01440764_train_adv_meta.json
        ├── ...
        └── feature_index.json

- 4-GPU parallel via multiprocessing.
- Batch size 128 per GPU.
- Features shape: (n_images, n_tokens, d_lat)  e.g. (100, 49, 12288)
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

SPLITS_TYPES = [("train", "clean"), ("train", "adv"), ("val", "clean"), ("val", "adv")]


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Model / SAE builders
# ------------------------------------------------------------------
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


def build_sae(device: torch.device, sae_ckpt_path: str):
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
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg.update({"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)})
    return sae, norm_mean, norm_std, resolved_cfg


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def extract_features_from_loader(
    loader: DataLoader,
    model: torch.nn.Module,
    sae: torch.nn.Module,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    device: torch.device,
):
    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    all_features = []
    all_names = []
    d_lat = None
    n_tokens = None

    with torch.no_grad():
        for images, names in loader:
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

            z_cpu = z.detach().cpu().numpy().astype(np.float32)
            z_per_image = z_cpu.reshape(bsz, n_tokens, d_lat)

            all_features.append(z_per_image)
            all_names.extend(list(names))

    handle.remove()

    features = np.concatenate(all_features, axis=0)  # (n_images, n_tokens, d_lat)
    meta = {
        "shape": features.shape,
        "n_images": int(features.shape[0]),
        "n_tokens": int(features.shape[1]),
        "d_lat": int(features.shape[2]),
        "stage": stage_idx,
    }
    return features, all_names, meta


# ------------------------------------------------------------------
# Per-GPU worker
# ------------------------------------------------------------------
def worker(args_tuple):
    (
        gpu_id,
        task_list,
        base_ckpt,
        sae_ckpt,
        output_root,
        batch_size,
        num_workers,
        stage,
    ) = args_tuple

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    model = build_model(device, base_ckpt)
    sae, norm_mean, norm_std, sae_cfg = build_sae(device, sae_ckpt)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    for wnid, class_idx, split, img_type, input_dir in tqdm(
        task_list,
        desc=f"GPU-{gpu_id}",
        position=gpu_id,
        leave=True,
    ):
        dataset = ImageFolderDataset(str(input_dir), transform=transform)
        if len(dataset) == 0:
            tqdm.write(f"[WARN] No images in {input_dir}")
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        features, names, meta = extract_features_from_loader(
            loader, model, sae, norm_mean, norm_std, stage, device
        )

        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = f"{wnid}_{split}_{img_type}"
        out_npy = out_dir / f"{out_prefix}_features.npy"
        out_meta = out_dir / f"{out_prefix}_meta.json"

        np.save(out_npy, features)

        meta.update({
            "wnid": wnid,
            "class_idx": int(class_idx),
            "split": split,
            "type": img_type,
            "input_dir": str(input_dir),
            "base_ckpt": base_ckpt,
            "sae_ckpt": sae_ckpt,
            "sae_config": sae_cfg,
            "image_names": names,
            "feature_npy": str(out_npy),
        })
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    del model, sae
    torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SAE latent features for 100-class adv samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples"),
        help="Root directory containing {wnid}/train/clean etc.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            REPO_ROOT / "SAE" / "adversarial_samples" / "sae_latent"
        ),
        help="Output directory for .npy / .json / feature_index.json",
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
    parser.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-gpus", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Verify GPU availability
    n_available = torch.cuda.device_count()
    if args.n_gpus > n_available:
        raise RuntimeError(
            f"Requested {args.n_gpus} GPUs but only {n_available} available"
        )

    # Scan class directories (exclude sae_latent if nested)
    class_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    class_dirs = [d for d in class_dirs if d.name != "sae_latent"]
    print(f"Found {len(class_dirs)} class directories")

    # Build task list: (wnid, class_idx, split, img_type, input_dir)
    task_list = []
    for class_dir in class_dirs:
        wnid = class_dir.name

        # Read class_idx from labels.txt
        labels_path = class_dir / "labels.txt"
        class_idx = 0
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                class_idx = int(parts[-1])
                                break
                            except ValueError:
                                pass
        else:
            print(f"[WARN] labels.txt not found for {wnid}, defaulting class_idx=0")

        for split, img_type in SPLITS_TYPES:
            input_dir = class_dir / split / img_type
            if input_dir.is_dir():
                task_list.append((wnid, class_idx, split, img_type, input_dir))
            else:
                print(f"[WARN] Directory not found: {input_dir}")

    print(f"Total extraction tasks: {len(task_list)}")

    # Verify stage / SAE d_in consistency
    expected_din = {0: 192, 1: 384, 2: 768, 3: 1536}[args.stage]
    # Quick-load SAE config to verify
    sae_ckpt_quick = torch.load(args.sae_ckpt, map_location="cpu")
    sae_config_quick = sae_ckpt_quick.get("config", {})
    d_in_quick = sae_ckpt_quick.get("d_in", sae_config_quick.get("d_in", None))
    if d_in_quick is not None and int(d_in_quick) != expected_din:
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.stage} expects d_in={expected_din}, "
            f"SAE has d_in={d_in_quick}"
        )
    print(f"Stage {args.stage} check passed (d_in={expected_din})")

    # Distribute tasks round-robin across GPUs
    tasks_per_gpu = [[] for _ in range(args.n_gpus)]
    for i, task in enumerate(task_list):
        tasks_per_gpu[i % args.n_gpus].append(task)

    for g in range(args.n_gpus):
        print(f"GPU {g}: {len(tasks_per_gpu[g])} tasks")

    # Launch workers
    mp.set_start_method("spawn", force=True)
    worker_args = [
        (
            gpu_id,
            tasks_per_gpu[gpu_id],
            args.base_ckpt,
            args.sae_ckpt,
            str(output_root),
            args.batch_size,
            args.num_workers,
            args.stage,
        )
        for gpu_id in range(args.n_gpus)
    ]

    processes = []
    for wargs in worker_args:
        p = mp.Process(target=worker, args=(wargs,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Build feature_index.json from all generated meta files
    print("\nBuilding feature_index.json ...")
    index = {}
    meta_files = sorted(output_root.glob("*_meta.json"))
    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        wnid = meta["wnid"]
        class_idx = meta["class_idx"]
        split = meta["split"]
        img_type = meta["type"]
        key = f"{split}_{img_type}"

        if wnid not in index:
            index[wnid] = {"class_idx": class_idx}
        index[wnid][key] = {
            "npy": meta["feature_npy"],
            "meta": str(meta_path),
            "n_images": meta["n_images"],
        }

    index_path = output_root / "feature_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Saved feature index: {index_path}")
    print(f"Total meta files indexed: {len(meta_files)}")
    print("Done!")


if __name__ == "__main__":
    main()
