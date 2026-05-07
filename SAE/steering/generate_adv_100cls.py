#!/usr/bin/env python3
"""
Generate adversarial samples for 100 ImageNet classes (train set).

- 20 fixed classes (from run_untar_20classes.py) + 80 random classes.
- 150 images per class, split into 100 (train-train) / 50 (train-val).
- Attack: APGD-CE, L2, eps=3.0, steps=100, n_restarts=1.
- 8-GPU parallel via multiprocessing (1 model per GPU).

Output structure:
    {output_root}/
        class_split.json
        {wnid}/
            labels.txt
            train/
                clean/{fname}.JPEG
                adv/{fname}_adv.JPEG
            val/
                clean/{fname}.JPEG
                adv/{fname}_adv.JPEG

Usage:
    cd /Data_share/hongyi/DAT/SAE/steering
    python generate_adv_100cls.py

Optional args:
    --n-gpus 8
    --batch-size 32
    --eps 3.0
    --steps 100
    --n-restarts 1
    --seed 42
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Resolve DAT root and insert paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from rebm.attacks.attack_steps import L2Step

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

# 20 representative classes from run_untar_20classes.py
FIXED_CLASSES = [
    ("n01440764", 0),
    ("n01530575", 10),
    ("n01641577", 30),
    ("n01806143", 84),
    ("n01871265", 101),
    ("n02077923", 150),
    ("n02123045", 281),
    ("n02128385", 288),
    ("n02129604", 292),
    ("n02165456", 301),
    ("n03063599", 504),
    ("n03085013", 508),
    ("n03250847", 542),
    ("n03445777", 574),
    ("n03770439", 655),
    ("n03888257", 701),
    ("n04146614", 779),
    ("n04285008", 817),
    ("n07720875", 945),
    ("n07747607", 950),
]


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        fname = Path(path).name
        return x, fname


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def build_model(device: torch.device, checkpoint: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, checkpoint, weights_only=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------------------------------------------------
# APGD-CE L2 attack (hand-rolled, consistent with DAT codebase)
# ------------------------------------------------------------------
def apgd_ce_attack_l2(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 3.0,
    steps: int = 100,
    step_size: float = None,
    n_restarts: int = 1,
) -> torch.Tensor:
    """
    APGD-CE under L2 threat model.
    n_restarts=1  => single run with random start.
    n_restarts>1  => multiple random restarts, keep the one with highest CE loss.
    """
    if step_size is None:
        step_size = eps / 4.0

    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    best_adv = None
    best_loss = None

    for _ in range(n_restarts):
        x0 = x.clone().detach()
        step = L2Step(eps=eps, orig_input=x0, step_size=step_size)
        x_curr = step.random_perturb(x0)

        for _ in range(steps):
            x_curr = x_curr.clone().detach().requires_grad_(True)
            logits = model(x_curr)
            loss = F.cross_entropy(logits, labels)

            (grad,) = torch.autograd.grad(
                outputs=loss,
                inputs=[x_curr],
                grad_outputs=None,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )

            with torch.no_grad():
                x_curr = step.step(x_curr, grad)
                x_curr = step.project(x_curr)

        # Track best among restarts (highest CE loss = most adversarial)
        with torch.no_grad():
            logits = model(x_curr)
            curr_loss = F.cross_entropy(logits, labels, reduction="none")

            if best_adv is None:
                best_adv = x_curr.clone().detach()
                best_loss = curr_loss
            else:
                improve = curr_loss > best_loss
                improve = improve.view(-1, 1, 1, 1)
                best_adv = torch.where(improve, x_curr, best_adv)
                best_loss = torch.where(improve.view(-1), curr_loss, best_loss)

    return best_adv


# ------------------------------------------------------------------
# Image saving
# ------------------------------------------------------------------
def save_image_tensor(tensor: torch.Tensor, path: Path):
    img = torch.clamp(tensor, 0.0, 1.0)
    pil = T.ToPILImage()(img.cpu())
    pil.save(str(path), quality=95)


# ------------------------------------------------------------------
# Sampling / splitting helpers
# ------------------------------------------------------------------
def sample_images(class_dir: Path, n_samples: int, seed: int):
    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTS
    ]
    if len(images) < n_samples:
        raise ValueError(
            f"Class {class_dir.name} has only {len(images)} images, "
            f"required {n_samples}"
        )
    # Use a per-class deterministic sub-seed
    rng = random.Random(seed ^ hash(class_dir.name))
    return rng.sample(images, k=n_samples)


def split_train_val(image_paths, n_train: int, seed: int):
    rng = random.Random(seed ^ hash(image_paths[0].parent.name) ^ 0xABCD)
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


# ------------------------------------------------------------------
# Per-GPU worker
# ------------------------------------------------------------------
def worker(args_tuple):
    (
        gpu_id,
        task_list,
        checkpoint,
        output_root,
        eps,
        steps,
        n_restarts,
        batch_size,
        num_workers,
        seed,
    ) = args_tuple

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Re-seed per process for determinism
    torch.manual_seed(seed + gpu_id)
    np.random.seed(seed + gpu_id)
    random.seed(seed + gpu_id)

    model = build_model(device, checkpoint)
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    for wnid, class_idx, split_dict in tqdm(
        task_list, desc=f"GPU-{gpu_id} classes", position=gpu_id, leave=True
    ):
        class_out = Path(output_root) / wnid
        labels_records = []

        for split, image_paths in split_dict.items():
            if not image_paths:
                continue

            clean_dir = class_out / split / "clean"
            adv_dir = class_out / split / "adv"
            clean_dir.mkdir(parents=True, exist_ok=True)
            adv_dir.mkdir(parents=True, exist_ok=True)

            dataset = ImagePathDataset(image_paths, transform)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            for images, filenames in tqdm(
                loader,
                desc=f"{wnid}/{split}",
                leave=False,
                position=gpu_id + 8,
            ):
                images = images.to(device)
                labels = torch.full(
                    (images.size(0),),
                    class_idx,
                    dtype=torch.long,
                    device=device,
                )

                # Generate adversarial examples
                adv_images = apgd_ce_attack_l2(
                    model,
                    images,
                    labels,
                    eps=eps,
                    steps=steps,
                    n_restarts=n_restarts,
                )

                # Save clean + adv images
                for i in range(images.size(0)):
                    fname = filenames[i]
                    stem = Path(fname).stem

                    clean_path = clean_dir / fname
                    adv_path = adv_dir / f"{stem}_adv.JPEG"

                    save_image_tensor(images[i], clean_path)
                    save_image_tensor(adv_images[i], adv_path)

                    labels_records.append(
                        (f"{split}/clean/{fname}", class_idx)
                    )
                    labels_records.append(
                        (f"{split}/adv/{stem}_adv.JPEG", class_idx)
                    )

        # Write per-class labels.txt
        if labels_records:
            labels_path = class_out / "labels.txt"
            with open(labels_path, "w") as f:
                for rel_path, idx in labels_records:
                    f.write(f"{rel_path} {idx}\n")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate adv samples for 100 ImageNet classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "ImageNet" / "train"),
        help="ImageNet train directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "adversarial_samples"),
        help="Output root directory",
    )
    parser.add_argument("--n-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--eps", type=float, default=3.0, help="L2 epsilon")
    parser.add_argument("--steps", type=int, default=100, help="APGD steps")
    parser.add_argument("--n-restarts", type=int, default=1, help="APGD restarts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-samples", type=int, default=150, help="Images sampled per class"
    )
    parser.add_argument(
        "--n-train", type=int, default=100, help="Train split size per class"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed main process
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Validate GPUs
    n_available = torch.cuda.device_count()
    if args.n_gpus > n_available:
        raise RuntimeError(
            f"Requested {args.n_gpus} GPUs but only {n_available} available"
        )

    train_dir = Path(args.train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Collect class directories (sorted => standard ImageNet class order)
    all_class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    all_wnids = [d.name for d in all_class_dirs]
    class_to_idx = {d.name: i for i, d in enumerate(all_class_dirs)}

    print(f"Found {len(all_class_dirs)} class directories in {train_dir}")

    # Build 100-class list
    fixed_wnids = {c[0] for c in FIXED_CLASSES}
    missing_fixed = fixed_wnids - set(all_wnids)
    if missing_fixed:
        raise ValueError(f"Fixed classes not found: {missing_fixed}")

    remaining = [w for w in all_wnids if w not in fixed_wnids]
    rng = random.Random(args.seed)
    rng.shuffle(remaining)
    random_80 = remaining[:80]

    selected_classes = FIXED_CLASSES + [(w, class_to_idx[w]) for w in random_80]
    print(f"Selected {len(selected_classes)} classes: 20 fixed + 80 random")

    # Sample & split
    tasks = []  # [(wnid, class_idx, {"train": [...], "val": [...]}), ...]
    for wnid, class_idx in selected_classes:
        class_dir = train_dir / wnid
        images = sample_images(class_dir, args.n_samples, args.seed)
        train_paths, val_paths = split_train_val(images, args.n_train, args.seed)
        tasks.append((wnid, class_idx, {"train": train_paths, "val": val_paths}))

    # Save metadata
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    random_classes_meta = [(w, class_to_idx[w]) for w in random_80]
    meta = {
        "seed": args.seed,
        "n_samples_per_class": args.n_samples,
        "n_train": args.n_train,
        "n_val": args.n_samples - args.n_train,
        "eps": args.eps,
        "steps": args.steps,
        "n_restarts": args.n_restarts,
        "batch_size": args.batch_size,
        "n_gpus": args.n_gpus,
        "fixed_classes": [{"wnid": w, "class_idx": i} for w, i in FIXED_CLASSES],
        "random_classes": [{"wnid": w, "class_idx": i} for w, i in random_classes_meta],
    }
    with open(output_root / "class_split.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {output_root / 'class_split.json'}")

    # Distribute tasks across GPUs (round-robin for balance)
    tasks_per_gpu = [[] for _ in range(args.n_gpus)]
    for i, task in enumerate(tasks):
        tasks_per_gpu[i % args.n_gpus].append(task)

    for g in range(args.n_gpus):
        print(f"GPU {g}: {len(tasks_per_gpu[g])} classes")

    # Launch workers
    mp.set_start_method("spawn", force=True)
    worker_args = [
        (
            gpu_id,
            tasks_per_gpu[gpu_id],
            args.checkpoint,
            str(output_root),
            args.eps,
            args.steps,
            args.n_restarts,
            args.batch_size,
            args.num_workers,
            args.seed,
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

    print("\n" + "=" * 60)
    print("All tasks completed.")
    print(f"Output directory: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
