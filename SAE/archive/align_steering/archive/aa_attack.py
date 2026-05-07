#!/usr/bin/env python3
"""
AutoAttack (AA) 攻击脚本：支持原始模型（No SAE）与挂载 SAE 但不 steering 的对比。

使用 AutoAttack 的 4 组件攻击（APGD-CE, APGD-DLR, FAB-T, Square），
生成对抗样本并评估模型鲁棒性。
输出保存到 adv_samples/aa/ 目录下，与 baseline 结果完全隔离。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Resolve DAT root
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

# AutoAttack
from autoattack import AutoAttack


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: T.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        class_name = path.parent.name
        return x, class_name, str(path)


def build_model(device: torch.device, checkpoint: str):
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


def make_sae_hook(sae_model, norm_mean, norm_std):
    def hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        x_reconstruct, _, _ = sae_model(flat_norm)
        x_out = x_reconstruct * norm_std + norm_mean
        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return hook


def get_class_dirs(val_dir: Path) -> List[Path]:
    class_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])
    return class_dirs


def sample_images_from_class(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]

    if not images:
        return []

    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eps: float,
    class_to_idx: dict,
) -> Tuple[dict, torch.Tensor]:
    """
    评估模型在干净样本和 AutoAttack 对抗样本上的准确率。

    Returns:
        results: 包含各项指标的字典
        x_adv:   对抗样本张量 [N, C, H, W]
    """
    all_images = []
    all_labels = []
    all_paths = []

    for images, class_names, paths in dataloader:
        labels = torch.tensor([class_to_idx[name] for name in class_names])
        all_images.append(images)
        all_labels.append(labels)
        all_paths.extend(paths)

    all_images = torch.cat(all_images, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    total = all_images.size(0)

    # Clean accuracy
    with torch.no_grad():
        logits_clean = model(all_images)
        preds_clean = logits_clean.argmax(dim=1)
        clean_correct = (preds_clean == all_labels).sum().item()

    print(f"\nRunning AutoAttack (eps={eps:.4f}, samples={total})...")
    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=eps,
        version="standard",
        device=str(device),
    )
    x_adv = adversary.run_standard_evaluation(all_images, all_labels, bs=16)

    # Adv accuracy
    with torch.no_grad():
        logits_adv = model(x_adv)
        preds_adv = logits_adv.argmax(dim=1)
        adv_correct = (preds_adv == all_labels).sum().item()

    print(f"  Clean Acc: {100.0 * clean_correct / total:.2f}%")
    print(f"  Adv Acc:   {100.0 * adv_correct / total:.2f}%")
    print(f"  ASR:       {100.0 * (total - adv_correct) / total:.2f}%")

    results = {
        'clean_accuracy': 100.0 * clean_correct / total,
        'adversarial_accuracy': 100.0 * adv_correct / total,
        'attack_success_rate': 100.0 * (total - adv_correct) / total,
        'total_samples': total,
        'clean_correct': clean_correct,
        'adv_correct': adv_correct,
        'predictions': {
            'clean': preds_clean.cpu().tolist(),
            'adversarial': preds_adv.cpu().tolist(),
            'true': all_labels.cpu().tolist(),
            'paths': all_paths,
        }
    }

    return results, x_adv


def main():
    parser = argparse.ArgumentParser(
        description="AutoAttack on ImageNet Validation Set (No SAE or SAE-mounted without steering)"
    )
    parser.add_argument("--val-dir", type=str, default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=1, help="GPU index (逻辑卡，worker会绑定CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--class-name", type=str, default=None, help="指定类别 (如 n01440764)")
    parser.add_argument("--n-samples", type=int, default=50, help="每类采样图片数")
    parser.add_argument("--eps", type=float, default=8/255, help="Linf 扰动预算")
    parser.add_argument("--steps", type=int, default=100, help="(保留但忽略，AutoAttack 自动决定)")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size (AutoAttack 内部bs)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-adversarial", action="store_true", help="保存对抗样本图片")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "adv_samples" / "aa"), help="对抗样本保存根目录")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--use-sae", action="store_true", help="挂载 SAE 但不 steering")
    parser.add_argument("--sae-ckpt", type=str, default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"))
    parser.add_argument("--sae-stage", type=int, default=3)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0")
            args.gpu = 0
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    model = build_model(device=device, checkpoint=args.checkpoint)

    sae_hook_handle = None
    if args.use_sae:
        print(f"Loading SAE from: {args.sae_ckpt} (stage={args.sae_stage})")
        sae_model, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)
        print(f"SAE loaded: d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")
        sae_hook_handle = model.stages[args.sae_stage].register_forward_hook(
            make_sae_hook(sae_model, norm_mean, norm_std)
        )
        print("SAE mounted (no steering).")

    try:
        val_dir = Path(args.val_dir).resolve()
        class_dirs = get_class_dirs(val_dir)
        class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        print(f"Found {len(class_dirs)} classes in validation set")

        if args.class_name is not None:
            selected_class = args.class_name
            if selected_class not in class_to_idx:
                raise ValueError(f"Class {selected_class} not found")
        else:
            selected_class = random.choice([d.name for d in class_dirs])

        selected_class_idx = class_to_idx[selected_class]
        print(f"\nSelected class: {selected_class} (index: {selected_class_idx})")

        class_dir = val_dir / selected_class
        image_paths = sample_images_from_class(class_dir, args.n_samples)
        print(f"Sampled {len(image_paths)} images from class {selected_class}")

        if not image_paths:
            raise RuntimeError(f"No images found in {class_dir}")

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        dataset = ImagePathDataset(image_paths, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"\n{'='*60}")
        print(f"Attack Parameters:")
        print(f"  Method: AutoAttack (APGD-CE + APGD-DLR + FAB-T + Square)")
        print(f"  Epsilon (Linf): {args.eps:.4f} ({args.eps * 255:.1f}/255)")
        print(f"  Use SAE: {args.use_sae}")
        print(f"{'='*60}\n")

        results, x_adv = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            eps=args.eps,
            class_to_idx=class_to_idx,
        )

        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success Rate: {results['attack_success_rate']:.2f}%")
        print(f"{'='*60}")

        output_data = {
            "meta": {
                "class_name": selected_class,
                "class_index": int(selected_class_idx),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "eps": float(args.eps),
                "attack_method": "autoattack",
                "n_samples": int(args.n_samples),
                "seed": int(args.seed),
                "use_sae": bool(args.use_sae),
                "sae_stage": int(args.sae_stage) if args.use_sae else None,
                "sae_ckpt": str(Path(args.sae_ckpt).resolve()) if args.use_sae else None,
            },
            "accuracy": {
                "clean_accuracy": float(results['clean_accuracy']),
                "adversarial_accuracy": float(results['adversarial_accuracy']),
                "attack_success_rate": float(results['attack_success_rate']),
                "total_samples": int(results['total_samples']),
            }
        }

        condition_dir = f"sae_stage{args.sae_stage}" if args.use_sae else "no_sae"
        output_root = Path(args.output_dir) / condition_dir

        if args.output_json is None:
            output_json_path = output_root / f"attack_results_{selected_class}.json"
        else:
            output_json_path = Path(args.output_json)

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nAttack results saved to: {output_json_path}")

        if args.save_adversarial:
            output_dir = output_root / selected_class
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving adversarial samples to: {output_dir}")

            img_idx = 0
            paths = results['predictions']['paths']
            for i in range(x_adv.size(0)):
                adv_img = x_adv[i].cpu().detach()
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_pil = T.ToPILImage()(adv_img)
                orig_name = Path(paths[i]).stem
                save_path = output_dir / f"{orig_name}_adv.png"
                adv_pil.save(save_path)
                img_idx += 1

            print(f"Saved {img_idx} adversarial samples")

        print("\nDone!")

    finally:
        if sae_hook_handle is not None:
            sae_hook_handle.remove()
            print("SAE hook removed.")


if __name__ == "__main__":
    main()
