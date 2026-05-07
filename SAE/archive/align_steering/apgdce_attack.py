#!/usr/bin/env python3
"""
APGD-CE 攻击基线脚本：支持原始模型（No SAE）与挂载 SAE 但不 steering 的对比。

随机从 ImageNet validation set 中抽取一个类别的图片，
生成对抗样本并评估模型鲁棒性。
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
import torch.nn.functional as F
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
from rebm.attacks.attack_steps import LinfStep
from sae_core.model import TopKAutoencoder


class ImagePathDataset(Dataset):
    """简单的图像路径数据集"""
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
    """构建并加载基础 ConvNeXT 模型"""
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
    """生成一个 forward hook，在指定 stage 后挂载 SAE encode/decode（无 steering）。"""
    def hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        x_reconstruct, _, _ = sae_model(flat_norm)
        x_out = x_reconstruct * norm_std + norm_mean
        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return hook


def get_class_dirs(val_dir: Path) -> List[Path]:
    """获取验证集中所有类别文件夹"""
    class_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])
    return class_dirs


def sample_images_from_class(class_dir: Path, n_samples: int) -> List[Path]:
    """从指定类别文件夹中随机采样图片"""
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]

    if not images:
        return []

    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


def apgd_ce_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.LongTensor,
    eps: float = 8/255,
    steps: int = 100,
    step_size: float = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    APGD-CE (AutoAttack PGD with Cross Entropy) 攻击

    Args:
        model: 目标模型
        x: 干净样本 [B, C, H, W]
        labels: 真实标签 [B]
        eps: 最大扰动大小 (Linf)
        steps: 迭代步数
        step_size: 步长，默认 eps/4
        random_start: 是否随机初始化

    Returns:
        对抗样本
    """
    if step_size is None:
        step_size = eps / 4

    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    x0 = x.clone().detach()
    step = LinfStep(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)

        loss = F.cross_entropy(logits, labels)

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )

        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x.clone().detach()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eps: float,
    steps: int,
    step_size: float,
    class_to_idx: dict,
) -> dict:
    """
    评估模型在干净样本和对抗样本上的准确率

    Returns:
        包含各项指标的字典
    """
    clean_correct = 0
    adv_correct = 0
    total = 0

    all_clean_preds = []
    all_adv_preds = []
    all_labels = []
    all_paths = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, (images, class_names, paths) in enumerate(pbar):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)

        batch_size = images.size(0)
        total += batch_size

        with torch.no_grad():
            logits_clean = model(images)
            preds_clean = logits_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()

        x_adv = apgd_ce_attack(
            model=model,
            x=images,
            labels=labels,
            eps=eps,
            steps=steps,
            step_size=step_size,
            random_start=True,
        )

        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()

        all_clean_preds.extend(preds_clean.cpu().tolist())
        all_adv_preds.extend(preds_adv.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_paths.extend(paths)

        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        pbar.set_postfix({
            'Clean Acc': f'{clean_acc:.2f}%',
            'Adv Acc': f'{adv_acc:.2f}%',
            'ASR': f'{100 - adv_acc:.2f}%'
        })

    results = {
        'clean_accuracy': 100.0 * clean_correct / total,
        'adversarial_accuracy': 100.0 * adv_correct / total,
        'attack_success_rate': 100.0 * (total - adv_correct) / total,
        'total_samples': total,
        'clean_correct': clean_correct,
        'adv_correct': adv_correct,
        'predictions': {
            'clean': all_clean_preds,
            'adversarial': all_adv_preds,
            'true': all_labels,
            'paths': all_paths,
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="APGD-CE Attack on ImageNet Validation Set (Baseline: No SAE or SAE-mounted without steering)"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "ImageNet" / "val"),
        help="ImageNet validation directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="使用的 GPU 编号 (默认 1，即第二张显卡)"
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="指定类别名称 (如 n01440764)，不指定则随机选择"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="每个类别采样的图片数量"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8/255,
        help="Linf 扰动预算 (默认 8/255)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="APGD 迭代步数"
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="步长 (默认 eps/4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批处理大小"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--save-adversarial",
        action="store_true",
        help="保存对抗样本图片"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "adv_samples"),
        help="对抗样本保存根目录"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="攻击结果 JSON 输出文件 (默认自动生成)"
    )
    parser.add_argument(
        "--use-sae",
        action="store_true",
        help="挂载 SAE 但不进行 steering 干预（用于对比 SAE 自身引入的影响）"
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
        help="SAE 检查点路径"
    )
    parser.add_argument(
        "--sae-stage",
        type=int,
        default=3,
        help="挂载 SAE 的 stage 索引 (0-3)"
    )

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
                raise ValueError(f"Class {selected_class} not found in validation set")
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
        print(f"  Epsilon (Linf): {args.eps:.4f} ({args.eps * 255:.1f}/255)")
        print(f"  Steps: {args.steps}")
        print(f"  Step size: {args.step_size or args.eps/4:.4f}")
        print(f"  Use SAE: {args.use_sae}")
        print(f"{'='*60}\n")

        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            eps=args.eps,
            steps=args.steps,
            step_size=args.step_size,
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
                "steps": int(args.steps),
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

            dataset_no_transform = ImagePathDataset(image_paths, T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]))
            loader = DataLoader(dataset_no_transform, batch_size=args.batch_size, shuffle=False)

            img_idx = 0
            for images, class_names, paths in tqdm(loader, desc="Saving adv samples"):
                labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
                images = images.to(device)

                x_adv = apgd_ce_attack(
                    model=model,
                    x=images,
                    labels=labels,
                    eps=args.eps,
                    steps=args.steps,
                    step_size=args.step_size,
                    random_start=True,
                )

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
