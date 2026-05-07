#!/usr/bin/env python3
"""
批量生成 eps=3/255 对抗样本（50个类），只加载一次模型。
输出到 adv_samples_eps3/no_sae/{class_name}/
"""

import argparse
import json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from rebm.attacks.attack_steps import LinfStep


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
        class_name = path.parent.name
        return x, class_name, str(path.name)


def build_model(device, checkpoint):
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


def apgd_ce_attack(model, x, labels, eps, steps, step_size=None, random_start=True):
    if step_size is None:
        step_size = eps / 4
    x0 = x.clone().detach()
    step = LinfStep(eps=eps, orig_input=x0, step_size=step_size)
    if random_start:
        x = step.random_perturb(x)
    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        (grad,) = torch.autograd.grad(outputs=loss, inputs=[x], grad_outputs=None,
                                       retain_graph=False, create_graph=False,
                                       only_inputs=True, allow_unused=False)
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x.clone().detach()


def evaluate_and_save(model, dataloader, device, eps, steps, step_size,
                      class_to_idx, save_dir, save_images):
    clean_correct = 0
    adv_correct = 0
    total = 0
    all_adv_images = []
    all_labels = []
    all_names = []

    for images, class_names, names in tqdm(dataloader, desc="Attack", leave=False):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)
        B = images.size(0)
        total += B

        with torch.no_grad():
            logits_clean = model(images)
            preds_clean = logits_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()

        x_adv = apgd_ce_attack(model, images, labels, eps, steps, step_size, random_start=True)

        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()

        if save_images:
            all_adv_images.append(x_adv.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)

    acc = {
        "clean_accuracy": 100.0 * clean_correct / total,
        "adversarial_accuracy": 100.0 * adv_correct / total,
        "attack_success_rate": 100.0 * (total - adv_correct) / total,
        "total_samples": total,
    }

    if save_images and all_adv_images:
        save_dir.mkdir(parents=True, exist_ok=True)
        adv_tensor = torch.cat(all_adv_images, dim=0)
        for i in range(adv_tensor.size(0)):
            img = torch.clamp(adv_tensor[i], 0, 1)
            T.ToPILImage()(img).save(save_dir / f"{Path(all_names[i]).stem}_adv.png")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Batch generate eps=3 adversarial samples")
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eps", type=float, default=3/255)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples_eps3" / "no_sae"))
    parser.add_argument("--class-list", type=str, default=None,
                        help="指定类别列表文件（每行一个类别名），不指定则处理所有类")
    parser.add_argument("--n-classes", type=int, default=0,
                        help="处理的类别数（0=全部，仅在未指定class-list时有效）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    model = build_model(device, args.checkpoint)

    val_dir = Path(args.val_dir).resolve()
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    # 确定要处理的类别
    if args.class_list:
        with open(args.class_list, "r") as f:
            selected_classes = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(selected_classes)} classes from {args.class_list}")
    else:
        selected_classes = [d.name for d in class_dirs]
        if args.n_classes > 0:
            selected_classes = selected_classes[:args.n_classes]
        print(f"Processing first {len(selected_classes)} classes")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    all_results = {}

    for class_name in tqdm(selected_classes, desc="Classes", unit="class"):
        if class_name not in class_to_idx:
            tqdm.write(f"[Skip] {class_name} not found in val set")
            continue

        class_dir = val_dir / class_name
        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
        if not images:
            tqdm.write(f"[Skip] {class_name}: no images")
            continue

        n = min(args.n_samples, len(images))
        sampled = random.sample(images, k=n)
        dataset = ImagePathDataset(sampled, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        save_dir = output_root / class_name
        acc = evaluate_and_save(
            model=model, dataloader=loader, device=device,
            eps=args.eps, steps=args.steps, step_size=args.step_size,
            class_to_idx=class_to_idx, save_dir=save_dir, save_images=True,
        )

        all_results[class_name] = acc
        tqdm.write(f"[{class_name}] Clean={acc['clean_accuracy']:.1f}%  Adv={acc['adversarial_accuracy']:.1f}%  ASR={acc['attack_success_rate']:.1f}%")

        # Save per-class JSON
        json_path = output_root / f"attack_results_{class_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"class_name": class_name, "eps": float(args.eps), "steps": args.steps,
                         "n_samples": acc["total_samples"], "checkpoint": str(Path(args.checkpoint).resolve())},
                "accuracy": acc,
            }, f, indent=2)

    # Summary
    clean_list = [r["clean_accuracy"] for r in all_results.values()]
    adv_list = [r["adversarial_accuracy"] for r in all_results.values()]
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} classes)")
    print(f"  Clean Acc:  {np.mean(clean_list):.2f}% ± {np.std(clean_list):.2f}%")
    print(f"  Adv Acc:    {np.mean(adv_list):.2f}% ± {np.std(adv_list):.2f}%")
    print(f"  ASR:        {100 - np.mean(adv_list):.2f}%")
    print(f"{'='*60}")
    print(f"Adversarial images saved to: {output_root}")


if __name__ == "__main__":
    main()
