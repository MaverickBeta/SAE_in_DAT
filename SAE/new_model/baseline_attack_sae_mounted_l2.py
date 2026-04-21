#!/usr/bin/env python3
"""
L2 PGD-CE 攻击挂载 SAE 的原始模型（保留原始 FC Head）。

SAE 挂载在 stage 3 后，做 encode/decode（无 steering），
然后走原始 norm_pre + head 输出 logits。
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

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

from rebm.attacks.attack_steps import L2Step
from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


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
        return x, class_name, str(path.name)


def get_class_dirs(val_dir: Path) -> List[Path]:
    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def sample_images(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        return []
    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


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
    """生成一个 forward hook，在 stage 后挂载 SAE encode/decode（无 steering）。"""
    def hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        x_reconstruct, _, _ = sae_model(flat_norm)
        x_out = x_reconstruct * norm_std + norm_mean
        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return hook


def l2_pgd_ce_attack(model, x, labels, eps=3.0, steps=110, step_size=None, random_start=True):
    """L2 PGD-CE：使用 L2Step 进行攻击。"""
    x0 = x.clone().detach()
    if step_size is None:
        step_size = eps / 4

    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    step = L2Step(eps=eps, orig_input=x0, step_size=step_size)

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


def evaluate(model, dataloader, device, eps, steps, step_size, class_to_idx):
    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, class_names, _ in tqdm(dataloader, desc="Evaluating"):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)
        B = images.size(0)
        total += B

        # Clean
        with torch.no_grad():
            logits_clean = model(images)
            preds_clean = logits_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()

        # Attack
        x_adv = l2_pgd_ce_attack(model, images, labels, eps, steps, step_size)

        # Adv
        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()

    return {
        "clean_accuracy": 100.0 * clean_correct / total,
        "adversarial_accuracy": 100.0 * adv_correct / total,
        "total_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(description="L2 PGD-CE on SAE-mounted model (original head)")
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--eps", type=float, default=3.0,
                        help="L2 perturbation budget (default 3.0, matching training config)")
    parser.add_argument("--steps", type=int, default=110,
                        help="PGD steps (default 110, matching training config)")
    parser.add_argument("--step-size", type=float, default=3.0,
                        help="L2 step size (default 3.0, matching training config)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-ckpt", type=str, default=None)
    parser.add_argument("--sae-stage", type=int, default=3,
                        help="Which stage to mount SAE after (default: 3)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip class if output JSON already exists")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples_l2_sae_mounted"))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print(f"Loading model from: {args.checkpoint}")
    model = build_model(device=device, checkpoint=args.checkpoint)

    # Mount SAE
    sae_hook_handle = None
    if args.sae_ckpt:
        print(f"Loading SAE from: {args.sae_ckpt} (stage={args.sae_stage})")
        sae_model, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)
        print(f"SAE loaded: d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")
        sae_hook_handle = model.stages[args.sae_stage].register_forward_hook(
            make_sae_hook(sae_model, norm_mean, norm_std)
        )
        print("SAE mounted (no steering).")
    else:
        print("No SAE checkpoint provided, running baseline model.")

    try:
        # Select class
        val_dir = Path(args.val_dir).resolve()
        class_dirs = get_class_dirs(val_dir)
        class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        if args.class_name:
            selected_class = args.class_name
            if selected_class not in class_to_idx:
                raise ValueError(f"Class {selected_class} not found")
        else:
            selected_class = random.choice([d.name for d in class_dirs])

        selected_class_idx = class_to_idx[selected_class]
        print(f"\nSelected class: {selected_class} (index: {selected_class_idx})")

        image_paths = sample_images(val_dir / selected_class, args.n_samples)
        print(f"Sampled {len(image_paths)} images")

        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        dataset = ImagePathDataset(image_paths, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        # Evaluate
        print(f"\n{'='*60}")
        print(f"L2 PGD-CE on {'SAE-mounted' if args.sae_ckpt else 'baseline'} model")
        print(f"  L2 Epsilon: {args.eps}")
        print(f"  Steps: {args.steps}")
        print(f"  Step size: {args.step_size}")
        print(f"{'='*60}")

        results = evaluate(model, dataloader, device, args.eps, args.steps, args.step_size, class_to_idx)

        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"  Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success Rate: {100 - results['adversarial_accuracy']:.2f}%")
        print(f"{'='*60}")

        # Save
        output_data = {
            "meta": {
                "class_name": selected_class,
                "class_index": int(selected_class_idx),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "sae_ckpt": str(Path(args.sae_ckpt).resolve()) if args.sae_ckpt else None,
                "sae_stage": args.sae_stage,
                "eps": float(args.eps),
                "steps": args.steps,
                "step_size": args.step_size,
                "n_samples": results["total_samples"],
                "seed": args.seed,
            },
            "accuracy": {
                "clean_accuracy": float(results["clean_accuracy"]),
                "adversarial_accuracy": float(results["adversarial_accuracy"]),
                "attack_success_rate": float(100 - results["adversarial_accuracy"]),
                "total_samples": int(results["total_samples"]),
            }
        }

        if args.output_json is None:
            out_root = Path(args.output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            output_json_path = out_root / f"sae_mounted_l2_attack_results_{selected_class}.json"
        else:
            output_json_path = Path(args.output_json)

        if args.skip_existing and output_json_path.exists():
            print(f"\n[SKIP] {output_json_path} already exists, skipping.")
            return

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json_path}")
        print("\nDone!")

    finally:
        if sae_hook_handle is not None:
            sae_hook_handle.remove()
            print("SAE hook removed.")


if __name__ == "__main__":
    main()
