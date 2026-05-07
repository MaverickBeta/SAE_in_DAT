#!/usr/bin/env python3
"""
SAE Reconstruction Fidelity Test for WRN34x10 on CIFAR-10.

Tests whether passing activations through SAE reconstruction degrades
classification accuracy on CIFAR-10 test set.

Usage:
    python eval_reconstruction_fidelity.py \
        --sae_ckpt /path/to/epoch_8.pt \
        --gpu_id 0
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.models.wide_resnet_innoutrobustness import WideResNet34x10
from sae_core.model import TopKAutoencoder


class WRNWithSAE(torch.nn.Module):
    """Wrap WRN to insert SAE reconstruction at the activation layer."""

    def __init__(self, base_model, sae, norm_mean, norm_std):
        super().__init__()
        self.base_model = base_model
        self.sae = sae
        self.register_buffer("norm_mean", norm_mean)
        self.register_buffer("norm_std", norm_std)
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

    def forward(self, x):
        # --- Front half: up to activation layer ---
        x = self.base_model._normalize_input(x)
        out = self.base_model.conv1(x)
        out = self.base_model.block1(out)
        out = self.base_model.block2(out)
        out = self.base_model.block3(out)
        out = self.base_model.activation(self.base_model.bn1(out))  # [B, 640, 8, 8]

        # --- SAE reconstruction ---
        b, c, h, w = out.shape
        flat = out.permute(0, 2, 3, 1).reshape(-1, c)  # [B*64, 640]
        flat_norm = (flat - self.norm_mean) / self.norm_std
        flat_recon, _, _ = self.sae(flat_norm)
        flat_recon = flat_recon * self.norm_std + self.norm_mean
        out = flat_recon.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # --- Back half: pool + classifier ---
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.base_model.nChannels)
        out = self.base_model.fc(out)
        return out


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="SAE reconstruction fidelity test on CIFAR-10")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="/Data_share/hongyi/DAT/data")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth",
    )
    parser.add_argument("--sae_ckpt", type=str, required=True, help="Path to SAE checkpoint")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CIFAR-10 test set (auto-download if not present)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset: CIFAR-10 test set, {len(test_dataset)} images")

    # Load base WRN model
    print(f"\n[1/3] Loading base model: {args.model_ckpt}")
    base_model = WideResNet34x10(
        num_classes=10,
        activation="relu",
        dropRate=0.0,
        return_feature_map=False,
        normalize_input=True,
        use_batchnorm=True,
    )
    base_model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))
    base_model = base_model.to(device).eval()

    # Baseline accuracy
    print("[2/3] Baseline accuracy (no SAE)...")
    baseline_acc = evaluate(base_model, test_loader, device)
    print(f"  Baseline accuracy: {baseline_acc:.4f} ({baseline_acc:.2%})")

    # Load SAE
    print(f"\n[3/3] Loading SAE: {args.sae_ckpt}")
    sae_ckpt = torch.load(args.sae_ckpt, map_location="cpu")
    config = sae_ckpt.get("config", {})
    d_in = sae_ckpt.get("d_in", config.get("d_in"))
    d_lat = sae_ckpt.get("d_lat", config.get("d_lat"))
    k = sae_ckpt.get("k", config.get("k"))
    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(sae_ckpt["model_state_dict"])
    sae = sae.to(device).eval()
    norm_mean = sae_ckpt["norm_mean"].to(device)
    norm_std = sae_ckpt["norm_std"].to(device)
    print(f"  SAE config: d_in={d_in}, d_lat={d_lat}, k={k}")

    # Wrapped model with SAE reconstruction
    print("\n[4/4] Accuracy with SAE reconstruction...")
    wrapped_model = WRNWithSAE(base_model, sae, norm_mean, norm_std).to(device)
    sae_acc = evaluate(wrapped_model, test_loader, device)
    print(f"  SAE accuracy:      {sae_acc:.4f} ({sae_acc:.2%})")

    # Summary
    drop = baseline_acc - sae_acc
    print(f"\n{'='*50}")
    print(f"  Baseline:  {baseline_acc:.4f} ({baseline_acc:.2%})")
    print(f"  SAE recon: {sae_acc:.4f} ({sae_acc:.2%})")
    print(f"  Drop:      {drop:.4f} ({drop*100:.2f} pp)")
    print(f"{'='*50}")
    if drop < 0.005:
        print("  ✅ SAE reconstruction fidelity is excellent (< 0.5% drop)")
    elif drop < 0.01:
        print("  ⚠️  SAE reconstruction has minor degradation (0.5-1% drop)")
    else:
        print("  ❌ SAE reconstruction significantly degrades accuracy (> 1% drop)")


if __name__ == "__main__":
    main()
