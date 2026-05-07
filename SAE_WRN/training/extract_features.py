#!/usr/bin/env python3
"""
Extract activation features from WideResNet34x10 (DAT checkpoint) on CIFAR-10.

Hook location: model.activation (post-block3, pre-global-pool)
Output shape: [N, 640, 8, 8] per class
"""

import argparse
import gc
import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT")
from rebm.models.wide_resnet_innoutrobustness import WideResNet34x10


def parse_args():
    parser = argparse.ArgumentParser(description="Extract WRN34x10 activation features for CIFAR-10 SAE")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="/Data_share/hongyi/DAT/data")
    parser.add_argument("--out_dir", type=str, default="/Data_share/hongyi/DAT/SAE_WRN/features")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--build_merged", action="store_true", help="Also build a single merged.pt file")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.out_dir}")

    # CIFAR-10 transform: model has internal normalization, so only ToTensor here
    transform = transforms.Compose([transforms.ToTensor()])

    print(f"Loading CIFAR-10 (download if not exists at {args.data_root})...")
    dataset = CIFAR10(root=args.data_root, train=(args.split == "train"), download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataset loaded: {len(dataset)} images, {len(dataset.classes)} classes")

    # Load model
    print(f"Loading checkpoint: {args.ckpt_path}")
    model = WideResNet34x10(
        num_classes=10,
        activation="relu",
        dropRate=0.0,
        return_feature_map=False,
        normalize_input=True,
        use_batchnorm=True,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Hook activation layer (post-block3, pre-avg_pool)
    captured = {}

    def hook_fn(_module, _input, output):
        captured["feat"] = output.detach().cpu()  # [B, 640, 8, 8]

    handle = model.activation.register_forward_hook(hook_fn)

    os.makedirs(args.out_dir, exist_ok=True)

    class_names = dataset.classes  # ['airplane', 'automobile', ...]
    class_buffers = {i: [] for i in range(10)}

    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extract"):
            images = images.to(device)
            _ = model(images)

            feat = captured["feat"]  # [B, 640, 8, 8]
            for i in range(feat.size(0)):
                label = labels[i].item()
                class_buffers[label].append(feat[i].clone())

    handle.remove()

    # Save per-class features
    print("\nSaving per-class features...")
    for cls_idx in range(10):
        if not class_buffers[cls_idx]:
            continue
        tensor = torch.stack(class_buffers[cls_idx])  # [N_cls, 640, 8, 8]
        out_path = os.path.join(args.out_dir, f"{class_names[cls_idx]}.pt")
        torch.save(tensor, out_path)
        print(f"  {class_names[cls_idx]:12s}: {tuple(tensor.shape)} -> {out_path}")

    # Optionally build merged file
    if args.build_merged:
        print("\nBuilding merged file...")
        all_tensors = []
        for cls_idx in range(10):
            if class_buffers[cls_idx]:
                all_tensors.append(torch.stack(class_buffers[cls_idx]))
        merged = torch.cat(all_tensors, dim=0)  # [N_total, 640, 8, 8]
        merged_path = os.path.join(args.out_dir, "merged.pt")
        torch.save({"features": merged}, merged_path)
        print(f"  Merged: {tuple(merged.shape)} -> {merged_path}")
        del merged
        gc.collect()

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
