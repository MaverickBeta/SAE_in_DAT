#!/usr/bin/env python3
"""
批量提取对抗样本 SAE latent（只加载一次模型）。
遍历指定目录下的所有类别子文件夹，为每个类别生成 *_spatial.npz。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
from sae_core.model import TopKAutoencoder

STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}


class AdvImageDataset(Dataset):
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
        return x, path.name


def build_base_model(device, checkpoint):
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


def build_sae(device, sae_ckpt_path):
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
    return sae, norm_mean, norm_std, {"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)}


def extract_single_class(model, sae_model, norm_mean, norm_std, stage_idx, image_paths, batch_size, num_workers, device, topk):
    captured = {}
    def hook_fn(_, __, output):
        captured["feat"] = output
    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = AdvImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    image_names_all = []
    spatial_indices_list = []
    spatial_activations_list = []

    with torch.no_grad():
        for images, names in tqdm(loader, desc="Extract", unit="batch", leave=False):
            images = images.to(device)
            _ = model(images)
            feat = captured["feat"]
            b, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)
            z_per_image = z.reshape(b, h * w, -1).cpu().numpy()

            for i in range(b):
                spatial_vec = z_per_image[i]
                k = min(topk, spatial_vec.shape[1])
                indices = np.argsort(spatial_vec, axis=1)[:, ::-1][:, :k]
                rows = np.arange(spatial_vec.shape[0])[:, None]
                values = spatial_vec[rows, indices]
                image_names_all.append(names[i])
                spatial_indices_list.append(indices.astype(np.int32))
                spatial_activations_list.append(values.astype(np.float32))

    handle.remove()
    return (
        image_names_all,
        np.stack(spatial_indices_list, axis=0),
        np.stack(spatial_activations_list, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Batch extract adversarial SAE latent")
    parser.add_argument("--adv-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sae-ckpt", type=str, required=True)
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-classes", type=int, default=0, help="0 = all classes")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading base model...")
    model = build_base_model(device, args.checkpoint)
    print("Loading SAE...")
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(f"Stage mismatch: stage{args.sae_stage} expects d_in={expected_din}, SAE has {sae_cfg['d_in']}")
    print(f"SAE: stage={args.sae_stage}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")

    adv_root = Path(args.adv_root)
    class_dirs = sorted([d for d in adv_root.iterdir() if d.is_dir()])
    if args.n_classes > 0:
        class_dirs = class_dirs[:args.n_classes]
    print(f"Found {len(class_dirs)} class directories to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    for class_dir in tqdm(class_dirs, desc="Classes", unit="class"):
        class_name = class_dir.name
        out_npz = output_dir / f"{class_name}_spatial.npz"
        if out_npz.exists():
            tqdm.write(f"[Skip] {class_name} already exists")
            continue

        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts])
        if not image_paths:
            tqdm.write(f"[Skip] {class_name}: no images")
            continue

        names, indices, activations = extract_single_class(
            model=model, sae_model=sae_model, norm_mean=norm_mean, norm_std=norm_std,
            stage_idx=args.sae_stage, image_paths=image_paths, batch_size=args.batch_size,
            num_workers=args.num_workers, device=device, topk=args.topk,
        )

        np.savez_compressed(
            out_npz,
            image_names=np.array(names, dtype=object),
            spatial_indices=indices,
            spatial_activations=activations,
        )
        tqdm.write(f"[Saved] {class_name}: {indices.shape} -> {out_npz.name}")

    print(f"\nAll done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
