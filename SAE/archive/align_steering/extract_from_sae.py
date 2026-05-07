#!/usr/bin/env python3
"""
加载对抗样本图片，通过挂载 SAE 的模型提取 Top64 激活特征的索引和激活值。
"""

import argparse
import json
import os
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
from sae_core.model import TopKAutoencoder


STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}


class AdvImageDataset(Dataset):
    """对抗样本图片数据集"""
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
        return x, str(path.name)


def build_base_model(device: torch.device, checkpoint: str):
    """构建基础 ConvNeXT 模型"""
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


def infer_condition_dir(adv_dir: Path) -> str:
    """根据对抗样本目录路径推断条件子目录。
    支持 baseline (adv_samples/no_sae/) 和 AA (adv_samples/aa/no_sae/)。
    """
    path_str = str(adv_dir).replace("\\", "/")
    is_aa = "/aa/" in path_str or "adv_samples/aa" in path_str
    prefix = "aa_" if is_aa else ""

    if "no_sae" in path_str:
        return f"{prefix}no_sae"
    if "sae_stage" in path_str:
        import re
        m = re.search(r"sae_stage(\d+)", path_str)
        if m:
            return f"{prefix}sae_stage{m.group(1)}"
        return f"{prefix}sae_stage3"
    return f"{prefix}other"


def build_sae(device: torch.device, sae_ckpt_path: str):
    """构建 SAE 模型"""
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


def extract_sae_features(
    model: torch.nn.Module,
    sae_model: TopKAutoencoder,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    dataloader: DataLoader,
    device: torch.device,
    topk: int = 64,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    提取每张图片的 SAE spatial TopK 特征

    Returns:
        image_names: 图片文件名列表 (N,)
        spatial_indices: [N, H*W, topk] int32 数组
        spatial_activations: [N, H*W, topk] float32 数组
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    image_names_all: List[str] = []
    spatial_indices_list: List[np.ndarray] = []
    spatial_activations_list: List[np.ndarray] = []

    with torch.no_grad():
        for images, image_names in tqdm(dataloader, desc="Extracting SAE spatial features"):
            images = images.to(device)
            batch_size = images.size(0)

            _ = model(images)
            feat = captured["feat"]

            b, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)  # [B*H*W, d_lat]

            z_per_image = z.reshape(batch_size, h * w, -1).cpu().numpy()  # [B, H*W, d_lat]

            for i in range(batch_size):
                img_name = image_names[i]
                spatial_vec = z_per_image[i]  # [H*W, d_lat]

                k = min(topk, spatial_vec.shape[1])
                indices = np.argsort(spatial_vec, axis=1)[:, ::-1][:, :k]  # [H*W, topk]
                # 使用 advanced indexing提取对应值
                rows = np.arange(spatial_vec.shape[0])[:, None]
                values = spatial_vec[rows, indices]  # [H*W, topk]

                image_names_all.append(img_name)
                spatial_indices_list.append(indices.astype(np.int32))
                spatial_activations_list.append(values.astype(np.float32))

    handle.remove()
    return (
        image_names_all,
        np.stack(spatial_indices_list, axis=0),
        np.stack(spatial_activations_list, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract SAE TopK features from adversarial samples"
    )
    parser.add_argument(
        "--adv-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "n01440764"),
        help="对抗样本目录"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
        help="基础模型检查点路径"
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
        help="SAE 对应的 ConvNeXT stage (默认 3)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=64,
        help="提取的 TopK 特征数量 (默认 64)"
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
        "--gpu",
        type=int,
        default=1,
        help="使用的 GPU 编号 (默认 1)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：输出 JSON 文件路径 (供人阅读)"
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="输出 NPZ 文件路径 (默认自动生成，推荐用于后续计算)"
    )

    args = parser.parse_args()

    # 设备设置
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0")
            args.gpu = 0
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 检查对抗样本目录
    adv_dir = Path(args.adv_dir).resolve()
    if not adv_dir.exists():
        raise FileNotFoundError(f"Adversarial samples directory not found: {adv_dir}")

    # 获取所有对抗样本图片
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_paths = sorted([p for p in adv_dir.iterdir() if p.is_file() and p.suffix in exts])

    if not image_paths:
        raise RuntimeError(f"No images found in {adv_dir}")

    print(f"Found {len(image_paths)} adversarial samples in {adv_dir}")

    # 加载模型
    print(f"Loading base model from: {args.checkpoint}")
    model = build_base_model(device=device, checkpoint=args.checkpoint)

    print(f"Loading SAE from: {args.sae_ckpt}")
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(
        device=device, sae_ckpt_path=args.sae_ckpt
    )

    # 验证 stage 匹配
    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.sae_stage} expects d_in={expected_din}, "
            f"SAE has d_in={sae_cfg['d_in']}"
        )

    print(
        f"SAE loaded: stage={args.sae_stage}, d_in={sae_cfg['d_in']}, "
        f"d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    # 数据预处理
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    # 创建数据集和数据加载器
    dataset = AdvImageDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 提取 SAE 特征
    print(f"\nExtracting Top-{args.topk} spatial SAE features...")
    image_names, spatial_indices, spatial_activations = extract_sae_features(
        model=model,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        stage_idx=args.sae_stage,
        dataloader=dataloader,
        device=device,
        topk=args.topk,
    )

    class_name = adv_dir.name
    spatial_size = spatial_indices.shape[1]

    # 默认 NPZ 路径
    condition = infer_condition_dir(adv_dir)
    if args.output_npz is None:
        npz_path = (
            Path(__file__).resolve().parent
            / "adv_samples"
            / "sae_latent"
            / condition
            / f"{class_name}_spatial.npz"
        )
    else:
        npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        image_names=np.array(image_names, dtype=object),
        spatial_indices=spatial_indices,          # [N, H*W, topk]
        spatial_activations=spatial_activations,  # [N, H*W, topk]
    )
    print(f"\nSaved NPZ: {npz_path}")
    print(f"  Shape: {spatial_indices.shape} (images={len(image_names)}, spatial={spatial_size}, topk={args.topk})")

    # 可选 JSON
    if args.output_json is not None:
        json_results = {}
        for i, img_name in enumerate(image_names):
            json_results[img_name] = {
                "spatial_indices": spatial_indices[i].tolist(),
                "spatial_activations": spatial_activations[i].tolist(),
            }
        output_data = {
            "meta": {
                "adv_dir": str(adv_dir),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "sae_ckpt": str(Path(args.sae_ckpt).resolve()),
                "sae_stage": int(args.sae_stage),
                "sae_d_in": int(sae_cfg["d_in"]),
                "sae_d_lat": int(sae_cfg["d_lat"]),
                "sae_k": int(sae_cfg["k"]),
                "topk_extracted": int(args.topk),
                "num_images": len(image_names),
                "spatial_size": int(spatial_size),
            },
            "features": json_results,
        }
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved JSON: {json_path}")

    print(f"\n{'='*60}")
    print(f"Total images processed: {len(image_names)}")
    print(f"{'='*60}")
    print("\nDone!")


if __name__ == "__main__":
    main()
