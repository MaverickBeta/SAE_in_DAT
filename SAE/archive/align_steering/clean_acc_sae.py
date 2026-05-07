#!/usr/bin/env python3
"""
测试干净样本在 SAE latent space 中的 Cosine/Jaccard TopK 准确率。

直接从 ImageNet val 集采样干净图片，通过挂载 SAE 的模型提取 spatial TopK 特征，
与 train set 统计的 class statistics 比较 cosine/jaccard similarity，
输出 Top1/Top3/Top5 准确率。同时报告原始分类头的 clean accuracy 作为对比。
"""

import argparse
import json
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


STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}


class ImagePathDataset(Dataset):
    """图像路径数据集"""
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
    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def sample_images_from_class(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        return []
    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


def extract_clean_sae_features(
    model: torch.nn.Module,
    sae_model: TopKAutoencoder,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    dataloader: DataLoader,
    device: torch.device,
    topk: int = 64,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    提取干净样本的 SAE spatial TopK 特征。

    Returns:
        image_names: 图片文件名列表 (N,)
        class_names: 类别名称列表 (N,)
        spatial_indices: [N, H*W, topk] int32
        spatial_activations: [N, H*W, topk] float32
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    image_names_all: List[str] = []
    class_names_all: List[str] = []
    spatial_indices_list: List[np.ndarray] = []
    spatial_activations_list: List[np.ndarray] = []

    with torch.no_grad():
        for images, class_names, image_names in tqdm(dataloader, desc="Extracting clean SAE features"):
            images = images.to(device)
            batch_size = images.size(0)

            # 同时获取模型原始分类结果
            logits = model(images)
            preds = logits.argmax(dim=1)

            feat = captured["feat"]
            b, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)  # [B*H*W, d_lat]

            z_per_image = z.reshape(batch_size, h * w, -1).cpu().numpy()

            for i in range(batch_size):
                spatial_vec = z_per_image[i]  # [H*W, d_lat]
                k = min(topk, spatial_vec.shape[1])
                indices = np.argsort(spatial_vec, axis=1)[:, ::-1][:, :k]
                rows = np.arange(spatial_vec.shape[0])[:, None]
                values = spatial_vec[rows, indices]

                image_names_all.append(image_names[i])
                class_names_all.append(class_names[i])
                spatial_indices_list.append(indices.astype(np.int32))
                spatial_activations_list.append(values.astype(np.float32))

    handle.remove()
    return (
        image_names_all,
        class_names_all,
        np.stack(spatial_indices_list, axis=0),
        np.stack(spatial_activations_list, axis=0),
    )


def build_dense_from_topk(indices: torch.Tensor, values: torch.Tensor, d_lat: int) -> torch.Tensor:
    shape = list(indices.shape[:-1]) + [d_lat]
    dense = torch.zeros(*shape, device=indices.device, dtype=values.dtype)
    dense.scatter_(-1, indices, values)
    return dense


def compute_batch_similarities(
    adv_indices: torch.Tensor,
    adv_activations: torch.Tensor,
    cls_indices: torch.Tensor,
    cls_activations: torch.Tensor,
    d_lat: int,
    device: torch.device,
) -> torch.Tensor:
    """计算 Cosine Similarity，返回 [N, num_classes]"""
    N, spatial_size, topk = adv_indices.shape
    num_classes = cls_indices.shape[0]

    sim_cosine = torch.zeros(N, num_classes, device=device, dtype=torch.float32)

    for p in range(spatial_size):
        adv_dense = build_dense_from_topk(
            adv_indices[:, p, :].to(device),
            adv_activations[:, p, :].to(device),
            d_lat,
        )
        cls_dense = build_dense_from_topk(
            cls_indices[:, p, :].to(device),
            cls_activations[:, p, :].to(device),
            d_lat,
        )

        adv_norm = torch.norm(adv_dense, p=2, dim=-1, keepdim=True)
        cls_norm = torch.norm(cls_dense, p=2, dim=-1, keepdim=True)
        cosine = torch.mm(adv_dense, cls_dense.t()) / (adv_norm * cls_norm.t() + 1e-8)
        sim_cosine += cosine

    sim_cosine /= spatial_size
    return sim_cosine


def main():
    parser = argparse.ArgumentParser(
        description="Clean Sample SAE Cosine TopK Accuracy Evaluation"
    )
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"),
                        help="ImageNet validation directory")
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
                        help="Model checkpoint path")
    parser.add_argument("--gpu", type=int, default=1, help="GPU index")
    parser.add_argument("--class-name", type=str, default=None,
                        help="指定类别名称 (如 n01440764)，不指定则随机选择")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="每个类别采样的图片数量")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-ckpt", type=str,
                        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"))
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--class-npz", type=str,
                        default=str(Path(__file__).resolve().parent / "sae_stat_results_v2.npz"),
                        help="Class statistics NPZ path (from sae_stat.py)")
    parser.add_argument("--topk", type=int, default=64,
                        help="SAE TopK extraction")
    parser.add_argument("--output-json", type=str, default=None,
                        help="输出 JSON 路径 (默认自动生成)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + SAE
    print(f"Loading model from: {args.checkpoint}")
    model = build_model(device=device, checkpoint=args.checkpoint)

    print(f"Loading SAE from: {args.sae_ckpt}")
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)
    print(f"SAE loaded: d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}")

    expected_din = STAGE_DIN_MAP[args.sae_stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(f"Stage/SAE mismatch: stage{args.sae_stage} expects d_in={expected_din}, SAE has d_in={sae_cfg['d_in']}")

    # Mount SAE hook
    sae_hook_handle = model.stages[args.sae_stage].register_forward_hook(
        make_sae_hook(sae_model, norm_mean, norm_std)
    )
    print(f"SAE mounted at stage {args.sae_stage} (no steering).")

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
        print(f"Sampled {len(image_paths)} clean images from class {selected_class}")

        if not image_paths:
            raise RuntimeError(f"No images found in {class_dir}")

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        dataset = ImagePathDataset(image_paths, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

        # ========== Step 1: Extract SAE features & compute model clean accuracy ==========
        print(f"\n{'='*60}")
        print("Step 1: Extracting SAE features and computing model clean accuracy...")
        print(f"{'='*60}")

        # We need to compute both: model clean acc (with SAE hook active) and SAE features
        image_names_all, class_names_all, spatial_indices, spatial_activations = \
            extract_clean_sae_features(
                model=model,
                sae_model=sae_model,
                norm_mean=norm_mean,
                norm_std=norm_std,
                stage_idx=args.sae_stage,
                dataloader=dataloader,
                device=device,
                topk=args.topk,
            )

        # Re-run model forward to get clean accuracy (with hook still active)
        clean_correct = 0
        total = 0
        with torch.no_grad():
            for images, class_names, _ in dataloader:
                labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                clean_correct += (preds == labels).sum().item()
                total += images.size(0)

        model_clean_acc = 100.0 * clean_correct / total
        print(f"\nModel Clean Accuracy (with SAE hook): {model_clean_acc:.2f}%")

        # ========== Step 2: Load class statistics ==========
        print(f"\n{'='*60}")
        print("Step 2: Loading class statistics from train set...")
        print(f"{'='*60}")

        cls_data = np.load(args.class_npz, allow_pickle=True)
        cls_names = list(cls_data["class_names"])
        cls_indices = cls_data["spatial_indices"]         # [1000, H*W, topk]
        cls_activations = cls_data["spatial_activations"] # [1000, H*W, topk]
        d_lat = int(sae_cfg["d_lat"])
        print(f"Loaded {len(cls_names)} classes, spatial shape={tuple(cls_indices.shape)}, d_lat={d_lat}")

        # ========== Step 3: Compute cosine similarities ==========
        print(f"\n{'='*60}")
        print("Step 3: Computing cosine similarities...")
        print(f"{'='*60}")

        adv_indices_t = torch.from_numpy(spatial_indices).long()
        adv_activations_t = torch.from_numpy(spatial_activations).float()
        cls_indices_t = torch.from_numpy(cls_indices).long()
        cls_activations_t = torch.from_numpy(cls_activations).float()

        sim_cosine = compute_batch_similarities(
            adv_indices_t, adv_activations_t,
            cls_indices_t, cls_activations_t,
            d_lat=d_lat, device=device
        )

        # ========== Step 4: Evaluate TopK ==========
        print(f"\n{'='*60}")
        print("Step 4: Evaluating TopK accuracy...")
        print(f"{'='*60}")

        # Get Top5 predictions for each image
        top5_values, top5_indices = torch.topk(sim_cosine, k=5, dim=-1)
        top5_indices = top5_indices.cpu().numpy()
        top5_values = top5_values.cpu().numpy()

        true_class = selected_class
        total_imgs = len(image_names_all)

        t1 = 0
        t3 = 0
        t5 = 0

        per_image_results = []
        for i in range(total_imgs):
            pred_classes = [cls_names[idx] for idx in top5_indices[i]]
            pred_sims = [float(top5_values[i][j]) for j in range(5)]

            is_t1 = pred_classes[0] == true_class
            is_t3 = true_class in pred_classes[:3]
            is_t5 = true_class in pred_classes[:5]

            if is_t1:
                t1 += 1
            if is_t3:
                t3 += 1
            if is_t5:
                t5 += 1

            per_image_results.append({
                "image": image_names_all[i],
                "top5_cosine": [
                    {"class": c, "similarity": round(s, 6)} for c, s in zip(pred_classes, pred_sims)
                ],
                "is_correct_top1": is_t1,
            })

        cosine_top1 = 100.0 * t1 / total_imgs
        cosine_top3 = 100.0 * t3 / total_imgs
        cosine_top5 = 100.0 * t5 / total_imgs

        print(f"\n{'='*60}")
        print(f"RESULTS for class: {selected_class} ({total_imgs} images)")
        print(f"{'='*60}")
        print(f"  Model Clean Accuracy (with SAE):  {model_clean_acc:.2f}%")
        print(f"  SAE Cosine Top-1:                 {cosine_top1:.2f}%")
        print(f"  SAE Cosine Top-3:                 {cosine_top3:.2f}%")
        print(f"  SAE Cosine Top-5:                 {cosine_top5:.2f}%")
        print(f"{'='*60}")
        print(f"  Gap (Model Acc - Cos Top1):       {model_clean_acc - cosine_top1:.2f}%")
        print(f"{'='*60}")

        # ========== Save results ==========
        output_data = {
            "meta": {
                "class_name": selected_class,
                "class_index": int(selected_class_idx),
                "n_samples": total_imgs,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "sae_ckpt": str(Path(args.sae_ckpt).resolve()),
                "sae_stage": args.sae_stage,
                "class_npz": str(Path(args.class_npz).resolve()),
                "seed": args.seed,
            },
            "accuracy": {
                "model_clean_accuracy": round(model_clean_acc, 2),
                "sae_cosine_top1": round(cosine_top1, 2),
                "sae_cosine_top3": round(cosine_top3, 2),
                "sae_cosine_top5": round(cosine_top5, 2),
                "gap_model_minus_cos_top1": round(model_clean_acc - cosine_top1, 2),
            },
            "per_image": per_image_results,
        }

        if args.output_json is None:
            output_json_path = Path(__file__).resolve().parent / "clean_sae_results" / f"clean_acc_{selected_class}.json"
        else:
            output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json_path}")

    finally:
        if sae_hook_handle is not None:
            sae_hook_handle.remove()
            print("SAE hook removed.")


if __name__ == "__main__":
    main()
