#!/usr/bin/env python3
"""
PyTorch GPU 版本：计算对抗样本 spatial SAE 特征与类别平均 spatial 特征之间的
Cosine Similarity 和 Jaccard Similarity。

输入为压缩后的 TopK 稀疏表示，按 spatial position 分批展开为 dense 计算，
充分利用 GPU 加速，适配大规模验证。
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from collections import Counter


def build_dense_from_topk(indices: torch.Tensor, values: torch.Tensor, d_lat: int) -> torch.Tensor:
    """从 TopK 稀疏表示重建 dense 向量。"""
    shape = list(indices.shape[:-1]) + [d_lat]
    dense = torch.zeros(*shape, device=indices.device, dtype=values.dtype)
    dense.scatter_(-1, indices, values)
    return dense


def build_mask_from_topk(indices: torch.Tensor, d_lat: int) -> torch.Tensor:
    """从 TopK 索引构建 0/1 mask（用于 Jaccard）。"""
    shape = list(indices.shape[:-1]) + [d_lat]
    mask = torch.zeros(*shape, device=indices.device, dtype=torch.float32)
    mask.scatter_(-1, indices, 1.0)
    return mask


def compute_batch_similarities(
    adv_indices: torch.Tensor,       # [N, spatial_size, topk]
    adv_activations: torch.Tensor,   # [N, spatial_size, topk]
    cls_indices: torch.Tensor,       # [num_classes, spatial_size, topk]
    cls_activations: torch.Tensor,   # [num_classes, spatial_size, topk]
    d_lat: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在 GPU 上按 spatial position 分批计算 Cosine 和 Jaccard similarity。"""
    N, spatial_size, topk = adv_indices.shape
    num_classes = cls_indices.shape[0]

    sim_cosine = torch.zeros(N, num_classes, device=device, dtype=torch.float32)
    sim_jaccard = torch.zeros(N, num_classes, device=device, dtype=torch.float32)

    for p in range(spatial_size):
        adv_dense = build_dense_from_topk(
            adv_indices[:, p, :].to(device),
            adv_activations[:, p, :].to(device),
            d_lat,
        )  # [N, d_lat]
        cls_dense = build_dense_from_topk(
            cls_indices[:, p, :].to(device),
            cls_activations[:, p, :].to(device),
            d_lat,
        )  # [num_classes, d_lat]

        # Cosine similarity: [N, d_lat] @ [d_lat, num_classes] -> [N, num_classes]
        adv_norm = torch.norm(adv_dense, p=2, dim=-1, keepdim=True)      # [N, 1]
        cls_norm = torch.norm(cls_dense, p=2, dim=-1, keepdim=True)      # [num_classes, 1]
        cosine = torch.mm(adv_dense, cls_dense.t()) / (adv_norm * cls_norm.t() + 1e-8)
        sim_cosine += cosine

        # Jaccard similarity
        adv_mask = build_mask_from_topk(adv_indices[:, p, :].to(device), d_lat)   # [N, d_lat]
        cls_mask = build_mask_from_topk(cls_indices[:, p, :].to(device), d_lat)   # [num_classes, d_lat]
        intersection = torch.mm(adv_mask, cls_mask.t())                           # [N, num_classes]
        union = adv_mask.sum(-1, keepdim=True) + cls_mask.sum(-1, keepdim=True).t() - intersection
        jaccard = intersection / (union + 1e-8)
        sim_jaccard += jaccard

    sim_cosine /= spatial_size
    sim_jaccard /= spatial_size
    return sim_cosine, sim_jaccard


def get_topk_classes(similarities: torch.Tensor, class_names: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
    """对 N 张图片同时取 TopK 最近类别。"""
    topk_idx = torch.topk(similarities, k=k, dim=-1).indices.cpu().numpy()
    results = []
    for i in range(topk_idx.shape[0]):
        row = [(class_names[idx], float(similarities[i, idx].item())) for idx in topk_idx[i]]
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial cosine/jaccard similarity (PyTorch GPU)"
    )
    parser.add_argument("--adv-npz", type=str, required=True)
    parser.add_argument("--class-npz", type=str, default=str(Path(__file__).resolve().parent / "sae_stat_results_v2.npz"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading adversarial features from: {args.adv_npz}")
    adv_data = np.load(args.adv_npz, allow_pickle=True)
    adv_names = adv_data["image_names"]
    adv_indices = adv_data["spatial_indices"]
    adv_activations = adv_data["spatial_activations"]
    print(f"  Loaded {len(adv_names)} adversarial samples, shape={tuple(adv_indices.shape)}")

    print(f"Loading class statistics from: {args.class_npz}")
    cls_data = np.load(args.class_npz, allow_pickle=True)
    cls_names = list(cls_data["class_names"])
    cls_indices = cls_data["spatial_indices"]
    cls_activations = cls_data["spatial_activations"]
    d_lat = 65536
    print(f"  Loaded {len(cls_names)} classes, shape={tuple(cls_indices.shape)}, d_lat={d_lat}")

    true_class = Path(args.adv_npz).stem.replace("_spatial", "")
    print(f"\nTrue class inferred: {true_class}")

    # 转为 torch
    adv_indices_t = torch.from_numpy(adv_indices).long()
    adv_activations_t = torch.from_numpy(adv_activations).float()
    cls_indices_t = torch.from_numpy(cls_indices).long()
    cls_activations_t = torch.from_numpy(cls_activations).float()

    print(f"\nComputing similarities on GPU...")
    sim_cosine, sim_jaccard = compute_batch_similarities(
        adv_indices_t, adv_activations_t,
        cls_indices_t, cls_activations_t,
        d_lat=d_lat, device=device
    )

    print(f"  Done. Cosine shape: {tuple(sim_cosine.shape)}")

    topk_cosine = get_topk_classes(sim_cosine, cls_names, k=args.topk)
    topk_jaccard = get_topk_classes(sim_jaccard, cls_names, k=args.topk)

    results = {}
    for i, img_name in enumerate(adv_names):
        results[str(img_name)] = {
            f"top{args.topk}_cosine": [
                {"class": cls, "similarity": round(sim, 6)} for cls, sim in topk_cosine[i]
            ],
            f"top{args.topk}_jaccard": [
                {"class": cls, "similarity": round(sim, 6)} for cls, sim in topk_jaccard[i]
            ],
        }

    statistics = {}
    for metric_name, topk_list in [("cosine", topk_cosine), ("jaccard", topk_jaccard)]:
        top1_preds = [row[0][0] for row in topk_list]
        top5_preds = [[cls for cls, _ in row] for row in topk_list]
        pred_counter = Counter(top1_preds)
        total = len(topk_list)
        t1 = pred_counter.get(true_class, 0)
        t3 = sum(1 for p in top5_preds if true_class in p[:3])
        t5 = sum(1 for p in top5_preds if true_class in p[:5])

        statistics[metric_name] = {
            "top1_accuracy": round(100.0 * t1 / total, 1),
            "top3_accuracy": round(100.0 * t3 / total, 1),
            "top5_accuracy": round(100.0 * t5 / total, 1),
            "true_class_count_top1": t1,
            "true_class_count_top3": t3,
            "true_class_count_top5": t5,
            "total_images": total,
            "top1_distribution": [
                {"class": cls, "count": count, "percentage": round(100 * count / total, 1)}
                for cls, count in pred_counter.most_common(5)
            ],
        }

    print(f"\n{'='*60}")
    print("Results:")
    for mn in ["cosine", "jaccard"]:
        print(f"  {mn.upper()} Top-1: {statistics[mn]['top1_accuracy']:.1f}%")
        print(f"  {mn.upper()} Top-3: {statistics[mn]['top3_accuracy']:.1f}%")
        print(f"  {mn.upper()} Top-5: {statistics[mn]['top5_accuracy']:.1f}%")
    print(f"{'='*60}")

    output_data = {
        "meta": {
            "true_class": true_class,
            "adv_npz": str(Path(args.adv_npz).resolve()),
            "class_npz": str(Path(args.class_npz).resolve()),
            "num_images": len(adv_names),
            "num_classes": len(cls_names),
            "topk": args.topk,
        },
        "statistics": statistics,
        "per_image_results": results,
    }

    if args.output_json is None:
        output_path = Path(args.adv_npz).parent / f"eval_dis_{Path(args.adv_npz).stem}.json"
    else:
        output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
