#!/usr/bin/env python3
"""
批量评估：对 adv_latent_dir 下的每个类别，分别用 class_stats_top64 和 top128 评估，
最后汇总所有类别的平均 Top-1/3/5 准确率。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from collections import Counter


def sparse_cosine_similarity(adv_indices, adv_activations, cls_indices, cls_activations, device, d_lat):
    N, P, k = adv_indices.shape
    C = cls_indices.shape[0]
    cls_norms = torch.norm(cls_activations, dim=-1).to(device)
    cosine_sum = torch.zeros(N, C, device=device, dtype=torch.float32)
    adv_idx_t = adv_indices.long().to(device)
    adv_val_t = adv_activations.to(device)
    cls_idx_t = cls_indices.long().to(device)
    cls_val_t = cls_activations.to(device)

    for p in range(P):
        cls_dense = torch.zeros(C, d_lat, device=device, dtype=torch.float32)
        cls_dense.scatter_(1, cls_idx_t[:, p, :], cls_val_t[:, p, :])
        dot = torch.zeros(N, C, device=device, dtype=torch.float32)
        for j in range(k):
            idx_j = adv_idx_t[:, p, j]
            val_j = adv_val_t[:, p, j]
            gathered = cls_dense[:, idx_j]
            dot += gathered.t() * val_j.unsqueeze(1)
        adv_norm = torch.norm(adv_val_t[:, p, :], dim=1)
        cosine_sum += dot / (adv_norm.unsqueeze(1) * cls_norms[:, p].unsqueeze(0) + 1e-8)

    return cosine_sum / P


def evaluate_single(adv_npz_path, cls_indices, cls_activations, cls_names, device, d_lat):
    adv_data = np.load(adv_npz_path, allow_pickle=True)
    adv_names = adv_data["image_names"]
    adv_indices = adv_data["spatial_indices"]
    adv_activations = adv_data["spatial_activations"]
    true_class = adv_npz_path.stem.replace("_spatial", "")

    adv_idx_t = torch.from_numpy(adv_indices).long()
    adv_val_t = torch.from_numpy(adv_activations).float()

    sim = sparse_cosine_similarity(adv_idx_t, adv_val_t, cls_indices, cls_activations, device, d_lat)
    top5_idx = torch.topk(sim, k=5, dim=-1).indices.cpu().numpy()

    top1_preds = [cls_names[top5_idx[i][0]] for i in range(len(adv_names))]
    top5_preds = [[cls_names[idx] for idx in top5_idx[i]] for i in range(len(adv_names))]
    pred_counter = Counter(top1_preds)
    total = len(adv_names)

    t1 = pred_counter.get(true_class, 0)
    t3 = sum(1 for p in top5_preds if true_class in p[:3])
    t5 = sum(1 for p in top5_preds if true_class in p[:5])

    return {
        "true_class": true_class,
        "num_images": total,
        "top1_accuracy": 100.0 * t1 / total,
        "top3_accuracy": 100.0 * t3 / total,
        "top5_accuracy": 100.0 * t5 / total,
        "top1_count": t1,
        "top3_count": t3,
        "top5_count": t5,
        "top1_distribution": [
            {"class": cls, "count": cnt, "percentage": 100.0 * cnt / total}
            for cls, cnt in pred_counter.most_common(5)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Batch eval top64 vs top128 class stats")
    parser.add_argument("--adv-latent-dir", type=str, required=True)
    parser.add_argument("--class-top64", type=str, required=True)
    parser.add_argument("--class-top128", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class stats (once each)
    print("Loading class stats (top64)...")
    cls64 = np.load(args.class_top64, allow_pickle=True)
    cls64_idx = torch.from_numpy(cls64["spatial_indices"]).long()
    cls64_val = torch.from_numpy(cls64["spatial_activations"]).float()
    cls_names = list(cls64["class_names"])
    d_lat64 = int(cls64_idx.max()) + 1
    print(f"  top64: {cls64_idx.shape}, d_lat={d_lat64}")

    print("Loading class stats (top128)...")
    cls128 = np.load(args.class_top128, allow_pickle=True)
    cls128_idx = torch.from_numpy(cls128["spatial_indices"]).long()
    cls128_val = torch.from_numpy(cls128["spatial_activations"]).float()
    d_lat128 = int(cls128_idx.max()) + 1
    print(f"  top128: {cls128_idx.shape}, d_lat={d_lat128}")

    adv_dir = Path(args.adv_latent_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adv_npzs = sorted(adv_dir.glob("*_spatial.npz"))
    print(f"\nFound {len(adv_npzs)} adversarial npz files to evaluate\n")

    rows64 = []
    rows128 = []

    for adv_npz in adv_npzs:
        class_name = adv_npz.stem.replace("_spatial", "")
        print(f"[{class_name}] Evaluating...")

        res64 = evaluate_single(adv_npz, cls64_idx, cls64_val, cls_names, device, d_lat64)
        res128 = evaluate_single(adv_npz, cls128_idx, cls128_val, cls_names, device, d_lat128)

        rows64.append(res64)
        rows128.append(res128)

        # Save per-class JSON
        with open(output_dir / f"eval_top64_{class_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"class_npz": str(Path(args.class_top64).resolve()), "adv_npz": str(adv_npz.resolve()), "num_images": res64["num_images"]},
                "statistics": {"cosine": res64},
            }, f, indent=2)
        with open(output_dir / f"eval_top128_{class_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"class_npz": str(Path(args.class_top128).resolve()), "adv_npz": str(adv_npz.resolve()), "num_images": res128["num_images"]},
                "statistics": {"cosine": res128},
            }, f, indent=2)

    # Aggregate
    def agg(rows):
        n = len(rows)
        return {
            "num_classes": n,
            "top1": sum(r["top1_accuracy"] for r in rows) / n,
            "top3": sum(r["top3_accuracy"] for r in rows) / n,
            "top5": sum(r["top5_accuracy"] for r in rows) / n,
            "top1_std": (sum((r["top1_accuracy"] - sum(x["top1_accuracy"] for x in rows)/n)**2 for r in rows)/n)**0.5,
            "top3_std": (sum((r["top3_accuracy"] - sum(x["top3_accuracy"] for x in rows)/n)**2 for r in rows)/n)**0.5,
            "top5_std": (sum((r["top5_accuracy"] - sum(x["top5_accuracy"] for x in rows)/n)**2 for r in rows)/n)**0.5,
        }

    agg64 = agg(rows64)
    agg128 = agg(rows128)

    summary = {
        "top64": agg64,
        "top128": agg128,
        "delta": {
            "top1": agg128["top1"] - agg64["top1"],
            "top3": agg128["top3"] - agg64["top3"],
            "top5": agg128["top5"] - agg64["top5"],
        },
        "per_class": {
            "top64": rows64,
            "top128": rows128,
        },
    }

    summary_path = output_dir / "aggregate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"{'Metric':<15} {'Top-64':>18} {'Top-128':>18} {'Delta':>18}")
    print("-" * 80)
    for metric in ["top1", "top3", "top5"]:
        m64 = agg64[metric]
        m128 = agg128[metric]
        delta = summary["delta"][metric]
        print(f"{metric.upper() + ' Acc':<15} {m64:>17.2f}% {m128:>17.2f}% {delta:>+17.2f}%")
    print("-" * 80)
    print(f"{'Std Top-1':<15} {agg64['top1_std']:>17.2f}% {agg128['top1_std']:>17.2f}%")
    print("=" * 80)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
