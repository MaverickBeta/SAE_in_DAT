#!/usr/bin/env python3
"""
对比不同 class-npz TopK 配置下的 eval_dis_spatial_fast.py 评估结果。

用法：
    python compare_topk_eval.py eval_top64.json eval_top128.json [eval_top256.json ...]
"""

import json
import sys
from pathlib import Path


def load_eval_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_stats(data):
    meta = data.get("meta", {})
    stats = data.get("statistics", {})
    cosine = stats.get("cosine", {})
    return {
        "label": Path(meta.get("class_npz", "unknown")).stem,
        "num_images": meta.get("num_images", 0),
        "num_classes": meta.get("num_classes", 0),
        "top1": cosine.get("top1_accuracy", 0.0),
        "top3": cosine.get("top3_accuracy", 0.0),
        "top5": cosine.get("top5_accuracy", 0.0),
        "true_class_count_top1": cosine.get("true_class_count_top1", 0),
        "true_class_count_top3": cosine.get("true_class_count_top3", 0),
        "true_class_count_top5": cosine.get("true_class_count_top5", 0),
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <eval_json1> [eval_json2] ...")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    rows = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] File not found: {p}")
            continue
        data = load_eval_json(p)
        rows.append(extract_stats(data))

    if not rows:
        print("No valid results to compare.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TOP-K CLASS STAT COMPARISON")
    print("=" * 80)
    print(f"{'Config':<30} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10} {'Images':>8}")
    print("-" * 80)

    baseline = rows[0]
    for r in rows:
        print(
            f"{r['label']:<30} "
            f"{r['top1']:>9.1f}% "
            f"{r['top3']:>9.1f}% "
            f"{r['top5']:>9.1f}% "
            f"{r['num_images']:>8}"
        )

    if len(rows) > 1:
        print("-" * 80)
        print(f"\nRelative to baseline ({baseline['label']}):")
        print(f"{'Config':<30} {'Δ Top-1':>12} {'Δ Top-3':>12} {'Δ Top-5':>12}")
        print("-" * 70)
        for r in rows[1:]:
            d1 = r["top1"] - baseline["top1"]
            d3 = r["top3"] - baseline["top3"]
            d5 = r["top5"] - baseline["top5"]
            print(f"{r['label']:<30} {d1:>+11.2f}% {d3:>+11.2f}% {d5:>+11.2f}%")

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
