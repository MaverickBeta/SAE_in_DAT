#!/usr/bin/env python3
"""
批量对比所有已完成类别的 before 和 after accuracy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_json(path: Path) -> dict:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_completed_classes(ad_samples_dir: Path) -> List[str]:
    """获取所有已完成的类别（有 attack_results 和 eval_dis 文件）"""
    attack_files = list(ad_samples_dir.glob("attack_results_*.json"))
    eval_files = list(ad_samples_dir.glob("eval_dis_sae_features_*.json"))
    
    # 提取类别名
    attack_classes = set(f.stem.replace("attack_results_", "") for f in attack_files)
    eval_classes = set(f.stem.replace("eval_dis_sae_features_", "") for f in eval_files)
    
    # 返回两个文件都存在的类别
    completed = sorted(attack_classes & eval_classes)
    return completed


def compare_class(ad_samples_dir: Path, class_name: str) -> Dict:
    """对比单个类别的 before 和 after accuracy"""
    # 加载 before accuracy (attack results)
    attack_file = ad_samples_dir / f"attack_results_{class_name}.json"
    before_data = load_json(attack_file)
    before = before_data.get("before_accuracy", {})
    
    # 加载 after accuracy (eval dis results)
    eval_file = ad_samples_dir / f"eval_dis_sae_features_{class_name}.json"
    after_data = load_json(eval_file)
    after_stats = after_data.get("statistics", {}).get("cosine", {})
    after_acc = after_stats.get("after_accuracy", {})
    
    return {
        "class_name": class_name,
        "before": {
            "clean_accuracy": before.get("clean_accuracy", 0),
            "adversarial_accuracy": before.get("adversarial_accuracy", 0),
            "attack_success_rate": before.get("attack_success_rate", 0),
        },
        "after": {
            "top1": after_acc.get("top1", 0),
            "top3": after_acc.get("top3", 0),
        }
    }


def print_comparison(results: List[Dict]):
    """打印对比结果"""
    print("\n" + "="*100)
    print(f"{'Class':<15} {'Before Clean':<12} {'Before Adv':<12} {'ASR':<8} {'After Top1':<12} {'After Top3':<12}")
    print("="*100)
    
    for result in results:
        cls = result["class_name"]
        before = result["before"]
        after = result["after"]
        
        print(f"{cls:<15} "
              f"{before['clean_accuracy']:>10.1f}%  "
              f"{before['adversarial_accuracy']:>10.1f}%  "
              f"{before['attack_success_rate']:>6.1f}%  "
              f"{after['top1']:>10.1f}%  "
              f"{after['top3']:>10.1f}%")
    
    print("="*100)


def print_statistics(results: List[Dict]):
    """打印统计信息"""
    if not results:
        print("No results to statistics.")
        return
    
    n = len(results)
    
    # 计算平均值
    avg_before_clean = sum(r["before"]["clean_accuracy"] for r in results) / n
    avg_before_adv = sum(r["before"]["adversarial_accuracy"] for r in results) / n
    avg_asr = sum(r["before"]["attack_success_rate"] for r in results) / n
    avg_after_top1 = sum(r["after"]["top1"] for r in results) / n
    avg_after_top3 = sum(r["after"]["top3"] for r in results) / n
    
    print("\n" + "="*100)
    print("STATISTICS (Average)")
    print("="*100)
    print(f"{'Metric':<30} {'Average':<15} {'Min':<15} {'Max':<15}")
    print("-"*100)
    
    metrics = [
        ("Before Clean Accuracy", [r["before"]["clean_accuracy"] for r in results]),
        ("Before Adversarial Accuracy", [r["before"]["adversarial_accuracy"] for r in results]),
        ("Attack Success Rate", [r["before"]["attack_success_rate"] for r in results]),
        ("After Cosine Top-1", [r["after"]["top1"] for r in results]),
        ("After Cosine Top-3", [r["after"]["top3"] for r in results]),
    ]
    
    for name, values in metrics:
        avg_val = sum(values) / n
        min_val = min(values)
        max_val = max(values)
        print(f"{name:<30} {avg_val:>13.2f}%  {min_val:>13.2f}%  {max_val:>13.2f}%")
    
    print("="*100)
    print(f"\nTotal classes: {n}")


def save_summary(results: List[Dict], output_file: Path):
    """保存汇总结果到 JSON"""
    output_data = {
        "individual_results": results,
        "statistics": {}
    }
    
    if results:
        n = len(results)
        output_data["statistics"] = {
            "count": n,
            "before_clean_accuracy_avg": sum(r["before"]["clean_accuracy"] for r in results) / n,
            "before_adversarial_accuracy_avg": sum(r["before"]["adversarial_accuracy"] for r in results) / n,
            "attack_success_rate_avg": sum(r["before"]["attack_success_rate"] for r in results) / n,
            "after_cosine_top1_avg": sum(r["after"]["top1"] for r in results) / n,
            "after_cosine_top3_avg": sum(r["after"]["top3"] for r in results) / n,
        }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="批量对比所有已完成类别的 before 和 after accuracy"
    )
    parser.add_argument(
        "--ad-samples-dir",
        type=str,
        default="/Data_share/hongyi/DAT/ad_samples",
        help="对抗样本目录"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--class-list",
        type=str,
        default=None,
        help="指定类别列表文件 (每行一个类别名)"
    )
    
    args = parser.parse_args()
    
    ad_samples_dir = Path(args.ad_samples_dir)
    
    # 获取要处理的类别
    if args.class_list:
        with open(args.class_list, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(classes)} classes from {args.class_list}")
    else:
        classes = get_completed_classes(ad_samples_dir)
        print(f"Found {len(classes)} completed classes")
    
    if not classes:
        print("No classes to process!")
        return
    
    # 对比每个类别
    results = []
    for class_name in classes:
        try:
            result = compare_class(ad_samples_dir, class_name)
            results.append(result)
        except Exception as e:
            print(f"Error processing {class_name}: {e}")
    
    # 打印对比结果
    print_comparison(results)
    
    # 打印统计信息
    print_statistics(results)
    
    # 保存汇总结果
    if args.output_json:
        output_file = Path(args.output_json)
    else:
        timestamp = Path().stat().st_mtime if results else 0
        output_file = ad_samples_dir / f"comparison_summary_{len(results)}classes.json"
    
    save_summary(results, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
