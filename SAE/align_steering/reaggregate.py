#!/usr/bin/env python3
"""
从已生成的 batch_eval.py 输出文件中重新聚合汇总结果。
无需重新跑攻击，直接读取现有 JSON 文件生成正确的 summary。
"""
 
import json
from collections import defaultdict
from pathlib import Path

ALIGN_STEERING_DIR = Path(__file__).resolve().parent


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def main():
    conditions = ["no_sae", "sae_stage3", "aa_no_sae", "aa_sae_stage3"]
    class_npz = ALIGN_STEERING_DIR / "sae_stat_results_v2.npz"

    # 扫描已完成的所有类
    per_class = {}

    for cond in conditions:
        is_aa = cond.startswith("aa_")
        base_cond = cond[3:] if is_aa else cond

        if is_aa:
            if base_cond == "no_sae":
                attack_dir = ALIGN_STEERING_DIR / "adv_samples" / "aa" / "no_sae"
                eval_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / "aa_no_sae"
            else:
                attack_dir = ALIGN_STEERING_DIR / "adv_samples" / "aa" / base_cond
                eval_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / cond
        else:
            if base_cond == "no_sae":
                attack_dir = ALIGN_STEERING_DIR / "adv_samples" / "no_sae"
                eval_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / "no_sae"
            else:
                attack_dir = ALIGN_STEERING_DIR / "adv_samples" / base_cond
                eval_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / base_cond

        if not attack_dir.exists():
            continue

        # 扫描 eval json 确定有哪些类已完成
        if not eval_dir.exists():
            continue

        for eval_json in eval_dir.glob("eval_dis_*_spatial.json"):
            cls = eval_json.stem.replace("eval_dis_", "").replace("_spatial", "")

            # attack json (平铺在 attack_dir 根目录)
            attack_json = attack_dir / f"attack_results_{cls}.json"

            if not attack_json.exists():
                continue

            attack_data = load_json(attack_json)
            eval_data = load_json(eval_json)

            if not attack_data or not eval_data:
                continue

            if cls not in per_class:
                per_class[cls] = {}

            acc = attack_data.get("accuracy", {})
            stats = eval_data.get("statistics", {})

            per_class[cls][cond] = {
                "before": {
                    "clean_accuracy": acc.get("clean_accuracy", 0.0),
                    "adversarial_accuracy": acc.get("adversarial_accuracy", 0.0),
                    "attack_success_rate": acc.get("attack_success_rate", 0.0),
                },
                "after": {
                    "cosine": {
                        "top1": stats.get("cosine", {}).get("top1_accuracy", 0.0),
                        "top3": stats.get("cosine", {}).get("top3_accuracy", 0.0),
                        "top5": stats.get("cosine", {}).get("top5_accuracy", 0.0),
                    },
                    "jaccard": {
                        "top1": stats.get("jaccard", {}).get("top1_accuracy", 0.0),
                        "top3": stats.get("jaccard", {}).get("top3_accuracy", 0.0),
                        "top5": stats.get("jaccard", {}).get("top5_accuracy", 0.0),
                    },
                },
            }

    # 聚合
    agg = defaultdict(lambda: {
        "count": 0,
        "clean_acc": [],
        "adv_acc": [],
        "asr": [],
        "cosine_top1": [],
        "cosine_top3": [],
        "cosine_top5": [],
        "jaccard_top1": [],
        "jaccard_top3": [],
        "jaccard_top5": [],
    })

    for cls, conds in per_class.items():
        for cond, entry in conds.items():
            agg[cond]["count"] += 1
            agg[cond]["clean_acc"].append(entry["before"]["clean_accuracy"])
            agg[cond]["adv_acc"].append(entry["before"]["adversarial_accuracy"])
            agg[cond]["asr"].append(entry["before"]["attack_success_rate"])
            agg[cond]["cosine_top1"].append(entry["after"]["cosine"]["top1"])
            agg[cond]["cosine_top3"].append(entry["after"]["cosine"]["top3"])
            agg[cond]["cosine_top5"].append(entry["after"]["cosine"]["top5"])
            agg[cond]["jaccard_top1"].append(entry["after"]["jaccard"]["top1"])
            agg[cond]["jaccard_top3"].append(entry["after"]["jaccard"]["top3"])
            agg[cond]["jaccard_top5"].append(entry["after"]["jaccard"]["top5"])

    aggregate = {}
    for cond in conditions:
        d = agg[cond]
        n = d["count"]
        if n == 0:
            continue
        aggregate[cond] = {
            "count": n,
            "avg_clean_accuracy": round(sum(d["clean_acc"]) / n, 2),
            "avg_adversarial_accuracy": round(sum(d["adv_acc"]) / n, 2),
            "avg_attack_success_rate": round(sum(d["asr"]) / n, 2),
            "avg_cosine_top1": round(sum(d["cosine_top1"]) / n, 2),
            "avg_cosine_top3": round(sum(d["cosine_top3"]) / n, 2),
            "avg_cosine_top5": round(sum(d["cosine_top5"]) / n, 2),
            "avg_jaccard_top1": round(sum(d["jaccard_top1"]) / n, 2),
            "avg_jaccard_top3": round(sum(d["jaccard_top3"]) / n, 2),
            "avg_jaccard_top5": round(sum(d["jaccard_top5"]) / n, 2),
        }

    summary = {
        "conditions": conditions,
        "class_npz": str(class_npz),
        "per_class": per_class,
        "aggregate": aggregate,
    }

    summary_file = ALIGN_STEERING_DIR / "adv_samples" / "batch_summary_reaggregated.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("RE-AGGREGATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total classes found: {len(per_class)}")
    for cond in conditions:
        if cond not in aggregate:
            continue
        d = aggregate[cond]
        label = cond.upper()
        print(f"\n[{label}]  ({d['count']} classes)")
        print(f"  Before - Clean Acc:      {d['avg_clean_accuracy']:.2f}%")
        print(f"  Before - Adv Acc:        {d['avg_adversarial_accuracy']:.2f}%")
        print(f"  Before - ASR:            {d['avg_attack_success_rate']:.2f}%")
        print(f"  After  - Cosine Top-1:   {d['avg_cosine_top1']:.2f}%")
        print(f"  After  - Cosine Top-3:   {d['avg_cosine_top3']:.2f}%")
        print(f"  After  - Cosine Top-5:   {d['avg_cosine_top5']:.2f}%")
        print(f"  After  - Jaccard Top-1:  {d['avg_jaccard_top1']:.2f}%")
        print(f"  After  - Jaccard Top-3:  {d['avg_jaccard_top3']:.2f}%")
        print(f"  After  - Jaccard Top-5:  {d['avg_jaccard_top5']:.2f}%")
    print(f"{'='*80}")
    print(f"\nSaved to: {summary_file}")


if __name__ == "__main__":
    main()
