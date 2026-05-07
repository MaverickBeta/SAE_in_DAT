#!/usr/bin/env python3
"""
临时汇总脚本：读取 compare_sae_results/ 下的已有 JSON 结果，
直接在终端输出三组对照的 Clean Acc / Adv Acc / ASR 统计。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_condition(cond_dir: Path):
    files = sorted(cond_dir.glob("attack_results_*.json"))
    if not files:
        return None

    clean, adv, asr = [], [], []
    for f in files:
        data = load_json(f)
        acc = data.get("accuracy", {})
        clean.append(acc.get("clean_accuracy", 0.0))
        adv.append(acc.get("adversarial_accuracy", 0.0))
        asr.append(acc.get("attack_success_rate", 0.0))

    n = len(files)
    return {
        "count": n,
        "clean_avg": sum(clean) / n,
        "adv_avg": sum(adv) / n,
        "asr_avg": sum(asr) / n,
        "clean_std": (sum((x - sum(clean)/n) ** 2 for x in clean) / n) ** 0.5,
        "adv_std": (sum((x - sum(adv)/n) ** 2 for x in adv) / n) ** 0.5,
        "asr_std": (sum((x - sum(asr)/n) ** 2 for x in asr) / n) ** 0.5,
    }


def main():
    base_dir = Path("./compare_sae_results")
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])

    conditions = {
        "no_sae": base_dir / "no_sae",
        "sae_64k": base_dir / "sae_64k",
        "sae_exp32": base_dir / "sae_exp32",
    }

    print("\n" + "=" * 90)
    print(f"SAE COMPARISON SUMMARY  |  Dir: {base_dir.resolve()}")
    print("=" * 90)
    print(f"{'Condition':<15} {'Count':>8} {'Clean Acc':>18} {'Adv Acc':>18} {'ASR':>18}")
    print("-" * 90)

    for name, cond_dir in conditions.items():
        stats = aggregate_condition(cond_dir)
        if stats is None:
            print(f"{name:<15} {'—':>8} {'—':>18} {'—':>18} {'—':>18}")
            continue

        clean_str = f"{stats['clean_avg']:.2f}% ± {stats['clean_std']:.2f}%"
        adv_str = f"{stats['adv_avg']:.2f}% ± {stats['adv_std']:.2f}%"
        asr_str = f"{stats['asr_avg']:.2f}% ± {stats['asr_std']:.2f}%"
        print(f"{name:<15} {stats['count']:>8} {clean_str:>18} {adv_str:>18} {asr_str:>18}")

    print("=" * 90)

    # 额外打印相对于 no_sae 的差值
    no_sae_stats = aggregate_condition(conditions["no_sae"])
    if no_sae_stats:
        print("\nRelative to no_sae:")
        print(f"{'Condition':<15} {'Δ Clean':>12} {'Δ Adv':>12} {'Δ ASR':>12}")
        print("-" * 55)
        for name in ["sae_64k", "sae_exp32"]:
            stats = aggregate_condition(conditions[name])
            if stats is None:
                continue
            d_clean = stats["clean_avg"] - no_sae_stats["clean_avg"]
            d_adv = stats["adv_avg"] - no_sae_stats["adv_avg"]
            d_asr = stats["asr_avg"] - no_sae_stats["asr_avg"]
            print(f"{name:<15} {d_clean:>+11.2f}% {d_adv:>+11.2f}% {d_asr:>+11.2f}%")
        print("=" * 55)

    print()


if __name__ == "__main__":
    main()
