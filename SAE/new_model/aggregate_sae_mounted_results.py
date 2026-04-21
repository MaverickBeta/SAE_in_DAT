#!/usr/bin/env python3
"""汇总 adv_samples_l2_sae_mounted 下的攻击结果，输出表格和 CSV。"""
import argparse
import csv
import json
import statistics
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Aggregate SAE-mounted L2 attack results")
    parser.add_argument("--input-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "adv_samples_l2_sae_mounted"))
    parser.add_argument("--output-csv", type=str,
                        default=str(Path(__file__).resolve().parent / "results" / "adv_samples_l2_sae_mounted" / "summary.csv"))
    args = parser.parse_args()

    root = Path(args.input_dir)
    ckpts = sorted([d for d in root.iterdir() if d.is_dir()])

    rows = []
    print(f"{'Checkpoint':<50s} {'Clean':>12s} {'Adv':>12s} {'ASR':>12s} {'Classes':>8s}")
    print("=" * 100)

    for ckpt_dir in ckpts:
        jsons = list(ckpt_dir.glob("sae_mounted_l2_attack_results_*.json"))
        if not jsons:
            continue
        clean_list, adv_list, asr_list = [], [], []
        for p in jsons:
            with open(p) as f:
                data = json.load(f)
            acc = data.get("accuracy", {})
            clean_list.append(acc.get("clean_accuracy", 0))
            adv_list.append(acc.get("adversarial_accuracy", 0))
            asr_list.append(acc.get("attack_success_rate", 0))

        def fmt(vals):
            if len(vals) < 2:
                return f"{vals[0]:.2f}" if vals else "N/A"
            return f"{statistics.mean(vals):.2f}±{statistics.stdev(vals):.2f}"

        def num(vals):
            return statistics.mean(vals) if vals else 0.0

        print(f"{ckpt_dir.name:<50s} {fmt(clean_list):>12s} {fmt(adv_list):>12s} {fmt(asr_list):>12s} {len(jsons):>8d}")
        rows.append({
            "checkpoint": ckpt_dir.name,
            "clean_acc_mean": round(num(clean_list), 2),
            "clean_acc_std": round(statistics.stdev(clean_list), 2) if len(clean_list) > 1 else 0.0,
            "adv_acc_mean": round(num(adv_list), 2),
            "adv_acc_std": round(statistics.stdev(adv_list), 2) if len(adv_list) > 1 else 0.0,
            "asr_mean": round(num(asr_list), 2),
            "asr_std": round(statistics.stdev(asr_list), 2) if len(asr_list) > 1 else 0.0,
            "n_classes": len(jsons),
        })

    print("=" * 100)

    # Write CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "clean_acc_mean", "clean_acc_std",
            "adv_acc_mean", "adv_acc_std", "asr_mean", "asr_std", "n_classes"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
