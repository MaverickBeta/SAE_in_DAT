#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sharded attack eval artifacts.")
    p.add_argument(
        "--base-root",
        default="/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py",
        help="Root containing shard_00, shard_01, ...",
    )
    p.add_argument(
        "--attack-name",
        default="fab_untar_sl2py",
        help="Attack folder under each shard directory.",
    )
    p.add_argument("--num-shards", type=int, default=4, help="Expected shard count.")
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <base-root>/<attack-name>",
    )
    p.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep shard_XX folders after merge. Default behavior deletes them.",
    )
    return p.parse_args()


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def shard_dir(base_root: str, idx: int) -> str:
    return os.path.join(base_root, f"shard_{idx:02d}")


def merge(args: argparse.Namespace) -> Dict:
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")

    out_dir = args.out_dir or os.path.join(args.base_root, args.attack_name)
    os.makedirs(out_dir, exist_ok=True)

    # Remove previous merged artifacts in destination attack directory.
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)

    shard_infos = []
    merged_rows: List[Dict] = []

    total = 0
    success_count = 0
    target_conf_sum = 0.0
    source_conf_sum = 0.0

    for i in range(args.num_shards):
        attack_dir = os.path.join(shard_dir(args.base_root, i), args.attack_name)
        summary_path = os.path.join(attack_dir, "summary.json")
        csv_path = os.path.join(attack_dir, "eval_results.csv")

        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing csv: {csv_path}")

        s = read_json(summary_path)
        rows = read_csv_rows(csv_path)

        for r in rows:
            rr = dict(r)
            rr["shard"] = f"{i:02d}"
            merged_rows.append(rr)

        for name in os.listdir(attack_dir):
            src = os.path.join(attack_dir, name)
            if not os.path.isfile(src):
                continue
            low = name.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png") or low.endswith(".bmp") or low.endswith(".webp")):
                continue
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst):
                raise RuntimeError(f"Duplicate image filename while merging: {name}")
            shutil.copy2(src, dst)

        t = int(s.get("total", 0))
        sc = int(s.get("success_count", 0))
        total += t
        success_count += sc
        target_conf_sum += float(s.get("avg_target_conf", 0.0)) * t
        source_conf_sum += float(s.get("avg_source_conf", 0.0)) * t

        shard_infos.append(
            {
                "shard_index": i,
                "attack_dir": attack_dir,
                "total": t,
                "success_count": sc,
                "success_rate": float(s.get("success_rate", 0.0)),
                "escape_rate": float(s.get("escape_rate", 0.0)),
            }
        )

    merged_csv = os.path.join(out_dir, "eval_results.csv")
    fieldnames = [
        "filename",
        "pred",
        "top1_conf",
        "target_conf_62",
        "source_conf_150",
        "success",
        "shard",
    ]
    with open(merged_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in merged_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    success_rate = (success_count / total) if total > 0 else 0.0
    merged_summary = {
        "attack_name": args.attack_name,
        "num_shards": args.num_shards,
        "base_root": args.base_root,
        "total": total,
        "success_count": success_count,
        "fail_count": total - success_count,
        "success_rate": success_rate,
        "escape_count": success_count,
        "escape_rate": success_rate,
        "avg_target_conf": (target_conf_sum / total) if total > 0 else 0.0,
        "avg_source_conf": (source_conf_sum / total) if total > 0 else 0.0,
        "shards": shard_infos,
        "eval_csv": merged_csv,
    }

    merged_summary_path = os.path.join(out_dir, "summary.json")
    with open(merged_summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, indent=2)

    deleted_shards = []
    if not args.keep_shards:
        for i in range(args.num_shards):
            d = shard_dir(args.base_root, i)
            if os.path.isdir(d):
                shutil.rmtree(d)
                deleted_shards.append(d)

    return {
        "out_dir": out_dir,
        "merged_csv": merged_csv,
        "merged_summary": merged_summary_path,
        "total": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "deleted_shards": deleted_shards,
    }


def main():
    args = parse_args()
    result = merge(args)
    print(f"Merged OK | out_dir={result['out_dir']}")
    print(f"eval_results.csv={result['merged_csv']}")
    print(f"summary.json={result['merged_summary']}")
    print(
        f"total={result['total']} success_count={result['success_count']} "
        f"success_rate={result['success_rate']:.4f}"
    )
    if result["deleted_shards"]:
        print("deleted_shards:")
        for d in result["deleted_shards"]:
            print(d)


if __name__ == "__main__":
    main()
