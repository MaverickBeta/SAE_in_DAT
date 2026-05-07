#!/usr/bin/env python3
"""
批量测试干净样本的 SAE Cosine TopK 准确率。
支持多 GPU 并行，每个 worker 跑一个类。
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGN_STEERING_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list) -> bool:
    print(f"[RUN] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {e}")
        return False


def worker(gpu_id: int, task_queue, results_list, config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker GPU {gpu_id}] Started")

    while True:
        try:
            task = task_queue.get(timeout=10)
        except Empty:
            print(f"[Worker GPU {gpu_id}] Queue empty, exiting")
            break

        if task is None:
            break

        class_name = task
        print(f"[Worker GPU {gpu_id}] Starting {class_name}")

        cmd = [
            sys.executable,
            str(ALIGN_STEERING_DIR / "clean_acc_sae.py"),
            "--class-name", class_name,
            "--gpu", "0",
            "--n-samples", str(config["n_samples"]),
            "--batch-size", str(config["batch_size"]),
            "--sae-stage", str(config["sae_stage"]),
        ]

        if config.get("class_npz"):
            cmd.extend(["--class-npz", str(config["class_npz"])])

        success = run_cmd(cmd)
        results_list.append({
            "class_name": class_name,
            "gpu": gpu_id,
            "success": success,
        })

        if success:
            print(f"[Worker GPU {gpu_id}] {class_name} done")
        else:
            print(f"[Worker GPU {gpu_id}] {class_name} FAILED")


def aggregate_results(result_dir: Path):
    """汇总所有 clean_acc_*.json 结果。"""
    all_results = []
    for p in sorted(result_dir.glob("clean_acc_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results.append(data)

    if not all_results:
        print("No results found.")
        return

    agg = defaultdict(list)
    for r in all_results:
        acc = r["accuracy"]
        for k, v in acc.items():
            if isinstance(v, (int, float)):
                agg[k].append(v)

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(all_results)} classes)")
    print("=" * 70)
    for k, vals in sorted(agg.items()):
        avg = sum(vals) / len(vals)
        print(f"  {k:35s}: {avg:6.2f} (std={json_std(vals):.2f})")
    print("=" * 70)


def json_std(vals: list) -> float:
    import math
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description="Batch clean SAE cosine evaluation")
    parser.add_argument("--n-classes", type=int, default=50, help="类别数")
    parser.add_argument("--n-samples", type=int, default=50, help="每类图片数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7], help="GPU 列表")
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--class-list", type=str, default=None, help="类别列表文件")
    parser.add_argument("--class-npz", type=str, default=str(ALIGN_STEERING_DIR / "sae_stat_results_v2.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    val_dir = REPO_ROOT / "data" / "ImageNet" / "val"
    all_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    if args.class_list:
        with open(args.class_list, "r") as f:
            selected = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(selected)} classes from {args.class_list}")
    else:
        selected = random.sample(all_classes, k=min(args.n_classes, len(all_classes)))
        selected = sorted(selected)
        print(f"Randomly selected {len(selected)} classes")

    # 保存列表
    class_list_file = ALIGN_STEERING_DIR / "clean_sae_results" / "selected_classes.txt"
    class_list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(class_list_file, "w") as f:
        for cls in selected:
            f.write(f"{cls}\n")
    print(f"Class list saved to: {class_list_file}")

    # 任务队列
    manager = Manager()
    task_queue = manager.Queue()
    results_list = manager.list()

    for cls in selected:
        task_queue.put(cls)

    config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "sae_stage": args.sae_stage,
        "class_npz": Path(args.class_npz),
    }

    processes = []
    for gpu_id in args.gpus:
        p = Process(target=worker, args=(gpu_id, task_queue, results_list, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll workers finished. Results: {len(results_list)}")

    # 汇总
    result_dir = ALIGN_STEERING_DIR / "clean_sae_results"
    aggregate_results(result_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
