#!/usr/bin/env python3
"""
批量 L2 PGD-CE 攻击挂载 SAE 的原始模型：多 GPU 并行。

每个 worker 跑一个类，完成后取下一个类。
已存在的 JSON 会自动跳过（--skip-existing）。
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty

SCRIPT_DIR = Path(__file__).resolve().parent


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
            str(SCRIPT_DIR / "baseline_attack_sae_mounted_l2.py"),
            "--class-name", class_name,
            "--gpu", "0",
            "--n-samples", str(config["n_samples"]),
            "--eps", str(config["eps"]),
            "--steps", str(config["steps"]),
            "--step-size", str(config["step_size"]),
            "--batch-size", str(config["batch_size"]),
            "--output-dir", str(config["output_dir"]),
            "--skip-existing",
        ]
        if config.get("sae_ckpt"):
            cmd += ["--sae-ckpt", str(config["sae_ckpt"])]
        if config.get("sae_stage") is not None:
            cmd += ["--sae-stage", str(config["sae_stage"])]

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


def aggregate_results(output_dir: Path):
    all_results = []
    for p in sorted(output_dir.glob("sae_mounted_l2_attack_results_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results.append(data)

    if not all_results:
        print("No results found.")
        return

    agg = defaultdict(list)
    for r in all_results:
        a = r.get("accuracy", {})
        for k, v in a.items():
            if isinstance(v, (int, float)):
                agg[k].append(v)

    import math
    def std(vals):
        n = len(vals)
        if n < 2:
            return 0.0
        m = sum(vals) / n
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({len(all_results)} classes)")
    print("=" * 70)
    for k, vals in sorted(agg.items()):
        avg = sum(vals) / len(vals)
        print(f"  {k:40s}: {avg:6.2f} (std={std(vals):.2f})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Batch L2 PGD-CE on SAE-mounted model")
    parser.add_argument("--n-classes", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=110)
    parser.add_argument("--step-size", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--class-list", type=str, default=None)
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR / "adv_samples_l2_sae_mounted"))
    parser.add_argument("--sae-ckpt", type=str, default=None,
                        help="Path to SAE checkpoint (overrides default)")
    parser.add_argument("--sae-stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    val_dir = Path(__file__).resolve().parents[2] / "data" / "ImageNet" / "val"
    all_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    if args.class_list:
        with open(args.class_list, "r") as f:
            selected = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(selected)} classes from {args.class_list}")
    else:
        selected = random.sample(all_classes, k=min(args.n_classes, len(all_classes)))
        selected = sorted(selected)
        print(f"Randomly selected {len(selected)} classes")

    list_file = Path(args.output_dir) / "selected_classes.txt"
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w") as f:
        for cls in selected:
            f.write(f"{cls}\n")
    print(f"Class list saved to: {list_file}")

    manager = Manager()
    task_queue = manager.Queue()
    results_list = manager.list()

    for cls in selected:
        task_queue.put(cls)

    config = {
        "n_samples": args.n_samples,
        "eps": args.eps,
        "steps": args.steps,
        "step_size": args.step_size,
        "batch_size": args.batch_size,
        "output_dir": Path(args.output_dir),
        "sae_ckpt": Path(args.sae_ckpt) if args.sae_ckpt else None,
        "sae_stage": args.sae_stage,
    }

    processes = []
    for gpu_id in args.gpus:
        p = Process(target=worker, args=(gpu_id, task_queue, results_list, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll workers finished. Results: {len(results_list)}")
    aggregate_results(Path(args.output_dir))
    print("\nDone!")


if __name__ == "__main__":
    main()
