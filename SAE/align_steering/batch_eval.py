#!/usr/bin/env python3
"""
多 GPU 并行批量评估脚本。

每张卡独立运行一个类别的完整 pipeline（所有实验条件），完成后取下一个类别。
流程：apgdce_attack.py → extract_from_sae.py → eval_dis_spatial.py
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGN_STEERING_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], desc: str) -> bool:
    """运行子命令，打印并返回是否成功。"""
    print(f"[{desc}] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc} failed: {e}")
        return False


def run_pipeline(class_name: str, condition: str, attack_script: str, gpu_id: int, config: Dict) -> Dict:
    """
    对一个类别的一个条件运行完整 3-stage pipeline。

    condition: "no_sae" 或 "sae_stage{数字}"
    attack_script: "baseline_attack.py" 或 "aa_attack.py"
    """
    result = {
        "class_name": class_name,
        "condition": condition,
        "attack_script": attack_script,
        "gpu": gpu_id,
        "success": False,
        "skipped": False,
        "steps": [],
        "error": None,
    }

    is_aa = attack_script == "aa_attack.py"
    root_dir = ALIGN_STEERING_DIR / "adv_samples" / "aa" if is_aa else ALIGN_STEERING_DIR / "adv_samples"

    # 路径映射
    if condition == "no_sae":
        adv_dir = root_dir / "no_sae" / class_name
        attack_json = root_dir / "no_sae" / f"attack_results_{class_name}.json"
        sae_latent_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / ("aa_no_sae" if is_aa else "no_sae")
        npz_name = f"{class_name}_spatial.npz"
        eval_json = sae_latent_dir / f"eval_dis_{class_name}_spatial.json"
    elif condition.startswith("sae_stage"):
        stage = condition.replace("sae_stage", "")
        adv_dir = root_dir / condition / class_name
        attack_json = root_dir / condition / f"attack_results_{class_name}.json"
        sae_latent_dir = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / (f"aa_sae_stage{stage}" if is_aa else condition)
        npz_name = f"{class_name}_spatial.npz"
        eval_json = sae_latent_dir / f"eval_dis_{class_name}_spatial.json"
    else:
        result["error"] = f"Unknown condition: {condition}"
        return result

    sae_latent_dir.mkdir(parents=True, exist_ok=True)

    # Skip existing
    if config.get("skip_existing") and eval_json.exists():
        print(f"[SKIP] {class_name}/{condition}/{attack_script} already done")
        result["success"] = True
        result["skipped"] = True
        return result

    # Step 1: attack
    attack_cmd = [
        sys.executable,
        str(ALIGN_STEERING_DIR / attack_script),
        "--class-name", class_name,
        "--gpu", "0",
        "--n-samples", str(config["n_samples"]),
        "--eps", str(config["eps"]),
        "--steps", str(config["steps"]),
        "--batch-size", str(config["batch_size"]),
        "--save-adversarial",
    ]
    if condition.startswith("sae_stage"):
        attack_cmd.extend(["--use-sae", "--sae-stage", stage])

    if not run_cmd(attack_cmd, f"GPU{gpu_id} {class_name}/{condition}/{attack_script} attack"):
        result["error"] = "attack failed"
        return result
    result["steps"].append("attack")

    # Step 2: extract_from_sae.py
    extract_cmd = [
        sys.executable,
        str(ALIGN_STEERING_DIR / "extract_from_sae.py"),
        "--adv-dir", str(adv_dir),
        "--gpu", "0",
    ]
    if not run_cmd(extract_cmd, f"GPU{gpu_id} {class_name}/{condition}/{attack_script} extract"):
        result["error"] = "extract failed"
        return result
    result["steps"].append("extract")

    # Step 3: eval_dis_spatial.py
    npz_path = sae_latent_dir / npz_name
    eval_cmd = [
        sys.executable,
        str(ALIGN_STEERING_DIR / "eval_dis_spatial.py"),
        "--adv-npz", str(npz_path),
        "--class-npz", str(config["class_npz"]),
        "--gpu", "0",
    ]
    if not run_cmd(eval_cmd, f"GPU{gpu_id} {class_name}/{condition}/{attack_script} eval"):
        result["error"] = "eval failed"
        return result
    result["steps"].append("eval")

    result["success"] = True
    return result


def worker(gpu_id: int, task_queue, results_list, config: Dict):
    """每个 GPU 一个 worker，依次从队列取任务执行。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker GPU {gpu_id}] Started")

    while True:
        try:
            task = task_queue.get(timeout=10)
        except Empty:
            print(f"[Worker GPU {gpu_id}] Queue empty, exiting")
            break

        if task is None:
            print(f"[Worker GPU {gpu_id}] Received stop signal")
            break

        class_name, attack_script, conditions = task
        print(f"[Worker GPU {gpu_id}] Starting {class_name} ({attack_script}) with {conditions}")

        for condition in conditions:
            result = run_pipeline(class_name, condition, attack_script, gpu_id, config)
            results_list.append(result)
            if result.get("skipped"):
                print(f"[Worker GPU {gpu_id}] {class_name}/{condition}/{attack_script} skipped (exists)")
            elif result["success"]:
                print(f"[Worker GPU {gpu_id}] {class_name}/{condition}/{attack_script} done")
            else:
                print(f"[Worker GPU {gpu_id}] {class_name}/{condition}/{attack_script} FAILED: {result['error']}")

        print(f"[Worker GPU {gpu_id}] Finished {class_name} ({attack_script})")


def load_json_safe(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def aggregate_results(results: List[Dict], conditions: List[str], class_npz: Path) -> Dict:
    """汇总所有已完成的类别的 before/after 指标。"""
    summary = {
        "conditions": conditions,
        "class_npz": str(class_npz),
        "per_class": {},
    }

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

    for r in results:
        if not r["success"]:
            continue

        cls = r["class_name"]
        cond = r["condition"]

        if cls not in summary["per_class"]:
            summary["per_class"][cls] = {}

        entry = {}

        # 读取 attack JSON (before)
        if cond == "no_sae":
            attack_json = ALIGN_STEERING_DIR / "adv_samples" / "no_sae" / cls / f"attack_results_{cls}.json"
        else:
            stage = cond.replace("sae_stage", "")
            attack_json = ALIGN_STEERING_DIR / "adv_samples" / cond / cls / f"attack_results_{cls}_sae_stage{stage}.json"

        attack_data = load_json_safe(attack_json)
        acc = attack_data.get("accuracy", {})
        entry["before"] = {
            "clean_accuracy": acc.get("clean_accuracy", 0.0),
            "adversarial_accuracy": acc.get("adversarial_accuracy", 0.0),
            "attack_success_rate": acc.get("attack_success_rate", 0.0),
        }

        # 读取 eval JSON (after)
        if cond == "no_sae":
            eval_json = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / "no_sae" / f"eval_dis_{cls}_spatial.json"
        else:
            eval_json = ALIGN_STEERING_DIR / "adv_samples" / "sae_latent" / cond / f"eval_dis_{cls}_spatial.json"

        eval_data = load_json_safe(eval_json)
        stats = eval_data.get("statistics", {})
        entry["after"] = {
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
        }

        summary["per_class"][cls][cond] = entry

        # 聚合
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

    # 计算平均值
    summary["aggregate"] = {}
    for cond in conditions:
        d = agg[cond]
        n = d["count"]
        if n == 0:
            continue
        summary["aggregate"][cond] = {
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

    return summary


def print_summary(summary: Dict):
    """打印汇总表格到终端。"""
    print("\n" + "=" * 100)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 100)

    conditions = summary.get("conditions", [])
    agg = summary.get("aggregate", {})

    for cond in conditions:
        if cond not in agg:
            continue
        d = agg[cond]
        print(f"\n[{cond.upper()}]  ({d['count']} classes)")
        print(f"  Before - Clean Acc:      {d['avg_clean_accuracy']:.2f}%")
        print(f"  Before - Adv Acc:        {d['avg_adversarial_accuracy']:.2f}%")
        print(f"  Before - ASR:            {d['avg_attack_success_rate']:.2f}%")
        print(f"  After  - Cosine Top-1:   {d['avg_cosine_top1']:.2f}%")
        print(f"  After  - Cosine Top-3:   {d['avg_cosine_top3']:.2f}%")
        print(f"  After  - Cosine Top-5:   {d['avg_cosine_top5']:.2f}%")
        print(f"  After  - Jaccard Top-1:  {d['avg_jaccard_top1']:.2f}%")
        print(f"  After  - Jaccard Top-3:  {d['avg_jaccard_top3']:.2f}%")
        print(f"  After  - Jaccard Top-5:  {d['avg_jaccard_top5']:.2f}%")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch evaluation: attack → extract → eval"
    )
    parser.add_argument("--n-classes", type=int, default=50, help="随机选择的类别数")
    parser.add_argument("--n-samples", type=int, default=50, help="每类采样图片数")
    parser.add_argument("--eps", type=float, default=8 / 255, help="APGD epsilon")
    parser.add_argument("--steps", type=int, default=100, help="APGD 迭代步数")
    parser.add_argument("--batch-size", type=int, default=16, help="攻击 batch size")
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[4, 5, 6, 7],
        help="使用的 GPU 列表 (默认 4 5 6 7)"
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["no_sae", "sae_stage3"],
        help="实验条件列表 (默认 no_sae sae_stage3)"
    )
    parser.add_argument(
        "--attack-methods", type=str, nargs="+",
        default=["baseline"],
        choices=["baseline", "aa"],
        help="攻击方法列表 (默认 baseline，可选 aa)"
    )
    parser.add_argument("--class-list", type=str, default=None, help="指定类别列表文件")
    parser.add_argument(
        "--class-npz", type=str,
        default=str(ALIGN_STEERING_DIR / "sae_stat_results_v2.npz"),
        help="类别统计 NPZ 路径"
    )
    parser.add_argument("--skip-existing", action="store_true", help="跳过已完成的类别")
    parser.add_argument("--output-summary", type=str, default=None, help="汇总 JSON 输出路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    random.seed(args.seed)

    # 获取类别列表
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

    print(f"Classes: {selected[:5]}... (showing first 5)")
    print(f"Conditions: {args.conditions}")
    print(f"Attack methods: {args.attack_methods}")
    print(f"GPUs: {args.gpus}")

    # 保存选中的类别列表
    class_list_file = ALIGN_STEERING_DIR / "adv_samples" / "selected_classes.txt"
    class_list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(class_list_file, "w") as f:
        for cls in selected:
            f.write(f"{cls}\n")
    print(f"Selected classes saved to: {class_list_file}")

    # 构建任务队列：每个任务 = (class_name, attack_script, [conditions])
    manager = Manager()
    task_queue = manager.Queue()
    results_list = manager.list()

    attack_script_map = {"baseline": "apgdce_attack.py", "aa": "aa_attack.py"}
    for cls in selected:
        for method in args.attack_methods:
            task_queue.put((cls, attack_script_map[method], args.conditions))

    # 启动 workers
    config = {
        "n_samples": args.n_samples,
        "eps": args.eps,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "class_npz": Path(args.class_npz),
        "skip_existing": args.skip_existing,
    }

    processes = []
    for gpu_id in args.gpus:
        p = Process(target=worker, args=(gpu_id, task_queue, results_list, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll workers finished. Total result entries: {len(results_list)}")
    print("\nRunning reaggregation from disk files...")

    reagg_cmd = [
        sys.executable,
        str(ALIGN_STEERING_DIR / "reaggregate.py"),
    ]
    try:
        subprocess.run(reagg_cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Reaggregation failed: {e}")
        print("You can manually run: python reaggregate.py")

    print("Done!")


if __name__ == "__main__":
    main()
