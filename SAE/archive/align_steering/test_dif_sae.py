#!/usr/bin/env python3
"""
批量对比两组 SAE 在 DAT 模型上的 Clean Acc 与 APGD-CE Robust Acc。

三组对照条件：
  1) no_sae   : 原始模型，无 SAE
  2) sae_64k  : 旧版 merged-feature SAE (d_lat=65536)
  3) sae_exp32: 新版 stage3 SAE (expansion_rate=32, d_lat=49152)

支持多 GPU 并行，每张卡独立取任务执行，崩溃隔离。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
import multiprocessing
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from rebm.attacks.attack_steps import LinfStep
from sae_core.model import TopKAutoencoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: T.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        x = self.transform(rgb)
        class_name = path.parent.name
        return x, class_name, str(path)


# ---------------------------------------------------------------------------
# Model / SAE builders
# ---------------------------------------------------------------------------
def build_model(device: torch.device, checkpoint: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, checkpoint)
    model = model.to(device)
    model.eval()
    return model


def build_sae(device: torch.device, sae_ckpt_path: str):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = config.get("k", ckpt.get("k", None))

    if d_in is None:
        raise KeyError("Cannot resolve d_in from SAE checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from SAE checkpoint")
        d_lat = int(d_in) * int(expansion_rate)
    if k is None:
        raise KeyError("Cannot resolve k from SAE checkpoint")

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg["d_in"] = int(d_in)
    resolved_cfg["d_lat"] = int(d_lat)
    resolved_cfg["k"] = int(k)
    return sae, norm_mean, norm_std, resolved_cfg


def make_sae_hook(sae_model, norm_mean, norm_std):
    def hook(module, input, output):
        B, C, H, W = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - norm_mean) / norm_std
        x_reconstruct, _, _ = sae_model(flat_norm)
        x_out = x_reconstruct * norm_std + norm_mean
        return x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return hook


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_class_dirs(val_dir: Path) -> List[Path]:
    return sorted([p for p in val_dir.iterdir() if p.is_dir()])


def sample_images_from_class(class_dir: Path, n_samples: int) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not images:
        return []
    n_samples = min(n_samples, len(images))
    return random.sample(images, k=n_samples)


# ---------------------------------------------------------------------------
# APGD-CE attack
# ---------------------------------------------------------------------------
def apgd_ce_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.LongTensor,
    eps: float = 8 / 255,
    steps: int = 100,
    step_size: float = None,
    random_start: bool = True,
) -> torch.Tensor:
    if step_size is None:
        step_size = eps / 4

    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    x0 = x.clone().detach()
    step = LinfStep(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )

        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x.clone().detach()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eps: float,
    steps: int,
    step_size: float,
    class_to_idx: dict,
) -> dict:
    clean_correct = 0
    adv_correct = 0
    total = 0

    all_clean_preds = []
    all_adv_preds = []
    all_labels = []
    all_paths = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    for batch_idx, (images, class_names, paths) in enumerate(pbar):
        labels = torch.tensor([class_to_idx[name] for name in class_names], device=device)
        images = images.to(device)

        batch_size = images.size(0)
        total += batch_size

        with torch.no_grad():
            logits_clean = model(images)
            preds_clean = logits_clean.argmax(dim=1)
            clean_correct += (preds_clean == labels).sum().item()

        x_adv = apgd_ce_attack(
            model=model,
            x=images,
            labels=labels,
            eps=eps,
            steps=steps,
            step_size=step_size,
            random_start=True,
        )

        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()

        all_clean_preds.extend(preds_clean.cpu().tolist())
        all_adv_preds.extend(preds_adv.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_paths.extend(paths)

        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        pbar.set_postfix({
            'Clean': f'{clean_acc:.2f}%',
            'Adv': f'{adv_acc:.2f}%',
        })

    return {
        'clean_accuracy': 100.0 * clean_correct / total,
        'adversarial_accuracy': 100.0 * adv_correct / total,
        'attack_success_rate': 100.0 * (total - adv_correct) / total,
        'total_samples': total,
        'clean_correct': clean_correct,
        'adv_correct': adv_correct,
        'predictions': {
            'clean': all_clean_preds,
            'adversarial': all_adv_preds,
            'true': all_labels,
            'paths': all_paths,
        }
    }


# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------
def run_single_condition(
    class_name: str,
    condition: str,
    sae_ckpt: Optional[str],
    config: dict,
    device: torch.device,
) -> Dict:
    """
    对单个类别、单个条件进行评估。
    Returns: {"success": bool, "results": dict or None, "error": str or None}
    """
    val_dir = Path(config["val_dir"]).resolve()
    class_dirs = get_class_dirs(val_dir)
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    if class_name not in class_to_idx:
        return {"success": False, "results": None, "error": f"Class {class_name} not found"}

    class_dir = val_dir / class_name
    image_paths = sample_images_from_class(class_dir, config["n_samples"])
    if not image_paths:
        return {"success": False, "results": None, "error": f"No images in {class_dir}"}

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = build_model(device=device, checkpoint=config["checkpoint"])
    sae_hook_handle = None

    try:
        if condition != "no_sae" and sae_ckpt is not None:
            sae_model, norm_mean, norm_std, sae_cfg = build_sae(
                device=device, sae_ckpt_path=sae_ckpt
            )
            sae_hook_handle = model.stages[config["sae_stage"]].register_forward_hook(
                make_sae_hook(sae_model, norm_mean, norm_std)
            )

        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            eps=config["eps"],
            steps=config["steps"],
            step_size=config.get("step_size"),
            class_to_idx=class_to_idx,
        )

        return {"success": True, "results": results, "error": None}

    except Exception as e:
        return {"success": False, "results": None, "error": str(e)}

    finally:
        if sae_hook_handle is not None:
            sae_hook_handle.remove()
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker(gpu_id: int, task_queue, results_list, config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Worker GPU {gpu_id}] Started on {device}")

    while True:
        try:
            task = task_queue.get(timeout=10)
        except Empty:
            print(f"[Worker GPU {gpu_id}] Queue empty, exiting")
            break

        if task is None:
            print(f"[Worker GPU {gpu_id}] Received stop signal")
            break

        class_name, condition, sae_ckpt = task
        print(f"[Worker GPU {gpu_id}] {class_name} | {condition}")

        result = run_single_condition(
            class_name=class_name,
            condition=condition,
            sae_ckpt=sae_ckpt,
            config=config,
            device=device,
        )

        results_list.append({
            "gpu": gpu_id,
            "class_name": class_name,
            "condition": condition,
            "success": result["success"],
            "error": result["error"],
            "results": result["results"],
        })

        if result["success"]:
            # 保存到文件
            out_dir = Path(config["output_dir"]) / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"attack_results_{class_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "meta": {
                        "class_name": class_name,
                        "condition": condition,
                        "checkpoint": config["checkpoint"],
                        "eps": config["eps"],
                        "steps": config["steps"],
                        "sae_ckpt": sae_ckpt,
                        "sae_stage": config["sae_stage"] if condition != "no_sae" else None,
                    },
                    "accuracy": {
                        k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in result["results"].items()
                        if k not in ("predictions",)
                    }
                }, f, indent=2)
            print(f"[Worker GPU {gpu_id}] {class_name} | {condition} -> saved")
        else:
            print(f"[Worker GPU {gpu_id}] {class_name} | {condition} FAILED: {result['error']}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(output_dir: Path, conditions: List[str]):
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    agg = defaultdict(lambda: {"count": 0, "clean": [], "adv": [], "asr": []})

    for condition in conditions:
        cond_dir = output_dir / condition
        if not cond_dir.exists():
            continue
        for f in sorted(cond_dir.glob("attack_results_*.json")):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            acc = data.get("accuracy", {})
            agg[condition]["count"] += 1
            agg[condition]["clean"].append(acc.get("clean_accuracy", 0.0))
            agg[condition]["adv"].append(acc.get("adversarial_accuracy", 0.0))
            agg[condition]["asr"].append(acc.get("attack_success_rate", 0.0))

    header = f"{'Condition':<15} {'Classes':>8} {'Clean Acc':>12} {'Adv Acc':>12} {'ASR':>12}"
    print(header)
    print("-" * len(header))

    for condition in conditions:
        d = agg[condition]
        n = d["count"]
        if n == 0:
            print(f"{condition:<15} {'—':>8} {'—':>12} {'—':>12} {'—':>12}")
            continue
        clean_avg = sum(d["clean"]) / n
        adv_avg = sum(d["adv"]) / n
        asr_avg = sum(d["asr"]) / n
        print(f"{condition:<15} {n:>8} {clean_avg:>11.2f}% {adv_avg:>11.2f}% {asr_avg:>11.2f}%")

    print("=" * 80)

    # 保存汇总 JSON
    summary = {}
    for condition in conditions:
        d = agg[condition]
        n = d["count"]
        if n == 0:
            continue
        summary[condition] = {
            "count": n,
            "avg_clean_accuracy": round(sum(d["clean"]) / n, 2),
            "avg_adversarial_accuracy": round(sum(d["adv"]) / n, 2),
            "avg_attack_success_rate": round(sum(d["asr"]) / n, 2),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # PyTorch requires 'spawn' when CUDA is used in subprocesses
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Batch compare two SAE checkpoints on Clean + APGD-CE Robust Acc"
    )
    parser.add_argument("--val-dir", type=str,
                        default=str(REPO_ROOT / "data" / "ImageNet" / "val"),
                        help="ImageNet validation directory")
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
                        help="DAT model checkpoint")
    parser.add_argument("--n-classes", type=int, default=50,
                        help="Number of classes to evaluate")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Images per class")
    parser.add_argument("--eps", type=float, default=8 / 255,
                        help="APGD Linf epsilon")
    parser.add_argument("--steps", type=int, default=100,
                        help="APGD steps")
    parser.add_argument("--step-size", type=float, default=None,
                        help="APGD step size (default eps/4)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU IDs to use")
    parser.add_argument("--sae-stage", type=int, default=3,
                        help="Stage index to mount SAE")
    parser.add_argument("--class-list", type=str, default=None,
                        help="Optional file with one class name per line")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "compare_sae_results"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already evaluated class/condition combinations")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------------------------
    # 定义三组对照条件
    # -----------------------------------------------------------------------
    conditions = {
        "no_sae": None,
        "sae_64k": str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
        "sae_exp32": str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "stage3" / "k64_exp32" / "sae_stage3_din1536_exp32_k64_step_50000.pt"),
    }

    # -----------------------------------------------------------------------
    # 选取类别（支持 resume 复用之前的列表）
    # -----------------------------------------------------------------------
    val_dir = Path(args.val_dir).resolve()
    all_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_list_file = output_dir / "selected_classes.txt"

    if args.resume and class_list_file.exists():
        with open(class_list_file, "r") as f:
            selected = [line.strip() for line in f if line.strip()]
        selected = [c for c in selected if c in all_classes]
        print(f"[Resume] Loaded {len(selected)} classes from existing {class_list_file}")
    elif args.class_list is not None:
        with open(args.class_list, "r") as f:
            selected = [line.strip() for line in f if line.strip()]
        selected = [c for c in selected if c in all_classes]
        print(f"Loaded {len(selected)} valid classes from {args.class_list}")
    else:
        selected = random.sample(all_classes, k=min(args.n_classes, len(all_classes)))
        selected = sorted(selected)
        print(f"Randomly selected {len(selected)} classes")

    # 保存类别列表
    with open(class_list_file, "w") as f:
        for c in selected:
            f.write(f"{c}\n")
    print(f"Class list saved to: {class_list_file}")

    # -----------------------------------------------------------------------
    # 构建任务队列：(class_name, condition, sae_ckpt)
    # -----------------------------------------------------------------------
    manager = Manager()
    task_queue = manager.Queue()
    results_list = manager.list()

    skipped = 0
    for class_name in selected:
        for cond_name, sae_ckpt in conditions.items():
            if args.resume:
                result_path = output_dir / cond_name / f"attack_results_{class_name}.json"
                if result_path.exists():
                    skipped += 1
                    continue
            task_queue.put((class_name, cond_name, sae_ckpt))

    if skipped:
        print(f"[Resume] Skipped {skipped} already completed tasks")
    print(f"Total tasks to run: {task_queue.qsize() - len(args.gpus)}")  # minus None sentinels

    # 发送终止信号
    for _ in args.gpus:
        task_queue.put(None)

    config = {
        "val_dir": str(val_dir),
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "eps": args.eps,
        "steps": args.steps,
        "step_size": args.step_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sae_stage": args.sae_stage,
        "output_dir": str(output_dir),
    }

    # -----------------------------------------------------------------------
    # 启动 workers
    # -----------------------------------------------------------------------
    processes = []
    for gpu_id in args.gpus:
        p = Process(target=worker, args=(gpu_id, task_queue, results_list, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll workers finished. Total results: {len(results_list)}")

    # -----------------------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------------------
    aggregate_results(output_dir, list(conditions.keys()))
    print("\nDone!")


if __name__ == "__main__":
    main()
