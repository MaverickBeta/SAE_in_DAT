#!/usr/bin/env python3
"""
Fine-grained accuracy evaluation for global entry steering.

Measures 5 key metrics per parameter combo:
  1. clean_acc        — clean images, no hook (ceiling)
  2. robust_acc       — ALL adv images, no hook (baseline)
  3. steer_robust_acc — ALL adv images, with steering
  4. succ_steer_acc   — SUCCESSFUL adv only, with steering
  5. fail_steer_acc   — FAILED adv only, with steering

Also computes:
  - recovery_rate (relative to clean ceiling)
  - robust_delta (steer_robust - robust, net effect)
  - succ_delta (succ_steer - succ_control)
  - fail_delta (fail_steer - fail_control, THE KEY METRIC)

Usage:
  python grid_search_acc.py \
      --cf 0.88 --min-classes 8 10 12 --con 0.6 0.7 0.8 \
      --dir-thresh 0.5 1.0 2.0 --cv 0.5 0.8 1.0 --alpha 1.0 5.0
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
from pathlib import Path
from itertools import product
from multiprocessing import Process, set_start_method
from PIL import Image
import torchvision.transforms as transforms

BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"
FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20classes")
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20classes_global_entry")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
CLASSES = [
    ("n01440764",   0, "tench"),
    ("n01530575",  10, "brambling"),
    ("n01641577",  30, "bullfrog"),
    ("n01806143",  84, "peacock"),
    ("n01871265", 101, "tusker"),
    ("n02077923", 150, "sea_lion"),
    ("n02123045", 281, "tabby_cat"),
    ("n02128385", 288, "leopard"),
    ("n02129604", 292, "tiger"),
    ("n02165456", 301, "ladybug"),
    ("n03063599", 504, "coffee_mug"),
    ("n03085013", 508, "computer_keyboard"),
    ("n03250847", 542, "drum"),
    ("n03445777", 574, "golf_ball"),
    ("n03770439", 655, "miniskirt"),
    ("n03888257", 701, "parachute"),
    ("n04146614", 779, "school_bus"),
    ("n04285008", 817, "sports_car"),
    ("n07720875", 945, "artichoke"),
    ("n07747607", 950, "orange"),
]
N_CLASSES = len(CLASSES)
N_TOKENS = 49
N_CHANNELS = 12288

_PREPROCESS = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


def _load_np(path):
    img = Image.open(path).convert("RGB")
    img = _PREPROCESS(img)
    return np.array(img)


def preload_all_images():
    print("[Main] Preloading all images into CPU memory...")
    all_data = {}
    for wnid, cls_idx, name in CLASSES:
        clean_dir = IMAGENET_VAL / wnid
        clean_paths = sorted(clean_dir.glob("*.JPEG")) if clean_dir.is_dir() else []
        clean_arrays = [_load_np(p) for p in clean_paths]
        run_name = f"{wnid}_cls{cls_idx}_apgd_ce_l2_eps3_steps100"
        adv_dir = ADV_ROOT / run_name
        adv_entries = []
        if adv_dir.is_dir():
            for path in sorted(adv_dir.glob("*.JPEG")):
                if path.parent != adv_dir:
                    continue
                arr = _load_np(path)
                is_succ = "_succ" in path.name
                adv_entries.append((arr, is_succ))
        all_data[name] = {
            "cls_idx": cls_idx,
            "clean": clean_arrays,
            "adv": adv_entries,
        }
    total = sum(len(d["clean"]) + len(d["adv"]) for d in all_data.values())
    print(f"[Main] Loaded {total} images")
    return all_data


def compute_feature_stats():
    print("[Main] Loading raw SAE features...")
    clean_freq = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    clean_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    adv_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    for i, (wnid, cls_idx, name) in enumerate(CLASSES):
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        clean_feat = np.load(clean_path)
        adv_feat = np.load(adv_path)
        clean_freq[i] = (clean_feat != 0).mean(axis=0)
        clean_mean[i] = clean_feat.mean(axis=0)
        adv_mean[i] = adv_feat.mean(axis=0)
        del clean_feat, adv_feat
    delta = adv_mean - clean_mean
    print(f"[Main] Feature stats computed")
    return clean_freq, clean_mean, adv_mean, delta


def filter_entries(clean_freq, delta, cf_thresh, con_thresh, cv_thresh, dir_thresh, min_classes):
    valid_mask = clean_freq >= cf_thresh
    n_valid = valid_mask.sum(axis=0)
    candidate_mask = n_valid >= min_classes
    candidate_indices = np.argwhere(candidate_mask)
    entries = []
    for tok, ch in candidate_indices:
        vm = valid_mask[:, tok, ch]
        valid_deltas = delta[:, tok, ch][vm]
        n_suppress = int(np.sum(valid_deltas < -dir_thresh))
        n_enhance = int(np.sum(valid_deltas > dir_thresh))
        consistency = max(n_suppress, n_enhance) / vm.sum() if vm.sum() > 0 else 0.0
        if consistency < con_thresh:
            continue
        directional_mask = np.abs(valid_deltas) > dir_thresh
        directional_deltas = valid_deltas[directional_mask]
        if len(directional_deltas) < 2:
            continue
        mean_delta = float(np.mean(directional_deltas))
        std_delta = float(np.std(directional_deltas, ddof=0))
        mean_abs_delta = float(np.mean(np.abs(directional_deltas)))
        if mean_abs_delta < 1e-12:
            continue
        cv = std_delta / mean_abs_delta
        if cv >= cv_thresh:
            continue
        entries.append({"token": int(tok), "channel": int(ch), "mean_delta": mean_delta})
    return entries


def worker(gpu_id, param_combos, out_path, all_data_np, clean_freq, delta):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stagger = (gpu_id - 4) * 5
    if stagger > 0:
        time.sleep(stagger)

    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
    sys.path.insert(0, "/Data_share/hongyi/DAT")
    sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

    from rebm.training.utils_architecture import create_convnext_model
    from rebm.training.modeling import load_checkpoint
    from sae_core.model import TopKAutoencoder

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    all_data = {}
    for name, d in all_data_np.items():
        cls_idx = d["cls_idx"]
        all_data[name] = {
            "cls_idx": cls_idx,
            "clean": [(to_tensor(img).to(DEVICE), cls_idx) for img in d["clean"]],
            "all_adv": [(to_tensor(img).to(DEVICE), cls_idx) for img, _ in d["adv"]],
            "succ_adv": [(to_tensor(img).to(DEVICE), cls_idx) for img, is_succ in d["adv"] if is_succ],
            "fail_adv": [(to_tensor(img).to(DEVICE), cls_idx) for img, is_succ in d["adv"] if not is_succ],
        }

    print(f"[GPU {gpu_id}] Loading model...")
    model = create_convnext_model(
        model_type="convnext_large", num_classes=1000,
        normalize_input=False, use_layernorm=True, use_convstem=True,
    )
    load_checkpoint(model, BASE_CKPT)
    model = model.to(DEVICE)
    model.eval()

    sae_ckpt = torch.load(SAE_CKPT, map_location=DEVICE)
    cfg = sae_ckpt.get("config", {})
    d_in = sae_ckpt.get("d_in", cfg.get("d_in", None))
    d_lat = sae_ckpt.get("d_lat", cfg.get("d_lat", None))
    k = cfg.get("k", sae_ckpt.get("k", None))
    if d_lat is None:
        d_lat = int(d_in) * int(cfg.get("expansion_rate", 8))

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(sae_ckpt["model_state_dict"])
    sae = sae.to(DEVICE)
    sae.eval()

    norm_mean = sae_ckpt["norm_mean"].to(DEVICE)
    norm_std = sae_ckpt["norm_std"].to(DEVICE)

    @torch.no_grad()
    def evaluate(images, hook_fn=None):
        if hook_fn is not None:
            handle = model.stages[3].register_forward_hook(hook_fn)
        correct = 0
        total = 0
        try:
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE]
                imgs = torch.stack([img for img, _ in batch])
                labels = torch.tensor([label for _, label in batch], device=DEVICE)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        finally:
            if hook_fn is not None:
                handle.remove()
        return correct / total if total > 0 else 0.0

    def make_steering_hook(entries, alpha):
        def hook_fn(module, input, output):
            bsz, channels, height, width = output.shape
            flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
            flat_norm = (flat - norm_mean) / norm_std
            z = sae.encode(flat_norm)
            z_spatial = z.reshape(bsz, 7, 7, d_lat)
            for e in entries:
                row = e["token"] // 7
                col = e["token"] % 7
                z_spatial[:, row, col, e["channel"]] -= alpha * e["mean_delta"]
            z = z_spatial.reshape(bsz * 49, d_lat)
            recon_norm = (z @ sae.W_dec) + sae.b_dec
            recon_flat = recon_norm * norm_std + norm_mean
            recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
            return recon
        return hook_fn

    # Precompute control metrics (no hook)
    print(f"[GPU {gpu_id}] Precomputing control metrics...")
    control_results = {}
    for name, data in all_data.items():
        control_results[name] = {
            "clean_acc": evaluate(data["clean"]) if data["clean"] else 0.0,
            "robust_acc": evaluate(data["all_adv"]) if data["all_adv"] else 0.0,
            "succ_control_acc": evaluate(data["succ_adv"]) if data["succ_adv"] else 0.0,
            "fail_control_acc": evaluate(data["fail_adv"]) if data["fail_adv"] else 0.0,
            "n_clean": len(data["clean"]),
            "n_all_adv": len(data["all_adv"]),
            "n_succ": len(data["succ_adv"]),
            "n_fail": len(data["fail_adv"]),
        }

    total_clean = sum(r["n_clean"] for r in control_results.values())
    total_all_adv = sum(r["n_all_adv"] for r in control_results.values())
    total_succ = sum(r["n_succ"] for r in control_results.values())
    total_fail = sum(r["n_fail"] for r in control_results.values())

    overall_clean_acc = sum(r["clean_acc"] * r["n_clean"] for r in control_results.values()) / total_clean
    overall_robust_acc = sum(r["robust_acc"] * r["n_all_adv"] for r in control_results.values()) / total_all_adv
    overall_succ_control = sum(r["succ_control_acc"] * r["n_succ"] for r in control_results.values() if r["n_succ"] > 0) / total_succ if total_succ > 0 else 0
    overall_fail_control = sum(r["fail_control_acc"] * r["n_fail"] for r in control_results.values() if r["n_fail"] > 0) / total_fail if total_fail > 0 else 0

    summary = []
    print(f"[GPU {gpu_id}] Evaluating {len(param_combos)} combos...")

    for combo in tqdm(param_combos, desc=f"GPU{gpu_id}"):
        entries = filter_entries(
            clean_freq, delta,
            cf_thresh=combo["cf"],
            con_thresh=combo["con"],
            cv_thresh=combo["cv"],
            dir_thresh=combo["dir_thresh"],
            min_classes=combo["min_classes"],
        )
        n_entries = len(entries)
        alpha = combo["alpha"]

        total_all_steer_correct = 0
        total_succ_steer_correct = 0
        total_fail_steer_correct = 0
        total_clean_steer_correct = 0

        for name, data in all_data.items():
            cr = control_results[name]
            hook = make_steering_hook(entries, alpha)

            if data["all_adv"]:
                steer_robust = evaluate(data["all_adv"], hook)
                total_all_steer_correct += int(steer_robust * cr["n_all_adv"])

                if data["succ_adv"]:
                    succ_steer = evaluate(data["succ_adv"], hook)
                    total_succ_steer_correct += int(succ_steer * cr["n_succ"])

                if data["fail_adv"]:
                    fail_steer = evaluate(data["fail_adv"], hook)
                    total_fail_steer_correct += int(fail_steer * cr["n_fail"])

            if data["clean"]:
                clean_steer = evaluate(data["clean"], hook)
                total_clean_steer_correct += int(clean_steer * cr["n_clean"])

        overall_steer_robust = total_all_steer_correct / total_all_adv
        overall_succ_steer = total_succ_steer_correct / total_succ if total_succ > 0 else 0
        overall_fail_steer = total_fail_steer_correct / total_fail if total_fail > 0 else 0
        overall_clean_steer = total_clean_steer_correct / total_clean if total_clean > 0 else 0

        recovery = None
        if overall_clean_acc > overall_succ_control:
            recovery = (overall_succ_steer - overall_succ_control) / (overall_clean_acc - overall_succ_control) * 100

        robust_delta = overall_steer_robust - overall_robust_acc
        succ_delta = overall_succ_steer - overall_succ_control
        fail_delta = overall_fail_steer - overall_fail_control
        clean_delta = overall_clean_steer - overall_clean_acc

        summary.append({
            "cf": combo["cf"],
            "min_classes": combo["min_classes"],
            "con": combo["con"],
            "dir_thresh": combo["dir_thresh"],
            "cv": combo["cv"],
            "alpha": alpha,
            "n_entries": n_entries,
            "clean_acc": overall_clean_acc,
            "robust_acc": overall_robust_acc,
            "succ_control": overall_succ_control,
            "fail_control": overall_fail_control,
            "clean_steer": overall_clean_steer,
            "steer_robust_acc": overall_steer_robust,
            "succ_steer": overall_succ_steer,
            "fail_steer": overall_fail_steer,
            "clean_delta": clean_delta,
            "robust_delta": robust_delta,
            "succ_delta": succ_delta,
            "fail_delta": fail_delta,
            "recovery_rate": recovery,
        })

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GPU {gpu_id}] Done. Saved {len(summary)} results")


def main():
    parser = argparse.ArgumentParser(description="Fine-grained accuracy grid search")
    parser.add_argument("--cf", nargs="+", type=float, default=[0.88])
    parser.add_argument("--min-classes", nargs="+", type=int, default=[8, 10, 12], dest="min_classes")
    parser.add_argument("--con", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    parser.add_argument("--dir-thresh", nargs="+", type=float, default=[0.5, 1.0, 2.0], dest="dir_thresh")
    parser.add_argument("--cv", nargs="+", type=float, default=[0.5, 0.8, 1.0])
    parser.add_argument("--alpha", nargs="+", type=float, default=[1.0, 5.0])
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_data_np = preload_all_images()
    clean_freq, clean_mean, adv_mean, delta = compute_feature_stats()
    del clean_mean, adv_mean

    all_params = list(product(args.cf, args.min_classes, args.con, args.dir_thresh, args.cv, args.alpha))
    n_gpus = len(args.gpus)

    param_dicts = []
    for cf, min_classes, con, dir_thresh, cv, alpha in all_params:
        param_dicts.append({
            "cf": cf, "min_classes": min_classes, "con": con,
            "dir_thresh": dir_thresh, "cv": cv, "alpha": alpha,
        })

    print(f"{'='*80}")
    print(f"GRID SEARCH — Fine-grained accuracy evaluation")
    print(f"  GPUs:      {args.gpus}")
    print(f"  Combos:    {len(param_dicts)}")
    print(f"  cf:        {args.cf}")
    print(f"  min_cls:   {args.min_classes}")
    print(f"  con:       {args.con}")
    print(f"  dir:       {args.dir_thresh}")
    print(f"  cv:        {args.cv}")
    print(f"  alpha:     {args.alpha}")
    print(f"{'='*80}\n")

    random.shuffle(param_dicts)
    chunks = [param_dicts[i::n_gpus] for i in range(n_gpus)]
    for i, gpu_id in enumerate(args.gpus):
        print(f"  GPU {gpu_id}: {len(chunks[i])} combos")

    set_start_method("spawn", force=True)
    processes = []
    tmp_paths = []
    for i, gpu_id in enumerate(args.gpus):
        tmp_path = OUT_DIR / f"grid_acc_tmp_gpu{gpu_id}.json"
        tmp_paths.append(tmp_path)
        tmp_path.unlink(missing_ok=True)
        p = Process(target=worker, args=(gpu_id, chunks[i], tmp_path, all_data_np, clean_freq, delta))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    for tmp_path in tmp_paths:
        if tmp_path.exists():
            with open(tmp_path) as f:
                all_results.extend(json.load(f))

    if not all_results:
        print("No results!")
        return

    all_results = [r for r in all_results if r["recovery_rate"] is not None]
    all_results.sort(key=lambda x: x["recovery_rate"], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP-20 RESULTS (by Recovery Rate)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Params':>35} {'N':>3} {'Clean':>6} {'ClSteer':>7} {'Robust':>6} {'StRob':>6} {'Succ+':>6} {'Fail+':>6} {'Recov':>6}")
    print("-" * 110)
    for i, r in enumerate(all_results[:20], 1):
        suffix = f"cf{r['cf']:.2f}/min{r['min_classes']}/con{r['con']}/dir{r['dir_thresh']}/cv{r['cv']}/a{r['alpha']}"
        print(f"{i:>4} {suffix:>35} {r['n_entries']:>3} "
              f"{r['clean_acc']*100:>5.1f}% {r['clean_steer']*100:>6.1f}% "
              f"{r['robust_acc']*100:>5.1f}% {r['steer_robust_acc']*100:>5.1f}% "
              f"{r['succ_steer']*100:>5.1f}% {r['fail_steer']*100:>5.1f}% {r['recovery_rate']:>5.1f}%")

    print(f"\n{'='*100}")
    print(f"TOP-10 BY FAIL SAFETY (least negative fail_delta)")
    print(f"{'='*100}")
    safe_results = sorted(all_results, key=lambda x: x["fail_delta"], reverse=True)
    print(f"{'Rank':>4} {'Params':>35} {'CleanΔ':>7} {'FailΔ':>7} {'SuccΔ':>7} {'RobustΔ':>8} {'Recov':>6}")
    print("-" * 110)
    for i, r in enumerate(safe_results[:10], 1):
        suffix = f"cf{r['cf']:.2f}/min{r['min_classes']}/con{r['con']}/dir{r['dir_thresh']}/cv{r['cv']}/a{r['alpha']}"
        print(f"{i:>4} {suffix:>35} {r['clean_delta']*100:>+6.2f}% {r['fail_delta']*100:>+6.2f}% "
              f"{r['succ_delta']*100:>+6.2f}% {r['robust_delta']*100:>+7.2f}% {r['recovery_rate']:>5.1f}%")

    out_json = OUT_DIR / "grid_search_acc_results.json"
    with open(out_json, "w") as f:
        json.dump({"config": {"gpus": args.gpus, "alphas": args.alpha}, "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_json}")

    if all_results:
        best = all_results[0]
        print(f"\nBest: recovery={best['recovery_rate']:.2f}%, fail_delta={best['fail_delta']*100:+.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
