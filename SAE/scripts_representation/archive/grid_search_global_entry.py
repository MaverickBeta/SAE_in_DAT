#!/usr/bin/env python3
"""
Multi-GPU grid search over global entry filtering hyperparameters.

Generates entries ON-THE-FLY from raw SAE features instead of relying on a
pre-filtered JSON. This means --cf / --con / --cv actually expand the candidate
pool rather than just sub-sampling a fixed set.
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

# ── Paths & constants ───────────────────────────────────────────────
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

# Preprocessing for image loading (CPU-only)
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
    print(f"[Main] Loaded {total} images (~{total * 224 * 224 * 3 / 1e6:.0f} MB as uint8)")
    return all_data


def compute_feature_stats():
    """Load raw SAE features and compute per-class per-entry statistics."""
    print("[Main] Loading raw SAE features...")
    clean_freq = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    clean_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)
    adv_mean = np.zeros((N_CLASSES, N_TOKENS, N_CHANNELS), dtype=np.float32)

    for i, (wnid, cls_idx, name) in enumerate(CLASSES):
        clean_path = FEAT_DIR / f"clean_{wnid}_cls{cls_idx}_stage3_k256_features.npy"
        adv_path = FEAT_DIR / f"adv_{wnid}_cls{cls_idx}_stage3_k256_features.npy"

        clean_feat = np.load(clean_path)   # (n_clean, 49, 12288)
        adv_feat = np.load(adv_path)       # (n_adv, 49, 12288)

        clean_freq[i] = (clean_feat != 0).mean(axis=0)
        clean_mean[i] = clean_feat.mean(axis=0)
        adv_mean[i] = adv_feat.mean(axis=0)

        del clean_feat, adv_feat

    delta = adv_mean - clean_mean
    print(f"[Main] Feature stats computed: {clean_freq.nbytes / 1e6:.1f} MB per array")
    return clean_freq, clean_mean, adv_mean, delta


def filter_entries_from_arrays(clean_freq, delta, cf_thresh, con_thresh, cv_thresh,
                               dir_thresh, min_classes):
    """
    Probe global entries directly from statistic arrays.
    Logic mirrors probe_global_entries.py exactly.
    """
    valid_mask = clean_freq >= cf_thresh          # (N_CLASSES, N_TOKENS, N_CHANNELS)
    n_valid = valid_mask.sum(axis=0)              # (N_TOKENS, N_CHANNELS)
    candidate_mask = n_valid >= min_classes       # (N_TOKENS, N_CHANNELS)
    candidate_indices = np.argwhere(candidate_mask)

    entries = []
    for tok, ch in candidate_indices:
        vm = valid_mask[:, tok, ch]
        valid_deltas = delta[:, tok, ch][vm]

        n_suppress = int(np.sum(valid_deltas < -dir_thresh))
        n_enhance = int(np.sum(valid_deltas > dir_thresh))
        n_neutral = vm.sum() - n_suppress - n_enhance

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

        entries.append({
            "token": int(tok),
            "channel": int(ch),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "cv": cv,
            "consistency": consistency,
            "n_valid": int(vm.sum()),
            "direction": "SUPPRESS" if n_suppress > n_enhance else "ENHANCE",
        })

    return entries


# ── Per-GPU worker ──────────────────────────────────────────────────
def worker(gpu_id, params, out_path, all_data_np, clean_freq, delta):
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

    # Convert numpy images to GPU tensors
    to_tensor = transforms.ToTensor()
    all_data = {}
    for name, d in all_data_np.items():
        cls_idx = d["cls_idx"]
        clean_tensors = [(to_tensor(img).to(DEVICE), cls_idx) for img in d["clean"]]
        all_adv = [(to_tensor(img).to(DEVICE), cls_idx) for img, _ in d["adv"]]
        succ_adv = [(to_tensor(img).to(DEVICE), cls_idx) for img, is_succ in d["adv"] if is_succ]
        all_data[name] = {
            "cls_idx": cls_idx,
            "clean": clean_tensors,
            "all_adv": all_adv,
            "succ_adv": succ_adv,
        }

    # Load model & SAE
    print(f"[GPU {gpu_id}] Loading model...")
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
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
    def evaluate(images, model, hook_fn=None):
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

    def make_universal_steering_hook(entries):
        def hook_fn(module, input, output):
            bsz, channels, height, width = output.shape
            flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
            flat_norm = (flat - norm_mean) / norm_std
            z = sae.encode(flat_norm)
            z_spatial = z.reshape(bsz, 7, 7, d_lat)
            for e in entries:
                token, channel = e["token"], e["channel"]
                row = token // 7
                col = token % 7
                mean_delta = e["mean_delta"]
                z_spatial[:, row, col, channel] = z_spatial[:, row, col, channel] - mean_delta
            z = z_spatial.reshape(bsz * 49, d_lat)
            recon_norm = (z @ sae.W_dec) + sae.b_dec
            recon_flat = recon_norm * norm_std + norm_mean
            recon = recon_flat.reshape(bsz, 7, 7, channels).permute(0, 3, 1, 2)
            return recon
        return hook_fn

    # Precompute control metrics
    print(f"[GPU {gpu_id}] Precomputing control metrics...")
    control_results = {}
    for name, data in all_data.items():
        control_results[name] = {
            "clean_acc": evaluate(data["clean"], model) if data["clean"] else 0.0,
            "robust_control_acc": evaluate(data["all_adv"], model) if data["all_adv"] else 0.0,
            "succ_control_acc": evaluate(data["succ_adv"], model) if data["succ_adv"] else 0.0,
            "n_clean": len(data["clean"]),
            "n_all_adv": len(data["all_adv"]),
            "n_succ": len(data["succ_adv"]),
        }

    total_all_adv = sum(r["n_all_adv"] for r in control_results.values())
    total_succ = sum(r["n_succ"] for r in control_results.values())
    overall_clean_acc = sum(r["clean_acc"] * r["n_clean"] for r in control_results.values()) / \
                        sum(r["n_clean"] for r in control_results.values())
    overall_robust_control = sum(r["robust_control_acc"] * r["n_all_adv"] for r in control_results.values()) / total_all_adv
    overall_succ_control = sum(r["succ_control_acc"] * r["n_succ"] for r in control_results.values() if r["n_succ"] > 0) / total_succ if total_succ > 0 else 0

    # Evaluate parameter combinations
    print(f"[GPU {gpu_id}] Evaluating {len(params)} combinations...")
    summary = []
    for cf, min_classes, con, dir_thresh, cv in tqdm(params, desc=f"GPU{gpu_id}", position=gpu_id, leave=False):
        entries_info = filter_entries_from_arrays(
            clean_freq, delta,
            cf_thresh=cf,
            con_thresh=con,
            cv_thresh=cv,
            dir_thresh=dir_thresh,
            min_classes=min_classes,
        )
        n_entries = len(entries_info)

        total_all_steering_correct = 0
        total_succ_steering_correct = 0

        for name, data in all_data.items():
            cr = control_results[name]
            if not data["all_adv"]:
                continue

            hook = make_universal_steering_hook(entries_info)
            robust_steering_acc = evaluate(data["all_adv"], model, hook)
            total_all_steering_correct += int(robust_steering_acc * cr["n_all_adv"])

            if data["succ_adv"]:
                hook = make_universal_steering_hook(entries_info)
                succ_steering_acc = evaluate(data["succ_adv"], model, hook)
                total_succ_steering_correct += int(succ_steering_acc * cr["n_succ"])

        overall_robust_steering = total_all_steering_correct / total_all_adv
        overall_succ_steering = total_succ_steering_correct / total_succ if total_succ > 0 else 0

        recovery = None
        if overall_clean_acc > overall_succ_control:
            recovery = (overall_succ_steering - overall_succ_control) / (overall_clean_acc - overall_succ_control) * 100

        suffix = f"cf_{int(cf*100):d}_min{min_classes}_con_{int(con*100):d}_dir{int(dir_thresh*10):d}_cv_{int(cv*100):d}"
        summary.append({
            "suffix": suffix,
            "cf": cf,
            "min_classes": min_classes,
            "con": con,
            "dir_thresh": dir_thresh,
            "cv": cv,
            "n_entries": n_entries,
            "robust_control": overall_robust_control,
            "robust_steering": overall_robust_steering,
            "robust_improvement": overall_robust_steering - overall_robust_control,
            "succ_control": overall_succ_control,
            "succ_steering": overall_succ_steering,
            "recovery_rate": recovery,
        })

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GPU {gpu_id}] Done. Saved {len(summary)} results to {out_path}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU grid search global entry steering (on-the-fly probing)")
    parser.add_argument("--cf", nargs="+", type=float,
                        default=[0.92, 0.90, 0.88],
                        help="List of clean-freq thresholds")
    parser.add_argument("--con", nargs="+", type=float,
                        default=[0.80, 0.70, 0.60],
                        help="List of consistency thresholds")
    parser.add_argument("--cv", nargs="+", type=float,
                        default=[0.50, 0.80, 1.00],
                        help="List of CV thresholds")
    parser.add_argument("--dir-thresh", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="List of |delta| thresholds (default: 0.5 1.0 2.0)")
    parser.add_argument("--min-classes", nargs="+", type=int, default=[12, 10, 8],
                        help="List of minimum valid classes (default: 12 10 8)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7],
                        help="GPU IDs to use")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Print top-k results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling parameter combos")
    args = parser.parse_args()

    random.seed(args.seed)

    # Preload images (CPU only)
    all_data_np = preload_all_images()

    # Compute feature statistics (CPU only)
    clean_freq, clean_mean, adv_mean, delta = compute_feature_stats()

    # We only need clean_freq and delta for entry probing; free the rest
    del clean_mean, adv_mean

    all_params = list(product(args.cf, args.min_classes, args.con, args.dir_thresh, args.cv))
    n_gpus = len(args.gpus)

    print(f"{'='*80}")
    print(f"MULTI-GPU GRID SEARCH (on-the-fly probing)")
    print(f"  GPUs:      {args.gpus}")
    print(f"  Combos:    {len(all_params)} total")
    print(f"  cf:        {args.cf}")
    print(f"  con:       {args.con}")
    print(f"  cv:        {args.cv}")
    print(f"  dir_thresh:{args.dir_thresh}")
    print(f"{'='*80}\n")

    # Shuffle then chunk so each GPU gets a mix of cf/con/cv values
    random.shuffle(all_params)
    chunks = [all_params[i::n_gpus] for i in range(n_gpus)]
    for i, gpu_id in enumerate(args.gpus):
        print(f"  GPU {gpu_id}: {len(chunks[i])} combos")

    # Launch one process per GPU
    set_start_method("spawn", force=True)
    processes = []
    tmp_paths = []
    for i, gpu_id in enumerate(args.gpus):
        tmp_path = OUT_DIR / f"grid_search_tmp_gpu{gpu_id}.json"
        tmp_paths.append(tmp_path)
        tmp_path.unlink(missing_ok=True)
        p = Process(target=worker, args=(
            gpu_id, chunks[i], tmp_path, all_data_np, clean_freq, delta
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results
    all_results = []
    for tmp_path in tmp_paths:
        if tmp_path.exists():
            with open(tmp_path) as f:
                all_results.extend(json.load(f))

    if not all_results:
        print("No results produced!")
        return

    all_results = [r for r in all_results if r["recovery_rate"] is not None]
    all_results.sort(key=lambda x: x["recovery_rate"], reverse=True)

    # Print top-k
    print(f"\n{'='*80}")
    print(f"TOP-{args.top_k} RESULTS (by Recovery Rate)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Suffix':>20} {'N_ent':>5} {'Robust_C':>8} {'Robust_S':>8} {'Imp':>6} {'Succ_C':>7} {'Succ_S':>7} {'Recov%':>7}")
    print("-" * 80)
    for i, r in enumerate(all_results[:args.top_k], 1):
        print(f"{i:>4} {r['suffix']:>20} {r['n_entries']:>5} "
              f"{r['robust_control']*100:>7.1f}% {r['robust_steering']*100:>7.1f}% "
              f"{r['robust_improvement']*100:>+5.1f}% "
              f"{r['succ_control']*100:>6.1f}% {r['succ_steering']*100:>6.1f}% "
              f"{r['recovery_rate']:>6.1f}%")

    summary_path = OUT_DIR / "grid_search_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "gpus": args.gpus,
                "cf_vals": args.cf,
                "con_vals": args.con,
                "cv_vals": args.cv,
                "dir_thresh": args.dir_thresh,
                "min_classes": args.min_classes,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")

    if all_results:
        best = all_results[0]
        print(f"\nBest config: {best['suffix']} (recovery={best['recovery_rate']:.1f}%, n_entries={best['n_entries']})")
        print("Run for detailed per-class results:")
        print(f"  python global_entry_steering.py --cf {best['cf']} --con {best['con']} --cv {best['cv']} --min-classes {best['min_classes']} --dir-thresh {best['dir_thresh']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
