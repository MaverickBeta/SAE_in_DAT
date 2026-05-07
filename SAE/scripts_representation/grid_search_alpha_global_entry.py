#!/usr/bin/env python3
"""
Grid search over steering alpha for ALL param combos with recovery > 3.0%.

For each param combo, tests alpha = [0.2, 0.5, 1.0, 2.0, 5.0].
Runs on 4 GPUs in parallel.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, set_start_method

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint
from sae_core.model import TopKAutoencoder

# ── Paths ───────────────────────────────────────────────────────────
BASE_CKPT = "/Data_share/hongyi/DAT/checkpoints/convnext_large_model_bestfid.pth"
SAE_CKPT = "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage3/k256_exp8/sae_stage3_din1536_exp8_k256_step_50000.pt"
FEAT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation/features_20classes")
ADV_ROOT = Path("/Data_share/hongyi/DAT/SAE/results_representation/untar_samples_20classes")
IMAGENET_VAL = Path("/Data_share/hongyi/imagenet/val")
PREV_JSON = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20classes_global_entry/grid_search_summary.json")
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


def load_images():
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
    return all_data


def load_feature_stats():
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
    del clean_mean, adv_mean
    return clean_freq, delta


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
        mean_abs_delta = float(np.mean(np.abs(directional_deltas)))
        if mean_abs_delta < 1e-12:
            continue
        std_delta = float(np.std(directional_deltas, ddof=0))
        cv = std_delta / mean_abs_delta
        if cv >= cv_thresh:
            continue
        entries.append({
            "token": int(tok),
            "channel": int(ch),
            "mean_delta": float(np.mean(directional_deltas)),
        })
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

    # Convert images
    to_tensor = transforms.ToTensor()
    all_data = {}
    for name, d in all_data_np.items():
        cls_idx = d["cls_idx"]
        all_data[name] = {
            "cls_idx": cls_idx,
            "clean": [(to_tensor(img).to(DEVICE), cls_idx) for img in d["clean"]],
            "all_adv": [(to_tensor(img).to(DEVICE), cls_idx) for img, _ in d["adv"]],
            "succ_adv": [(to_tensor(img).to(DEVICE), cls_idx) for img, is_succ in d["adv"] if is_succ],
        }

    # Load model
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

    def make_steering_hook(entries, alpha):
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
                z_spatial[:, row, col, channel] -= alpha * mean_delta
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

    alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
    summary = []

    print(f"[GPU {gpu_id}] Evaluating {len(param_combos)} param combos × {len(alphas)} alphas...")
    for combo in tqdm(param_combos, desc=f"GPU{gpu_id}"):
        # Filter entries once per combo
        entries = filter_entries(
            clean_freq, delta,
            cf_thresh=combo["cf"],
            con_thresh=combo["con"],
            cv_thresh=combo["cv"],
            dir_thresh=combo["dir_thresh"],
            min_classes=combo["min_classes"],
        )
        n_entries = len(entries)

        for alpha in alphas:
            total_all_steering_correct = 0
            total_succ_steering_correct = 0

            for name, data in all_data.items():
                cr = control_results[name]
                if not data["all_adv"]:
                    continue

                hook = make_steering_hook(entries, alpha)
                robust_steering_acc = evaluate(data["all_adv"], model, hook)
                total_all_steering_correct += int(robust_steering_acc * cr["n_all_adv"])

                if data["succ_adv"]:
                    hook = make_steering_hook(entries, alpha)
                    succ_steering_acc = evaluate(data["succ_adv"], model, hook)
                    total_succ_steering_correct += int(succ_steering_acc * cr["n_succ"])

            overall_robust_steering = total_all_steering_correct / total_all_adv
            overall_succ_steering = total_succ_steering_correct / total_succ if total_succ > 0 else 0

            recovery = None
            if overall_clean_acc > overall_succ_control:
                recovery = (overall_succ_steering - overall_succ_control) / (overall_clean_acc - overall_succ_control) * 100

            suffix = f"cf_{int(combo['cf']*100):d}_min{combo['min_classes']}_con_{int(combo['con']*100):d}_dir{int(combo['dir_thresh']*10):d}_cv_{int(combo['cv']*100):d}_a{int(alpha*10):d}"

            summary.append({
                "suffix": suffix,
                "base_suffix": combo["suffix"],
                "cf": combo["cf"],
                "min_classes": combo["min_classes"],
                "con": combo["con"],
                "dir_thresh": combo["dir_thresh"],
                "cv": combo["cv"],
                "alpha": alpha,
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


def main():
    # Load previous results and filter recovery > 3.0%
    print("[Alpha] Loading previous grid search results...")
    with open(PREV_JSON) as f:
        prev_results = json.load(f)["results"]

    selected = [r for r in prev_results if r["recovery_rate"] > 3.0]
    print(f"Selected {len(selected)} param combos with recovery > 3.0%")

    # Load images and features in main process
    all_data_np = load_images()
    clean_freq, delta = load_feature_stats()

    n_gpus = 4
    gpu_ids = [4, 5, 6, 7]
    chunks = [selected[i::n_gpus] for i in range(n_gpus)]

    print(f"\n{'='*80}")
    print(f"ALPHA GRID SEARCH")
    print(f"  Param combos: {len(selected)}")
    print(f"  Alphas: [0.2, 0.5, 1.0, 2.0, 5.0]")
    print(f"  Total evaluations: {len(selected) * 5}")
    print(f"  GPUs: {gpu_ids}")
    print(f"{'='*80}\n")

    for i, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(chunks[i])} param combos ({len(chunks[i])*5} evaluations)")

    # Launch workers
    set_start_method("spawn", force=True)
    processes = []
    tmp_paths = []
    for i, gpu_id in enumerate(gpu_ids):
        tmp_path = OUT_DIR / f"alpha_search_tmp_gpu{gpu_id}.json"
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

    # Print top-20
    print(f"\n{'='*80}")
    print(f"TOP-20 RESULTS (by Recovery Rate)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Suffix':>40} {'N_ent':>5} {'Alpha':>5} {'Robust_S':>8} {'Recov%':>7}")
    print("-" * 80)
    for i, r in enumerate(all_results[:20], 1):
        print(f"{i:>4} {r['suffix']:>40} {r['n_entries']:>5} {r['alpha']:>5.1f} {r['robust_steering']*100:>7.1f}% {r['recovery_rate']:>6.1f}%")

    # Save summary
    out_json = OUT_DIR / "alpha_grid_search_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "gpus": gpu_ids,
                "alphas": [0.2, 0.5, 1.0, 2.0, 5.0],
                "n_param_combos": len(selected),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {out_json}")

    if all_results:
        best = all_results[0]
        print(f"\nBest: {best['suffix']} (recovery={best['recovery_rate']:.1f}%, alpha={best['alpha']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
