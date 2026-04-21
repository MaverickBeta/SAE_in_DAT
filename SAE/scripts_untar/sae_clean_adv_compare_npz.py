#!/usr/bin/env python3
"""Compare SAE representations between clean and adversarial samples from shard files.

Inputs:
- One run directory containing rank folders (or one rank folder directly)
- Shards saved by gen_evaluate_imagenet.py: shard_*.npz or shard_*.pt

Outputs:
- summary JSON with distribution/paired statistics
- visualization figures for mean activation comparison and top changed features
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Resolve DAT repository root from current script location: DAT/SAE/scripts_untar/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Use local repository modules.
import sys

sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


def find_rank_dirs(run_dir: Path) -> List[Path]:
    rank_dirs = []
    if (run_dir / "metrics.json").exists():
        rank_dirs.append(run_dir)
    rank_dirs.extend(sorted([p for p in run_dir.glob("rank*") if p.is_dir() and (p / "metrics.json").exists()]))

    uniq = []
    seen = set()
    for p in rank_dirs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if not uniq:
        raise FileNotFoundError(f"No rank directories found under: {run_dir}")
    return uniq


def load_shard_arrays(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        d = np.load(path)
        return {
            "x_clean": d["x_clean"],
            "x_adv": d["x_adv"],
            "sample_ids": d["sample_ids"],
        }
    if path.suffix == ".pt":
        d = torch.load(path, map_location="cpu")
        return {
            "x_clean": d["x_clean"].cpu().numpy(),
            "x_adv": d["x_adv"].cpu().numpy(),
            "sample_ids": d["sample_ids"].cpu().numpy(),
        }
    raise ValueError(f"Unsupported shard extension: {path}")


def to_float_chw(arr: np.ndarray) -> np.ndarray:
    # npz shards from gen_evaluate_imagenet.py are uint8 in [0, 255].
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def load_all_pairs(run_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank_dirs = find_rank_dirs(run_dir)

    clean_parts = []
    adv_parts = []
    id_parts = []

    for rank_dir in rank_dirs:
        shard_files = sorted(list(rank_dir.glob("shard_*.npz")) + list(rank_dir.glob("shard_*.pt")))
        if not shard_files:
            continue
        for shard in shard_files:
            d = load_shard_arrays(shard)
            clean = to_float_chw(d["x_clean"])
            adv = to_float_chw(d["x_adv"])
            ids = d["sample_ids"].astype(np.int64)

            clean_parts.append(clean)
            adv_parts.append(adv)
            id_parts.append(ids)

    if not clean_parts:
        raise RuntimeError(f"No shard data found under: {run_dir}")

    clean_all = np.concatenate(clean_parts, axis=0)
    adv_all = np.concatenate(adv_parts, axis=0)
    ids_all = np.concatenate(id_parts, axis=0)

    # Keep deterministic order by sample id to ensure pairing consistency.
    order = np.argsort(ids_all)
    clean_all = clean_all[order]
    adv_all = adv_all[order]
    ids_all = ids_all[order]

    if max_samples > 0:
        clean_all = clean_all[:max_samples]
        adv_all = adv_all[:max_samples]
        ids_all = ids_all[:max_samples]

    return clean_all, adv_all, ids_all


def build_base_model(device: torch.device, checkpoint: str):
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
    if k is None:
        raise KeyError("Cannot resolve k from SAE checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from SAE checkpoint")
        d_lat = int(d_in) * int(expansion_rate)

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


def resolve_stage(sae_cfg: dict, sae_stage: int = None) -> int:
    if sae_stage is not None:
        if sae_stage not in STAGE_DIN_MAP:
            raise ValueError(f"Invalid sae_stage={sae_stage}, expected 0..3")
        expected = STAGE_DIN_MAP[sae_stage]
        if int(sae_cfg["d_in"]) != expected:
            raise ValueError(
                f"Stage/SAE mismatch: stage{sae_stage} expects d_in={expected}, SAE has d_in={sae_cfg['d_in']}"
            )
        return sae_stage

    inferred = DIN_STAGE_MAP.get(int(sae_cfg["d_in"]), None)
    if inferred is None:
        raise ValueError(f"Cannot infer stage from SAE d_in={sae_cfg['d_in']}; pass --sae-stage explicitly")
    return inferred


def extract_sae_stats(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    clean_np: np.ndarray,
    adv_np: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    captured = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    clean_sum = None
    adv_sum = None
    clean_sq_sum = None
    adv_sq_sum = None
    token_count_total = 0

    l2_img_list = []
    l1_img_list = []
    cos_img_list = []

    with torch.no_grad():
        n = clean_np.shape[0]
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)

            clean_batch = torch.from_numpy(clean_np[i:j]).to(device)
            _ = model(clean_batch)
            feat_clean = captured["feat"]
            b, c, h, w = feat_clean.shape
            flat_clean = feat_clean.permute(0, 2, 3, 1).reshape(-1, c)
            flat_clean = (flat_clean - norm_mean) / norm_std
            z_clean = sae_model.encode(flat_clean)

            adv_batch = torch.from_numpy(adv_np[i:j]).to(device)
            _ = model(adv_batch)
            feat_adv = captured["feat"]
            flat_adv = feat_adv.permute(0, 2, 3, 1).reshape(-1, c)
            flat_adv = (flat_adv - norm_mean) / norm_std
            z_adv = sae_model.encode(flat_adv)

            if clean_sum is None:
                clean_sum = z_clean.sum(dim=0)
                adv_sum = z_adv.sum(dim=0)
                clean_sq_sum = (z_clean ** 2).sum(dim=0)
                adv_sq_sum = (z_adv ** 2).sum(dim=0)
            else:
                clean_sum += z_clean.sum(dim=0)
                adv_sum += z_adv.sum(dim=0)
                clean_sq_sum += (z_clean ** 2).sum(dim=0)
                adv_sq_sum += (z_adv ** 2).sum(dim=0)

            token_count = z_clean.shape[0]
            token_count_total += token_count

            # Image-level paired distance: one value per (clean, adv) image pair.
            # Reshape from (B*H*W, D) -> (B, H*W, D), then flatten token+feature dims.
            z_clean_img = z_clean.reshape(b, h * w, -1).reshape(b, -1)
            z_adv_img = z_adv.reshape(b, h * w, -1).reshape(b, -1)
            diff_img = z_adv_img - z_clean_img
            l2_img_list.append(torch.linalg.norm(diff_img, dim=1).detach().cpu())
            l1_img_list.append(torch.linalg.norm(diff_img, ord=1, dim=1).detach().cpu())
            cos_img = torch.nn.functional.cosine_similarity(z_clean_img, z_adv_img, dim=1)
            cos_img_list.append(cos_img.detach().cpu())

    handle.remove()

    clean_mean = (clean_sum / max(1, token_count_total)).cpu().numpy()
    adv_mean = (adv_sum / max(1, token_count_total)).cpu().numpy()

    clean_var = (clean_sq_sum / max(1, token_count_total) - (clean_sum / max(1, token_count_total)) ** 2).cpu().numpy()
    adv_var = (adv_sq_sum / max(1, token_count_total) - (adv_sum / max(1, token_count_total)) ** 2).cpu().numpy()

    l2_vals = torch.cat(l2_img_list, dim=0).numpy()
    l1_vals = torch.cat(l1_img_list, dim=0).numpy()
    cos_vals = torch.cat(cos_img_list, dim=0).numpy()

    return {
        "clean_mean": clean_mean,
        "adv_mean": adv_mean,
        "clean_var": clean_var,
        "adv_var": adv_var,
        "l2_vals": l2_vals,
        "l1_vals": l1_vals,
        "cos_vals": cos_vals,
        "token_count": int(token_count_total),
    }


def compute_feature_mask(
    clean_mean: np.ndarray,
    adv_mean: np.ndarray,
    drop_both_zero_features: bool,
    zero_filter_threshold: float,
) -> np.ndarray:
    if not drop_both_zero_features:
        return np.ones_like(clean_mean, dtype=bool)
    return (np.abs(clean_mean) > zero_filter_threshold) | (np.abs(adv_mean) > zero_filter_threshold)


def compute_relative_delta(
    clean_mean: np.ndarray,
    adv_mean: np.ndarray,
    relative_mode: str,
    relative_eps: float,
) -> np.ndarray:
    delta_abs = np.abs(adv_mean - clean_mean)
    if relative_mode == "baseline":
        denom = np.abs(clean_mean) + relative_eps
    else:
        # Symmetric relative change is more stable near 0.
        denom = (np.abs(clean_mean) + np.abs(adv_mean)) + relative_eps
    return delta_abs / denom


def save_enhanced_plots(
    clean_mean: np.ndarray,
    adv_mean: np.ndarray,
    out_dir: Path,
    topk: int,
    drop_both_zero_features: bool,
    zero_filter_threshold: float,
    relative_mode: str,
    relative_eps: float,
    delta_threshold: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = compute_feature_mask(
        clean_mean=clean_mean,
        adv_mean=adv_mean,
        drop_both_zero_features=drop_both_zero_features,
        zero_filter_threshold=zero_filter_threshold,
    )
    idx_valid = np.where(mask)[0]
    if idx_valid.shape[0] == 0:
        idx_valid = np.arange(clean_mean.shape[0])

    clean_v = clean_mean[idx_valid]
    adv_v = adv_mean[idx_valid]
    delta_v = adv_v - clean_v

    # Apply threshold on absolute delta before ranking.
    thr_mask = np.abs(delta_v) > delta_threshold
    if np.any(thr_mask):
        idx_thr = idx_valid[thr_mask]
        clean_thr = clean_mean[idx_thr]
        adv_thr = adv_mean[idx_thr]
        delta_thr = adv_thr - clean_thr
    else:
        idx_thr = idx_valid
        clean_thr = clean_v
        adv_thr = adv_v
        delta_thr = delta_v

    # Absolute ranking
    abs_order_local = np.argsort(np.abs(delta_thr))[::-1]
    top_abs_local = abs_order_local[: min(topk, abs_order_local.shape[0])]
    top_abs_idx = idx_thr[top_abs_local]

    # Relative ranking
    rel_v = compute_relative_delta(
        clean_mean=clean_thr,
        adv_mean=adv_thr,
        relative_mode=relative_mode,
        relative_eps=relative_eps,
    )
    rel_order_local = np.argsort(rel_v)[::-1]
    top_rel_local = rel_order_local[: min(topk, rel_order_local.shape[0])]
    top_rel_idx = idx_thr[top_rel_local]

    # Combined plot: x-axis uses feature IDs ordered by rank.
    abs_sorted_vals = np.abs(delta_thr[abs_order_local])
    abs_sorted_ids = idx_thr[abs_order_local]
    rel_sorted_vals = rel_v[rel_order_local]
    rel_sorted_ids = idx_thr[rel_order_local]

    def _set_feature_id_ticks(ax, sorted_ids: np.ndarray):
        n = sorted_ids.shape[0]
        if n == 0:
            return
        max_ticks = 40
        step = max(1, int(np.ceil(n / max_ticks)))
        tick_pos = np.arange(0, n, step)
        tick_labels = [str(int(sorted_ids[t])) for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=False)
    axes[0].plot(abs_sorted_vals, color="#8e44ad", linewidth=1.0)
    axes[0].set_title(f"|delta| sorted by rank (x labels are feature IDs, threshold>{delta_threshold:g})")
    axes[0].set_xlabel("Feature ID (rank-sorted)")
    axes[0].set_ylabel("|adv_mean - clean_mean|")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    _set_feature_id_ticks(axes[0], abs_sorted_ids)

    axes[1].plot(rel_sorted_vals, color="#16a085", linewidth=1.0)
    axes[1].set_title(f"Relative delta sorted by rank (mode={relative_mode}, threshold>{delta_threshold:g})")
    axes[1].set_xlabel("Feature ID (rank-sorted)")
    axes[1].set_ylabel("relative |delta|")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    _set_feature_id_ticks(axes[1], rel_sorted_ids)

    plt.tight_layout()
    fig.savefig(out_dir / "sae_delta_sorted_combined.png", dpi=300)
    plt.close(fig)

    return {
        "feature_mask_kept": int(idx_valid.shape[0]),
        "feature_mask_total": int(clean_mean.shape[0]),
        "threshold_kept": int(idx_thr.shape[0]),
        "top_abs_idx": top_abs_idx,
        "top_rel_idx": top_rel_idx,
        "abs_rank_sorted_feature_ids": abs_sorted_ids,
        "rel_rank_sorted_feature_ids": rel_sorted_ids,
        "relative_values_kept": rel_v,
        "delta_values_kept": delta_thr,
    }


def summarize(
    stats: Dict,
    sample_ids: np.ndarray,
    topk: int,
    drop_both_zero_features: bool,
    zero_filter_threshold: float,
    relative_mode: str,
    relative_eps: float,
    delta_threshold: float,
) -> Dict:
    clean_mean = stats["clean_mean"]
    adv_mean = stats["adv_mean"]
    delta = adv_mean - clean_mean

    top_idx = np.argsort(np.abs(delta))[::-1][: min(topk, delta.shape[0])]

    mask = compute_feature_mask(
        clean_mean=clean_mean,
        adv_mean=adv_mean,
        drop_both_zero_features=drop_both_zero_features,
        zero_filter_threshold=zero_filter_threshold,
    )
    idx_valid = np.where(mask)[0]
    if idx_valid.shape[0] == 0:
        idx_valid = np.arange(clean_mean.shape[0])

    rel_all = compute_relative_delta(
        clean_mean=clean_mean,
        adv_mean=adv_mean,
        relative_mode=relative_mode,
        relative_eps=relative_eps,
    )

    delta_valid = np.abs(delta[idx_valid])
    thr_mask = delta_valid > delta_threshold
    if np.any(thr_mask):
        idx_thr = idx_valid[thr_mask]
    else:
        idx_thr = idx_valid

    rel_thr = rel_all[idx_thr]
    rel_order = np.argsort(rel_thr)[::-1]
    top_rel_idx = idx_thr[rel_order[: min(topk, rel_order.shape[0])]]

    l2_vals = stats["l2_vals"]
    l1_vals = stats["l1_vals"]
    cos_vals = stats["cos_vals"]

    summary = {
        "num_samples": int(sample_ids.shape[0]),
        "token_count": int(stats["token_count"]),
        "paired_distance": {
            "l2_mean": float(np.mean(l2_vals)),
            "l2_std": float(np.std(l2_vals)),
            "l1_mean": float(np.mean(l1_vals)),
            "l1_std": float(np.std(l1_vals)),
            "cos_mean": float(np.mean(cos_vals)),
            "cos_std": float(np.std(cos_vals)),
        },
        "global_activation": {
            "clean_mean_abs": float(np.mean(np.abs(clean_mean))),
            "adv_mean_abs": float(np.mean(np.abs(adv_mean))),
            "delta_abs_mean": float(np.mean(np.abs(delta))),
            "delta_abs_max": float(np.max(np.abs(delta))),
            "relative_delta_mean_kept": float(np.mean(rel_thr)),
            "relative_delta_max_kept": float(np.max(rel_thr)),
        },
        "feature_filter": {
            "drop_both_zero_features": bool(drop_both_zero_features),
            "zero_filter_threshold": float(zero_filter_threshold),
            "kept_features": int(idx_valid.shape[0]),
            "delta_threshold": float(delta_threshold),
            "threshold_kept_features": int(idx_thr.shape[0]),
            "total_features": int(clean_mean.shape[0]),
            "relative_mode": relative_mode,
            "relative_eps": float(relative_eps),
        },
        "top_changed_features": [
            {
                "feature": int(i),
                "clean_mean": float(clean_mean[i]),
                "adv_mean": float(adv_mean[i]),
                "delta": float(delta[i]),
                "abs_delta": float(abs(delta[i])),
            }
            for i in top_idx
        ],
        "top_changed_features_relative": [
            {
                "feature": int(i),
                "clean_mean": float(clean_mean[i]),
                "adv_mean": float(adv_mean[i]),
                "delta": float(delta[i]),
                "relative_abs_delta": float(rel_all[i]),
            }
            for i in top_rel_idx
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare clean vs adv SAE distributions from shard files")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing rank folders/shards")
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
    )
    parser.add_argument("--sae-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument(
        "--keep-both-zero-features",
        action="store_true",
        default=False,
        help="Keep features where both clean/adv mean activations are near zero",
    )
    parser.add_argument(
        "--zero-filter-threshold",
        type=float,
        default=1e-12,
        help="Threshold for near-zero filtering when both-zero features are dropped",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Keep only features with |adv_mean-clean_mean| greater than this value for sorted delta plots",
    )
    parser.add_argument(
        "--relative-mode",
        type=str,
        choices=["baseline", "symmetric"],
        default="symmetric",
        help="Relative delta mode: baseline=|d|/(|clean|+eps), symmetric=|d|/(|clean|+|adv|+eps)",
    )
    parser.add_argument("--relative-eps", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "SAE" / "results_untar")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading run data from: {run_dir}")

    clean_np, adv_np, sample_ids = load_all_pairs(run_dir, max_samples=args.max_samples)
    print(f"Loaded paired samples: {clean_np.shape[0]}")

    model = build_base_model(device, args.checkpoint)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)
    stage_idx = resolve_stage(sae_cfg, args.sae_stage)
    print(
        f"SAE: ckpt={args.sae_ckpt}, stage={stage_idx}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    stats = extract_sae_stats(
        model=model,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        stage_idx=stage_idx,
        clean_np=clean_np,
        adv_np=adv_np,
        batch_size=args.batch_size,
        device=device,
    )

    drop_both_zero_features = not args.keep_both_zero_features

    plot_meta = save_enhanced_plots(
        clean_mean=stats["clean_mean"],
        adv_mean=stats["adv_mean"],
        out_dir=out_dir,
        topk=args.topk,
        drop_both_zero_features=drop_both_zero_features,
        zero_filter_threshold=args.zero_filter_threshold,
        relative_mode=args.relative_mode,
        relative_eps=args.relative_eps,
        delta_threshold=args.delta_threshold,
    )
    summary = summarize(
        stats,
        sample_ids=sample_ids,
        topk=args.topk,
        drop_both_zero_features=drop_both_zero_features,
        zero_filter_threshold=args.zero_filter_threshold,
        relative_mode=args.relative_mode,
        relative_eps=args.relative_eps,
        delta_threshold=args.delta_threshold,
    )
    summary["meta"] = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "sae_ckpt": args.sae_ckpt,
        "sae_stage": int(stage_idx),
        "batch_size": int(args.batch_size),
        "max_samples": int(args.max_samples),
        "plot_feature_kept": int(plot_meta["feature_mask_kept"]),
        "plot_threshold_kept": int(plot_meta["threshold_kept"]),
        "plot_feature_total": int(plot_meta["feature_mask_total"]),
    }
    summary["plot_feature_ids"] = {
        "abs_rank_sorted": [int(x) for x in plot_meta["abs_rank_sorted_feature_ids"].tolist()],
        "rel_rank_sorted": [int(x) for x in plot_meta["rel_rank_sorted_feature_ids"].tolist()],
        "abs_topk": [int(x) for x in plot_meta["top_abs_idx"].tolist()],
        "rel_topk": [int(x) for x in plot_meta["top_rel_idx"].tolist()],
    }

    out_json = out_dir / "sae_clean_adv_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Summary JSON: {out_json}")
    print(f"Combined delta plot: {out_dir / 'sae_delta_sorted_combined.png'}")


if __name__ == "__main__":
    main()
