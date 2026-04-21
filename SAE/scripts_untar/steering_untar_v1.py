#!/usr/bin/env python3
"""Clean-only anomaly-triggered SAE steering (v1).

This script implements a practical first research version of conditional steering:
1) Build a candidate feature set from sae_clean_adv_summary.json.
2) Calibrate per-feature clean statistics from clean samples only.
3) Compute per-image anomaly scores s_i = |z_i - mu_i_clean| / sigma_i_clean.
4) Trigger steering only for anomalous features and pull them toward clean center.
5) Report clean/robust top1/top5 and probability metrics before/after steering.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Resolve DAT repository root from current script location: DAT/SAE/scripts_untar/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


@dataclass
class EvalMetrics:
    clean_top1: float
    clean_top5: float
    clean_true_prob_mean: float
    clean_top5_mass_mean: float
    robust_top1: float
    robust_top5: float
    robust_true_prob_mean: float
    robust_top5_mass_mean: float


def find_rank_dirs(run_dir: Path) -> List[Path]:
    rank_dirs: List[Path] = []
    if (run_dir / "metrics.json").exists():
        rank_dirs.append(run_dir)
    rank_dirs.extend(sorted([p for p in run_dir.glob("rank*") if p.is_dir() and (p / "metrics.json").exists()]))

    out: List[Path] = []
    seen = set()
    for p in rank_dirs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    if not out:
        raise FileNotFoundError(f"No rank directories found under: {run_dir}")
    return out


def load_shard_arrays(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        d = np.load(path)
        out = {
            "x_clean": d["x_clean"],
            "x_adv": d["x_adv"],
            "sample_ids": d["sample_ids"],
        }
        if "y_true" in d:
            out["y_true"] = d["y_true"]
        return out

    if path.suffix == ".pt":
        d = torch.load(path, map_location="cpu")
        out = {
            "x_clean": d["x_clean"].cpu().numpy(),
            "x_adv": d["x_adv"].cpu().numpy(),
            "sample_ids": d["sample_ids"].cpu().numpy(),
        }
        if "y_true" in d:
            out["y_true"] = d["y_true"].cpu().numpy()
        return out

    raise ValueError(f"Unsupported shard extension: {path}")


def to_float_chw(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def load_all_data(run_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rank_dirs = find_rank_dirs(run_dir)

    clean_parts: List[np.ndarray] = []
    adv_parts: List[np.ndarray] = []
    id_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for rank_dir in rank_dirs:
        shard_files = sorted(list(rank_dir.glob("shard_*.npz")) + list(rank_dir.glob("shard_*.pt")))
        for shard in shard_files:
            d = load_shard_arrays(shard)
            if "y_true" not in d:
                raise KeyError(
                    f"Missing y_true in shard {shard}. "
                    "Please use outputs from gen_evaluate_imagenet.py."
                )
            clean_parts.append(to_float_chw(d["x_clean"]))
            adv_parts.append(to_float_chw(d["x_adv"]))
            id_parts.append(d["sample_ids"].astype(np.int64))
            y_parts.append(d["y_true"].astype(np.int64))

    if not clean_parts:
        raise RuntimeError(f"No shard data found under: {run_dir}")

    clean_all = np.concatenate(clean_parts, axis=0)
    adv_all = np.concatenate(adv_parts, axis=0)
    ids_all = np.concatenate(id_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    order = np.argsort(ids_all)
    clean_all = clean_all[order]
    adv_all = adv_all[order]
    y_all = y_all[order]
    ids_all = ids_all[order]

    if max_samples > 0:
        clean_all = clean_all[:max_samples]
        adv_all = adv_all[:max_samples]
        y_all = y_all[:max_samples]
        ids_all = ids_all[:max_samples]

    return clean_all, adv_all, y_all, ids_all


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


def resolve_stage(sae_cfg: dict, sae_stage: Optional[int] = None) -> int:
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


def parse_summary_features(
    summary_json: Path,
    min_abs_delta: float,
    min_rel_delta: float,
    max_features: int,
    feature_ids_override: Optional[List[int]] = None,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], List[int]]:
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    abs_map: Dict[int, float] = {}
    rel_map: Dict[int, float] = {}
    direction_map: Dict[int, float] = {}

    for item in summary.get("top_changed_features", []):
        fid = int(item["feature"])
        abs_map[fid] = float(item.get("abs_delta", 0.0))
        direction_map[fid] = float(item["clean_mean"] - item["adv_mean"])

    for item in summary.get("top_changed_features_relative", []):
        fid = int(item["feature"])
        rel_map[fid] = float(item.get("relative_abs_delta", 0.0))
        direction_map[fid] = float(item["clean_mean"] - item["adv_mean"])

    if feature_ids_override:
        selected = [int(x) for x in feature_ids_override]
    else:
        # Candidate features must be high in both absolute and relative deltas.
        common = [fid for fid in abs_map if fid in rel_map]
        selected = [
            fid for fid in common
            if abs_map.get(fid, 0.0) >= min_abs_delta and rel_map.get(fid, 0.0) >= min_rel_delta
        ]

        # Keep deterministic order from plot abs ranking if available.
        order_source = summary.get("plot_feature_ids", {}).get("abs_rank_sorted", [])
        order_idx = {int(fid): i for i, fid in enumerate(order_source)}
        selected.sort(key=lambda x: order_idx.get(x, 10**9))

        if max_features > 0:
            selected = selected[:max_features]

    if not selected:
        raise ValueError(
            "No selected features found. Lower --min-abs-delta/--min-rel-delta or pass --feature-ids explicitly."
        )

    missing = [fid for fid in selected if fid not in direction_map]
    if missing:
        raise ValueError(f"Missing direction data in summary for feature IDs: {missing}")

    return selected, abs_map, rel_map, selected


def collect_feature_image_means(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    x_np: np.ndarray,
    feature_ids: List[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Collect per-image mean latent values for selected features.

    Returns an array with shape [N_images, N_features].
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)
    out_parts: List[np.ndarray] = []

    idx_t = torch.tensor(feature_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        n = x_np.shape[0]
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            xb = torch.from_numpy(x_np[i:j]).to(device)
            _ = model(xb)

            feat = captured["feat"]
            b, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat).reshape(b, h * w, -1)

            z_sel_mean = z[:, :, idx_t].mean(dim=1)  # [b, m]
            out_parts.append(z_sel_mean.detach().cpu().numpy())

    handle.remove()
    return np.concatenate(out_parts, axis=0)


def build_clean_calibration(
    clean_img_means: np.ndarray,
    tau_quantile: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build clean-only calibration: mu, sigma, and per-feature tau on z-score."""
    mu = clean_img_means.mean(axis=0)
    sigma = clean_img_means.std(axis=0)
    sigma = np.maximum(sigma, eps)

    s_clean = np.abs(clean_img_means - mu[None, :]) / sigma[None, :]
    tau = np.quantile(s_clean, tau_quantile, axis=0)
    tau = np.maximum(tau, eps)
    return mu, sigma, tau


def make_conditional_clean_pull_hook(
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    feature_ids: List[int],
    mu_clean: np.ndarray,
    sigma_clean: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    max_gate: float,
    eps: float,
):
    feat_idx = torch.tensor(feature_ids, dtype=torch.long, device=norm_mean.device)
    mu_t = torch.tensor(mu_clean, dtype=norm_mean.dtype, device=norm_mean.device)
    sigma_t = torch.tensor(sigma_clean, dtype=norm_mean.dtype, device=norm_mean.device)
    tau_t = torch.tensor(tau, dtype=norm_mean.dtype, device=norm_mean.device)

    def hook_fn(_, __, output):
        bsz, channels, height, width = output.shape
        flat_raw = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat_raw - norm_mean) / norm_std

        z = sae_model.encode(flat_norm).reshape(bsz, height * width, -1)

        # Per-image, per-feature anomaly score on image-mean latent value.
        z_img = z[:, :, feat_idx].mean(dim=1)  # [b, m]
        s = torch.abs(z_img - mu_t.unsqueeze(0)) / (sigma_t.unsqueeze(0) + eps)

        # Soft gate after threshold crossing.
        gate = torch.clamp((s - tau_t.unsqueeze(0)) / (tau_t.unsqueeze(0) + eps), min=0.0, max=max_gate)

        # Pull image-level mean toward clean center mu.
        delta = (z_img - mu_t.unsqueeze(0))  # [b, m]
        shift = alpha * gate * delta  # [b, m]

        z[:, :, feat_idx] = z[:, :, feat_idx] - shift.unsqueeze(1)

        z_flat = z.reshape(-1, z.shape[-1])
        recon_norm = (z_flat @ sae_model.W_dec) + sae_model.b_dec
        recon_raw = recon_norm * norm_std + norm_mean
        return recon_raw.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)

    return hook_fn


def evaluate_logits_metrics(
    model,
    x_np: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    y_t = torch.from_numpy(y_true).to(device)

    top1_correct = 0
    top5_correct = 0
    true_prob_sum = 0.0
    top5_mass_sum = 0.0
    n = x_np.shape[0]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            xb = torch.from_numpy(x_np[i:j]).to(device)
            yb = y_t[i:j]

            logits = model(xb)
            probs = F.softmax(logits, dim=1)

            pred_top1 = torch.argmax(probs, dim=1)
            top1_correct += int((pred_top1 == yb).sum().item())

            top5_idx = torch.topk(probs, k=5, dim=1).indices
            top5_match = (top5_idx == yb.unsqueeze(1)).any(dim=1)
            top5_correct += int(top5_match.sum().item())

            true_prob_sum += float(probs[torch.arange(probs.shape[0], device=device), yb].sum().item())
            top5_mass_sum += float(torch.topk(probs, k=5, dim=1).values.sum(dim=1).sum().item())

    n_f = float(max(1, n))
    return (
        top1_correct / n_f,
        top5_correct / n_f,
        true_prob_sum / n_f,
        top5_mass_sum / n_f,
    )


def evaluate_clean_adv(
    model,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> EvalMetrics:
    c_top1, c_top5, c_truep, c_top5_mass = evaluate_logits_metrics(
        model=model,
        x_np=x_clean,
        y_true=y_true,
        batch_size=batch_size,
        device=device,
    )
    r_top1, r_top5, r_truep, r_top5_mass = evaluate_logits_metrics(
        model=model,
        x_np=x_adv,
        y_true=y_true,
        batch_size=batch_size,
        device=device,
    )
    return EvalMetrics(
        clean_top1=c_top1,
        clean_top5=c_top5,
        clean_true_prob_mean=c_truep,
        clean_top5_mass_mean=c_top5_mass,
        robust_top1=r_top1,
        robust_top5=r_top5,
        robust_true_prob_mean=r_truep,
        robust_top5_mass_mean=r_top5_mass,
    )


def main():
    parser = argparse.ArgumentParser(description="Clean-only anomaly-triggered SAE steering (v1)")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
    )
    parser.add_argument("--sae-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--feature-ids", type=int, nargs="+", default=None, help="Optional manual feature list")
    parser.add_argument("--max-features", type=int, default=8, help="Max selected features when auto-selecting")
    parser.add_argument("--min-abs-delta", type=float, default=0.1)
    parser.add_argument("--min-rel-delta", type=float, default=0.1)

    parser.add_argument(
        "--clean-calib-ratio",
        type=float,
        default=0.5,
        help="Ratio of clean samples used for calibration (rest used in eval)",
    )
    parser.add_argument(
        "--tau-quantile",
        type=float,
        default=0.95,
        help="Per-feature anomaly threshold quantile on clean z-scores",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-gate", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_json = Path(args.summary_json).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_clean_all, x_adv_all, y_all, ids_all = load_all_data(run_dir, max_samples=args.max_samples)
    n_all = x_clean_all.shape[0]
    print(f"Loaded samples: {n_all}")

    feature_ids, abs_map, rel_map, _ = parse_summary_features(
        summary_json=summary_json,
        min_abs_delta=args.min_abs_delta,
        min_rel_delta=args.min_rel_delta,
        max_features=args.max_features,
        feature_ids_override=args.feature_ids,
    )
    print(f"Selected feature IDs ({len(feature_ids)}): {feature_ids}")

    model = build_base_model(device, args.checkpoint)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)
    stage_idx = resolve_stage(sae_cfg, args.sae_stage)
    print(
        f"SAE: ckpt={args.sae_ckpt}, stage={stage_idx}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    # Split clean set into calibration and eval partitions.
    n_calib = int(round(n_all * args.clean_calib_ratio))
    n_calib = max(1, min(n_all - 1, n_calib)) if n_all > 1 else 1

    x_clean_calib = x_clean_all[:n_calib]
    x_clean_eval = x_clean_all[n_calib:] if n_calib < n_all else x_clean_all
    x_adv_eval = x_adv_all[n_calib:] if n_calib < n_all else x_adv_all
    y_eval = y_all[n_calib:] if n_calib < n_all else y_all
    ids_eval = ids_all[n_calib:] if n_calib < n_all else ids_all

    print(f"Calibration samples: {x_clean_calib.shape[0]}, Eval samples: {x_clean_eval.shape[0]}")

    clean_means_calib = collect_feature_image_means(
        model=model,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        stage_idx=stage_idx,
        x_np=x_clean_calib,
        feature_ids=feature_ids,
        batch_size=args.batch_size,
        device=device,
    )

    mu_clean, sigma_clean, tau = build_clean_calibration(
        clean_img_means=clean_means_calib,
        tau_quantile=args.tau_quantile,
        eps=args.eps,
    )

    # Baseline on eval split.
    baseline = evaluate_clean_adv(
        model=model,
        x_clean=x_clean_eval,
        x_adv=x_adv_eval,
        y_true=y_eval,
        batch_size=args.batch_size,
        device=device,
    )

    # Conditional steering on eval split.
    hook = model.stages[stage_idx].register_forward_hook(
        make_conditional_clean_pull_hook(
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            feature_ids=feature_ids,
            mu_clean=mu_clean,
            sigma_clean=sigma_clean,
            tau=tau,
            alpha=args.alpha,
            max_gate=args.max_gate,
            eps=args.eps,
        )
    )
    try:
        steered = evaluate_clean_adv(
            model=model,
            x_clean=x_clean_eval,
            x_adv=x_adv_eval,
            y_true=y_eval,
            batch_size=args.batch_size,
            device=device,
        )
    finally:
        hook.remove()

    print("Baseline vs Steered (eval split):")
    print(
        f"  Robust top1: {baseline.robust_top1:.4f} -> {steered.robust_top1:.4f}, "
        f"top5: {baseline.robust_top5:.4f} -> {steered.robust_top5:.4f}"
    )
    print(
        f"  Clean  top1: {baseline.clean_top1:.4f} -> {steered.clean_top1:.4f}, "
        f"top5: {baseline.clean_top5:.4f} -> {steered.clean_top5:.4f}"
    )

    out = {
        "meta": {
            "run_dir": str(run_dir),
            "summary_json": str(summary_json),
            "checkpoint": args.checkpoint,
            "sae_ckpt": args.sae_ckpt,
            "sae_stage": int(stage_idx),
            "num_samples_total": int(n_all),
            "num_samples_calib": int(x_clean_calib.shape[0]),
            "num_samples_eval": int(x_clean_eval.shape[0]),
            "batch_size": int(args.batch_size),
            "clean_calib_ratio": float(args.clean_calib_ratio),
            "tau_quantile": float(args.tau_quantile),
            "alpha": float(args.alpha),
            "max_gate": float(args.max_gate),
            "eps": float(args.eps),
            "min_abs_delta": float(args.min_abs_delta),
            "min_rel_delta": float(args.min_rel_delta),
        },
        "selected_features": [
            {
                "feature": int(fid),
                "abs_delta": float(abs_map.get(fid, 0.0)),
                "rel_delta": float(rel_map.get(fid, 0.0)),
                "mu_clean": float(mu_clean[i]),
                "sigma_clean": float(sigma_clean[i]),
                "tau": float(tau[i]),
            }
            for i, fid in enumerate(feature_ids)
        ],
        "baseline": {
            "clean_top1": float(baseline.clean_top1),
            "clean_top5": float(baseline.clean_top5),
            "clean_true_prob_mean": float(baseline.clean_true_prob_mean),
            "clean_top5_mass_mean": float(baseline.clean_top5_mass_mean),
            "robust_top1": float(baseline.robust_top1),
            "robust_top5": float(baseline.robust_top5),
            "robust_true_prob_mean": float(baseline.robust_true_prob_mean),
            "robust_top5_mass_mean": float(baseline.robust_top5_mass_mean),
        },
        "steered": {
            "clean_top1": float(steered.clean_top1),
            "clean_top5": float(steered.clean_top5),
            "clean_true_prob_mean": float(steered.clean_true_prob_mean),
            "clean_top5_mass_mean": float(steered.clean_top5_mass_mean),
            "robust_top1": float(steered.robust_top1),
            "robust_top5": float(steered.robust_top5),
            "robust_true_prob_mean": float(steered.robust_true_prob_mean),
            "robust_top5_mass_mean": float(steered.robust_top5_mass_mean),
        },
        "delta": {
            "clean_top1": float(steered.clean_top1 - baseline.clean_top1),
            "clean_top5": float(steered.clean_top5 - baseline.clean_top5),
            "clean_true_prob_mean": float(steered.clean_true_prob_mean - baseline.clean_true_prob_mean),
            "robust_top1": float(steered.robust_top1 - baseline.robust_top1),
            "robust_top5": float(steered.robust_top5 - baseline.robust_top5),
            "robust_true_prob_mean": float(steered.robust_true_prob_mean - baseline.robust_true_prob_mean),
        },
        "eval_sample_ids": [int(x) for x in ids_eval.tolist()],
    }

    if args.output_json:
        out_path = Path(args.output_json).resolve()
    else:
        out_path = summary_json.parent / "steering_untar_v1_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
