#!/usr/bin/env python3
"""Search SAE feature steering combinations to improve robust accuracy.

This script consumes:
- a run directory with shard_*.npz/.pt (contains x_clean, x_adv, y_true, sample_ids)
- a summary JSON produced by sae_clean_adv_compare_npz.py (contains plot_feature_ids)
- model and SAE checkpoints

It evaluates steering feature combinations on existing adversarial images and reports
which combinations improve robust top-1/top-5 metrics.
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

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
class EvalResult:
    robust_top1: float
    robust_top5: float
    robust_true_prob_mean: float
    robust_top5_mass_mean: float
    clean_top1: float
    clean_top5: float
    clean_true_prob_mean: float
    clean_top5_mass_mean: float

    def score(self, w_top1: float, w_top5: float, w_trueprob: float) -> float:
        return (
            w_top1 * self.robust_top1
            + w_top5 * self.robust_top5
            + w_trueprob * self.robust_true_prob_mean
        )


@dataclass
class Candidate:
    pool_name: str
    feature_ids: Tuple[int, ...]
    alpha: float
    metrics: EvalResult
    score: float


def find_rank_dirs(run_dir: Path) -> List[Path]:
    rank_dirs = []
    if (run_dir / "metrics.json").exists():
        rank_dirs.append(run_dir)
    rank_dirs.extend(sorted([p for p in run_dir.glob("rank*") if p.is_dir() and (p / "metrics.json").exists()]))

    uniq: List[Path] = []
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
        if not shard_files:
            continue

        for shard in shard_files:
            d = load_shard_arrays(shard)
            if "y_true" not in d:
                raise KeyError(
                    f"Missing y_true in shard {shard}. "
                    "Use run outputs produced by gen_evaluate_imagenet.py."
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


def load_summary_feature_info(summary_json: Path) -> Tuple[Dict[str, List[int]], Dict[int, float]]:
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_ids = summary.get("plot_feature_ids", None)
    if not plot_ids:
        raise KeyError("summary JSON missing plot_feature_ids. Re-run sae_clean_adv_compare_npz.py first.")

    pools = {
        "abs_rank_sorted": [int(x) for x in plot_ids.get("abs_rank_sorted", [])],
        "rel_rank_sorted": [int(x) for x in plot_ids.get("rel_rank_sorted", [])],
        "abs_topk": [int(x) for x in plot_ids.get("abs_topk", [])],
        "rel_topk": [int(x) for x in plot_ids.get("rel_topk", [])],
    }

    # Direction map from summary: steer z toward clean mean => shift by (clean - adv).
    direction_map: Dict[int, float] = {}
    for item in summary.get("top_changed_features", []):
        fid = int(item["feature"])
        direction_map[fid] = float(item["clean_mean"] - item["adv_mean"])
    for item in summary.get("top_changed_features_relative", []):
        fid = int(item["feature"])
        direction_map[fid] = float(item["clean_mean"] - item["adv_mean"])

    return pools, direction_map


def make_steering_hook(
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    feature_ids: Sequence[int],
    direction_values: Sequence[float],
    alpha: float,
):
    feature_idx = torch.tensor(feature_ids, dtype=torch.long, device=norm_mean.device)
    direction = torch.tensor(direction_values, dtype=norm_mean.dtype, device=norm_mean.device)

    def hook_fn(_, __, output):
        bsz, channels, height, width = output.shape
        flat_raw = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat_raw - norm_mean) / norm_std

        z = sae_model.encode(flat_norm)
        if feature_idx.numel() > 0:
            z[:, feature_idx] += alpha * direction.unsqueeze(0)

        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        return recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)

    return hook_fn


def evaluate_model(
    model,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    y_t = torch.from_numpy(y_true).to(device)

    def _collect_metrics(x_np: np.ndarray):
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

    clean_top1, clean_top5, clean_true_prob, clean_top5_mass = _collect_metrics(x_clean)
    robust_top1, robust_top5, robust_true_prob, robust_top5_mass = _collect_metrics(x_adv)

    return EvalResult(
        robust_top1=robust_top1,
        robust_top5=robust_top5,
        robust_true_prob_mean=robust_true_prob,
        robust_top5_mass_mean=robust_top5_mass,
        clean_top1=clean_top1,
        clean_top5=clean_top5,
        clean_true_prob_mean=clean_true_prob,
        clean_top5_mass_mean=clean_top5_mass,
    )


def pool_candidates(pool: List[int], max_pool_size: int) -> List[int]:
    seen = set()
    out = []
    for fid in pool:
        if fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
        if len(out) >= max_pool_size:
            break
    return out


def beam_search_for_pool(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    pool_name: str,
    pool_feature_ids: List[int],
    direction_map: Dict[int, float],
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
    device: torch.device,
    alphas: List[float],
    max_features: int,
    beam_width: int,
    w_top1: float,
    w_top5: float,
    w_trueprob: float,
) -> List[Candidate]:
    stage_module = model.stages[stage_idx]

    def eval_combo(feature_ids: Tuple[int, ...], alpha: float) -> Candidate:
        dirs = [direction_map.get(fid, 0.0) for fid in feature_ids]
        hook = stage_module.register_forward_hook(
            make_steering_hook(
                sae_model=sae_model,
                norm_mean=norm_mean,
                norm_std=norm_std,
                feature_ids=feature_ids,
                direction_values=dirs,
                alpha=alpha,
            )
        )
        try:
            metrics = evaluate_model(
                model=model,
                x_clean=x_clean,
                x_adv=x_adv,
                y_true=y_true,
                batch_size=batch_size,
                device=device,
            )
        finally:
            hook.remove()

        score = metrics.score(w_top1=w_top1, w_top5=w_top5, w_trueprob=w_trueprob)
        return Candidate(
            pool_name=pool_name,
            feature_ids=feature_ids,
            alpha=alpha,
            metrics=metrics,
            score=score,
        )

    # Baseline is not returned from here; handled by caller.
    beam: List[Candidate] = []
    all_seen: Dict[Tuple[Tuple[int, ...], float], Candidate] = {}

    for k in range(1, max_features + 1):
        proposals: List[Tuple[int, ...]] = []
        if k == 1:
            proposals = [(fid,) for fid in pool_feature_ids]
        else:
            base_sets = [c.feature_ids for c in beam]
            next_sets = set()
            for base in base_sets:
                used = set(base)
                max_pos = max(pool_feature_ids.index(fid) for fid in base)
                for pos in range(max_pos + 1, len(pool_feature_ids)):
                    fid = pool_feature_ids[pos]
                    if fid in used:
                        continue
                    cand = tuple(sorted(base + (fid,), key=lambda x: pool_feature_ids.index(x)))
                    next_sets.add(cand)
            proposals = sorted(next_sets, key=lambda t: [pool_feature_ids.index(x) for x in t])

        level_candidates: List[Candidate] = []
        work_items = len(proposals) * len(alphas)
        if tqdm is not None:
            pbar = tqdm(total=work_items, desc=f"{pool_name} k={k}", leave=False)
        else:
            pbar = None

        for combo in proposals:
            for alpha in alphas:
                key = (combo, float(alpha))
                if key in all_seen:
                    level_candidates.append(all_seen[key])
                    if pbar is not None:
                        pbar.update(1)
                    continue
                c = eval_combo(combo, alpha)
                all_seen[key] = c
                level_candidates.append(c)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        level_candidates.sort(key=lambda x: x.score, reverse=True)
        beam = level_candidates[:beam_width]

    final = sorted(all_seen.values(), key=lambda x: x.score, reverse=True)
    return final


def main():
    parser = argparse.ArgumentParser(description="Search feature steering combinations for robust accuracy")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing shard files")
    parser.add_argument("--summary-json", type=str, required=True, help="sae_clean_adv_summary.json path")
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
    )
    parser.add_argument("--sae-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--max-pool-size", type=int, default=20, help="Use first-N features per pool")
    parser.add_argument("--max-features", type=int, default=3, help="Max features in one steering combo")
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
        help="Steering strength candidates",
    )
    parser.add_argument("--w-top1", type=float, default=1.0)
    parser.add_argument("--w-top5", type=float, default=0.25)
    parser.add_argument("--w-trueprob", type=float, default=0.1)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs for pool-parallel search, e.g. '0,1,2,3'",
    )
    parser.add_argument(
        "--parallel-pools",
        action="store_true",
        default=False,
        help="Run multiple search pools in parallel across provided GPUs",
    )
    parser.add_argument(
        "--only-pool",
        type=str,
        default=None,
        choices=["abs_rank_sorted", "rel_rank_sorted", "abs_topk", "rel_topk"],
        help="Internal option: run search on one pool only",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_json = Path(args.summary_json).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    pools, direction_map = load_summary_feature_info(summary_json)
    pool_keys = ["abs_rank_sorted", "rel_rank_sorted", "abs_topk", "rel_topk"]
    if args.only_pool is not None:
        pool_keys = [args.only_pool]
    for k in pool_keys:
        if not pools.get(k):
            raise ValueError(f"Feature pool '{k}' is empty in summary JSON")

    # Orchestrator mode: one pool per subprocess/GPU.
    if args.parallel_pools and args.only_pool is None and len(pool_keys) > 1:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        if not gpu_ids:
            if torch.cuda.is_available():
                gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
            else:
                gpu_ids = [""]

        assign = {}
        for i, pool_name in enumerate(pool_keys):
            assign[pool_name] = gpu_ids[i % len(gpu_ids)]

        print(f"Parallel pool search enabled, pool->gpu: {assign}")

        out_root = Path(args.output_json).resolve().parent if args.output_json else summary_json.parent
        out_root.mkdir(parents=True, exist_ok=True)

        procs = []
        pool_out = {}
        for pool_name in pool_keys:
            child_out = out_root / f".steering_{pool_name}.json"
            pool_out[pool_name] = child_out

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-dir",
                str(run_dir),
                "--summary-json",
                str(summary_json),
                "--checkpoint",
                str(args.checkpoint),
                "--sae-ckpt",
                str(args.sae_ckpt),
                "--batch-size",
                str(args.batch_size),
                "--max-samples",
                str(args.max_samples),
                "--max-pool-size",
                str(args.max_pool_size),
                "--max-features",
                str(args.max_features),
                "--beam-width",
                str(args.beam_width),
                "--w-top1",
                str(args.w_top1),
                "--w-top5",
                str(args.w_top5),
                "--w-trueprob",
                str(args.w_trueprob),
                "--only-pool",
                pool_name,
                "--output-json",
                str(child_out),
            ]
            if args.sae_stage is not None:
                cmd.extend(["--sae-stage", str(args.sae_stage)])
            if args.alphas:
                cmd.append("--alphas")
                cmd.extend([str(a) for a in args.alphas])

            env = os.environ.copy()
            gpu_id = assign[pool_name]
            if gpu_id != "":
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

            p = subprocess.Popen(cmd, env=env)
            procs.append((pool_name, p))

        done = set()
        if tqdm is not None:
            pbar = tqdm(total=len(procs), desc="Pools done")
        else:
            pbar = None

        while len(done) < len(procs):
            for pool_name, p in procs:
                if pool_name in done:
                    continue
                ret = p.poll()
                if ret is None:
                    continue
                if ret != 0:
                    raise RuntimeError(f"Subprocess failed for pool={pool_name}, exit={ret}")
                done.add(pool_name)
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f"[done] {pool_name} ({len(done)}/{len(procs)})")
            time.sleep(0.2)

        if pbar is not None:
            pbar.close()

        merged: List[Dict] = []
        baseline = None
        baseline_score = None
        meta = None
        for pool_name in pool_keys:
            child_path = pool_out[pool_name]
            with open(child_path, "r", encoding="utf-8") as f:
                child = json.load(f)
            if baseline is None:
                baseline = child.get("baseline")
                baseline_score = baseline.get("score", None) if baseline else None
                meta = child.get("meta")
            merged.extend(child.get("top_candidates", []))

        merged.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
        top = merged[:50]
        best = top[0] if top else None

        output = {
            "meta": {
                **(meta or {}),
                "parallel_pools": True,
                "gpu_ids": gpu_ids,
            },
            "baseline": baseline,
            "best": None,
            "top_candidates": top,
        }

        if best is not None and baseline is not None:
            output["best"] = {
                "pool": best["pool"],
                "feature_ids": best["feature_ids"],
                "alpha": best["alpha"],
                "score": best["score"],
                "delta_vs_baseline": {
                    "robust_top1": float(best["robust_top1"] - baseline["robust_top1"]),
                    "robust_top5": float(best["robust_top5"] - baseline["robust_top5"]),
                    "robust_true_prob_mean": float(
                        best["robust_true_prob_mean"] - baseline["robust_true_prob_mean"]
                    ),
                    "clean_top1": float(best["clean_top1"] - baseline["clean_top1"]),
                    "clean_top5": float(best["clean_top5"] - baseline["clean_top5"]),
                },
                "metrics": {
                    "robust_top1": float(best["robust_top1"]),
                    "robust_top5": float(best["robust_top5"]),
                    "robust_true_prob_mean": float(best["robust_true_prob_mean"]),
                    "clean_top1": float(best["clean_top1"]),
                    "clean_top5": float(best["clean_top5"]),
                },
            }

        out_path = Path(args.output_json).resolve() if args.output_json else (summary_json.parent / "steering_search_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print("\nTop steering candidates:")
        for i, cand in enumerate(top[:10], start=1):
            print(
                f"{i:02d}. pool={cand['pool']} features={cand['feature_ids']} alpha={cand['alpha']:g} "
                f"score={cand['score']:.4f} robust_top1={cand['robust_top1']:.4f} "
                f"robust_top5={cand['robust_top5']:.4f} true_prob={cand['robust_true_prob_mean']:.4f}"
            )

        print(f"\nSaved steering results: {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_clean, x_adv, y_true, sample_ids = load_all_data(run_dir=run_dir, max_samples=args.max_samples)
    print(f"Loaded samples: {x_clean.shape[0]}")

    model = build_base_model(device, args.checkpoint)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)
    stage_idx = resolve_stage(sae_cfg, args.sae_stage)

    print(
        f"SAE: ckpt={args.sae_ckpt}, stage={stage_idx}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    # Baseline without steering hook.
    baseline = evaluate_model(
        model=model,
        x_clean=x_clean,
        x_adv=x_adv,
        y_true=y_true,
        batch_size=args.batch_size,
        device=device,
    )
    baseline_score = baseline.score(args.w_top1, args.w_top5, args.w_trueprob)

    print("Baseline metrics:")
    print(
        f"  Robust top1={baseline.robust_top1:.4f}, top5={baseline.robust_top5:.4f}, "
        f"true_prob={baseline.robust_true_prob_mean:.4f}, score={baseline_score:.4f}"
    )

    all_results: List[Candidate] = []
    for pool_name in pool_keys:
        ids = pool_candidates(pools[pool_name], args.max_pool_size)
        print(f"Searching pool={pool_name}, candidates={len(ids)}")
        pool_res = beam_search_for_pool(
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            stage_idx=stage_idx,
            pool_name=pool_name,
            pool_feature_ids=ids,
            direction_map=direction_map,
            x_clean=x_clean,
            x_adv=x_adv,
            y_true=y_true,
            batch_size=args.batch_size,
            device=device,
            alphas=[float(a) for a in args.alphas],
            max_features=args.max_features,
            beam_width=args.beam_width,
            w_top1=args.w_top1,
            w_top5=args.w_top5,
            w_trueprob=args.w_trueprob,
        )
        all_results.extend(pool_res)

    all_results.sort(key=lambda x: x.score, reverse=True)
    best = all_results[0] if all_results else None

    print("\nTop steering candidates:")
    for i, cand in enumerate(all_results[:10], start=1):
        print(
            f"{i:02d}. pool={cand.pool_name} features={list(cand.feature_ids)} alpha={cand.alpha:g} "
            f"score={cand.score:.4f} robust_top1={cand.metrics.robust_top1:.4f} "
            f"robust_top5={cand.metrics.robust_top5:.4f} true_prob={cand.metrics.robust_true_prob_mean:.4f}"
        )

    output = {
        "meta": {
            "run_dir": str(run_dir),
            "summary_json": str(summary_json),
            "checkpoint": args.checkpoint,
            "sae_ckpt": args.sae_ckpt,
            "sae_stage": int(stage_idx),
            "num_samples": int(x_clean.shape[0]),
            "batch_size": int(args.batch_size),
            "max_pool_size": int(args.max_pool_size),
            "max_features": int(args.max_features),
            "beam_width": int(args.beam_width),
            "alphas": [float(a) for a in args.alphas],
            "weights": {
                "top1": float(args.w_top1),
                "top5": float(args.w_top5),
                "trueprob": float(args.w_trueprob),
            },
        },
        "baseline": {
            "robust_top1": baseline.robust_top1,
            "robust_top5": baseline.robust_top5,
            "robust_true_prob_mean": baseline.robust_true_prob_mean,
            "robust_top5_mass_mean": baseline.robust_top5_mass_mean,
            "clean_top1": baseline.clean_top1,
            "clean_top5": baseline.clean_top5,
            "clean_true_prob_mean": baseline.clean_true_prob_mean,
            "clean_top5_mass_mean": baseline.clean_top5_mass_mean,
            "score": baseline_score,
        },
        "best": None,
        "top_candidates": [],
    }

    if best is not None:
        output["best"] = {
            "pool": best.pool_name,
            "feature_ids": [int(x) for x in best.feature_ids],
            "alpha": float(best.alpha),
            "score": float(best.score),
            "delta_vs_baseline": {
                "robust_top1": float(best.metrics.robust_top1 - baseline.robust_top1),
                "robust_top5": float(best.metrics.robust_top5 - baseline.robust_top5),
                "robust_true_prob_mean": float(best.metrics.robust_true_prob_mean - baseline.robust_true_prob_mean),
                "clean_top1": float(best.metrics.clean_top1 - baseline.clean_top1),
                "clean_top5": float(best.metrics.clean_top5 - baseline.clean_top5),
            },
            "metrics": {
                "robust_top1": float(best.metrics.robust_top1),
                "robust_top5": float(best.metrics.robust_top5),
                "robust_true_prob_mean": float(best.metrics.robust_true_prob_mean),
                "robust_top5_mass_mean": float(best.metrics.robust_top5_mass_mean),
                "clean_top1": float(best.metrics.clean_top1),
                "clean_top5": float(best.metrics.clean_top5),
                "clean_true_prob_mean": float(best.metrics.clean_true_prob_mean),
                "clean_top5_mass_mean": float(best.metrics.clean_top5_mass_mean),
            },
        }

    for cand in all_results[:50]:
        output["top_candidates"].append(
            {
                "pool": cand.pool_name,
                "feature_ids": [int(x) for x in cand.feature_ids],
                "alpha": float(cand.alpha),
                "score": float(cand.score),
                "robust_top1": float(cand.metrics.robust_top1),
                "robust_top5": float(cand.metrics.robust_top5),
                "robust_true_prob_mean": float(cand.metrics.robust_true_prob_mean),
                "clean_top1": float(cand.metrics.clean_top1),
                "clean_top5": float(cand.metrics.clean_top5),
            }
        )

    if args.output_json:
        out_path = Path(args.output_json).resolve()
    else:
        out_path = summary_json.parent / "steering_search_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved steering results: {out_path}")


if __name__ == "__main__":
    main()
