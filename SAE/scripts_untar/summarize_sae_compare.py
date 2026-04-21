#!/usr/bin/env python3
"""Summarize SAE on/off comparison from saved eval outputs.

Inputs are run directories that contain one or more rank folders with:
- metrics.json
- shard_XXXX.npz or shard_XXXX.pt

The script reports:
- per-group (SAE off/on) mean/std clean & robust accuracy
- paired sample-level flip statistics (off -> on)
- a compact text table suitable for experiment logs
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def find_rank_dirs(run_dir: Path) -> List[Path]:
    rank_dirs = []
    if (run_dir / "metrics.json").exists():
        rank_dirs.append(run_dir)
    rank_dirs.extend(sorted([p for p in run_dir.glob("rank*") if p.is_dir()]))
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for p in rank_dirs:
        if p in seen:
            continue
        seen.add(p)
        if (p / "metrics.json").exists():
            out.append(p)
    if not out:
        raise FileNotFoundError(f"No rank directories with metrics.json found under: {run_dir}")
    return out


def load_shard(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        d = np.load(path)
        return {
            "y_true": d["y_true"],
            "y_pred_clean": d["y_pred_clean"],
            "y_pred_adv": d["y_pred_adv"],
            "sample_ids": d["sample_ids"],
        }
    if path.suffix == ".pt":
        d = torch.load(path, map_location="cpu")
        return {
            "y_true": d["y_true"].cpu().numpy(),
            "y_pred_clean": d["y_pred_clean"].cpu().numpy(),
            "y_pred_adv": d["y_pred_adv"].cpu().numpy(),
            "sample_ids": d["sample_ids"].cpu().numpy(),
        }
    raise ValueError(f"Unsupported shard extension: {path}")


def load_run(run_dir: Path) -> Dict:
    rank_dirs = find_rank_dirs(run_dir)

    weighted_clean_sum = 0.0
    weighted_robust_sum = 0.0
    n_total = 0

    sample_map = {}

    for rank_dir in rank_dirs:
        metrics_path = rank_dir / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        n = int(m.get("num_samples", 0))
        weighted_clean_sum += float(m.get("clean_accuracy", 0.0)) * n
        weighted_robust_sum += float(m.get("robust_accuracy", 0.0)) * n
        n_total += n

        shard_files = sorted(list(rank_dir.glob("shard_*.npz")) + list(rank_dir.glob("shard_*.pt")))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in: {rank_dir}")

        for shard in shard_files:
            sd = load_shard(shard)
            y_true = sd["y_true"].astype(np.int64)
            y_pred_clean = sd["y_pred_clean"].astype(np.int64)
            y_pred_adv = sd["y_pred_adv"].astype(np.int64)
            sample_ids = sd["sample_ids"].astype(np.int64)

            for i in range(sample_ids.shape[0]):
                sid = int(sample_ids[i])
                sample_map[sid] = (
                    int(y_true[i]),
                    int(y_pred_clean[i]),
                    int(y_pred_adv[i]),
                )

    clean_acc = weighted_clean_sum / max(1, n_total)
    robust_acc = weighted_robust_sum / max(1, n_total)

    return {
        "run_dir": str(run_dir),
        "num_samples": int(n_total),
        "clean_accuracy": float(clean_acc),
        "robust_accuracy": float(robust_acc),
        "samples": sample_map,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_flip_stats(off_samples: Dict[int, Tuple[int, int, int]], on_samples: Dict[int, Tuple[int, int, int]]) -> Dict:
    ids = sorted(set(off_samples.keys()) & set(on_samples.keys()))
    if not ids:
        return {"n_paired": 0}

    clean_gain = 0
    clean_loss = 0
    robust_gain = 0
    robust_loss = 0
    robust_rr = 0  # off robust right, on robust right
    robust_rw = 0  # off robust right, on robust wrong
    robust_wr = 0  # off robust wrong, on robust right
    robust_ww = 0  # off robust wrong, on robust wrong

    off_clean_correct = 0
    on_clean_correct = 0
    off_robust_correct = 0
    on_robust_correct = 0

    label_mismatch = 0

    for sid in ids:
        y_off, c_off, r_off = off_samples[sid]
        y_on, c_on, r_on = on_samples[sid]

        if y_off != y_on:
            label_mismatch += 1
            continue

        off_c = int(c_off == y_off)
        on_c = int(c_on == y_on)
        off_r = int(r_off == y_off)
        on_r = int(r_on == y_on)

        off_clean_correct += off_c
        on_clean_correct += on_c
        off_robust_correct += off_r
        on_robust_correct += on_r

        clean_gain += int((off_c == 0) and (on_c == 1))
        clean_loss += int((off_c == 1) and (on_c == 0))
        robust_gain += int((off_r == 0) and (on_r == 1))
        robust_loss += int((off_r == 1) and (on_r == 0))

        if off_r == 1 and on_r == 1:
            robust_rr += 1
        elif off_r == 1 and on_r == 0:
            robust_rw += 1
        elif off_r == 0 and on_r == 1:
            robust_wr += 1
        else:
            robust_ww += 1

    n_valid = len(ids) - label_mismatch
    return {
        "n_paired": len(ids),
        "n_valid": n_valid,
        "label_mismatch": label_mismatch,
        "off_clean_acc": off_clean_correct / max(1, n_valid),
        "on_clean_acc": on_clean_correct / max(1, n_valid),
        "off_robust_acc": off_robust_correct / max(1, n_valid),
        "on_robust_acc": on_robust_correct / max(1, n_valid),
        "clean_gain_count": clean_gain,
        "clean_loss_count": clean_loss,
        "robust_gain_count": robust_gain,
        "robust_loss_count": robust_loss,
        "robust_rr": robust_rr,
        "robust_rw": robust_rw,
        "robust_wr": robust_wr,
        "robust_ww": robust_ww,
    }


def summarize_group(run_summaries: List[Dict]) -> Dict:
    clean_vals = [r["clean_accuracy"] for r in run_summaries]
    robust_vals = [r["robust_accuracy"] for r in run_summaries]
    clean_mean, clean_std = mean_std(clean_vals)
    robust_mean, robust_std = mean_std(robust_vals)
    return {
        "num_runs": len(run_summaries),
        "clean_mean": clean_mean,
        "clean_std": clean_std,
        "robust_mean": robust_mean,
        "robust_std": robust_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize SAE on/off comparison runs")
    parser.add_argument("--off-dirs", nargs="+", required=True, help="Run directories for SAE OFF")
    parser.add_argument("--on-dirs", nargs="+", required=True, help="Run directories for SAE ON")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save summary JSON")
    args = parser.parse_args()

    off_dirs = [Path(p).resolve() for p in args.off_dirs]
    on_dirs = [Path(p).resolve() for p in args.on_dirs]

    off_runs = [load_run(p) for p in off_dirs]
    on_runs = [load_run(p) for p in on_dirs]

    off_group = summarize_group(off_runs)
    on_group = summarize_group(on_runs)

    pair_count = min(len(off_runs), len(on_runs))
    paired = []
    for i in range(pair_count):
        paired.append(
            {
                "off_run": off_runs[i]["run_dir"],
                "on_run": on_runs[i]["run_dir"],
                "stats": paired_flip_stats(off_runs[i]["samples"], on_runs[i]["samples"]),
            }
        )

    # Aggregate paired stats across matched runs.
    agg = {
        "n_paired": 0,
        "n_valid": 0,
        "label_mismatch": 0,
        "clean_gain_count": 0,
        "clean_loss_count": 0,
        "robust_gain_count": 0,
        "robust_loss_count": 0,
        "robust_rr": 0,
        "robust_rw": 0,
        "robust_wr": 0,
        "robust_ww": 0,
    }
    for p in paired:
        s = p["stats"]
        for k in agg:
            agg[k] += int(s.get(k, 0))

    agg["off_clean_acc"] = 0.0
    agg["on_clean_acc"] = 0.0
    agg["off_robust_acc"] = 0.0
    agg["on_robust_acc"] = 0.0
    if agg["n_valid"] > 0:
        # Reconstruct from confusion totals.
        agg["off_robust_acc"] = (agg["robust_rr"] + agg["robust_rw"]) / agg["n_valid"]
        agg["on_robust_acc"] = (agg["robust_rr"] + agg["robust_wr"]) / agg["n_valid"]
        # clean accs derive from gain/loss only if we know off baseline; use weighted from groups instead.
        agg["off_clean_acc"] = off_group["clean_mean"]
        agg["on_clean_acc"] = on_group["clean_mean"]

    summary = {
        "off_group": off_group,
        "on_group": on_group,
        "paired_runs": paired,
        "paired_aggregate": agg,
    }

    print("=== SAE Comparison Summary ===")
    print(f"OFF runs: {len(off_runs)} | ON runs: {len(on_runs)} | Paired: {pair_count}")
    print("")
    print("Metric                 OFF(mean+-std)        ON(mean+-std)")
    print(
        f"Clean Acc              {off_group['clean_mean']:.4f} +- {off_group['clean_std']:.4f}    "
        f"{on_group['clean_mean']:.4f} +- {on_group['clean_std']:.4f}"
    )
    print(
        f"Robust Acc             {off_group['robust_mean']:.4f} +- {off_group['robust_std']:.4f}    "
        f"{on_group['robust_mean']:.4f} +- {on_group['robust_std']:.4f}"
    )
    print("")
    print("Paired flip stats (OFF -> ON)")
    print(f"Paired valid samples   {agg['n_valid']} (mismatch labels: {agg['label_mismatch']})")
    print(f"Clean gain/loss        +{agg['clean_gain_count']} / -{agg['clean_loss_count']}")
    print(f"Robust gain/loss       +{agg['robust_gain_count']} / -{agg['robust_loss_count']}")
    print(
        f"Robust 2x2 (off->on)   RR={agg['robust_rr']}  RW={agg['robust_rw']}  "
        f"WR={agg['robust_wr']}  WW={agg['robust_ww']}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary JSON: {out_path}")


if __name__ == "__main__":
    main()
