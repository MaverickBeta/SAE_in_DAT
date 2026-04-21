#!/usr/bin/env python3
"""Class-wise SAE TopK overlap and class-unique feature analysis.

Workflow:
1) Randomly sample N classes from imagenet/train.
2) For each class, sample fixed K images and compute class mean SAE activation per feature.
3) Build class TopK feature sets from the largest class-mean activations.
4) Report overlap statistics for TopK sets, including per-feature activation values by class.
5) Select class-unique features (top N): for feature f in class c,
   max_{c'!=c}(mu_{c',f}) < unique_ratio * mu_{c,f}.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Resolve DAT repository root from current script location: DAT/SAE/scripts_untar/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


@dataclass
class ClassStats:
    class_name: str
    image_count: int
    token_count: int
    mean_vec: np.ndarray


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


def resolve_stage(sae_cfg: dict, sae_stage: Optional[int]) -> int:
    if sae_stage is not None:
        if sae_stage not in STAGE_DIN_MAP:
            raise ValueError(f"Invalid sae_stage={sae_stage}, expected 0..3")
        if int(sae_cfg["d_in"]) != STAGE_DIN_MAP[sae_stage]:
            raise ValueError(
                f"Stage/SAE mismatch: stage{sae_stage} expects d_in={STAGE_DIN_MAP[sae_stage]}, "
                f"SAE has d_in={sae_cfg['d_in']}"
            )
        return sae_stage

    inferred = DIN_STAGE_MAP.get(int(sae_cfg["d_in"]), None)
    if inferred is None:
        raise ValueError(f"Cannot infer stage from SAE d_in={sae_cfg['d_in']}; pass --sae-stage explicitly")
    return inferred


def list_class_dirs(imagenet_train_dir: Path) -> List[Path]:
    if not imagenet_train_dir.exists():
        raise FileNotFoundError(f"Directory not found: {imagenet_train_dir}")

    out = sorted([p for p in imagenet_train_dir.iterdir() if p.is_dir()])
    if not out:
        raise RuntimeError(f"No class directories found in {imagenet_train_dir}")
    return out


def list_images_in_class(class_dir: Path) -> List[Path]:
    exts = {".jpeg", ".jpg", ".png"}
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def load_image_tensor(path: Path, transform: T.Compose) -> torch.Tensor:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return transform(rgb)


def chunked(seq: Sequence[Path], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def extract_class_latent_stats(
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    image_paths: List[Path],
    batch_size: int,
    device: torch.device,
    transform: T.Compose,
) -> Tuple[np.ndarray, int]:
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_, __, output):
        captured["feat"] = output

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    z_parts: List[torch.Tensor] = []
    token_count = 0

    with torch.no_grad():
        for paths_batch in chunked(image_paths, batch_size):
            xb = torch.stack([load_image_tensor(p, transform) for p in paths_batch], dim=0).to(device)
            _ = model(xb)

            feat = captured["feat"]
            b, c, h, w = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)

            z_parts.append(z.detach().cpu())
            token_count += int(z.shape[0])

    handle.remove()

    if not z_parts or token_count == 0:
        raise RuntimeError("No features were extracted for class")

    z_all = torch.cat(z_parts, dim=0)
    mean_vec_t = z_all.mean(dim=0)
    return mean_vec_t.numpy(), token_count


def build_topk_sets(class_stats: List[ClassStats], topk: int) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for cs in class_stats:
        order = np.argsort(cs.mean_vec)[::-1]
        out[cs.class_name] = [int(x) for x in order[:topk]]
    return out


def compute_overlap_report(
    topk_sets: Dict[str, List[int]],
    class_stats: List[ClassStats],
    unique_by_class: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    sets = {k: set(v) for k, v in topk_sets.items()}
    mean_map = {cs.class_name: cs.mean_vec for cs in class_stats}

    by_class: Dict[str, Dict[str, object]] = {}
    total_unique = 0
    total_overlapped = 0
    total_non_overlapped = 0

    for cname in sorted(unique_by_class.keys()):
        unique_feats = unique_by_class.get(cname, [])
        overlapped_features: List[Dict[str, object]] = []
        non_overlapped_features: List[Dict[str, object]] = []

        for item in unique_feats:
            fid = int(item["feature"])
            classes_with_f = sorted([k for k, s in sets.items() if fid in s])
            if not classes_with_f:
                # Should not happen in normal flow, but keep output robust.
                classes_with_f = [cname]

            mean_activation_by_class = {c: float(mean_map[c][fid]) for c in classes_with_f}
            record = {
                "feature": fid,
                "overlap_class_count": int(len(classes_with_f)),
                "overlap_classes": classes_with_f,
                "mean_activation_by_class": mean_activation_by_class,
                "class_mean_activation": float(item.get("class_mean_activation", mean_map[cname][fid])),
                "others_max_mean_activation": float(item.get("others_max_mean_activation", 0.0)),
                "others_to_class_ratio": float(item.get("others_to_class_ratio", 0.0)),
            }

            if len(classes_with_f) > 1:
                overlapped_features.append(record)
            else:
                non_overlapped_features.append(record)

        by_class[cname] = {
            "unique_feature_count": int(len(unique_feats)),
            "overlapped_feature_count": int(len(overlapped_features)),
            "non_overlapped_feature_count": int(len(non_overlapped_features)),
            "overlapped_features": overlapped_features,
            "non_overlapped_features": non_overlapped_features,
        }

        total_unique += len(unique_feats)
        total_overlapped += len(overlapped_features)
        total_non_overlapped += len(non_overlapped_features)

    return {
        "summary": {
            "class_count": int(len(by_class)),
            "total_unique_features_checked": int(total_unique),
            "total_overlapped_features": int(total_overlapped),
            "total_non_overlapped_features": int(total_non_overlapped),
        },
        "by_class": by_class,
    }


def select_unique_features(
    class_stats: List[ClassStats],
    class_topk: Dict[str, List[int]],
    unique_ratio: float,
    topn: int,
    eps: float,
) -> Dict[str, List[Dict[str, object]]]:
    class_names = [cs.class_name for cs in class_stats]
    mean_matrix = np.stack([cs.mean_vec for cs in class_stats], axis=0)  # [C, D]
    out: Dict[str, List[Dict[str, object]]] = {}

    for ci, cname in enumerate(class_names):
        row = mean_matrix[ci]
        # Restrict candidate features to this class TopK set, then sort by class activation.
        cand = class_topk.get(cname, [])
        order = sorted(cand, key=lambda fid: float(row[fid]), reverse=True)
        picked: List[Dict[str, object]] = []

        for fid in order:
            this_val = float(row[fid])
            if this_val <= 0.0:
                continue

            if mean_matrix.shape[0] == 1:
                others_max = 0.0
            else:
                others = np.delete(mean_matrix[:, fid], ci)
                others_max = float(np.max(others)) if others.size > 0 else 0.0

            threshold = unique_ratio * this_val
            if others_max < threshold:
                picked.append(
                    {
                        "feature": int(fid),
                        "class_mean_activation": this_val,
                        "others_max_mean_activation": others_max,
                        "others_to_class_ratio": float(others_max / (this_val + eps)),
                    }
                )
                if len(picked) >= topn:
                    break

        out[cname] = picked

    return out


def main():
    parser = argparse.ArgumentParser(description="Class-wise SAE TopK overlap analysis")
    parser.add_argument("--imagenet-train-dir", type=str, default="/Data_share/hongyi/imagenet/train")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"))
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt"),
    )
    parser.add_argument("--sae-stage", type=int, default=None)

    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--unique-ratio", type=float, default=0.1)
    parser.add_argument("--unique-topn", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "SAE" / "scripts_untar" / "sae_class_cluster_results.json"),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = Path(args.imagenet_train_dir).resolve()
    class_dirs = list_class_dirs(train_dir)
    if args.num_classes <= 0:
        raise ValueError("num-classes must be > 0")

    if args.num_classes > len(class_dirs):
        raise ValueError(
            f"Requested num_classes={args.num_classes} but only {len(class_dirs)} classes exist in {train_dir}"
        )
    if args.images_per_class <= 0:
        raise ValueError("images-per-class must be > 0")
    if args.topk <= 0:
        raise ValueError("topk must be > 0")
    if args.unique_topn <= 0:
        raise ValueError("unique-topn must be > 0")
    if not (0.0 < args.unique_ratio < 1.0):
        raise ValueError("unique-ratio must be in (0, 1)")

    sampled_class_dirs = random.sample(class_dirs, k=args.num_classes)
    sampled_class_dirs = sorted(sampled_class_dirs, key=lambda p: p.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_base_model(device=device, checkpoint=args.checkpoint)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)
    stage_idx = resolve_stage(sae_cfg, sae_stage=args.sae_stage)
    print(
        f"SAE loaded: stage={stage_idx}, d_in={sae_cfg['d_in']}, d_lat={sae_cfg['d_lat']}, k={sae_cfg['k']}"
    )

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    class_stats: List[ClassStats] = []

    class_pbar = tqdm(sampled_class_dirs, desc="Classes", unit="class")
    for class_dir in class_pbar:
        image_paths = list_images_in_class(class_dir)
        if not image_paths:
            continue

        take_n = min(len(image_paths), args.images_per_class)
        if len(image_paths) > take_n:
            image_paths = random.sample(image_paths, k=take_n)

        mean_vec, token_count = extract_class_latent_stats(
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            stage_idx=stage_idx,
            image_paths=image_paths,
            batch_size=args.batch_size,
            device=device,
            transform=transform,
        )

        class_stats.append(
            ClassStats(
                class_name=class_dir.name,
                image_count=len(image_paths),
                token_count=token_count,
                mean_vec=mean_vec,
            )
        )

    if not class_stats:
        raise RuntimeError("No class statistics were computed. Check imagenet path and image extensions.")

    class_meta: Dict[str, Dict[str, object]] = {}

    class_topk = build_topk_sets(class_stats, topk=args.topk)

    for cs in class_stats:
        topk_ids = class_topk[cs.class_name]

        class_meta[cs.class_name] = {
            "image_count": int(cs.image_count),
            "token_count": int(cs.token_count),
            "topk_count": int(len(topk_ids)),
        }

    unique_by_class = select_unique_features(
        class_stats=class_stats,
        class_topk=class_topk,
        unique_ratio=args.unique_ratio,
        topn=args.unique_topn,
        eps=args.eps,
    )
    overlap = compute_overlap_report(class_topk, class_stats, unique_by_class)

    out = {
        "meta": {
            "imagenet_train_dir": str(train_dir),
            "num_classes_requested": int(args.num_classes),
            "num_classes_used": int(len(class_stats)),
            "seed": int(args.seed),
            "images_per_class": int(args.images_per_class),
            "batch_size": int(args.batch_size),
            "checkpoint": str(args.checkpoint),
            "sae_ckpt": str(args.sae_ckpt),
            "sae_stage": int(stage_idx),
            "sae_d_in": int(sae_cfg["d_in"]),
            "sae_d_lat": int(sae_cfg["d_lat"]),
            "sae_k": int(sae_cfg["k"]),
            "topk": int(args.topk),
            "unique_ratio": float(args.unique_ratio),
            "unique_topn": int(args.unique_topn),
            "eps": float(args.eps),
        },
        "sampled_classes": [cs.class_name for cs in class_stats],
        "global": {
            "total_token_count": int(sum(cs.token_count for cs in class_stats)),
            "latent_dim": int(class_stats[0].mean_vec.shape[0]),
        },
        "class_meta": class_meta,
        "overlap": overlap,
        "unique_features_topn_by_class": unique_by_class,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Classes used: {len(class_stats)}")
    print(
        "Overlap summary (on unique_topn features): "
        f"total_unique_checked={overlap['summary']['total_unique_features_checked']}, "
        f"overlapped={overlap['summary']['total_overlapped_features']}, "
        f"non_overlapped={overlap['summary']['total_non_overlapped_features']}"
    )


if __name__ == "__main__":
    main()
