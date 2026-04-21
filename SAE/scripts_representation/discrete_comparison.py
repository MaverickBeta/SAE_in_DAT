import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


class ImageRecordDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, rec["class_idx"], rec["wnid"], os.path.basename(rec["path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clustering separability metrics between pre-SAE tensor and dense SAE latent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--imagenet-val-dir", type=str, default="/Data_share/hongyi/DAT/data/ImageNet/val")
    parser.add_argument("--base-ckpt", type=str, default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
    )
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--aggregation-level",
        type=str,
        choices=["token", "image"],
        default="image",
        help="Compute metrics on token-level or image-level representations.",
    )
    parser.add_argument(
        "--metric-max-points",
        type=int,
        default=5000,
        help="Max sampled points for metric computation (tokens or images depending on aggregation-level).",
    )
    parser.add_argument(
        "--post-active-dim-cap",
        type=int,
        default=12000,
        help="After dense-z zero-column pruning, cap active dims by non-zero count. 0 disables cap.",
    )
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="/Data_share/hongyi/DAT/SAE/results_representation")
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def list_images_in_dir(folder: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in paths if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def validate_paths(args: argparse.Namespace):
    if not os.path.isdir(args.imagenet_val_dir):
        raise FileNotFoundError(f"ImageNet val directory not found: {args.imagenet_val_dir}")
    if not os.path.exists(args.base_ckpt):
        raise FileNotFoundError(f"ConvNeXt checkpoint not found: {args.base_ckpt}")
    if not os.path.exists(args.sae_ckpt):
        raise FileNotFoundError(f"SAE checkpoint not found: {args.sae_ckpt}")


def sample_records(imagenet_val_dir: str, num_classes: int, images_per_class: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    class_dirs = [d for d in sorted(glob.glob(os.path.join(imagenet_val_dir, "*"))) if os.path.isdir(d)]

    eligible = []
    for d in class_dirs:
        wnid = os.path.basename(d)
        images = list_images_in_dir(d)
        if len(images) >= images_per_class:
            eligible.append((wnid, d, images))

    if len(eligible) < num_classes:
        raise RuntimeError(
            f"Not enough eligible classes: need {num_classes}, found {len(eligible)} with >= {images_per_class} images."
        )

    selected = random.sample(eligible, k=num_classes)
    class_meta = []
    records = []

    for class_idx, (wnid, class_dir, images) in enumerate(selected):
        chosen = images[:images_per_class]
        class_meta.append(
            {
                "class_idx": int(class_idx),
                "wnid": wnid,
                "class_dir": class_dir,
                "num_images": int(len(chosen)),
            }
        )
        for p in chosen:
            records.append({"path": p, "class_idx": int(class_idx), "wnid": wnid})

    return records, class_meta


def build_model_from_ckpt(device: torch.device, ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


def build_sae(device: torch.device, sae_ckpt_path: str):
    ckpt = torch.load(sae_ckpt_path, map_location=device)
    config = ckpt.get("config", {})

    d_in = ckpt.get("d_in", config.get("d_in", None))
    d_lat = ckpt.get("d_lat", config.get("d_lat", None))
    k = ckpt.get("k", config.get("k", None))

    if d_in is None:
        raise KeyError("Cannot resolve SAE d_in from checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve SAE d_lat from checkpoint")
        d_lat = int(d_in) * int(expansion_rate)
    if k is None:
        raise KeyError("Cannot resolve SAE k from checkpoint")

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    return sae, norm_mean, norm_std, {"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)}


def check_sae_stage_alignment(sae_ckpt_path: str, target_stage: int) -> Dict:
    ckpt = torch.load(sae_ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})
    d_in = ckpt.get("d_in", config.get("d_in", None))
    if d_in is None:
        raise KeyError("Cannot resolve SAE d_in from checkpoint.")

    inferred_stage = DIN_STAGE_MAP.get(int(d_in), None)
    is_aligned = inferred_stage == int(target_stage)
    expected_din = STAGE_DIN_MAP[int(target_stage)]

    return {
        "sae_ckpt": sae_ckpt_path,
        "sae_d_in": int(d_in),
        "inferred_stage": None if inferred_stage is None else int(inferred_stage),
        "target_stage": int(target_stage),
        "target_stage_expected_d_in": int(expected_din),
        "is_aligned": bool(is_aligned),
    }


def reservoir_update(
    rng: np.random.Generator,
    res_pre: List[np.ndarray],
    res_post: List[np.ndarray],
    res_label: List[int],
    batch_pre: np.ndarray,
    batch_post: np.ndarray,
    batch_label: np.ndarray,
    seen_count: int,
    max_points: int,
) -> int:
    for i in range(batch_pre.shape[0]):
        seen_count += 1
        if len(res_pre) < max_points:
            res_pre.append(batch_pre[i])
            res_post.append(batch_post[i])
            res_label.append(int(batch_label[i]))
        else:
            j = int(rng.integers(0, seen_count))
            if j < max_points:
                res_pre[j] = batch_pre[i]
                res_post[j] = batch_post[i]
                res_label[j] = int(batch_label[i])
    return seen_count


def extract_token_samples(
    model,
    sae,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    records: List[Dict],
    stage_idx: int,
    batch_size: int,
    num_workers: int,
    metric_max_points: int,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    dataset = ImageRecordDataset(records=records, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    rng = np.random.default_rng(seed)
    res_pre: List[np.ndarray] = []
    res_post: List[np.ndarray] = []
    res_label: List[int] = []
    seen_tokens = 0
    total_tokens = 0
    latent_dim = None

    with torch.no_grad():
        for images, class_idx, _, _ in tqdm(loader, desc=f"Extract stage{stage_idx}", leave=False):
            images = images.to(device)
            _ = model(images)

            stage_out = captured["stage"]
            bsz, channels, h, w = stage_out.shape
            flat = stage_out.permute(0, 2, 3, 1).reshape(-1, channels)
            flat_norm = (flat - norm_mean) / norm_std
            z = sae.encode(flat_norm)
            latent_dim = int(z.shape[1])

            labels_token = class_idx.to(device).view(bsz, 1).repeat(1, h * w).reshape(-1)

            batch_pre = flat_norm.detach().cpu().numpy().astype(np.float32)
            batch_post = z.detach().cpu().to(torch.float16).numpy()
            batch_label = labels_token.detach().cpu().numpy().astype(np.int32)

            total_tokens += int(batch_pre.shape[0])
            seen_tokens = reservoir_update(
                rng=rng,
                res_pre=res_pre,
                res_post=res_post,
                res_label=res_label,
                batch_pre=batch_pre,
                batch_post=batch_post,
                batch_label=batch_label,
                seen_count=seen_tokens,
                max_points=int(metric_max_points),
            )

    handle.remove()

    if len(res_pre) == 0:
        raise RuntimeError("No token samples collected for metric computation.")

    pre = np.stack(res_pre, axis=0).astype(np.float32)
    post = np.stack(res_post, axis=0)
    labels = np.asarray(res_label, dtype=np.int32)

    meta = {
        "total_tokens_seen": int(total_tokens),
        "sampled_tokens": int(pre.shape[0]),
        "latent_dim": int(latent_dim) if latent_dim is not None else None,
        "post_repr_dim": int(post.shape[1]),
    }
    return pre, post, labels, meta


def extract_image_samples(
    model,
    sae,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    records: List[Dict],
    stage_idx: int,
    batch_size: int,
    num_workers: int,
    metric_max_points: int,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    dataset = ImageRecordDataset(records=records, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    rng = np.random.default_rng(seed)
    res_pre: List[np.ndarray] = []
    res_post: List[np.ndarray] = []
    res_label: List[int] = []
    seen_images = 0
    latent_dim = None

    with torch.no_grad():
        for images, class_idx, _, _ in tqdm(loader, desc=f"Extract stage{stage_idx}", leave=False):
            images = images.to(device)
            _ = model(images)

            stage_out = captured["stage"]
            bsz, channels, h, w = stage_out.shape

            batch_pre_list = []
            batch_post_list = []
            batch_label_list = []

            # Process image-by-image to avoid creating very large (bsz*h*w, d_lat) tensors in GPU memory.
            for i in range(bsz):
                flat_i = stage_out[i].permute(1, 2, 0).reshape(-1, channels)
                flat_i = (flat_i - norm_mean) / norm_std
                z_i = sae.encode(flat_i)

                latent_dim = int(z_i.shape[1])
                pre_i = flat_i.mean(dim=0)
                post_i = z_i.mean(dim=0)

                batch_pre_list.append(pre_i.detach().cpu().numpy().astype(np.float32))
                batch_post_list.append(post_i.detach().cpu().to(torch.float16).numpy())
                batch_label_list.append(int(class_idx[i].item()))

            batch_pre = np.stack(batch_pre_list, axis=0)
            batch_post = np.stack(batch_post_list, axis=0)
            batch_label = np.asarray(batch_label_list, dtype=np.int32)

            seen_images = reservoir_update(
                rng=rng,
                res_pre=res_pre,
                res_post=res_post,
                res_label=res_label,
                batch_pre=batch_pre,
                batch_post=batch_post,
                batch_label=batch_label,
                seen_count=seen_images,
                max_points=int(metric_max_points),
            )

    handle.remove()

    if len(res_pre) == 0:
        raise RuntimeError("No image samples collected for metric computation.")

    pre = np.stack(res_pre, axis=0).astype(np.float32)
    post = np.stack(res_post, axis=0)
    labels = np.asarray(res_label, dtype=np.int32)

    meta = {
        "total_images_seen": int(len(records)),
        "sampled_images": int(pre.shape[0]),
        "latent_dim": int(latent_dim) if latent_dim is not None else None,
        "post_repr_dim": int(post.shape[1]),
    }
    return pre, post, labels, meta


def prepare_post_features(post_feats: np.ndarray, post_active_dim_cap: int) -> Tuple[np.ndarray, Dict]:
    active_cols = np.where(np.any(post_feats != 0, axis=0))[0]
    if active_cols.size == 0:
        raise RuntimeError("All sampled SAE latent values are zero; cannot compute post-SAE metrics.")

    post_active = post_feats[:, active_cols]
    if int(post_active_dim_cap) > 0 and post_active.shape[1] > int(post_active_dim_cap):
        nnz = np.count_nonzero(post_active, axis=0)
        keep_local = np.argsort(nnz)[::-1][: int(post_active_dim_cap)]
        post_active = post_active[:, keep_local]

    post_active = post_active.astype(np.float32)
    meta = {
        "latent_total_dim": int(post_feats.shape[1]),
        "latent_nonzero_dim": int(active_cols.size),
        "latent_selected_dim": int(post_active.shape[1]),
        "post_active_dim_cap": int(post_active_dim_cap),
    }
    return post_active, meta


def compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    knn_k: int,
    knn_folds: int,
) -> Dict:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    metrics = {
        "silhouette_true_labels": float(silhouette_score(x, y)),
    }

    n_classes = int(np.unique(y).shape[0])
    km = KMeans(n_clusters=n_classes, random_state=int(random_state), n_init=10)
    pred = km.fit_predict(x)
    metrics["kmeans_ari"] = float(adjusted_rand_score(y, pred))
    metrics["kmeans_nmi"] = float(normalized_mutual_info_score(y, pred))

    cv = StratifiedKFold(n_splits=int(knn_folds), shuffle=True, random_state=int(random_state))
    knn = KNeighborsClassifier(n_neighbors=int(knn_k))
    acc = cross_val_score(knn, x, y, cv=cv, scoring="accuracy")
    metrics["knn_cv_acc_mean"] = float(acc.mean())
    metrics["knn_cv_acc_std"] = float(acc.std())
    metrics["knn_k"] = int(knn_k)
    metrics["knn_folds"] = int(knn_folds)
    return metrics


def make_out_dir(results_dir: str, run_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    if run_name:
        leaf = run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        leaf = f"discrete_cmp_{ts}"
    out_dir = os.path.join(results_dir, "discrete_comparison", leaf)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    args = parse_args()
    validate_paths(args)

    out_dir = make_out_dir(args.results_dir, args.run_name)

    records, class_meta = sample_records(
        imagenet_val_dir=args.imagenet_val_dir,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        seed=args.random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sampled classes: {len(class_meta)}")
    print(f"Total sampled images: {len(records)}")
    print(f"Output dir: {out_dir}")

    model = build_model_from_ckpt(device=device, ckpt_path=args.base_ckpt)
    sae, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)

    sae_check = check_sae_stage_alignment(sae_ckpt_path=args.sae_ckpt, target_stage=args.stage)
    print(
        "SAE stage alignment | "
        f"d_in={sae_check['sae_d_in']}, inferred_stage={sae_check['inferred_stage']}, "
        f"target_stage={sae_check['target_stage']}, aligned={sae_check['is_aligned']}"
    )
    if not sae_check["is_aligned"]:
        raise ValueError("SAE and selected stage are not aligned. Please use a matching SAE checkpoint.")

    if args.aggregation_level == "token":
        pre_feats, post_feats, labels, sampling_meta = extract_token_samples(
            model=model,
            sae=sae,
            norm_mean=norm_mean,
            norm_std=norm_std,
            records=records,
            stage_idx=args.stage,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            metric_max_points=args.metric_max_points,
            seed=args.random_state,
            device=device,
        )
    else:
        pre_feats, post_feats, labels, sampling_meta = extract_image_samples(
            model=model,
            sae=sae,
            norm_mean=norm_mean,
            norm_std=norm_std,
            records=records,
            stage_idx=args.stage,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            metric_max_points=args.metric_max_points,
            seed=args.random_state,
            device=device,
        )

    post_metric_input, post_prepare_meta = prepare_post_features(
        post_feats=post_feats,
        post_active_dim_cap=args.post_active_dim_cap,
    )

    pre_metrics = compute_metrics(
        x=pre_feats,
        y=labels,
        random_state=args.random_state,
        knn_k=args.knn_k,
        knn_folds=args.knn_folds,
    )
    post_metrics = compute_metrics(
        x=post_metric_input,
        y=labels,
        random_state=args.random_state,
        knn_k=args.knn_k,
        knn_folds=args.knn_folds,
    )

    class_json = os.path.join(out_dir, "sampled_classes.json")
    labels_npy = os.path.join(out_dir, "labels.npy")
    pre_npy = os.path.join(out_dir, "pre_sae_tensor_samples.npy")
    post_npy = os.path.join(out_dir, "post_sae_latent_samples.npy")
    post_metric_npy = os.path.join(out_dir, "post_metric_input.npy")

    with open(class_json, "w", encoding="utf-8") as f:
        json.dump(class_meta, f, indent=2)

    np.save(labels_npy, labels)
    np.save(pre_npy, pre_feats)
    np.save(post_npy, post_feats)
    np.save(post_metric_npy, post_metric_input)

    summary = {
        "imagenet_val_dir": args.imagenet_val_dir,
        "base_ckpt": args.base_ckpt,
        "sae_ckpt": args.sae_ckpt,
        "sae_config": sae_cfg,
        "stage": int(args.stage),
        "aggregation_level": args.aggregation_level,
        "num_classes": int(args.num_classes),
        "images_per_class": int(args.images_per_class),
        "num_samples": int(len(records)),
        "sampling": sampling_meta,
        "pre_feature_shape": list(pre_feats.shape),
        "post_feature_shape": list(post_feats.shape),
        "post_metric_input_shape": list(post_metric_input.shape),
        "post_prepare": post_prepare_meta,
        "metrics": {
            "pre_input_tensor": pre_metrics,
            "post_dense_latent": post_metrics,
        },
        "sae_stage_check": sae_check,
        "files": {
            "sampled_classes_json": class_json,
            "labels_npy": labels_npy,
            "pre_sae_tensor_samples_npy": pre_npy,
            "post_sae_latent_samples_npy": post_npy,
            "post_metric_input_npy": post_metric_npy,
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
