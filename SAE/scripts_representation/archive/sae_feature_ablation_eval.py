import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import matplotlib.pyplot as plt
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


class FileListDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None, cache_images: bool = False):
        self.image_paths = image_paths
        self.transform = transform
        self.cache_images = cache_images
        self.cached = None

        if self.cache_images:
            self.cached = []
            for path in self.image_paths:
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                self.cached.append((image, os.path.basename(path)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cached is not None:
            return self.cached[idx]

        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate causal impact of SAE feature ablation on classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-ckpt", type=str, default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=(
            "/Data_share/hongyi/DAT/SAE/project/checkpoints/stage0/"
            "k32_exp32/sae_stage0_din192_exp32_k32_step_50000.pt"
        ),
    )
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--imagenet-val-dir", type=str, default="/Data_share/hongyi/DAT/data/ImageNet/val")
    parser.add_argument("--source-wnid", type=str, default="n02077923")
    parser.add_argument("--source-cls", type=int, default=150)
    parser.add_argument("--source-dir", type=str, default="")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topn-ablate", type=int, default=10)
    parser.add_argument("--low-pool-size", type=int, default=300)
    parser.add_argument("--random-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="If >0, set torch intra-op and inter-op thread count to this value.",
    )
    parser.add_argument(
        "--cache-images",
        action="store_true",
        default=False,
        help="Cache transformed images in RAM to reduce repeated CPU decoding overhead.",
    )
    parser.add_argument(
        "--recon-mode",
        type=str,
        choices=["raw", "normalized"],
        default="raw",
        help=(
            "raw: decode -> denorm back to stage space before feeding model; "
            "normalized: decode output fed directly (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/ablation_eval",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def list_images_in_dir(folder: str) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def resolve_source_dir(args: argparse.Namespace) -> str:
    if args.source_dir:
        return args.source_dir
    return os.path.join(args.imagenet_val_dir, args.source_wnid)


def infer_stage_from_din(d_in: int) -> Optional[int]:
    return DIN_STAGE_MAP.get(int(d_in))


def infer_k_tag(k: int, sae_ckpt_path: str) -> str:
    if int(k) == 32:
        return "k32"
    if int(k) == 64:
        return "k64"
    name = os.path.basename(sae_ckpt_path).lower()
    if "k32" in name:
        return "k32"
    if "k64" in name:
        return "k64"
    return f"k{int(k)}"


def build_model(device: torch.device, base_ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, base_ckpt_path)
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
        raise KeyError("Cannot resolve d_in from checkpoint")
    if k is None:
        raise KeyError("Cannot resolve k from checkpoint")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat from checkpoint")
        d_lat = int(d_in) * int(expansion_rate)

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)

    resolved_cfg = dict(config)
    resolved_cfg["d_in"] = int(d_in)
    resolved_cfg["d_lat"] = int(d_lat)
    resolved_cfg["k"] = int(k)
    return sae, norm_mean, norm_std, resolved_cfg


def compute_mean_activation(
    loader: DataLoader,
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    device: torch.device,
) -> np.ndarray:
    captured = {}

    def hook_fn(_, __, output):
        captured["stage"] = output.detach()

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    total = None
    total_tokens = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Compute mean activations", leave=False):
            images = images.to(device)
            _ = model(images)

            stage_out = captured["stage"]
            _, channels, _, _ = stage_out.shape
            flat = stage_out.permute(0, 2, 3, 1).reshape(-1, channels)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)

            zsum = z.sum(dim=0).detach().cpu()
            total = zsum if total is None else total + zsum
            total_tokens += z.shape[0]

    handle.remove()

    if total_tokens == 0:
        raise RuntimeError("No tokens extracted when computing mean activation")

    return (total / total_tokens).numpy()


def evaluate_condition(
    loader: DataLoader,
    model,
    source_cls: int,
    device: torch.device,
    use_sae: bool,
    stage_idx: int,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    recon_mode: str,
    ablate_indices: Optional[List[int]] = None,
) -> Dict:
    handle = None
    ablate_tensor = None

    if use_sae:
        if ablate_indices:
            ablate_tensor = torch.tensor(ablate_indices, dtype=torch.long, device=device)

        def hook_fn(_, __, output):
            bsz, channels, height, width = output.shape
            flat = output.permute(0, 2, 3, 1).reshape(-1, channels)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)
            if ablate_tensor is not None and ablate_tensor.numel() > 0:
                z[:, ablate_tensor] = 0.0
            recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
            if recon_mode == "raw":
                recon_flat = recon_norm * norm_std + norm_mean
            else:
                recon_flat = recon_norm
            recon = recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
            return recon

        handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    total = 0
    source_correct = 0
    source_escape = 0
    source_logit_sum = 0.0
    source_prob_sum = 0.0
    source_margin_sum = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            source_logits = logits[:, source_cls]
            other_logits = logits.clone()
            other_logits[:, source_cls] = float("-inf")
            best_other_logits = other_logits.max(dim=1).values
            margins = source_logits - best_other_logits

            source_correct += int((preds == source_cls).sum().item())
            source_escape += int((preds != source_cls).sum().item())
            source_logit_sum += float(source_logits.sum().item())
            source_prob_sum += float(probs[:, source_cls].sum().item())
            source_margin_sum += float(margins.sum().item())
            total += images.size(0)

    if handle is not None:
        handle.remove()

    if total == 0:
        raise RuntimeError("No images evaluated")

    return {
        "num_images": int(total),
        "source_top1_acc": float(source_correct / total),
        "escape_rate": float(source_escape / total),
        "avg_source_logit": float(source_logit_sum / total),
        "avg_source_prob": float(source_prob_sum / total),
        "avg_source_margin": float(source_margin_sum / total),
    }


def build_output_dir(results_dir: str, stage_idx: int, k_tag: str, source_wnid: str, run_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    leaf = run_name if run_name else f"{source_wnid}_stage{stage_idx}_{k_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(results_dir, f"stage{stage_idx}", k_tag, leaf)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_rows_csv(rows: List[Dict], out_csv: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    out_png: str,
    title: str,
    x: List[int],
    curves: Dict[str, List[float]],
    ylabel: str,
    baseline_lines: Optional[Dict[str, float]] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, vals in curves.items():
        ax.plot(x, vals, marker="o", linewidth=2, label=name)

    if baseline_lines:
        colors = ["#2c3e50", "#7f8c8d", "#95a5a6", "#34495e"]
        for i, (name, value) in enumerate(baseline_lines.items()):
            color = colors[i % len(colors)]
            ax.axhline(value, linestyle="--", linewidth=1.5, color=color, alpha=0.9, label=name)

    ax.set_title(title)
    ax.set_xlabel("Number of ablated features")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(args.torch_threads)

    source_dir = resolve_source_dir(args)
    if not os.path.exists(args.base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_ckpt}")
    if not os.path.exists(args.sae_ckpt):
        raise FileNotFoundError(f"SAE checkpoint not found: {args.sae_ckpt}")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    image_paths = list_images_in_dir(source_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if len(image_paths) == 0:
        raise RuntimeError("No images found for evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Source dir: {source_dir}")
    print(f"Images used: {len(image_paths)}")
    print(f"SAE reconstruction mode: {args.recon_mode}")
    print(f"Cache images: {args.cache_images}")
    if args.torch_threads and args.torch_threads > 0:
        print(f"Torch threads: {args.torch_threads}")

    dataset = FileListDataset(image_paths, transform=get_transform(), cache_images=args.cache_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    model = build_model(device, args.base_ckpt)
    sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

    inferred_stage = infer_stage_from_din(int(sae_cfg["d_in"]))
    if inferred_stage is not None and inferred_stage != args.stage:
        raise ValueError(
            f"Stage/SAE mismatch: --stage={args.stage} but SAE d_in implies stage{inferred_stage}"
        )
    if STAGE_DIN_MAP[args.stage] != int(sae_cfg["d_in"]):
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.stage} expects d_in={STAGE_DIN_MAP[args.stage]}, "
            f"SAE has d_in={sae_cfg['d_in']}"
        )

    k_tag = infer_k_tag(int(sae_cfg["k"]), args.sae_ckpt)
    out_dir = build_output_dir(args.results_dir, args.stage, k_tag, args.source_wnid, args.run_name)
    print(f"Output dir: {out_dir}")

    mean_act = compute_mean_activation(
        loader=loader,
        model=model,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        stage_idx=args.stage,
        device=device,
    )

    d_lat = mean_act.shape[0]
    n_max = min(args.topn_ablate, d_lat)
    low_pool_size = min(args.low_pool_size, d_lat)

    order_desc = np.argsort(mean_act)[::-1]
    order_asc = np.argsort(mean_act)

    high_topn = order_desc[:n_max].tolist()
    low_pool = order_asc[:low_pool_size].tolist()

    baseline_original = evaluate_condition(
        loader=loader,
        model=model,
        source_cls=args.source_cls,
        device=device,
        use_sae=False,
        stage_idx=args.stage,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        recon_mode=args.recon_mode,
        ablate_indices=None,
    )

    baseline_sae_recon = evaluate_condition(
        loader=loader,
        model=model,
        source_cls=args.source_cls,
        device=device,
        use_sae=True,
        stage_idx=args.stage,
        sae_model=sae_model,
        norm_mean=norm_mean,
        norm_std=norm_std,
        recon_mode=args.recon_mode,
        ablate_indices=None,
    )

    rows = []

    rows.append({"condition": "original_passthrough", "n_ablated": 0, "repeat": 0, **baseline_original})
    rows.append({"condition": "sae_recon_no_ablation", "n_ablated": 0, "repeat": 0, **baseline_sae_recon})

    high_acc_curve = []
    low_acc_curve = []
    rnd_acc_curve = []
    high_logit_curve = []
    low_logit_curve = []
    rnd_logit_curve = []

    for n in range(1, n_max + 1):
        high_idx = high_topn[:n]
        low_idx = low_pool[:n]

        high_metrics = evaluate_condition(
            loader=loader,
            model=model,
            source_cls=args.source_cls,
            device=device,
            use_sae=True,
            stage_idx=args.stage,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            recon_mode=args.recon_mode,
            ablate_indices=high_idx,
        )
        low_metrics = evaluate_condition(
            loader=loader,
            model=model,
            source_cls=args.source_cls,
            device=device,
            use_sae=True,
            stage_idx=args.stage,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            recon_mode=args.recon_mode,
            ablate_indices=low_idx,
        )

        rows.append({"condition": "high_activation", "n_ablated": n, "repeat": 0, **high_metrics})
        rows.append({"condition": "low_activation", "n_ablated": n, "repeat": 0, **low_metrics})

        high_acc_curve.append(high_metrics["source_top1_acc"])
        low_acc_curve.append(low_metrics["source_top1_acc"])
        high_logit_curve.append(high_metrics["avg_source_logit"])
        low_logit_curve.append(low_metrics["avg_source_logit"])

        rnd_metrics_list = []
        for r in range(args.random_repeats):
            rnd_idx = rng.choice(d_lat, size=n, replace=False).tolist()
            rnd_metrics = evaluate_condition(
                loader=loader,
                model=model,
                source_cls=args.source_cls,
                device=device,
                use_sae=True,
                stage_idx=args.stage,
                sae_model=sae_model,
                norm_mean=norm_mean,
                norm_std=norm_std,
                recon_mode=args.recon_mode,
                ablate_indices=rnd_idx,
            )
            rnd_metrics_list.append(rnd_metrics)
            rows.append({"condition": "random_activation", "n_ablated": n, "repeat": r + 1, **rnd_metrics})

        rnd_acc_curve.append(float(np.mean([m["source_top1_acc"] for m in rnd_metrics_list])))
        rnd_logit_curve.append(float(np.mean([m["avg_source_logit"] for m in rnd_metrics_list])))

    x = list(range(1, n_max + 1))
    plot_curve(
        out_png=os.path.join(out_dir, "curve_source_top1_acc.png"),
        title=(
            f"Stage{args.stage} {k_tag} | Source Top1 Acc vs Ablation Count\n"
            f"{args.source_wnid} ({len(image_paths)} images)"
        ),
        x=x,
        curves={
            "high_activation": high_acc_curve,
            "low_activation": low_acc_curve,
            "random_mean": rnd_acc_curve,
        },
        ylabel="Source Top1 Accuracy",
        baseline_lines={
            "original_passthrough": baseline_original["source_top1_acc"],
            "sae_recon_no_ablation": baseline_sae_recon["source_top1_acc"],
        },
    )
    plot_curve(
        out_png=os.path.join(out_dir, "curve_avg_source_logit.png"),
        title=(
            f"Stage{args.stage} {k_tag} | Avg Source Logit vs Ablation Count\n"
            f"{args.source_wnid} ({len(image_paths)} images)"
        ),
        x=x,
        curves={
            "high_activation": high_logit_curve,
            "low_activation": low_logit_curve,
            "random_mean": rnd_logit_curve,
        },
        ylabel="Average Source Logit",
        baseline_lines={
            "original_passthrough": baseline_original["avg_source_logit"],
            "sae_recon_no_ablation": baseline_sae_recon["avg_source_logit"],
        },
    )

    write_rows_csv(rows, os.path.join(out_dir, "ablation_metrics_rows.csv"))
    np.save(os.path.join(out_dir, "mean_activation.npy"), mean_act)

    payload = {
        "config": {
            "base_ckpt": args.base_ckpt,
            "sae_ckpt": args.sae_ckpt,
            "stage": args.stage,
            "source_wnid": args.source_wnid,
            "source_cls": args.source_cls,
            "num_images": len(image_paths),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "topn_ablate": n_max,
            "low_pool_size": low_pool_size,
            "random_repeats": args.random_repeats,
            "seed": args.seed,
            "recon_mode": args.recon_mode,
            "cache_images": args.cache_images,
            "torch_threads": args.torch_threads,
            "k_tag": k_tag,
            "sae_config": {
                "d_in": int(sae_cfg["d_in"]),
                "d_lat": int(sae_cfg["d_lat"]),
                "k": int(sae_cfg["k"]),
            },
            "preprocess": "Resize(256, bicubic) -> CenterCrop(224) -> ToTensor",
        },
        "selected_features": {
            "high_topn": high_topn,
            "low_pool": low_pool,
        },
        "baselines": {
            "original_passthrough": baseline_original,
            "sae_recon_no_ablation": baseline_sae_recon,
        },
        "curves": {
            "x_n_ablated": x,
            "source_top1_acc": {
                "high_activation": high_acc_curve,
                "low_activation": low_acc_curve,
                "random_mean": rnd_acc_curve,
            },
            "avg_source_logit": {
                "high_activation": high_logit_curve,
                "low_activation": low_logit_curve,
                "random_mean": rnd_logit_curve,
            },
        },
        "files": {
            "mean_activation_npy": os.path.join(out_dir, "mean_activation.npy"),
            "rows_csv": os.path.join(out_dir, "ablation_metrics_rows.csv"),
            "acc_curve_png": os.path.join(out_dir, "curve_source_top1_acc.png"),
            "logit_curve_png": os.path.join(out_dir, "curve_avg_source_logit.png"),
        },
    }

    out_json = os.path.join(out_dir, "ablation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done.")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
