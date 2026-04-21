import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

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
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick SAE reconstruction sanity check at one ConvNeXt stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-ckpt", type=str, default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
    parser.add_argument("--sae-ckpt", type=str, required=True)
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--imagenet-val-dir", type=str, default="/Data_share/hongyi/DAT/data/ImageNet/val")
    parser.add_argument("--source-wnid", type=str, default="n02077923")
    parser.add_argument("--source-dir", type=str, default="")
    parser.add_argument("--source-cls", type=int, default=150)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--recon-mode",
        type=str,
        choices=["raw", "normalized", "both"],
        default="both",
        help=(
            "raw: decode -> denorm back to stage space; "
            "normalized: decode output used directly; "
            "both: evaluate both modes."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/results_representation/reconstruct_eval",
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


def evaluate_original(loader: DataLoader, model, source_cls: int, device: torch.device) -> Dict:
    total = 0
    correct = 0
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
            best_other = other_logits.max(dim=1).values
            margin = source_logits - best_other

            total += images.size(0)
            correct += int((preds == source_cls).sum().item())
            source_logit_sum += float(source_logits.sum().item())
            source_prob_sum += float(probs[:, source_cls].sum().item())
            source_margin_sum += float(margin.sum().item())

    return {
        "num_images": int(total),
        "source_top1_acc": float(correct / max(1, total)),
        "avg_source_logit": float(source_logit_sum / max(1, total)),
        "avg_source_prob": float(source_prob_sum / max(1, total)),
        "avg_source_margin": float(source_margin_sum / max(1, total)),
    }


def evaluate_reconstruction(
    loader: DataLoader,
    model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    stage_idx: int,
    source_cls: int,
    device: torch.device,
    recon_mode: str,
) -> Dict:
    captured = {"mse_raw_sum": 0.0, "mse_norm_sum": 0.0, "cos_norm_sum": 0.0, "token_count": 0}

    def hook_fn(_, __, output):
        bsz, channels, height, width = output.shape
        flat_raw = output.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_norm = (flat_raw - norm_mean) / norm_std

        z = sae_model.encode(flat_norm)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec

        if recon_mode == "raw":
            recon_flat = recon_norm * norm_std + norm_mean
        else:
            recon_flat = recon_norm

        diff_raw = recon_flat - flat_raw
        mse_raw = (diff_raw * diff_raw).mean().item()

        diff_norm = recon_norm - flat_norm
        mse_norm = (diff_norm * diff_norm).mean().item()

        cos_norm = torch.nn.functional.cosine_similarity(flat_norm, recon_norm, dim=1).mean().item()

        captured["mse_raw_sum"] += mse_raw * flat_raw.shape[0]
        captured["mse_norm_sum"] += mse_norm * flat_raw.shape[0]
        captured["cos_norm_sum"] += cos_norm * flat_raw.shape[0]
        captured["token_count"] += int(flat_raw.shape[0])

        recon = recon_flat.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
        return recon

    handle = model.stages[stage_idx].register_forward_hook(hook_fn)

    total = 0
    correct = 0
    source_logit_sum = 0.0
    source_prob_sum = 0.0
    source_margin_sum = 0.0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Evaluate recon ({recon_mode})", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            source_logits = logits[:, source_cls]
            other_logits = logits.clone()
            other_logits[:, source_cls] = float("-inf")
            best_other = other_logits.max(dim=1).values
            margin = source_logits - best_other

            total += images.size(0)
            correct += int((preds == source_cls).sum().item())
            source_logit_sum += float(source_logits.sum().item())
            source_prob_sum += float(probs[:, source_cls].sum().item())
            source_margin_sum += float(margin.sum().item())

    handle.remove()

    tokens = max(1, captured["token_count"])
    return {
        "num_images": int(total),
        "source_top1_acc": float(correct / max(1, total)),
        "avg_source_logit": float(source_logit_sum / max(1, total)),
        "avg_source_prob": float(source_prob_sum / max(1, total)),
        "avg_source_margin": float(source_margin_sum / max(1, total)),
        "recon_mse_raw": float(captured["mse_raw_sum"] / tokens),
        "recon_mse_norm": float(captured["mse_norm_sum"] / tokens),
        "recon_cosine_norm": float(captured["cos_norm_sum"] / tokens),
    }


def build_output_dir(results_dir: str, stage_idx: int, k_tag: str, run_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    leaf = run_name if run_name else f"stage{stage_idx}_{k_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(results_dir, f"stage{stage_idx}", k_tag, leaf)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    args = parse_args()

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

    dataset = FileListDataset(image_paths, transform=get_transform())
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
    expected_din = STAGE_DIN_MAP[args.stage]
    if int(sae_cfg["d_in"]) != expected_din:
        raise ValueError(
            f"Stage/SAE mismatch: stage{args.stage} expects d_in={expected_din}, SAE has d_in={sae_cfg['d_in']}"
        )

    k_tag = infer_k_tag(int(sae_cfg["k"]), args.sae_ckpt)
    out_dir = build_output_dir(args.results_dir, args.stage, k_tag, args.run_name)
    print(f"Output dir: {out_dir}")

    baseline_original = evaluate_original(loader, model, args.source_cls, device)

    modes = ["raw", "normalized"] if args.recon_mode == "both" else [args.recon_mode]
    recon_results = {}
    for mode in modes:
        recon_results[mode] = evaluate_reconstruction(
            loader=loader,
            model=model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            stage_idx=args.stage,
            source_cls=args.source_cls,
            device=device,
            recon_mode=mode,
        )

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
            "recon_mode": args.recon_mode,
            "sae_config": {
                "d_in": int(sae_cfg["d_in"]),
                "d_lat": int(sae_cfg["d_lat"]),
                "k": int(sae_cfg["k"]),
                "k_tag": k_tag,
            },
            "preprocess": "Resize(256, bicubic) -> CenterCrop(224) -> ToTensor",
        },
        "baseline_original": baseline_original,
        "reconstruction": recon_results,
    }

    out_json = os.path.join(out_dir, "reconstruct_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done.")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
