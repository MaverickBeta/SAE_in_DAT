#!/usr/bin/env python3
"""
Evaluate SAE steering on val/adv samples across 100 classes.

Behavior:
  --sae-ckpt "" (empty)     : baseline, no SAE mounted
  --sae-ckpt <path> --alpha 0 : SAE encode-decode only, no steering
  --sae-ckpt <path> --alpha X : SAE steering with z -= alpha * delta
                                where delta = adv_mean - clean_mean
                                (positive alpha moves z toward clean_mean)

All entries in selected_entries.json are used for steering.

Output: DAT/SAE/steering/results/steering_eval_alpha{alpha}.json
"""
 
import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        paths = sorted(Path(folder_path).glob("*"))
        self.image_paths = [
            str(p) for p in paths if p.is_file() and p.suffix in IMAGE_EXTS
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(path).name


# ------------------------------------------------------------------
# Model / SAE builders
# ------------------------------------------------------------------
def build_model(device: torch.device, ckpt_path: str):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, ckpt_path, weights_only=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
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
    resolved_cfg.update({"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)})
    return sae, norm_mean, norm_std, resolved_cfg


# ------------------------------------------------------------------
# Hook factory
# ------------------------------------------------------------------
def make_sae_only_hook(sae_model, norm_mean, norm_std):
    """SAE encode-decode only, no steering."""

    def hook_fn(module, input, output):
        bsz, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std
        z = sae_model.encode(flat_norm)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return recon

    return hook_fn


def make_steering_hook(sae_model, norm_mean, norm_std, steering_entries, alpha: float):
    """
    steering_entries: list of dicts with keys:
        feature_idx, token_idx, clean_mean, adv_mean
    Operation: z[token, feature] -= alpha * (adv_mean - clean_mean)
    Positive alpha moves activation toward clean_mean.
    """

    def hook_fn(module, input, output):
        bsz, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - norm_mean) / norm_std

        z = sae_model.encode(flat_norm)
        z_spatial = z.reshape(bsz, h, w, -1)

        for entry in steering_entries:
            feat_idx = entry["feature_idx"]
            tok_idx = entry["token_idx"]
            clean_mean = entry["clean_mean"]
            adv_mean = entry["adv_mean"]
            delta = adv_mean - clean_mean
            row = tok_idx // w
            col = tok_idx % w
            z_spatial[:, row, col, feat_idx] -= alpha * delta

        z = z_spatial.reshape(bsz * h * w, -1)
        recon_norm = (z @ sae_model.W_dec) + sae_model.b_dec
        recon_flat = recon_norm * norm_std + norm_mean
        recon = recon_flat.reshape(bsz, h, w, c).permute(0, 3, 1, 2)
        return recon

    return hook_fn


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_class(model, dataloader, class_idx, device):
    correct = 0
    total = 0
    for images, _ in dataloader:
        images = images.to(device)
        labels = torch.full(
            (images.size(0),), class_idx, dtype=torch.long, device=device
        )
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total if total > 0 else 0.0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE steering on val/adv samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adv-root",
        type=str,
        default=str(REPO_ROOT / "SAE" / "steering_results"),
        help="Root directory containing per-class val/adv folders",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "model_bestfid.pth"),
    )
    parser.add_argument(
        "--sae-ckpt",
        type=str,
        default=str(
            REPO_ROOT
            / "SAE"
            / "project"
            / "checkpoints"
            / "stage3"
            / "k256_exp8"
            / "sae_stage3_din1536_exp8_k256_step_50000.pt"
        ),
        help="SAE checkpoint path. Leave empty or 'none' to run baseline without SAE.",
    )
    parser.add_argument(
        "--selected-entries",
        type=str,
        default=str(Path(__file__).resolve().parent / "selected_entries.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Steering strength. 0 = SAE encode-decode only. "
             "Positive value subtracts alpha*(adv_mean-clean_mean) from z.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3])
    return parser.parse_args()


def read_class_idx(class_dir: Path) -> int:
    labels_path = class_dir / "labels.txt"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            pass
    return 0


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine whether to use SAE
    sae_ckpt = args.sae_ckpt.strip()
    use_sae = sae_ckpt.lower() not in ("", "none", "null", "false")
    print(f"SAE enabled: {use_sae}")
    if use_sae:
        print(f"SAE checkpoint: {sae_ckpt}")
        print(f"Alpha (steering strength): {args.alpha}")

    # Load selected entries if SAE is used
    steering_entries = []
    if use_sae:
        with open(args.selected_entries, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
        steering_entries = selected_data.get("selected_entries", [])
        print(f"Loaded {len(steering_entries)} steering entries")

    # Scan classes
    adv_root = Path(args.adv_root)
    class_dirs = sorted(
        [d for d in adv_root.iterdir() if d.is_dir() and d.name != "sae_latent"]
    )
    print(f"Found {len(class_dirs)} classes")

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Build model (and SAE if needed)
    # ================================================================
    model = build_model(device, args.base_ckpt)
    handle = None
    sae = None

    if use_sae:
        sae, norm_mean, norm_std, sae_cfg = build_sae(device, sae_ckpt)
        expected_din = {0: 192, 1: 384, 2: 768, 3: 1536}[args.stage]
        if int(sae_cfg["d_in"]) != expected_din:
            raise ValueError(
                f"Stage/SAE mismatch: stage{args.stage} expects d_in={expected_din}, "
                f"SAE has d_in={sae_cfg['d_in']}"
            )

        if args.alpha == 0.0:
            hook_fn = make_sae_only_hook(sae, norm_mean, norm_std)
            print(f"SAE encode-decode only hook registered on stages[{args.stage}]")
        else:
            hook_fn = make_steering_hook(
                sae, norm_mean, norm_std, steering_entries, alpha=args.alpha
            )
            print(f"SAE steering hook registered on stages[{args.stage}] (alpha={args.alpha})")

        handle = model.stages[args.stage].register_forward_hook(hook_fn)

    # ================================================================
    # Evaluate
    # ================================================================
    run_tag = "baseline" if not use_sae else f"alpha{args.alpha}"
    print("\n" + "=" * 60)
    print(f"EVALUATING: {run_tag}")
    print("=" * 60)

    per_class = {}
    total_correct = 0
    total_samples = 0

    for class_dir in tqdm(class_dirs, desc=run_tag):
        wnid = class_dir.name
        val_adv_dir = class_dir / "val" / "adv"
        if not val_adv_dir.is_dir():
            continue

        class_idx = read_class_idx(class_dir)
        dataset = ImageFolderDataset(str(val_adv_dir), transform=transform)
        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        acc = evaluate_class(model, loader, class_idx, device)

        per_class[wnid] = {
            "class_idx": class_idx,
            "robust_acc": float(acc),
            "n_samples": len(dataset),
        }
        total_correct += int(acc * len(dataset))
        total_samples += len(dataset)

    if handle is not None:
        handle.remove()

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    print(
        f"\nOverall robust_acc: {overall*100:.2f}% "
        f"({total_correct}/{total_samples})"
    )

    del model
    if sae is not None:
        del sae
    torch.cuda.empty_cache()

    # ================================================================
    # Save results
    # ================================================================
    result = {
        "config": {
            "base_ckpt": args.base_ckpt,
            "sae_ckpt": sae_ckpt if use_sae else None,
            "selected_entries": args.selected_entries,
            "stage": args.stage,
            "alpha": args.alpha if use_sae else None,
        },
        "per_class": per_class,
        "summary": {
            "n_classes": len(class_dirs),
            "n_total_samples": total_samples,
            "robust_acc": float(overall),
        },
    }

    output_path = output_dir / f"steering_eval_{run_tag}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
