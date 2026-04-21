import csv
import glob
import json
import os
import sys
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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

TARGET_CLS = int(os.getenv("TARGET_CLS", "62"))
SOURCE_CLS = int(os.getenv("SOURCE_CLS", "150"))
SOURCE_WNID = os.getenv("SOURCE_WNID", "n02077923")

BASE_CKPT_PATH = os.getenv("BASE_CKPT_PATH", "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
SAE_CKPT_PATH = os.getenv(
    "SAE_CKPT_PATH",
    "/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
)
PGD_ADV_DIR = os.getenv(
    "PGD_ADV_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_tar/pgd_tar_sl2py"
)
PGD_EVAL_CSV = os.getenv("PGD_EVAL_CSV", os.path.join(PGD_ADV_DIR, "eval_results.csv"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/Data_share/hongyi/DAT/SAE/results")
IMAGENET_VAL_DIR = os.getenv("IMAGENET_VAL_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val")
CLEAN_SOURCE_DIR = os.getenv("CLEAN_SOURCE_DIR", os.path.join(IMAGENET_VAL_DIR, SOURCE_WNID))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
TOPK_FEATURES = int(os.getenv("TOPK_FEATURES", "300"))
ACT_THRESHOLD = float(os.getenv("ACT_THRESHOLD", "0.1"))
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "any").lower()  # any | all | success_source

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def validate_inputs():
    if not os.path.exists(BASE_CKPT_PATH):
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_CKPT_PATH}")
    if not os.path.exists(SAE_CKPT_PATH):
        raise FileNotFoundError(f"SAE checkpoint not found: {SAE_CKPT_PATH}")
    if not os.path.isdir(PGD_ADV_DIR):
        raise FileNotFoundError(f"PGD directory not found: {PGD_ADV_DIR}")
    if not os.path.exists(PGD_EVAL_CSV):
        raise FileNotFoundError(f"PGD eval csv not found: {PGD_EVAL_CSV}")
    if not os.path.isdir(CLEAN_SOURCE_DIR):
        raise FileNotFoundError(f"Clean source directory not found: {CLEAN_SOURCE_DIR}")


def list_images_in_dir(folder: str) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def resolve_image_path(adv_dir: str, filename: str) -> str:
    cand = os.path.join(adv_dir, filename)
    if os.path.exists(cand):
        return cand

    stem, ext = os.path.splitext(filename)
    if stem.endswith("_succ"):
        stem = stem[:-5]
    if stem.endswith("_fail"):
        stem = stem[:-5]

    fallback = os.path.join(adv_dir, stem + ext)
    if os.path.exists(fallback):
        return fallback

    # Last chance: search by stem prefix with known image extensions.
    for p in glob.glob(os.path.join(adv_dir, stem + "*")):
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS):
            return p

    raise FileNotFoundError(f"Cannot resolve image path for csv entry: {filename}")


def split_pgd_by_outcome(eval_csv_path: str, adv_dir: str) -> Dict[str, List[str]]:
    groups = {
        "targeted_success": [],
        "escape_non_target": [],
        "still_source": [],
    }

    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = int(row["pred"])
            img_path = resolve_image_path(adv_dir, row["filename"])

            if pred == TARGET_CLS:
                groups["targeted_success"].append(img_path)
            elif pred != SOURCE_CLS:
                groups["escape_non_target"].append(img_path)
            else:
                groups["still_source"].append(img_path)

    return groups


def build_model(device: torch.device):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, BASE_CKPT_PATH)
    model = model.to(device)
    model.eval()
    return model


def build_sae(device: torch.device):
    ckpt = torch.load(SAE_CKPT_PATH, map_location=device)
    config = ckpt["config"]

    sae = TopKAutoencoder(d_in=config["d_in"], d_lat=config["d_lat"], k=config["k"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    return sae, norm_mean, norm_std


def extract_group_mean_activations(
    image_paths: List[str],
    base_model,
    sae_model,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    if len(image_paths) == 0:
        return None

    dataset = FileListDataset(image_paths, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    captured = {}

    def hook_fn(_, __, output):
        captured["stage4"] = output.detach()

    handle = base_model.stages[3].register_forward_hook(hook_fn)

    total_acts = None
    total_tokens = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extract SAE activations", leave=False):
            images = images.to(device)
            _ = base_model(images)

            stage4 = captured["stage4"]
            _, c, _, _ = stage4.shape
            flat = stage4.permute(0, 2, 3, 1).reshape(-1, c)
            flat = (flat - norm_mean) / norm_std
            z = sae_model.encode(flat)

            zsum = z.sum(dim=0).detach().cpu()
            total_acts = zsum if total_acts is None else total_acts + zsum
            total_tokens += z.shape[0]

    handle.remove()

    if total_tokens == 0:
        return None
    return (total_acts / total_tokens).numpy()


def select_filtered_feature_indices(stack: np.ndarray, act_threshold: float, threshold_mode: str) -> np.ndarray:
    if threshold_mode == "all":
        mask = np.all(stack > act_threshold, axis=0)
    elif threshold_mode == "success_source":
        # Require both targeted-success and still-source means to be above threshold.
        mask = (stack[0] > act_threshold) & (stack[2] > act_threshold)
    else:
        # Default: keep dimensions where any outcome group is active enough.
        mask = np.max(stack, axis=0) > act_threshold

    idx = np.where(mask)[0]
    if idx.shape[0] == 0:
        raise RuntimeError(
            f"No latent dimensions pass threshold filtering: ACT_THRESHOLD={act_threshold}, THRESHOLD_MODE={threshold_mode}"
        )
    return idx


def plot_four_groups(group_means: Dict[str, np.ndarray], out_png: str):
    names = ["targeted_success", "escape_non_target", "still_source", "clean_source_original"]
    labels = [
        "1) Targeted Success (pred=target)",
        "2) Escape Non-Target (pred!=source and pred!=target)",
        "3) Still Source (pred=source)",
        "4) Clean Source Original",
    ]
    colors = ["#e67e22", "#16a085", "#2c3e50", "#5dade2"]

    for n in names:
        if group_means.get(n) is None:
            raise RuntimeError(f"Group {n} has 0 images, cannot draw 4-group comparison.")

    stack = np.stack([group_means[n] for n in names], axis=0)
    filtered_idx = select_filtered_feature_indices(stack, ACT_THRESHOLD, THRESHOLD_MODE)
    filtered_stack = stack[:, filtered_idx]
    k = min(TOPK_FEATURES, filtered_stack.shape[1])
    importance = np.max(filtered_stack, axis=0)
    top_local_idx = np.argsort(importance)[::-1][:k]
    top_idx = filtered_idx[top_local_idx]

    # Sort selected dimensions by how much they favor targeted success over still-source.
    order = np.argsort((stack[0] - stack[2])[top_idx])[::-1]
    feat_idx = top_idx[order]

    vals = stack[:, feat_idx]
    x = np.arange(k)
    w = 0.22

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(x - 1.5 * w, vals[0], width=w, color=colors[0], label=labels[0], zorder=3)
    ax.bar(x - 0.5 * w, vals[1], width=w, color=colors[1], label=labels[1], zorder=3)
    ax.bar(x + 0.5 * w, vals[2], width=w, color=colors[2], label=labels[2], zorder=3)
    ax.bar(x + 1.5 * w, vals[3], width=w, color=colors[3], label=labels[3], zorder=3)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title(
        f"SAE Hidden-Dimension Activation Distribution on PGD Outcomes (4 Groups)\n"
        f"Filtered: ACT_THRESHOLD>{ACT_THRESHOLD}, THRESHOLD_MODE={THRESHOLD_MODE}",
        fontsize=16,
    )
    ax.set_xlabel("SAE latent dimension (top-K selected, sorted by success-vs-source gap)", fontsize=12)
    ax.set_ylabel("Mean SAE activation", fontsize=12)

    tick_n = min(80, k)
    tick_pos = np.linspace(0, k - 1, num=tick_n, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(feat_idx[tick_pos], rotation=90, fontsize=8)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def export_top50_diff_csv(group_means: Dict[str, np.ndarray], out_csv: str):
    names = ["targeted_success", "escape_non_target", "still_source", "clean_source_original"]
    for n in names:
        if group_means.get(n) is None:
            raise RuntimeError(f"Group {n} has 0 images, cannot export top diff csv.")

    success = group_means["targeted_success"]
    escape = group_means["escape_non_target"]
    source = group_means["still_source"]
    clean_source = group_means["clean_source_original"]

    stack = np.stack([success, escape, source, clean_source], axis=0)
    filtered_idx = select_filtered_feature_indices(stack, ACT_THRESHOLD, THRESHOLD_MODE)

    diff = success - source
    diff_filtered = diff[filtered_idx]
    topk = min(50, diff_filtered.shape[0])
    local_top = np.argsort(diff_filtered)[::-1][:topk]
    top_idx = filtered_idx[local_top]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "latent_dim",
                "diff_success_minus_source",
                "success_mean_activation",
                "escape_non_target_mean_activation",
                "still_source_mean_activation",
                "clean_source_original_mean_activation",
            ],
        )
        writer.writeheader()
        for rank, dim in enumerate(top_idx, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "latent_dim": int(dim),
                    "diff_success_minus_source": float(diff[dim]),
                    "success_mean_activation": float(success[dim]),
                    "escape_non_target_mean_activation": float(escape[dim]),
                    "still_source_mean_activation": float(source[dim]),
                    "clean_source_original_mean_activation": float(clean_source[dim]),
                }
            )


def export_filtered_dims_csv(group_means: Dict[str, np.ndarray], out_csv: str):
    names = ["targeted_success", "escape_non_target", "still_source", "clean_source_original"]
    for n in names:
        if group_means.get(n) is None:
            raise RuntimeError(f"Group {n} has 0 images, cannot export filtered dims csv.")

    success = group_means["targeted_success"]
    escape = group_means["escape_non_target"]
    source = group_means["still_source"]
    clean_source = group_means["clean_source_original"]
    stack = np.stack([success, escape, source, clean_source], axis=0)

    filtered_idx = select_filtered_feature_indices(stack, ACT_THRESHOLD, THRESHOLD_MODE)
    diff = success - source
    order = np.argsort(diff[filtered_idx])[::-1]
    final_idx = filtered_idx[order]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_dim",
                "diff_success_minus_source",
                "success_mean_activation",
                "escape_non_target_mean_activation",
                "still_source_mean_activation",
                "clean_source_original_mean_activation",
            ],
        )
        writer.writeheader()
        for dim in final_idx:
            writer.writerow(
                {
                    "latent_dim": int(dim),
                    "diff_success_minus_source": float(diff[dim]),
                    "success_mean_activation": float(success[dim]),
                    "escape_non_target_mean_activation": float(escape[dim]),
                    "still_source_mean_activation": float(source[dim]),
                    "clean_source_original_mean_activation": float(clean_source[dim]),
                }
            )


def main():
    validate_inputs()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PGD_ADV_DIR={PGD_ADV_DIR}")
    print(f"PGD_EVAL_CSV={PGD_EVAL_CSV}")
    print(f"CLEAN_SOURCE_DIR={CLEAN_SOURCE_DIR}")

    groups = split_pgd_by_outcome(PGD_EVAL_CSV, PGD_ADV_DIR)
    groups["clean_source_original"] = list_images_in_dir(CLEAN_SOURCE_DIR)
    counts = {k: len(v) for k, v in groups.items()}
    print(f"Group counts: {counts}")

    base_model = build_model(device)
    sae_model, norm_mean, norm_std = build_sae(device)

    group_means = {}
    for name, paths in groups.items():
        print(f"Extracting group: {name} | images={len(paths)}")
        group_means[name] = extract_group_mean_activations(
            image_paths=paths,
            base_model=base_model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
        )

    out_png = os.path.join(RESULTS_DIR, "sae_pgd_four_groups_distribution.png")
    plot_four_groups(group_means, out_png)

    out_top50_csv = os.path.join(RESULTS_DIR, "sae_pgd_top50_success_minus_source.csv")
    export_top50_diff_csv(group_means, out_top50_csv)

    out_filtered_csv = os.path.join(RESULTS_DIR, "sae_pgd_filtered_dims_success_minus_source.csv")
    export_filtered_dims_csv(group_means, out_filtered_csv)

    out_json = os.path.join(RESULTS_DIR, "sae_pgd_three_outcomes_counts.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_class": SOURCE_CLS,
                "target_class": TARGET_CLS,
                "pgd_adv_dir": PGD_ADV_DIR,
                "pgd_eval_csv": PGD_EVAL_CSV,
                "clean_source_dir": CLEAN_SOURCE_DIR,
                "counts": counts,
            },
            f,
            indent=2,
        )

    print("Done.")
    print(f"Figure: {out_png}")
    print(f"Top50 diff csv: {out_top50_csv}")
    print(f"Filtered dims csv: {out_filtered_csv}")
    print(f"Counts: {out_json}")


if __name__ == "__main__":
    main()
