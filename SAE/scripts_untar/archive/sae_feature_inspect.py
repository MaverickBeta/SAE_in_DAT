import csv
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

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

SOURCE_CLS = int(os.getenv("SOURCE_CLS", "150"))
SOURCE_WNID = os.getenv("SOURCE_WNID", "n02077923")

BASE_CKPT_PATH = os.getenv("BASE_CKPT_PATH", "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
SAE_CKPT_PATH = os.getenv(
    "SAE_CKPT_PATH",
    "/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
)

UNTAR_ROOT = os.getenv("UNTAR_ROOT", "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_untar")
ATTACK_NAME = os.getenv("ATTACK_NAME", "").strip()
ATTACK_DIR = os.getenv("ATTACK_DIR", "").strip()

RESULTS_DIR = os.getenv("RESULTS_DIR", "/Data_share/hongyi/DAT/SAE/results_untar")
IMAGENET_VAL_DIR = os.getenv("IMAGENET_VAL_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val")
CLEAN_SOURCE_DIR = os.getenv("CLEAN_SOURCE_DIR", os.path.join(IMAGENET_VAL_DIR, SOURCE_WNID))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
TOPK_FEATURES = int(os.getenv("TOPK_FEATURES", "300"))
ACT_THRESHOLD = float(os.getenv("ACT_THRESHOLD", "0.4"))
THRESHOLD_MODE = os.getenv("THRESHOLD_MODE", "any").lower()  # any | all

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
    if not ATTACK_DIR:
        raise ValueError(
            "Please set ATTACK_DIR or ATTACK_NAME explicitly. "
            "Example: ATTACK_DIR=/.../apgd_ce_untar_sl2py python sae_feature_inspect.py"
        )

    if not os.path.isdir(ATTACK_DIR):
        raise FileNotFoundError(f"Attack directory not found: {ATTACK_DIR}")
    eval_csv = os.path.join(ATTACK_DIR, "eval_results.csv")
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Missing eval_results.csv: {eval_csv}")


def list_images_in_dir(folder: str) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def resolve_image_path(attack_dir: str, filename: str) -> str:
    cand = os.path.join(attack_dir, filename)
    if os.path.exists(cand):
        return cand

    stem, ext = os.path.splitext(filename)
    if stem.endswith("_succ"):
        stem = stem[:-5]
    if stem.endswith("_fail"):
        stem = stem[:-5]

    fallback = os.path.join(attack_dir, stem + ext)
    if os.path.exists(fallback):
        return fallback

    for p in glob.glob(os.path.join(attack_dir, stem + "*")):
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS):
            return p

    raise FileNotFoundError(f"Cannot resolve image path for csv entry: {filename}")


def split_attack_by_success(eval_csv_path: str, attack_dir: str) -> Tuple[Dict[str, List[str]], Counter, int]:
    groups = {
        "attack_success_escape": [],
        "attack_fail_still_source": [],
    }
    success_pred_counter = Counter()
    duplicate_rows = 0
    seen_filenames = set()

    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            if filename in seen_filenames:
                duplicate_rows += 1
                continue
            seen_filenames.add(filename)

            pred = int(row["pred"])
            img_path = resolve_image_path(attack_dir, filename)

            if pred != SOURCE_CLS:
                groups["attack_success_escape"].append(img_path)
                success_pred_counter[pred] += 1
            else:
                groups["attack_fail_still_source"].append(img_path)

    return groups, success_pred_counter, duplicate_rows


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
    return sae, norm_mean, norm_std, int(config["d_lat"])


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


def fill_missing_groups(group_means: Dict[str, np.ndarray], d_lat: int) -> Dict[str, np.ndarray]:
    out = dict(group_means)
    for name in ["attack_success_escape", "attack_fail_still_source"]:
        if out.get(name) is None:
            out[name] = np.zeros((d_lat,), dtype=np.float32)
    return out


def select_filtered_feature_indices(stack: np.ndarray, act_threshold: float, threshold_mode: str) -> np.ndarray:
    if threshold_mode == "all":
        mask = np.all(stack >= act_threshold, axis=0)
    else:
        mask = np.max(stack, axis=0) >= act_threshold

    idx = np.where(mask)[0]
    if idx.shape[0] == 0:
        # Backoff to top activation dims to guarantee a plot.
        k = min(TOPK_FEATURES, stack.shape[1])
        fallback = np.argsort(np.max(stack, axis=0))[::-1][:k]
        return np.sort(fallback)
    return idx


def plot_two_groups(group_means: Dict[str, np.ndarray], out_png: str, title_prefix: str):
    names = ["attack_success_escape", "attack_fail_still_source"]
    labels = [
        "1) Attack Success (escape from source)",
        "2) Attack Fail (still source)",
    ]
    colors = ["#e67e22", "#2c3e50"]

    stack = np.stack([group_means[n] for n in names], axis=0)
    filtered_idx = select_filtered_feature_indices(stack, ACT_THRESHOLD, THRESHOLD_MODE)
    filtered_stack = stack[:, filtered_idx]
    k = min(TOPK_FEATURES, filtered_stack.shape[1])

    # Sort x-axis by success-vs-fail activation gap.
    diff = filtered_stack[0] - filtered_stack[1]
    local_order = np.argsort(diff)[::-1][:k]
    feat_idx = filtered_idx[local_order]

    vals = stack[:, feat_idx]
    x = np.arange(k)
    w = 0.4

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(x - 0.5 * w, vals[0], width=w, color=colors[0], label=labels[0], zorder=3)
    ax.bar(x + 0.5 * w, vals[1], width=w, color=colors[1], label=labels[1], zorder=3)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title(
        f"{title_prefix} | SAE Mean Activation Distribution (Untargeted)\n"
        f"Filtered: activation >= {ACT_THRESHOLD}, mode={THRESHOLD_MODE}",
        fontsize=16,
    )
    ax.set_xlabel("Feature index (sorted by success - fail mean activation)", fontsize=12)
    ax.set_ylabel("Mean SAE Activation", fontsize=12)

    tick_n = min(80, k)
    tick_pos = np.linspace(0, k - 1, num=tick_n, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(feat_idx[tick_pos], rotation=90, fontsize=8)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    global ATTACK_DIR

    if not ATTACK_DIR:
        if ATTACK_NAME:
            ATTACK_DIR = os.path.join(UNTAR_ROOT, ATTACK_NAME)

    validate_inputs()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    attack_name = os.path.basename(os.path.normpath(ATTACK_DIR))
    eval_csv = os.path.join(ATTACK_DIR, "eval_results.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"ATTACK_DIR={ATTACK_DIR}")
    print(f"EVAL_CSV={eval_csv}")
    print(f"ACT_THRESHOLD={ACT_THRESHOLD}, THRESHOLD_MODE={THRESHOLD_MODE}")

    base_model = build_model(device)
    sae_model, norm_mean, norm_std, d_lat = build_sae(device)

    groups, success_pred_counter, duplicate_rows = split_attack_by_success(eval_csv, ATTACK_DIR)
    counts = {k: len(v) for k, v in groups.items()}

    print(f"Group counts: {counts}")
    if duplicate_rows > 0:
        print(f"Detected duplicate rows in eval csv and ignored: {duplicate_rows}")

    raw_means = {}
    for name, paths in groups.items():
        print(f"Extracting group: {name} | images={len(paths)}")
        raw_means[name] = extract_group_mean_activations(
            image_paths=paths,
            base_model=base_model,
            sae_model=sae_model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
        )

    group_means = fill_missing_groups(raw_means, d_lat)

    attack_out_dir = os.path.join(RESULTS_DIR, attack_name)
    os.makedirs(attack_out_dir, exist_ok=True)

    out_png = os.path.join(attack_out_dir, "sae_success_vs_fail_distribution.png")
    plot_two_groups(group_means, out_png, title_prefix=attack_name)

    success_count = counts["attack_success_escape"]
    top3 = success_pred_counter.most_common(3)
    top3_payload = []
    for pred_cls, n in top3:
        ratio = (n / success_count) if success_count > 0 else 0.0
        top3_payload.append({"pred_class": int(pred_cls), "count": int(n), "ratio_of_success": float(ratio)})

    out_json = os.path.join(attack_out_dir, "sae_groups_counts.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "attack_name": attack_name,
                "source_class": SOURCE_CLS,
                "attack_dir": ATTACK_DIR,
                "eval_csv": eval_csv,
                "counts": {
                    "attack_success_escape": counts["attack_success_escape"],
                    "attack_fail_still_source": counts["attack_fail_still_source"],
                },
                "success_rate": (
                    counts["attack_success_escape"]
                    / max(1, counts["attack_success_escape"] + counts["attack_fail_still_source"])
                ),
                "duplicate_rows_ignored": duplicate_rows,
                "top3_pred_classes_within_success": top3_payload,
                "act_threshold": ACT_THRESHOLD,
                "threshold_mode": THRESHOLD_MODE,
                "figure": out_png,
            },
            f,
            indent=2,
        )

    print("Done.")
    print(f"Figure: {out_png}")
    print(f"Counts: {out_json}")


if __name__ == "__main__":
    main()
