import csv
import glob
import json
import os
import sys
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder

TARGET_CLS = int(os.getenv("TARGET_CLS", "62"))
SOURCE_CLS = int(os.getenv("SOURCE_CLS", "150"))

BASE_CKPT_PATH = os.getenv("BASE_CKPT_PATH", "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
SAE_CKPT_PATH = os.getenv(
    "SAE_CKPT_PATH", "/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt"
)
PGD_ADV_DIR = os.getenv(
    "PGD_ADV_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_tar/pgd_tar_sl2py"
)
PGD_EVAL_CSV = os.getenv("PGD_EVAL_CSV", os.path.join(PGD_ADV_DIR, "eval_results.csv"))

RESULTS_DIR = os.getenv("RESULTS_DIR", "/Data_share/hongyi/DAT/SAE/results/steering")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

ZERO_FEATURES = [int(x.strip()) for x in os.getenv("ZERO_FEATURES", "19978,35639,30879").split(",") if x.strip()]
BOOST_FEATURES = [int(x.strip()) for x in os.getenv("BOOST_FEATURES", "27474").split(",") if x.strip()]
BOOST_VALUE = float(os.getenv("BOOST_VALUE", "10"))

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

    for p in glob.glob(os.path.join(adv_dir, stem + "*")):
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS):
            return p

    raise FileNotFoundError(f"Cannot resolve image path for csv entry: {filename}")


def load_targeted_success_images(eval_csv_path: str, adv_dir: str) -> List[str]:
    out = []
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = int(row["pred"])
            if pred == TARGET_CLS:
                out.append(resolve_image_path(adv_dir, row["filename"]))
    return out


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
    cfg = ckpt["config"]

    sae = TopKAutoencoder(d_in=cfg["d_in"], d_lat=cfg["d_lat"], k=cfg["k"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    d_lat = int(cfg["d_lat"])
    return sae, norm_mean, norm_std, d_lat


class SAEActivationSteeringHook:
    def __init__(
        self,
        sae_model,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        zero_features: List[int],
        boost_features: List[int],
        boost_value: float,
    ):
        self.sae_model = sae_model
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.zero_features = zero_features
        self.boost_features = boost_features
        self.boost_value = boost_value

    def __call__(self, _module, _input, output):
        b, c, h, w = output.shape
        flat = output.permute(0, 2, 3, 1).reshape(-1, c)
        flat_norm = (flat - self.norm_mean) / self.norm_std

        z = self.sae_model.encode(flat_norm)
        if self.zero_features:
            z[:, self.zero_features] = 0.0
        if self.boost_features:
            z[:, self.boost_features] = self.boost_value

        rec_norm = (z @ self.sae_model.W_dec) + self.sae_model.b_dec
        rec = rec_norm * self.norm_std + self.norm_mean
        rec = rec.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return rec


def run_eval(
    model,
    loader,
    device: torch.device,
    hook: SAEActivationSteeringHook = None,
):
    rows = []
    target_hits = 0

    handle = None
    if hook is not None:
        handle = model.stages[3].register_forward_hook(hook)

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
            top1 = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            tconf = probs[:, TARGET_CLS]
            sconf = probs[:, SOURCE_CLS]

            for i in range(images.size(0)):
                pred = int(preds[i].item())
                hit = int(pred == TARGET_CLS)
                target_hits += hit
                rows.append(
                    {
                        "filename": filenames[i],
                        "pred": pred,
                        "top1_conf": float(top1[i].item()),
                        "target_conf": float(tconf[i].item()),
                        "source_conf": float(sconf[i].item()),
                        "target_hit": hit,
                    }
                )

    if handle is not None:
        handle.remove()

    return rows, target_hits


def merge_rows(baseline_rows: List[dict], steered_rows: List[dict]) -> List[dict]:
    base_map = {r["filename"]: r for r in baseline_rows}
    steer_map = {r["filename"]: r for r in steered_rows}

    merged = []
    for fn in sorted(base_map.keys()):
        b = base_map[fn]
        s = steer_map[fn]
        merged.append(
            {
                "filename": fn,
                "baseline_pred": b["pred"],
                "baseline_top1_conf": b["top1_conf"],
                "baseline_target_conf": b["target_conf"],
                "baseline_source_conf": b["source_conf"],
                "steered_pred": s["pred"],
                "steered_top1_conf": s["top1_conf"],
                "steered_target_conf": s["target_conf"],
                "steered_source_conf": s["source_conf"],
                "changed_pred": int(b["pred"] != s["pred"]),
                "target_hit_baseline": b["target_hit"],
                "target_hit_steered": s["target_hit"],
            }
        )
    return merged


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    validate_inputs()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"ZERO_FEATURES={ZERO_FEATURES}")
    print(f"BOOST_FEATURES={BOOST_FEATURES}")
    print(f"BOOST_VALUE={BOOST_VALUE}")

    image_paths = load_targeted_success_images(PGD_EVAL_CSV, PGD_ADV_DIR)
    if len(image_paths) == 0:
        raise RuntimeError("No targeted-success images found in PGD eval csv.")

    print(f"Targeted-success images: {len(image_paths)}")

    dataset = FileListDataset(image_paths, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    model = build_model(device)
    sae, norm_mean, norm_std, d_lat = build_sae(device)

    invalid_zero = [i for i in ZERO_FEATURES if i < 0 or i >= d_lat]
    invalid_boost = [i for i in BOOST_FEATURES if i < 0 or i >= d_lat]
    if invalid_zero or invalid_boost:
        raise ValueError(
            f"Feature index out of range for d_lat={d_lat}. invalid_zero={invalid_zero}, invalid_boost={invalid_boost}"
        )

    print("Running baseline inference...")
    baseline_rows, base_hits = run_eval(model, loader, device, hook=None)

    print("Running steered inference...")
    hook = SAEActivationSteeringHook(
        sae_model=sae,
        norm_mean=norm_mean,
        norm_std=norm_std,
        zero_features=ZERO_FEATURES,
        boost_features=BOOST_FEATURES,
        boost_value=BOOST_VALUE,
    )
    steered_rows, steered_hits = run_eval(model, loader, device, hook=hook)

    merged = merge_rows(baseline_rows, steered_rows)

    csv_path = os.path.join(RESULTS_DIR, "ad_steering_targeted_success_results.csv")
    write_csv(csv_path, merged)

    total = len(merged)
    changed = sum(r["changed_pred"] for r in merged)
    summary = {
        "total_images": total,
        "target_class": TARGET_CLS,
        "source_class": SOURCE_CLS,
        "baseline_target_hits": base_hits,
        "baseline_target_hit_rate": (base_hits / total) if total else 0.0,
        "steered_target_hits": steered_hits,
        "steered_target_hit_rate": (steered_hits / total) if total else 0.0,
        "changed_pred_count": changed,
        "changed_pred_rate": (changed / total) if total else 0.0,
        "zero_features": ZERO_FEATURES,
        "boost_features": BOOST_FEATURES,
        "boost_value": BOOST_VALUE,
        "pgd_adv_dir": PGD_ADV_DIR,
        "pgd_eval_csv": PGD_EVAL_CSV,
        "base_ckpt": BASE_CKPT_PATH,
        "sae_ckpt": SAE_CKPT_PATH,
    }

    summary_path = os.path.join(RESULTS_DIR, "ad_steering_targeted_success_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
