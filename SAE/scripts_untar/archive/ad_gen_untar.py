import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torchattacks
except ImportError:
    print("Please install torchattacks first: pip install torchattacks")
    sys.exit(1)

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model

SOURCE_CLS = int(os.getenv("SOURCE_CLS", "150"))
SOURCE_WNID = os.getenv("SOURCE_WNID", "n02077923")

IMAGENET_VAL_DIR = os.getenv("IMAGENET_VAL_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val")
SOURCE_DIR = os.getenv("SOURCE_DIR", os.path.join(IMAGENET_VAL_DIR, SOURCE_WNID))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_untar")
ATTACK_FILTER = os.getenv("ATTACK_FILTER", "")
FORCE_REGENERATE = os.getenv("FORCE_REGENERATE", "0") == "1"

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
ATTACK_MAX_CHUNK = int(os.getenv("ATTACK_MAX_CHUNK", "0"))

THREAT_MODEL = os.getenv("THREAT_MODEL", "L2")  # L2 | Linf
EPS = float(os.getenv("EPS", "3.0"))

APGD_STEPS = int(os.getenv("APGD_STEPS", "100"))
APGD_BATCH_SIZE = int(os.getenv("APGD_BATCH_SIZE", "8"))
FAB_STEPS = int(os.getenv("FAB_STEPS", "100"))
FAB_BATCH_SIZE = int(os.getenv("FAB_BATCH_SIZE", "2"))
SQUARE_N_QUERIES = int(os.getenv("SQUARE_N_QUERIES", "5000"))
SQUARE_BATCH_SIZE = int(os.getenv("SQUARE_BATCH_SIZE", "2"))

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CROP_PCT = float(os.getenv("CROP_PCT", "0.875"))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_csv_items(v: str) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def should_run_attack(name: str, selected: set) -> bool:
    return (not selected) or (name in selected)


def get_transform(img_size: int = IMG_SIZE, crop_pct: float = CROP_PCT):
    scale_size = int(img_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def list_images(folder: str) -> List[str]:
    all_files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def clear_images(folder: str):
    for p in list_images(folder):
        os.remove(p)


def sanitize_suffix(name: str) -> str:
    stem, ext = os.path.splitext(name)
    if stem.endswith("_succ"):
        stem = stem[:-5]
    if stem.endswith("_fail"):
        stem = stem[:-5]
    return stem + ext


def reverse_transform_store_cpu(tensor_cpu: torch.Tensor, path: str):
    img = torch.clamp(tensor_cpu, 0, 1)
    transforms.ToPILImage()(img).save(path)


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None, fixed_label: int = SOURCE_CLS):
        all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        self.image_paths = [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]
        self.transform = transform
        self.fixed_label = fixed_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.fixed_label, os.path.basename(path)


class EvalImageDataset(Dataset):
    def __init__(self, folder_path: str, transform=None, fixed_label: int = SOURCE_CLS):
        all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        self.image_paths = [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]
        self.transform = transform
        self.fixed_label = fixed_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.fixed_label, os.path.basename(path)


def build_model(device: torch.device):
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    return model


def get_attack_specs(model, threat_model: str, eps: float) -> Dict[str, Dict]:
    norm = "L2" if threat_model.lower() == "l2" else "Linf"
    return {
        "apgd_ce_untar_sl2py": {
            "builder": lambda m: torchattacks.APGD(
                m,
                norm=norm,
                eps=eps,
                steps=APGD_STEPS,
                loss="ce",
            ),
            "batch_size": APGD_BATCH_SIZE,
            "params": {"norm": norm, "eps": eps, "steps": APGD_STEPS, "loss": "ce"},
        },
        "apgd_dlr_untar_sl2py": {
            "builder": lambda m: torchattacks.APGD(
                m,
                norm=norm,
                eps=eps,
                steps=APGD_STEPS,
                loss="dlr",
            ),
            "batch_size": APGD_BATCH_SIZE,
            "params": {"norm": norm, "eps": eps, "steps": APGD_STEPS, "loss": "dlr"},
        },
        "fab_untar_sl2py": {
            "builder": lambda m: torchattacks.FAB(
                m,
                norm=norm,
                eps=eps,
                steps=FAB_STEPS,
            ),
            "batch_size": FAB_BATCH_SIZE,
            "params": {"norm": norm, "eps": eps, "steps": FAB_STEPS},
        },
        "square_untar_sl2py": {
            "builder": lambda m: torchattacks.Square(
                m,
                norm=norm,
                eps=eps,
                n_queries=SQUARE_N_QUERIES,
            ),
            "batch_size": SQUARE_BATCH_SIZE,
            "params": {"norm": norm, "eps": eps, "n_queries": SQUARE_N_QUERIES},
        },
    }


def generate_attack_images(atk, source_dataset, save_dir, batch_size, device, attack_name):
    os.makedirs(save_dir, exist_ok=True)
    existing = len(list_images(save_dir))
    total = len(source_dataset)

    if FORCE_REGENERATE and existing > 0:
        print(f"[{attack_name}] FORCE_REGENERATE=1 -> remove existing outputs {existing}/{total}")
        clear_images(save_dir)

    loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    for batch_idx, (images, labels, filenames) in enumerate(tqdm(loader, desc=f"Generating {attack_name}"), start=1):
        images = images.to(device)
        labels = labels.to(device)

        chunk_size = images.size(0)
        if ATTACK_MAX_CHUNK > 0:
            chunk_size = min(chunk_size, ATTACK_MAX_CHUNK)
        chunk_size = max(1, chunk_size)

        adv_chunks = []
        start = 0
        while start < images.size(0):
            end = min(start + chunk_size, images.size(0))
            img_chunk = images[start:end]
            lbl_chunk = labels[start:end]
            try:
                adv_chunk = atk(img_chunk, lbl_chunk)
                adv_chunks.append(adv_chunk.detach())
                start = end
            except RuntimeError as err:
                if "out of memory" not in str(err).lower() or chunk_size == 1:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                print(f"[{attack_name}] OOM retry -> chunk_size={chunk_size}")

        adv_images = torch.cat(adv_chunks, dim=0)
        adv_images_cpu = adv_images.detach().cpu()
        for i in range(len(filenames)):
            reverse_transform_store_cpu(adv_images_cpu[i], os.path.join(save_dir, filenames[i]))

        print(f"[{attack_name}] batch {batch_idx}/{len(loader)} done")


def evaluate_attack_folder(model, folder_path, device, batch_size=EVAL_BATCH_SIZE):
    dataset = EvalImageDataset(folder_path, transform=get_transform(), fixed_label=SOURCE_CLS)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    rows = []
    robust_success_count = 0
    escape_success_count = 0

    with torch.no_grad():
        for images, true_labels, filenames in loader:
            images = images.to(device)
            true_labels = true_labels.to(device)

            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
            top1 = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            for i in range(images.size(0)):
                pred = int(preds[i].item())
                true_label = int(true_labels[i].item())

                robust_success = int(pred == true_label)
                escape_success = int(pred != SOURCE_CLS)

                robust_success_count += robust_success
                escape_success_count += escape_success

                rows.append(
                    {
                        "filename": filenames[i],
                        "true_label": true_label,
                        "pred": pred,
                        "top1_conf": float(top1[i].item()),
                        "robust_success": robust_success,
                        "escape_success": escape_success,
                    }
                )

    total = len(rows)
    robust_acc = (robust_success_count / total) if total > 0 else 0.0
    escape_rate = (escape_success_count / total) if total > 0 else 0.0

    summary = {
        "total": total,
        "source_class": SOURCE_CLS,
        "robust_success_count": robust_success_count,
        "robust_accuracy": robust_acc,
        "escape_success_count": escape_success_count,
        "escape_rate": escape_rate,
        "threat_model": THREAT_MODEL,
        "eps": EPS,
    }
    return rows, summary


def annotate_files_with_escape_suffix(folder_path: str, rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        old_name = row["filename"]
        clean_name = sanitize_suffix(old_name)
        base, ext = os.path.splitext(clean_name)
        suffix = "_succ" if int(row["escape_success"]) == 1 else "_fail"
        new_name = f"{base}{suffix}{ext}"

        old_path = os.path.join(folder_path, old_name)
        if not os.path.exists(old_path):
            alt_old = os.path.join(folder_path, clean_name)
            if os.path.exists(alt_old):
                old_path = alt_old

        new_path = os.path.join(folder_path, new_name)
        if os.path.abspath(old_path) != os.path.abspath(new_path) and os.path.exists(old_path):
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)

        rr = dict(row)
        rr["filename"] = new_name
        out.append(rr)

    return out


def write_eval_artifacts(folder_path: str, rows: List[Dict], summary: Dict, params: Dict, attack_name: str):
    os.makedirs(folder_path, exist_ok=True)

    csv_path = os.path.join(folder_path, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "true_label",
                "pred",
                "top1_conf",
                "robust_success",
                "escape_success",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(summary)
    payload["params"] = params
    payload["metadata"] = {
        "attack_name": attack_name,
        "attack_mode": "untargeted",
        "output_csv": csv_path,
    }

    summary_path = os.path.join(folder_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"SOURCE_DIR={SOURCE_DIR}")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"THREAT_MODEL={THREAT_MODEL}, EPS={EPS}")

    selected_set = set(parse_csv_items(ATTACK_FILTER))
    if selected_set:
        print(f"ATTACK_FILTER={sorted(selected_set)}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    model = build_model(device)
    source_dataset = ImageFolderDataset(folder_path=SOURCE_DIR, transform=get_transform(), fixed_label=SOURCE_CLS)
    print(f"source images={len(source_dataset)}")

    attack_specs = get_attack_specs(model, THREAT_MODEL, EPS)

    report = {
        "checkpoint": CHECKPOINT_PATH,
        "source_dir": SOURCE_DIR,
        "source_class": SOURCE_CLS,
        "threat_model": THREAT_MODEL,
        "eps": EPS,
        "attacks": [],
    }

    for attack_name, spec in attack_specs.items():
        if not should_run_attack(attack_name, selected_set):
            continue

        attack_dir = os.path.join(OUTPUT_ROOT, attack_name)
        print(f"\n=== Running {attack_name} ===")

        atk = spec["builder"](model)
        generate_attack_images(
            atk=atk,
            source_dataset=source_dataset,
            save_dir=attack_dir,
            batch_size=spec["batch_size"],
            device=device,
            attack_name=attack_name,
        )

        rows, summary = evaluate_attack_folder(model, attack_dir, device)
        rows = annotate_files_with_escape_suffix(attack_dir, rows)
        write_eval_artifacts(attack_dir, rows, summary, spec["params"], attack_name)

        report["attacks"].append(
            {
                "attack_name": attack_name,
                "robust_accuracy": summary["robust_accuracy"],
                "escape_rate": summary["escape_rate"],
                "total": summary["total"],
                "attack_dir": attack_dir,
            }
        )

    report_path = os.path.join(OUTPUT_ROOT, "untargeted_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Report: {report_path}")
    for item in report["attacks"]:
        print(
            f"[{item['attack_name']}] robust_accuracy={item['robust_accuracy']:.4f} | "
            f"escape_rate={item['escape_rate']:.4f} | total={item['total']}"
        )


if __name__ == "__main__":
    main()
