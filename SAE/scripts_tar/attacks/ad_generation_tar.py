import os
import sys
import csv
import json
import time
import gc
import glob
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torchattacks
except ImportError:
    print("Please install torchattacks first: pip install torchattacks")
    sys.exit(1)

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint

# Source/target classes for targeted attack: default Sea Lion (150) -> Rock Python (62)
TARGET_CLS = int(os.getenv("TARGET_CLS", "62"))
SOURCE_CLS = int(os.getenv("SOURCE_CLS", "150"))
SOURCE_WNID = os.getenv("SOURCE_WNID", "n02077923")

IMAGENET_VAL_DIR = os.getenv("IMAGENET_VAL_DIR", "/Data_share/hongyi/DAT/data/ImageNet/val")
SOURCE_DIR = os.getenv("SOURCE_DIR", "")
ATTACK_OUTPUT_ROOT = os.getenv("ATTACK_OUTPUT_ROOT", "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_tar")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))

# Single-GPU runtime options.
ATTACK_FILTER = os.getenv("ATTACK_FILTER", "")
FORCE_REGENERATE = os.getenv("FORCE_REGENERATE", "0") == "1"
ATTACK_MAX_CHUNK = int(os.getenv("ATTACK_MAX_CHUNK", "0"))
SKIP_IF_COMPLETE = os.getenv("SKIP_IF_COMPLETE", "1") == "1"

# Tunable params for targeted attacks.
PGD_EPS = float(os.getenv("PGD_EPS", str(12 / 255)))
PGD_ALPHA = float(os.getenv("PGD_ALPHA", str(2 / 255)))
PGD_STEPS = int(os.getenv("PGD_STEPS", "40"))

FAB_EPS = float(os.getenv("FAB_EPS", str(16 / 255)))
FAB_STEPS = int(os.getenv("FAB_STEPS", "80"))
FAB_N_RESTARTS = int(os.getenv("FAB_N_RESTARTS", "1"))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

TARGETED_ATTACKS = {
    "pgd_tar_sl2py": {
        "builder": lambda m: torchattacks.PGD(m, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS),
        "batch_size": 16,
        "params": {"eps": PGD_EPS, "alpha": PGD_ALPHA, "steps": PGD_STEPS},
    },
    "fab_tar_sl2py": {
        "builder": lambda m: torchattacks.FAB(m, eps=FAB_EPS, steps=FAB_STEPS, n_restarts=FAB_N_RESTARTS),
        "batch_size": 4,
        "params": {"eps": FAB_EPS, "steps": FAB_STEPS, "n_restarts": FAB_N_RESTARTS},
    },
}


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        self.image_paths = [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]
        if MAX_IMAGES > 0:
            self.image_paths = self.image_paths[:MAX_IMAGES]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


class EvalImageDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        all_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        self.image_paths = [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


def parse_csv_items(v: str) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def should_run_attack(name: str, selected: set) -> bool:
    return (not selected) or (name in selected)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def list_images(folder: str) -> List[str]:
    all_files = glob.glob(os.path.join(folder, "*"))
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


def params_changed(folder_path: str, new_params: Dict) -> bool:
    summary_path = os.path.join(folder_path, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            old_payload = json.load(f)
        return old_payload.get("params") != new_params
    except Exception:
        return False


def clear_cuda_memory(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_source_dir() -> str:
    if SOURCE_DIR:
        return SOURCE_DIR
    return os.path.join(IMAGENET_VAL_DIR, SOURCE_WNID)


def validate_runtime_inputs(source_dir: str):
    if TARGET_CLS < 0 or SOURCE_CLS < 0:
        raise ValueError(f"SOURCE_CLS/TARGET_CLS must be >= 0, got SOURCE_CLS={SOURCE_CLS}, TARGET_CLS={TARGET_CLS}")
    if SOURCE_CLS == TARGET_CLS:
        raise ValueError("SOURCE_CLS and TARGET_CLS cannot be the same for targeted attack")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")


def build_model(device: torch.device):
    print("Loading Base Model...")
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


def enable_targeted_mode(atk, device: torch.device, attack_name: str):
    try:
        atk.set_mode_targeted_by_function(
            target_map_function=lambda images, labels: torch.full(
                (images.size(0),), TARGET_CLS, dtype=torch.long, device=device
            )
        )
        print(f"[{attack_name}] targeted mode enabled -> class {TARGET_CLS}")
    except Exception as e:
        raise RuntimeError(f"{attack_name} does not support targeted mode in this environment: {e}")


def generate_attack_images(atk, source_dataset, save_dir, batch_size, device, attack_name, force_regenerate=False):
    os.makedirs(save_dir, exist_ok=True)
    total = len(source_dataset)
    existing = len(list_images(save_dir))

    if SKIP_IF_COMPLETE and (not force_regenerate) and total > 0 and existing == total:
        print(f"[{attack_name}] detected complete existing outputs ({existing}/{total}), skip generation")
        return

    if force_regenerate and existing > 0:
        print(f"[{attack_name}] FORCE_REGENERATE=1 -> removing existing outputs {existing}/{total}")
        clear_images(save_dir)

    loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print(f"\n================ Running Attack: {attack_name} ================")
    print(f"[{attack_name}] batch_size={batch_size}")

    timing = {"data_wait_s": 0.0, "attack_s": 0.0, "save_s": 0.0}
    oom_retries = 0
    loop_t0 = time.perf_counter()
    last_batch_end = loop_t0

    for batch_idx, (images, filenames) in enumerate(tqdm(loader, desc=f"Generating {attack_name}"), start=1):
        now = time.perf_counter()
        timing["data_wait_s"] += now - last_batch_end

        images = images.to(device)
        print(f"[{attack_name}] batch {batch_idx}/{len(loader)} start | raw_batch={images.size(0)}")
        max_chunk = images.size(0)
        if ATTACK_MAX_CHUNK > 0:
            max_chunk = min(max_chunk, ATTACK_MAX_CHUNK)
        chunk_size = max(1, max_chunk)

        adv_chunks = []
        start = 0
        while start < images.size(0):
            end = min(start + chunk_size, images.size(0))
            img_chunk = images[start:end]
            src_chunk = torch.full((img_chunk.size(0),), SOURCE_CLS, dtype=torch.long, device=device)
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                adv_chunk = atk(img_chunk, src_chunk)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timing["attack_s"] += time.perf_counter() - t0
                adv_chunks.append(adv_chunk.detach())
                start = end
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                oom_retries += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_size == 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                print(f"[{attack_name}] OOM -> retry with chunk_size={chunk_size}")

        adv_images = torch.cat(adv_chunks, dim=0)

        t_save0 = time.perf_counter()
        adv_images_cpu = adv_images.detach().cpu()
        for i in range(len(filenames)):
            reverse_transform_store_cpu(adv_images_cpu[i], os.path.join(save_dir, filenames[i]))
        timing["save_s"] += time.perf_counter() - t_save0
        print(f"[{attack_name}] batch {batch_idx}/{len(loader)} done")

        last_batch_end = time.perf_counter()

    total_s = max(time.perf_counter() - loop_t0, 1e-8)
    print(
        f"[{attack_name}] timing: total={total_s:.2f}s | "
        f"data_wait={timing['data_wait_s']:.2f}s ({timing['data_wait_s']/total_s*100:.1f}%) | "
        f"attack={timing['attack_s']:.2f}s ({timing['attack_s']/total_s*100:.1f}%) | "
        f"save={timing['save_s']:.2f}s ({timing['save_s']/total_s*100:.1f}%)"
    )
    if oom_retries > 0:
        print(f"[{attack_name}] OOM auto-retries={oom_retries}")


def evaluate_targeted_folder(model, folder_path, device, batch_size=32):
    dataset = EvalImageDataset(folder_path, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    rows = []
    success_count = 0
    escape_count = 0
    target_conf_sum = 0.0
    source_conf_sum = 0.0

    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            preds = probs.argmax(dim=1)
            top1 = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            target_confs = probs[:, TARGET_CLS]
            source_confs = probs[:, SOURCE_CLS]

            for i in range(images.size(0)):
                pred = int(preds[i].item())
                success = int(pred == TARGET_CLS)
                escape = int(pred != SOURCE_CLS)
                success_count += success
                escape_count += escape
                target_conf_sum += float(target_confs[i].item())
                source_conf_sum += float(source_confs[i].item())
                rows.append(
                    {
                        "filename": filenames[i],
                        "pred": pred,
                        "top1_conf": float(top1[i].item()),
                        "target_conf": float(target_confs[i].item()),
                        "source_conf": float(source_confs[i].item()),
                        "success": success,
                    }
                )

    total = len(rows)
    success_rate = (success_count / total) if total > 0 else 0.0
    summary = {
        "total": total,
        "success_count": success_count,
        "fail_count": total - success_count,
        "success_rate": success_rate,
        "escape_count": escape_count,
        "escape_rate": (escape_count / total) if total > 0 else 0.0,
        "target_class": TARGET_CLS,
        "source_class": SOURCE_CLS,
        "avg_target_conf": (target_conf_sum / total) if total > 0 else 0.0,
        "avg_source_conf": (source_conf_sum / total) if total > 0 else 0.0,
        "attack_mode": "targeted",
    }
    return rows, summary


def annotate_files_with_success(folder_path: str, rows: List[Dict]) -> List[Dict]:
    new_rows = []
    for row in rows:
        old_name = row["filename"]
        clean_name = sanitize_suffix(old_name)
        base, ext = os.path.splitext(clean_name)
        suffix = "_succ" if int(row["success"]) == 1 else "_fail"
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

        r = dict(row)
        r["filename"] = new_name
        new_rows.append(r)
    return new_rows


def write_eval_artifacts(folder_path: str, rows: List[Dict], summary: Dict, params: Dict, attack_name: str):
    os.makedirs(folder_path, exist_ok=True)

    csv_path = os.path.join(folder_path, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "pred", "top1_conf", "target_conf", "source_conf", "success"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(summary)
    payload["params"] = params
    payload["metadata"] = {
        "selected_from_sweep": False,
        "attack_name": attack_name,
        "attack_mode": "targeted",
    }

    with open(os.path.join(folder_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_targeted_attacks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"FORCE_REGENERATE={FORCE_REGENERATE}")
    print(f"SKIP_IF_COMPLETE={SKIP_IF_COMPLETE}")
    print(f"PGD params: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
    print(f"FAB params: eps={FAB_EPS}, steps={FAB_STEPS}, n_restarts={FAB_N_RESTARTS}")

    selected_set = set(parse_csv_items(ATTACK_FILTER))
    if selected_set:
        print(f"ATTACK_FILTER={sorted(selected_set)}")

    source_dir = resolve_source_dir()
    validate_runtime_inputs(source_dir)

    os.makedirs(ATTACK_OUTPUT_ROOT, exist_ok=True)
    model = build_model(device)

    source_dataset = ImageFolderDataset(source_dir, transform=get_transform())
    if len(source_dataset) == 0:
        raise RuntimeError(f"No images found in source dataset directory: {source_dir}")

    print(
        f"Source dataset: {source_dir} | images={len(source_dataset)} | "
        f"targeted attack: {SOURCE_CLS} -> {TARGET_CLS}"
    )

    report = {
        "checkpoint": CHECKPOINT_PATH,
        "source_dir": source_dir,
        "source_class": SOURCE_CLS,
        "target_class": TARGET_CLS,
        "mode": "targeted_only_single_gpu",
        "attacks": [],
    }

    for attack_name, spec in TARGETED_ATTACKS.items():
        if not should_run_attack(attack_name, selected_set):
            continue

        folder = os.path.join(ATTACK_OUTPUT_ROOT, attack_name)
        params = spec["params"]
        force = FORCE_REGENERATE or params_changed(folder, params)

        try:
            atk = spec["builder"](model)
            enable_targeted_mode(atk, device, attack_name)
        except Exception as e:
            print(f"[{attack_name}] skip due to targeted mode setup failure: {e}")
            report["attacks"].append(
                {
                    "attack_name": attack_name,
                    "attack_mode": "targeted",
                    "selected_params": params,
                    "status": "skipped_targeted_not_supported",
                    "error": str(e),
                }
            )
            continue

        generate_attack_images(
            atk=atk,
            source_dataset=source_dataset,
            save_dir=folder,
            batch_size=spec["batch_size"],
            device=device,
            attack_name=attack_name,
            force_regenerate=force,
        )

        rows, summary = evaluate_targeted_folder(model, folder, device)
        rows = annotate_files_with_success(folder, rows)
        write_eval_artifacts(folder, rows, summary, params, attack_name)
        clear_cuda_memory(device)

        report["attacks"].append(
            {
                "attack_name": attack_name,
                "attack_mode": "targeted",
                "selected_params": params,
                "targeted_success_rate": summary["success_rate"],
                "escape_rate": summary["escape_rate"],
                "final_dir": folder,
            }
        )

    report_path = os.path.join(ATTACK_OUTPUT_ROOT, "targeted_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nAll targeted attacks generated and evaluated.")
    print(f"Results root: {ATTACK_OUTPUT_ROOT}")
    print(f"Report: {report_path}")
    for item in report["attacks"]:
        if item.get("status") == "skipped_targeted_not_supported":
            print(f"[targeted] {item['attack_name']} | skipped ({item.get('error', 'unknown error')})")
            continue
        print(
            f"[targeted] {item['attack_name']} | "
            f"targeted_success_rate={item['targeted_success_rate']:.4f} | "
            f"escape_rate={item['escape_rate']:.4f}"
        )


if __name__ == "__main__":
    run_targeted_attacks()
