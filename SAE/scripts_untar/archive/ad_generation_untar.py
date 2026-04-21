import os
import sys
import csv
import json
import time
import gc
import glob
import shutil
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

# Source/target metadata kept for reporting compatibility.
ROCK_PYTHON_CLS = 62
SEA_LION_CLS = 150
SEA_LION_WNID = "n02077923"

IMAGENET_VAL_DIR = "/Data_share/hongyi/DAT/data/ImageNet/val"
ATTACK_OUTPUT_ROOT_BASE = "/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py_untar"
CHECKPOINT_PATH = "/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth"
NUM_WORKERS = 4

SOURCE_DIR = os.getenv("SOURCE_DIR", os.path.join(IMAGENET_VAL_DIR, SEA_LION_WNID))
ATTACK_OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", ATTACK_OUTPUT_ROOT_BASE)
NUM_SHARDS = int(os.getenv("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "0"))

# Single-GPU defaults suitable for RTX 3090.
ATTACK_MAX_CHUNK = int(os.getenv("ATTACK_MAX_CHUNK", "0"))
ATTACK_FILTER = os.getenv("ATTACK_FILTER", "")
FORCE_REGENERATE = os.getenv("FORCE_REGENERATE", "0") == "1"
SKIP_FAB = os.getenv("SKIP_FAB", "0") == "1"

# Runtime knobs for heavy attacks. Keep defaults practical for single RTX 3090.
FAB_STEPS = int(os.getenv("FAB_STEPS", "60"))
FAB_BATCH_SIZE = int(os.getenv("FAB_BATCH_SIZE", "2"))
SQUARE_N_QUERIES = int(os.getenv("SQUARE_N_QUERIES", "4000"))

# AA-like untargeted 4 attacks: APGD-CE, APGD-DLR, FAB, Square.
APGD_EPS = float(os.getenv("APGD_EPS", str(24 / 255)))
APGD_STEPS = int(os.getenv("APGD_STEPS", "300"))
FAB_EPS = float(os.getenv("FAB_EPS", str(16 / 255)))
SQUARE_EPS = float(os.getenv("SQUARE_EPS", str(24 / 255)))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

FIXED_ATTACKS = {
    "apgd_ce_untar_sl2py": {
        "builder": lambda m: torchattacks.APGD(m, eps=APGD_EPS, steps=APGD_STEPS, loss="ce"),
        "batch_size": 8,
        "params": {"eps": APGD_EPS, "steps": APGD_STEPS, "loss": "ce"},
    },
    "apgd_dlr_untar_sl2py": {
        "builder": lambda m: torchattacks.APGD(m, eps=APGD_EPS, steps=APGD_STEPS, loss="dlr"),
        "batch_size": 8,
        "params": {"eps": APGD_EPS, "steps": APGD_STEPS, "loss": "dlr"},
    },
    "fab_untar_sl2py": {
        "builder": lambda m: torchattacks.FAB(m, eps=FAB_EPS, steps=FAB_STEPS),
        "batch_size": FAB_BATCH_SIZE,
        "params": {"eps": FAB_EPS, "steps": FAB_STEPS},
    },
    "square_untar_sl2py": {
        "builder": lambda m: torchattacks.Square(m, eps=SQUARE_EPS, n_queries=SQUARE_N_QUERIES),
        "batch_size": 2,
        "params": {"eps": SQUARE_EPS, "n_queries": SQUARE_N_QUERIES},
    },
}


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str = "", transform=None, image_paths: List[str] = None):
        if image_paths is not None:
            self.image_paths = sorted(image_paths)
        else:
            self.image_paths = sorted(
                glob.glob(os.path.join(folder_path, "*.jpg"))
                + glob.glob(os.path.join(folder_path, "*.png"))
                + glob.glob(os.path.join(folder_path, "*.JPEG"))
            )
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


def shard_paths(paths: List[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 1:
        return list(paths)
    return [p for i, p in enumerate(paths) if (i % num_shards) == shard_index]


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


def build_model(device: torch.device):
    print("Loading Base Model...")
    if device.type == "cuda":
        # Force-create CUDA primary context early to avoid first-backward cublas warning.
        torch.cuda.init()
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


def generate_attack_images(atk, source_dataset, save_dir, batch_size, device, attack_name, force_regenerate=False):
    os.makedirs(save_dir, exist_ok=True)
    total = len(source_dataset)
    existing = len(list_images(save_dir))

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
        print(
            f"[{attack_name}] batch {batch_idx}/{len(loader)} start | "
            f"raw_batch={images.size(0)}"
        )
        max_chunk = images.size(0)
        if ATTACK_MAX_CHUNK > 0:
            max_chunk = min(max_chunk, ATTACK_MAX_CHUNK)
        chunk_size = max(1, max_chunk)

        adv_chunks = []
        start = 0
        while start < images.size(0):
            end = min(start + chunk_size, images.size(0))
            img_chunk = images[start:end]
            src_chunk = torch.full((img_chunk.size(0),), SEA_LION_CLS, dtype=torch.long, device=device)
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


def evaluate_untargeted_folder(model, folder_path, device, batch_size=32):
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
            target_confs = probs[:, ROCK_PYTHON_CLS]
            source_confs = probs[:, SEA_LION_CLS]

            for i in range(images.size(0)):
                pred = int(preds[i].item())
                # Untargeted success: escape from source class 150.
                success = int(pred != SEA_LION_CLS)
                success_count += success
                escape_count += success
                target_conf_sum += float(target_confs[i].item())
                source_conf_sum += float(source_confs[i].item())
                rows.append(
                    {
                        "filename": filenames[i],
                        "pred": pred,
                        "top1_conf": float(top1[i].item()),
                        "target_conf_62": float(target_confs[i].item()),
                        "source_conf_150": float(source_confs[i].item()),
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
        "escape_rate": success_rate,
        "target_class": ROCK_PYTHON_CLS,
        "source_class": SEA_LION_CLS,
        "avg_target_conf": (target_conf_sum / total) if total > 0 else 0.0,
        "avg_source_conf": (source_conf_sum / total) if total > 0 else 0.0,
        "attack_mode": "untargeted",
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
            fieldnames=["filename", "pred", "top1_conf", "target_conf_62", "source_conf_150", "success"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(summary)
    payload["params"] = params
    payload["metadata"] = {
        "selected_from_sweep": False,
        "attack_name": attack_name,
        "attack_mode": "untargeted",
    }

    with open(os.path.join(folder_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_untargeted_attacks():
    global ATTACK_OUTPUT_ROOT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"FORCE_REGENERATE={FORCE_REGENERATE}")
    print(f"SKIP_FAB={SKIP_FAB}")
    print(f"SOURCE_DIR={SOURCE_DIR}")
    print(f"NUM_SHARDS={NUM_SHARDS}, SHARD_INDEX={SHARD_INDEX}")
    print(
        f"AA-style untargeted attacks: [apgd-ce, apgd-dlr, fab, square] | "
        f"APGD_EPS={APGD_EPS}, APGD_STEPS={APGD_STEPS}, "
        f"FAB_EPS={FAB_EPS}, FAB_STEPS={FAB_STEPS}, "
        f"SQUARE_EPS={SQUARE_EPS}, SQUARE_N_QUERIES={SQUARE_N_QUERIES}"
    )

    selected_set = set(parse_csv_items(ATTACK_FILTER))
    if selected_set:
        print(f"ATTACK_FILTER={sorted(selected_set)}")

    if NUM_SHARDS < 1:
        raise ValueError(f"NUM_SHARDS must be >= 1, got {NUM_SHARDS}")
    if not (0 <= SHARD_INDEX < NUM_SHARDS):
        raise ValueError(
            f"SHARD_INDEX must satisfy 0 <= SHARD_INDEX < NUM_SHARDS, got {SHARD_INDEX} with NUM_SHARDS={NUM_SHARDS}"
        )

    if "OUTPUT_ROOT" not in os.environ and NUM_SHARDS > 1:
        ATTACK_OUTPUT_ROOT = os.path.join(ATTACK_OUTPUT_ROOT_BASE, f"shard_{SHARD_INDEX:02d}")

    os.makedirs(ATTACK_OUTPUT_ROOT, exist_ok=True)
    model = build_model(device)

    all_source_paths = sorted(
        p for p in glob.glob(os.path.join(SOURCE_DIR, "*")) if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)
    )
    shard_source_paths = shard_paths(all_source_paths, NUM_SHARDS, SHARD_INDEX)
    source_dataset = ImageFolderDataset(transform=get_transform(), image_paths=shard_source_paths)

    print(
        f"Source dataset: {SOURCE_DIR} | total_images={len(all_source_paths)} | shard_images={len(source_dataset)} | "
        f"source_class={SEA_LION_CLS} | untargeted success=escape from source"
    )

    report = {
        "checkpoint": CHECKPOINT_PATH,
        "source_dir": SOURCE_DIR,
        "num_source_images_total": len(all_source_paths),
        "num_source_images_shard": len(source_dataset),
        "num_shards": NUM_SHARDS,
        "shard_index": SHARD_INDEX,
        "source_class": SEA_LION_CLS,
        "target_class_ref": ROCK_PYTHON_CLS,
        "mode": "untargeted_only_single_gpu",
        "attacks": [],
    }

    for attack_name, spec in FIXED_ATTACKS.items():
        if not should_run_attack(attack_name, selected_set):
            continue
        if SKIP_FAB and attack_name == "fab_untar_sl2py":
            print("[fab_untar_sl2py] skipped by SKIP_FAB=1")
            continue

        folder = os.path.join(ATTACK_OUTPUT_ROOT, attack_name)
        params = spec["params"]
        force = FORCE_REGENERATE or params_changed(folder, params)
        atk = spec["builder"](model)
        generate_attack_images(
            atk=atk,
            source_dataset=source_dataset,
            save_dir=folder,
            batch_size=spec["batch_size"],
            device=device,
            attack_name=attack_name,
            force_regenerate=force,
        )

        rows, summary = evaluate_untargeted_folder(model, folder, device)
        rows = annotate_files_with_success(folder, rows)
        write_eval_artifacts(folder, rows, summary, params, attack_name)
        clear_cuda_memory(device)

        report["attacks"].append(
            {
                "attack_name": attack_name,
                "attack_mode": "untargeted",
                "selected_params": params,
                "escape_rate": summary["escape_rate"],
                "success_rate": summary["success_rate"],
                "final_dir": folder,
            }
        )

    report_path = os.path.join(ATTACK_OUTPUT_ROOT, "untargeted_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nAll untargeted attacks generated and evaluated.")
    print(f"Results root: {ATTACK_OUTPUT_ROOT}")
    print(f"Report: {report_path}")
    for item in report["attacks"]:
        print(
            f"[untargeted] {item['attack_name']} | "
            f"escape_rate={item['escape_rate']:.4f} | "
            f"success_rate={item['success_rate']:.4f}"
        )


if __name__ == "__main__":
    run_untargeted_attacks()
