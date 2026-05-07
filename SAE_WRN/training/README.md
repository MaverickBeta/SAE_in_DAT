# SAE for DAT WideResNet34x10 on CIFAR-10

This directory contains scripts to extract activation features from the DAT WideResNet34x10 checkpoint and train a TopK Sparse Autoencoder (SAE) on them.

## Model Architecture Recap

**WideResNet34x10** forward flow on CIFAR-10 (32×32):

| Layer | Output Shape | Description |
|:---|:---|:---|
| conv1 | `[B, 16, 32, 32]` | Initial conv |
| block1 | `[B, 160, 32, 32]` | 5 BasicBlocks, stride=1 |
| block2 | `[B, 320, 16, 16]` | 5 BasicBlocks, stride=2 |
| block3 | `[B, 640, 8, 8]` | 5 BasicBlocks, stride=2 |
| **activation** ⬅ hook here | **`[B, 640, 8, 8]`** | BN + ReLU (post-block3, pre-pool) |
| avg_pool | `[B, 640]` | Global average pool |
| fc | `[B, 10]` | Classifier |

**SAE input dimension**: `d_in = 640`
**Spatial tokens per image**: `8 × 8 = 64`

## Quick Start

### 1. Extract Features

```bash
cd /Data_share/hongyi/DAT/SAE_WRN

# Extract train set features (per-class .pt files)
python extract_features.py \
  --gpu_id 0 \
  --batch_size 128 \
  --split train \
  --data_root /Data_share/hongyi/DAT/data \
  --out_dir /Data_share/hongyi/DAT/SAE_WRN/features \
  --ckpt_path /Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth

# Also build a merged.pt (optional, for faster loading)
python extract_features.py \
  --gpu_id 0 \
  --batch_size 128 \
  --split train \
  --build_merged \
  --data_root /Data_share/hongyi/DAT/data \
  --out_dir /Data_share/hongyi/DAT/SAE_WRN/features \
  --ckpt_path /Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth
```

**Notes:**
- CIFAR-10 will be automatically downloaded to `--data_root` if not present.
- The model handles input normalization internally (`normalize_input=True`), so we only apply `ToTensor()`.
- Output per-class files have shape `[N_cls, 640, 8, 8]` (e.g., ~5000 images per class for train set).

### 2. Train SAE

#### Option A: Train from merged file
```bash
python train_sae.py \
  --data_path /Data_share/hongyi/DAT/SAE_WRN/features/merged.pt \
  --out_dir /Data_share/hongyi/DAT/SAE_WRN/checkpoints \
  --expansion_rate 32 \
  --k 64 \
  --batch_size 4096 \
  --steps 50000 \
  --gpu_id 0 \
  --preload
```

#### Option B: Train from per-class directory (no preload)
```bash
python train_sae.py \
  --data_path /Data_share/hongyi/DAT/SAE_WRN/features \
  --out_dir /Data_share/hongyi/DAT/SAE_WRN/checkpoints \
  --expansion_rate 32 \
  --k 64 \
  --batch_size 4096 \
  --steps 50000 \
  --gpu_id 0
```

**Key hyperparameters:**
| Param | Default | Description |
|:---|:---|:---|
| `--expansion_rate` | 32 | SAE latent dimension = `640 × expansion_rate` |
| `--k` | 64 | TopK sparsity (number of active latents) |
| `--batch_size` | 4096 | Training batch size (number of spatial tokens) |
| `--steps` | 50000 | Total training steps |
| `--lr` | 3e-4 | AdamW learning rate |
| `--dead_window` | 2500 | Steps between dead neuron checks |
| `--resample_every` | 5000 | Steps between dead neuron resampling |
| `--preload` | False | Preload all features to RAM (faster, needs ~8GB) |

### 3. Output Structure

```
SAE_WRN/
├── features/
│   ├── airplane.pt          # [5000, 640, 8, 8]
│   ├── automobile.pt
│   ├── ...
│   └── merged.pt            # Optional: {'features': [50000, 640, 8, 8]}
├── checkpoints/
│   └── k64_exp32/
│       ├── training_log.csv
│       ├── best_loss.pt
│       ├── best_ev.pt
│       ├── best_composite.pt
│       └── sae_wrn_din640_exp32_k64_step_50000.pt
```

## Checkpoint Format

Saved checkpoints contain:
```python
{
    "model_state_dict": {...},
    "config": {...},           # All training args
    "d_in": 640,
    "d_lat": 20480,            # 640 * 32
    "step": 50000,
    "norm_mean": [1, 640],     # Feature normalization mean
    "norm_std": [1, 640],      # Feature normalization std
}
```

## Reusing Your SAE

You can load the trained SAE alongside the base model for downstream analysis:

```python
import sys
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

import torch
from rebm.models.wide_resnet_innoutrobustness import WideResNet34x10
from sae_core.model import TopKAutoencoder

# Base model
model = WideResNet34x10(num_classes=10, normalize_input=True)
model.load_state_dict(torch.load("DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth", map_location="cpu"))
model.eval()

# SAE
sae_ckpt = torch.load("DAT/SAE_WRN/checkpoints/k64_exp32/best_ev.pt", map_location="cpu")
sae = TopKAutoencoder(d_in=640, d_lat=sae_ckpt["d_lat"], k=sae_ckpt["config"]["k"])
sae.load_state_dict(sae_ckpt["model_state_dict"])
sae.eval()

norm_mean = sae_ckpt["norm_mean"]
norm_std = sae_ckpt["norm_std"]

# Hook and encode
captured = {}
def hook_fn(m, inp, out):
    captured["feat"] = out  # [B, 640, 8, 8]

handle = model.activation.register_forward_hook(hook_fn)

with torch.no_grad():
    logits = model(images)          # [B, 10]
    feat = captured["feat"]         # [B, 640, 8, 8]
    b, c, h, w = feat.shape
    flat = feat.permute(0, 2, 3, 1).reshape(-1, c)  # [B*64, 640]
    flat_norm = (flat - norm_mean) / norm_std
    z = sae.encode(flat_norm)       # [B*64, d_lat] sparse latent

handle.remove()
```
