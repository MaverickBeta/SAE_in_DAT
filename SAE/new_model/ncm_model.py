#!/usr/bin/env python3
"""
ConvNeXt + SAE-NCM 模型。

Stage 3 后直接 SAE encode，输出 cosine NCM scores [B, 1000]，
完全替换原始 FC head 和 norm_head。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Resolve DAT root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SAE" / "project"))

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


def build_convnext_backbone(device: torch.device, checkpoint: str):
    """构建 ConvNeXt backbone（保留 stem + stages，丢弃 head）。"""
    model = create_convnext_model(
        model_type="convnext_large",
        num_classes=1000,
        normalize_input=False,
        use_layernorm=True,
        use_convstem=True,
    )
    load_checkpoint(model, checkpoint)
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
        raise KeyError("Cannot resolve d_in")
    if d_lat is None:
        expansion_rate = config.get("expansion_rate", None)
        if expansion_rate is None:
            raise KeyError("Cannot resolve d_lat")
        d_lat = int(d_in) * int(expansion_rate)
    if k is None:
        raise KeyError("Cannot resolve k")

    sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to(device)
    norm_std = ckpt["norm_std"].to(device)
    return sae, norm_mean, norm_std, {"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)}


class ConvNeXt_SAE_NCM(nn.Module):
    """
    ConvNeXt backbone + SAE-NCM 分类器。

    Forward:
        x -> stem -> stages -> SAE encode -> cosine NCM -> scores [B, 1000]

    不使用原始 head，不进行 SAE decode。
    """

    def __init__(self, convnext_model, sae_model, norm_mean, norm_std, cls_dense):
        """
        Args:
            convnext_model: ConvNeXt (含 stem, stages, norm_pre, head)
            sae_model: TopKAutoencoder
            norm_mean, norm_std: SAE 标准化参数
            cls_dense: [1000, 49, 65536] dense class mean latent
        """
        super().__init__()
        self.stem = convnext_model.stem
        self.stages = convnext_model.stages  # Sequential[4]

        # SAE 转 FP16，减少 encode 内部激活显存
        self.sae = sae_model.half()
        self.norm_mean = norm_mean.half()
        self.norm_std = norm_std.half()

        # 预计算 class statistics，常驻 GPU（FP16 减少显存）
        cls_dense_fp16 = cls_dense.half()
        self.register_buffer("cls_dense", cls_dense_fp16)                     # [C, P, D]
        self.register_buffer("cls_norm", torch.norm(cls_dense_fp16, dim=-1))  # [C, P]

        # Freeze SAE
        for p in self.sae.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """
        Returns:
            scores: [B, 1000]  average cosine similarity
        """
        # Backbone: stem + 4 stages
        out = self.stem(x)
        out = self.stages(out)           # [B, 1536, 7, 7]

        # SAE encode（输入转 FP16，SAE 权重也是 FP16，内部全 FP16）
        B, C, H, W = out.shape
        flat = out.permute(0, 2, 3, 1).reshape(-1, C)
        flat_norm = (flat - self.norm_mean) / self.norm_std
        z = self.sae.encode(flat_norm.half())   # [B*49, 65536] FP16
        z = z.reshape(B, H * W, -1)             # [B, 49, 65536] FP16

        # NCM scores（全 FP16 计算）
        scores = self.ncm_scores(z)   # [B, 1000] FP16
        return scores.float()         # 返回 FP32 供 CrossEntropy 使用

    def ncm_scores(self, z):
        """
        利用 TopK 稀疏性高效计算 cosine NCM scores。

        Args:
            z: [B, 49, 65536]  SAE encode 输出（每行仅 64 非零）

        Returns:
            scores: [B, 1000]  average cosine similarity
        """
        B, P, D = z.shape
        C = self.cls_dense.size(0)
        k = 64

        # 提取 z 的 TopK（z 本身已稀疏，topk 即非零元素）
        z_reshaped = z.reshape(-1, D)                # [B*P, D]
        z_topk_val, z_topk_idx = torch.topk(z_reshaped, k=k, dim=-1)
        z_idx = z_topk_idx.reshape(B, P, k)          # [B, P, 64]
        z_val = z_topk_val.reshape(B, P, k)          # [B, P, 64]

        # z 的 per-position norm
        z_norm = torch.norm(z_val, dim=-1) + 1e-8    # [B, P]

        # 向量化稀疏 cosine accumulation：把 49×64 次 launch 合并为 49 次
        cosine_sum = torch.zeros(B, C, device=z.device, dtype=torch.float32)

        for p in range(P):
            # z_idx[:, p, :] -> [B, 64], z_val[:, p, :] -> [B, 64]
            z_idx_p = z_idx[:, p, :]      # [B, k]
            z_val_p = z_val[:, p, :]      # [B, k]

            # cls_dense[:, p, z_idx_p] -> [C, B, k]
            gathered = self.cls_dense[:, p, z_idx_p]

            # [C, B, k] * [1, B, k] -> sum over k -> [C, B] -> [B, C]
            dot = (gathered * z_val_p.unsqueeze(0)).sum(dim=2).t()

            # 按 position 归一化并累加
            cosine_sum += dot / (
                z_norm[:, p].unsqueeze(1) * self.cls_norm[:, p].unsqueeze(0)
            )

        return cosine_sum / P


def build_ncm_model(device, checkpoint, sae_ckpt, class_npz_path):
    """
    构建完整的 SAE-NCM 模型。

    Args:
        device: torch.device
        checkpoint: ConvNeXt checkpoint path
        sae_ckpt: SAE checkpoint path
        class_npz_path: sae_stat_results_v2.npz path

    Returns:
        model: ConvNeXt_SAE_NCM (eval mode, on device)
        class_names: list of 1000 class names
    """
    # Backbone
    convnext = build_convnext_backbone(device, checkpoint)

    # SAE
    sae, norm_mean, norm_std, cfg = build_sae(device, sae_ckpt)
    print(f"SAE loaded: d_in={cfg['d_in']}, d_lat={cfg['d_lat']}, k={cfg['k']}")

    # Class statistics -> dense
    cls_data = np.load(class_npz_path, allow_pickle=True)
    cls_names = list(cls_data["class_names"])
    cls_indices = cls_data["spatial_indices"]          # [1000, 49, 64]
    cls_activations = cls_data["spatial_activations"]  # [1000, 49, 64]

    print(f"Building dense class means: {len(cls_names)} classes x 49 positions x 65536 dims ...")
    C, P, k = cls_indices.shape
    D = cfg["d_lat"]
    cls_dense = torch.zeros(C, P, D, dtype=torch.float32)
    for c in range(C):
        for p in range(P):
            idx = torch.from_numpy(cls_indices[c, p]).long()
            val = torch.from_numpy(cls_activations[c, p]).float()
            cls_dense[c, p, idx] = val

    # 先转 FP16 再搬 GPU，避免 FP32 + FP16 双副本同时占用显存
    cls_dense = cls_dense.half().to(device)
    mem_gb = cls_dense.element_size() * cls_dense.nelement() / (1024 ** 3)
    print(f"Class dense tensor: {tuple(cls_dense.shape)}, {mem_gb:.2f} GB on {device}")

    # Build model
    model = ConvNeXt_SAE_NCM(convnext, sae, norm_mean, norm_std, cls_dense)
    model = model.to(device)
    model.eval()

    return model, cls_names


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = str(REPO_ROOT / "checkpoints" / "model_bestfid.pth")
    sae_ckpt = str(REPO_ROOT / "SAE" / "project" / "checkpoints" / "k64" / "sae_64k_k64_step_50000.pt")
    class_npz = str(Path(__file__).resolve().parent.parent / "align_steering" / "sae_stat_results_v2.npz")

    model, class_names = build_ncm_model(device, checkpoint, sae_ckpt, class_npz)

    x = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        scores = model(x)
    print(f"\nTest forward:")
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(scores.shape)}")
    print(f"  Preds:  {[class_names[i] for i in scores.argmax(dim=1).tolist()]}")
