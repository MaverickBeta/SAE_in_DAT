#!/usr/bin/env python3
import argparse
import math
import os
import sys
import gc
import inspect
from pathlib import Path

# Resolve project roots robustly from this file location.
FILE_DIR = Path(__file__).resolve().parent
DAT_ROOT = FILE_DIR
while DAT_ROOT.name != "DAT" and DAT_ROOT.parent != DAT_ROOT:
    DAT_ROOT = DAT_ROOT.parent
if DAT_ROOT.name != "DAT":
    raise RuntimeError(f"Unable to locate DAT root from {FILE_DIR}")

# Keep DAT imports and local timm fork deterministic across launch locations.
sys.path.insert(0, str(DAT_ROOT))
sys.path.insert(0, str(DAT_ROOT / "pytorch-image-models"))
sys.path.insert(0, str(DAT_ROOT / "SAE" / "project"))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.utils.checkpoint as checkpoint

# 导入你原本的库与评估逻辑 (来自于 DAT 层级)
from evaluate_imagenet_robustbench import load_custom_model, get_preprocessing_function
import robustbench.data
import robustbench.eval
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy
try:
    from robustbench.eval import benchmark as rb_benchmark
except Exception:
    rb_benchmark = None

# 导入你现在的 SAE (来自于 SAE/project/ 层级)
from sae_core.model import TopKAutoencoder


class ConvNeXtWithSAE(nn.Module):
    """
    带有梯度检查点的混合网络。
    这是解决对抗攻击时长计算图导致巨量 OOM 的终极手段。
    """
    def __init__(self, base_model, sae, norm_mean, norm_std):
        super().__init__()
        self.base_model = base_model
        self.sae = sae
        self.register_buffer('norm_mean', norm_mean)
        self.register_buffer('norm_std', norm_std)
        # 清除 sae 的所有 requires_grad，防止梯度累积垃圾
        for p in self.sae.parameters():
            p.requires_grad = False

    def _sae_forward_wrapper(self, features_flat):
        """这部分将被 Checkpoint 包裹以释放庞大的内部图显存"""
        x_norm = (features_flat - self.norm_mean) / self.norm_std
        x_reconstruct, _, _ = self.sae(x_norm)
        features_recon = x_reconstruct * self.norm_std + self.norm_mean
        return features_recon

    def forward(self, x):
        # 1. 基础特征提取
        features = self.base_model.forward_features(x)
        B, C, H, W = features.shape

        # 2. 拉平
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)

        # ⭐️ 核心防御：梯度检查点 ⭐️
        if torch.is_grad_enabled() and features_flat.requires_grad:
            # use_reentrant=False 非重入模式，行为：前向时丢弃SAE中间激活，反向时重新计算SAE前向
            features_recon_flat = checkpoint.checkpoint(self._sae_forward_wrapper, features_flat, use_reentrant=False)
        else:
            features_recon_flat = self._sae_forward_wrapper(features_flat)

        # 3. 折叠回去并过最后的分类头
        features_recon = features_recon_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.base_model.forward_head(features_recon)
        return out


def fallback_benchmark(
    model,
    dataset,
    threat_model,
    eps,
    n_examples,
    data_dir,
    batch_size,
    device,
    preprocessing,
):
    """Fallback path that mirrors evaluate_imagenet_robustbench.py behavior."""
    if dataset != BenchmarkDataset.imagenet:
        raise NotImplementedError("Fallback benchmark currently supports ImageNet only")
    if threat_model not in {ThreatModel.L2, ThreatModel.Linf}:
        raise NotImplementedError("Fallback benchmark currently supports L2/Linf only")

    load_imagenet = robustbench.data.load_imagenet
    sig = inspect.signature(load_imagenet)
    kwargs = {
        "n_examples": n_examples,
        "data_dir": data_dir,
    }
    if "prepr" in sig.parameters:
        kwargs["prepr"] = preprocessing
    elif "preprocessing" in sig.parameters:
        kwargs["preprocessing"] = preprocessing
    elif "transform" in sig.parameters:
        kwargs["transform"] = preprocessing

    x_test, y_test = load_imagenet(**kwargs)

    clean_acc = clean_accuracy(
        model,
        x_test,
        y_test,
        batch_size=batch_size,
        device=device,
    )

    from autoattack import AutoAttack

    adversary = AutoAttack(
        model,
        norm=threat_model.value,
        eps=eps,
        version="standard",
        device=device,
    )
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    robust_acc = clean_accuracy(
        model,
        x_adv,
        y_test,
        batch_size=batch_size,
        device=device,
    )
    return clean_acc, robust_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt with or without SAE on RobustBench")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Base ConvNeXt checkpoint")
    parser.add_argument("--data_dir", type=str, default="../data/ImageNet", help="Path to ImageNet val dataset")
    parser.add_argument("--threat_model", type=str, default="L2", choices=["L2", "Linf", "corruptions"])
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--n_examples", type=int, default=5000, help="Evaluate on limited subset to save time")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=256)  # 把默认值修正为 256
    parser.add_argument("--sae_ckpt", type=str, default="none", help="Path to SAE. 'none' for vanilla.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # 提取多卡环境参数
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))

    use_sae = args.sae_ckpt.lower() != "none"
    model_mode_str = "SAE WRAPPED ConvNeXt" if use_sae else "VANILLA ConvNeXt (No SAE)"

    print(f"==> 1. Loading Base ConvNeXt from: {args.checkpoint}")
    base_model = load_custom_model(args.checkpoint, architecture="convnext_large")
    base_model.eval().to(device)

    # Keep evaluation assumptions aligned with evaluate_imagenet_robustbench.py.
    assert hasattr(base_model, 'normalize_input') and base_model.normalize_input == False, \
        f"convnext_large should have normalize_input=False (found: {getattr(base_model, 'normalize_input', 'N/A')})"

    if use_sae:
        print(f"==> 2. SAE MODE ENABLED. Loading SAE from: {args.sae_ckpt}")
        sae_ckpt = torch.load(args.sae_ckpt, map_location=device, weights_only=False)
        config = sae_ckpt['config']
        sae = TopKAutoencoder(d_in=config['d_in'], d_lat=config['d_lat'], k=config['k'])
        sae.load_state_dict(sae_ckpt['model_state_dict'])    
        sae.eval().to(device)
        
        norm_mean = sae_ckpt['norm_mean'].to(device)
        norm_std = sae_ckpt['norm_std'].to(device)

        print("==> 3. Creating Wrapped Hybrid Model with Gradient Checkpointing...")
        model = ConvNeXtWithSAE(base_model, sae, norm_mean, norm_std)
        model_info_str = f"({config['k']}k, k={config['k']})"
    else:
        print("==> 2. VANILLA MODE ENABLED.")
        model = base_model
        model_info_str = "(Original Checkpoint)"
    
    # 【非常重要】：如果是在外部大循环跑单机多卡 (且没有开启多进程切割)，则启用 DP。否则坚决只用单卡
    if num_gpus > 1 and world_size == 1:
        print(f"==> Using default DataParallel on {num_gpus} GPUs. (Warning: Could bottleneck with Checkpoints)")
        model = torch.nn.DataParallel(model)
    elif world_size > 1:
        print(f"==> [Rank {rank}] Multi-process chunking mode active! Disabling DataParallel, sticking to specific single GPU.")
    
    # === 统一强制所有层进入 eval() ===
    model.eval()
    for module in model.modules():
        module.eval()
    for param in model.parameters():
        param.requires_grad = False

    preprocessing = get_preprocessing_function(args.img_size)

    # ==========================
    # 4. 计算 Clean Accuracy (带数据集切块逻辑)
    # ==========================
    print(f"\n==> 4. [Rank {rank}] Computing Clean Accuracy - Mode: {model_mode_str} ...")
    val_dir = os.path.join(args.data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transform=preprocessing)
    
    # 限制最大样本数 (前 5000，改为随机抽取或均匀抽取保证囊括 1000 个类)
    import random
    if args.n_examples > 0 and args.n_examples < len(val_dataset):
        # 为了复现出真实分布，随机抽 5000 个索引并固定随机种子
        random.seed(42)
        indices = random.sample(range(len(val_dataset)), args.n_examples)
        val_dataset = Subset(val_dataset, indices)
        
    # 如果是多卡分发，只取自己该算的那一块
    if world_size > 1:
        chunk_size = math.ceil(len(val_dataset) / world_size)
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(start_idx, end_idx)))

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    clean_acc = correct / max(total, 1)
    print(f"\n[Rank {rank} Result] Clean Accuracy ({total} partial images): {clean_acc:.2%}")

    # ==========================
    # 🌟 垃圾清运环节
    # ==========================
    try:
        del images, labels, outputs
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================
    # 🌟 伪装属性骗过检测：强制接管数据集加载
    # ==========================
    # RobustBench 会试图读取 dataset 的 name 枚举。我们给这个已经切好的 Subset 手动赋个名。
    setattr(val_dataset, 'name', BenchmarkDataset.imagenet)

    # ==========================
    # 5. 启动攻击测评 (优先 RobustBench benchmark，失败再回退 AutoAttack)
    # ==========================
    print(f"\n==> 5. [Rank {rank}] Running adversarial evaluation ...")

    try:
        threat_model = ThreatModel(args.threat_model)
        samples_evaluated = args.n_examples
        if rb_benchmark is not None:
            _, robust_acc = rb_benchmark(
                model=model,
                dataset=BenchmarkDataset.imagenet,
                threat_model=threat_model,
                eps=args.eps,
                n_examples=args.n_examples,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                device=device,
                preprocessing=preprocessing,
            )
            clean_acc_local = clean_acc
        else:
            print("robustbench.eval.benchmark is unavailable; using fallback AutoAttack path.")
            clean_acc_local, robust_acc = fallback_benchmark(
                model=model,
                dataset=BenchmarkDataset.imagenet,
                threat_model=threat_model,
                eps=args.eps,
                n_examples=args.n_examples,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                device=device,
                preprocessing=preprocessing,
            )

        print("\n" + "=" * 60)
        print(f"[Rank {rank}] FINAL EVAL RESULT: {model_mode_str} {model_info_str}")
        print("=" * 60)
        print(f"Clean accuracy: {clean_acc_local:.4f}")
        print(f"Robust accuracy ({args.threat_model}, eps={args.eps}): {robust_acc:.4f}")
        print(f"Samples evaluated: {samples_evaluated}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FATAL ERROR CAUGHT on Rank {rank}] Type: {type(e).__name__}, Message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()