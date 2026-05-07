"""
Generate PGD adversarial examples for WRN34x10-CIFAR10.
Uses the same PGD attack as DAT training (eps=0.5, L2 norm).
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.models.wide_resnet_innoutrobustness import WideResNet34x10
from InNOutRobustness.utils.adversarial_attacks.pgd import PGD

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_cifar10_test_loader(batch_size=128, data_root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


def generate_adversarial(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = WideResNet34x10(num_classes=10, normalize_input=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Setup PGD attack: eps=0.5 (L2), matching DAT config
    # Standard PGD-50 with step size = 2*eps/iterations for L2
    attack = PGD(
        eps=args.eps,
        iterations=args.iterations,
        stepsize=args.stepsize,
        num_classes=10,
        momentum=0.9,
        norm='2',  # L2 norm
        loss='CrossEntropy',
        normalize_grad=True,
        restarts=args.restarts,
    )
    attack.set_model(model)

    # Load data
    loader = get_cifar10_test_loader(batch_size=args.batch_size, data_root=args.data_root)

    all_clean = []
    all_adv = []
    all_labels = []
    all_pred_clean = []
    all_pred_adv = []

    os.makedirs(args.out_dir, exist_ok=True)

    pbar = tqdm(loader, desc="Generating adversarial examples")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits_clean = model(images)
            pred_clean = logits_clean.argmax(dim=1)

        # Generate adversarial examples (untargeted)
        adv_images, _, _ = attack.perturb_inner(images, labels, targeted=False)

        with torch.no_grad():
            logits_adv = model(adv_images)
            pred_adv = logits_adv.argmax(dim=1)

        # Move to CPU for storage
        all_clean.append(images.cpu())
        all_adv.append(adv_images.cpu())
        all_labels.append(labels.cpu())
        all_pred_clean.append(pred_clean.cpu())
        all_pred_adv.append(pred_adv.cpu())

        # Running stats
        batch_clean_acc = (pred_clean == labels).float().mean().item()
        batch_adv_acc = (pred_adv == labels).float().mean().item()
        pbar.set_postfix({
            "clean_acc": f"{batch_clean_acc:.3f}",
            "adv_acc": f"{batch_adv_acc:.3f}",
        })

    # Concatenate all batches
    all_clean = torch.cat(all_clean, dim=0)       # [N, 3, 32, 32]
    all_adv = torch.cat(all_adv, dim=0)           # [N, 3, 32, 32]
    all_labels = torch.cat(all_labels, dim=0)     # [N]
    all_pred_clean = torch.cat(all_pred_clean, dim=0)
    all_pred_adv = torch.cat(all_pred_adv, dim=0)

    # Save
    save_path = os.path.join(args.out_dir, f"pgd_L2_eps{args.eps}_iter{args.iterations}.pt")
    torch.save({
        "clean_images": all_clean,
        "adv_images": all_adv,
        "labels": all_labels,
        "pred_clean": all_pred_clean,
        "pred_adv": all_pred_adv,
        "config": {
            "eps": args.eps,
            "iterations": args.iterations,
            "stepsize": args.stepsize,
            "restarts": args.restarts,
            "norm": "L2",
            "model": args.ckpt_path,
        }
    }, save_path)

    # Summary
    total = len(all_labels)
    clean_acc = (all_pred_clean == all_labels).float().mean().item()
    adv_acc = (all_pred_adv == all_labels).float().mean().item()
    attack_success_rate = 1.0 - adv_acc

    print(f"\n{'='*50}")
    print(f"Adversarial examples saved to: {save_path}")
    print(f"Total samples: {total}")
    print(f"Clean accuracy:  {clean_acc*100:.2f}%")
    print(f"Adv accuracy:    {adv_acc*100:.2f}%")
    print(f"Attack success:  {attack_success_rate*100:.2f}%")
    print(f"{'='*50}")

    # Also save a small visualization grid
    grid_path = os.path.join(args.out_dir, f"pgd_L2_eps{args.eps}_visual.png")
    save_visualization(all_clean, all_adv, all_labels, all_pred_clean, all_pred_adv, grid_path)
    print(f"Visualization saved to: {grid_path}")


def save_visualization(clean, adv, labels, pred_clean, pred_adv, path, n_samples=10):
    """Save a grid showing clean vs adversarial images."""
    import torchvision.utils as vutils
    import numpy as np

    # Pick n_samples with successful attacks
    success_mask = pred_clean == labels
    success_mask &= pred_adv != labels
    success_idx = torch.where(success_mask)[0]

    if len(success_idx) < n_samples:
        # Fallback: just pick first n_samples
        success_idx = torch.arange(n_samples)

    selected = success_idx[:n_samples]

    rows = []
    for i in selected:
        rows.append(clean[i])
        rows.append(adv[i])

    grid = vutils.make_grid(torch.stack(rows), nrow=2, padding=2, normalize=True, value_range=(0, 1))
    vutils.save_image(grid, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/Data_share/hongyi/DAT/checkpoints/cifar10-WRN3410-T40model_bestfid.pth")
    parser.add_argument("--data_root", type=str, default="/Data_share/hongyi/DAT/data")
    parser.add_argument("--out_dir", type=str, default="/Data_share/hongyi/DAT/SAE_WRN/attacks")
    parser.add_argument("--eps", type=float, default=0.5, help="L2 epsilon")
    parser.add_argument("--iterations", type=int, default=50, help="PGD iterations")
    parser.add_argument("--stepsize", type=float, default=0.02, help="PGD step size (default: 2*eps/iter)")
    parser.add_argument("--restarts", type=int, default=1, help="Number of random restarts")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if args.stepsize is None:
        args.stepsize = 2.0 * args.eps / args.iterations

    generate_adversarial(args)
