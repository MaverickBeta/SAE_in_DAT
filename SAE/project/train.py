import argparse
import csv
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from sae_core.model import TopKAutoencoder
from data.dataset import SAEFeatureLoader


STAGE_CHANNELS = {
    0: 192,
    1: 384,
    2: 768,
    3: 1536,
}


def resolve_train_config(args):
    if args.stage_idx not in STAGE_CHANNELS:
        raise ValueError(f"Unsupported stage_idx={args.stage_idx}, expected one of {sorted(STAGE_CHANNELS.keys())}")

    resolved_d_in = STAGE_CHANNELS[args.stage_idx]
    if args.d_in is not None and args.d_in != resolved_d_in:
        print(
            f"[Warn] Overriding provided d_in={args.d_in} with stage-based d_in={resolved_d_in} "
            f"for stage_idx={args.stage_idx}."
        )

    resolved_d_lat = resolved_d_in * args.expansion_rate

    if args.data_path is None:
        stage_dir = f"/Data_share/hongyi/DAT/data_SAE/imagenet_small_features/stage{args.stage_idx}"
        if os.path.isdir(stage_dir):
            resolved_data_path = stage_dir
        else:
            if args.stage_idx == 3:
                resolved_data_path = "/Data_share/hongyi/DAT/data_SAE/imagenet_small_features_merged.pt"
            else:
                resolved_data_path = f"/Data_share/hongyi/DAT/data_SAE/imagenet_small_features_stage{args.stage_idx}_merged.pt"
    else:
        resolved_data_path = args.data_path

    return resolved_data_path, resolved_d_in, resolved_d_lat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_aux_coeff(step, total_steps, warmup_steps=2000, finetune_start=None):
    if finetune_start is None:
        finetune_start = int(total_steps * 0.8)
    if step < warmup_steps:
        return 0.3
    elif step < finetune_start:
        return 0.1
    else:
        return 0.01


def resample_dead_neurons(model, optimizer, is_dead, activation_counts, noise_scale=0.20, max_ratio=0.50):
    dead_indices = torch.where(is_dead)[0]
    alive_indices = torch.where(~is_dead)[0]
    d_lat = is_dead.shape[0]

    n_dead = len(dead_indices)
    if n_dead == 0 or len(alive_indices) == 0:
        return 0

    # 限制单次 resample 比例，避免同质化灾难
    max_dead = int(d_lat * max_ratio)
    if n_dead > max_dead:
        perm = torch.randperm(n_dead, device=dead_indices.device)
        dead_indices = dead_indices[perm[:max_dead]]
        n_dead = len(dead_indices)

    probs = activation_counts[alive_indices].float()
    if probs.sum() < 1e-6:
        # activation_counts may have been zeroed; fall back to uniform
        probs = torch.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()

    sampled_idx = torch.multinomial(probs, n_dead, replacement=True)
    sampled_alive = alive_indices[sampled_idx]

    with torch.no_grad():
        model.W_enc[:, dead_indices] = model.W_enc[:, sampled_alive].clone()
        model.W_dec[dead_indices, :] = model.W_dec[sampled_alive, :].clone()

        model.W_enc[:, dead_indices] += (
            torch.randn_like(model.W_enc[:, dead_indices]) * noise_scale
        )
        model.W_dec[dead_indices, :] += (
            torch.randn_like(model.W_dec[dead_indices, :]) * noise_scale
        )

        model.b_enc[dead_indices] = 0.0
        model.set_decoder_norm_to_unit_norm()

    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            state = optimizer.state.get(p, {})
            if 'exp_avg' not in state or 'exp_avg_sq' not in state:
                continue

            if p is model.W_enc:
                state['exp_avg'][:, dead_indices] = 0.0
                state['exp_avg_sq'][:, dead_indices] = 0.0
            elif p is model.W_dec:
                state['exp_avg'][dead_indices, :] = 0.0
                state['exp_avg_sq'][dead_indices, :] = 0.0
            elif p is model.b_enc:
                state['exp_avg'][dead_indices] = 0.0
                state['exp_avg_sq'][dead_indices] = 0.0

    return n_dead


def save_checkpoint(path, model_state_dict, config, stage_idx, d_in, d_lat, step, norm_mean, norm_std):
    torch.save({
        'model_state_dict': model_state_dict,
        'config': config,
        'stage_idx': stage_idx,
        'd_in': d_in,
        'd_lat': d_lat,
        'step': step,
        'norm_mean': norm_mean.cpu(),
        'norm_std': norm_std.cpu(),
    }, path)


def train(args):
    set_seed(42)

    data_path, d_in, d_lat = resolve_train_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (mapped to physical GPU {args.gpu_id})")

    save_dir = os.path.join(
        "checkpoints",
        f"stage{args.stage_idx}",
        f"k{args.k}_exp{args.expansion_rate}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"🚀 Training TopK SAE | stage={args.stage_idx} | d_in={d_in} | "
        f"d_lat={d_lat} (exp={args.expansion_rate}) | k={args.k}"
    )
    print(f"Using data: {data_path}")
    print(f"Save dir: {save_dir}")
    print(f"Resample every: {args.resample_every} steps")
    print(f"Noise scale: {args.resample_noise_scale}")

    csv_path = os.path.join(save_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "loss", "mse_loss", "aux_loss", "lr",
        "dead_fraction", "l0", "explained_variance", "aux_coeff"
    ])
    print(f"CSV log: {csv_path}")

    loader = SAEFeatureLoader(
        data_path=data_path,
        batch_size=args.batch_size,
        device='cpu',
        files_per_batch=args.files_per_batch,
        cache_size=args.cache_size,
        hot_file_pool_size=args.hot_file_pool_size,
        pool_refresh_every=args.pool_refresh_every,
        preload=True,
    )

    if args.stats_batches is None:
        stats_batches = 200
    else:
        stats_batches = args.stats_batches

    if stats_batches <= 0:
        raise ValueError(f"stats_batches must be > 0, got {stats_batches}")

    print(f"Computing global normalization stats... mode={loader.mode}, stats_batches={stats_batches}")
    all_feats = []
    for _ in tqdm(range(stats_batches), desc="Loading stats"):
        all_feats.append(loader.get_batch())
    all_feats = torch.cat(all_feats).to(device)
    global_mean = all_feats.mean(dim=0, keepdim=True)
    global_std = all_feats.std(dim=0, keepdim=True) + 1e-6
    del all_feats
    torch.cuda.empty_cache()
    print("✅ Stats computed.")

    model = TopKAutoencoder(d_in=d_in, d_lat=d_lat, k=args.k).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)

    dead_neuron_window = args.dead_window
    activation_counts = torch.zeros(d_lat, device=device)

    best_loss = float('inf')
    best_ev = -float('inf')
    best_score = -float('inf')

    pbar = tqdm(range(args.steps), desc="Training SAE")

    for step in pbar:
        current_aux_coeff = get_aux_coeff(step, args.steps)

        x = loader.get_batch().to(device)
        x_norm = (x - global_mean) / global_std

        x_reconstruct, z, post_relu_acts = model(x_norm)

        mse_loss = F.mse_loss(x_reconstruct, x_norm)

        residual = x_norm - x_reconstruct.detach()
        did_fire = (z > 0).float().sum(dim=0)
        activation_counts += did_fire

        if step % dead_neuron_window == 0 and step > 0:
            is_dead = activation_counts < (0.001 * args.batch_size * dead_neuron_window)

            # 先不清零，等 backward + optimizer.step 之后再 resample 并清零
            pass
        elif step < dead_neuron_window:
            is_dead = torch.zeros(d_lat, dtype=torch.bool, device=device)
        else:
            # 在两次 window 之间，沿用上次计算的 is_dead
            pass

        aux_loss = 0.0
        if is_dead.sum() > 0:
            dead_acts = x_norm @ model.W_enc[:, is_dead] + model.b_enc[is_dead]
            dead_acts = F.relu(dead_acts)
            dead_dec = model.W_dec[is_dead, :].detach()
            aux_recon = dead_acts @ dead_dec
            aux_loss = F.mse_loss(aux_recon, residual)

        loss = mse_loss + current_aux_coeff * aux_loss

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = model.W_dec.grad
            if grad is not None:
                proj = (grad * model.W_dec.data).sum(dim=1, keepdim=True)
                grad.sub_(proj * model.W_dec.data / (model.W_dec.data.norm(dim=1, keepdim=True)**2 + 1e-8))

        model.set_decoder_norm_to_unit_norm()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            current_dead_fraction = is_dead.float().mean().item()
            l0_norm = (z > 0).float().sum(-1).mean().item()

            with torch.no_grad():
                total_var = torch.var(x_norm, dim=0).sum()
                residual_var = F.mse_loss(x_reconstruct, x_norm, reduction='sum') / x_norm.size(0)
                explained_var = 1 - (residual_var / total_var).item()

            aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
            lr_now = scheduler.get_last_lr()[0]

            csv_writer.writerow([
                step,
                f"{loss.item():.6f}",
                f"{mse_loss.item():.6f}",
                f"{aux_loss_val:.6f}",
                f"{lr_now:.8f}",
                f"{current_dead_fraction:.6f}",
                f"{l0_norm:.2f}",
                f"{explained_var:.6f}",
                f"{current_aux_coeff:.4f}",
            ])
            csv_file.flush()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MSE": f"{mse_loss.item():.4f}",
                "Dead": f"{current_dead_fraction:.2%}",
                "EV": f"{explained_var:.3f}",
                "Aux": f"{current_aux_coeff:.2f}",
            })

            if loss.item() < best_loss:
                best_loss = loss.item()
                path = os.path.join(save_dir, "best_loss.pt")
                save_checkpoint(
                    path, model.state_dict(), vars(args), args.stage_idx,
                    d_in, d_lat, step + 1, global_mean, global_std
                )

            if explained_var > best_ev:
                best_ev = explained_var
                path = os.path.join(save_dir, "best_ev.pt")
                save_checkpoint(
                    path, model.state_dict(), vars(args), args.stage_idx,
                    d_in, d_lat, step + 1, global_mean, global_std
                )

            score = explained_var * (1.0 - current_dead_fraction)
            if score > best_score:
                best_score = score
                path = os.path.join(save_dir, "best_composite.pt")
                save_checkpoint(
                    path, model.state_dict(), vars(args), args.stage_idx,
                    d_in, d_lat, step + 1, global_mean, global_std
                )

        # 8. Neuron Resampling: 必须在 backward + optimizer.step 之后执行，
        # 否则 in-place 修改 W_enc/W_dec 会导致 autograd 版本不匹配。
        if step % dead_neuron_window == 0 and step > 0:
            if args.resample_every > 0 and step % args.resample_every == 0:
                n_resampled = resample_dead_neurons(
                    model, optimizer, is_dead, activation_counts,
                    noise_scale=args.resample_noise_scale,
                    max_ratio=args.max_resample_ratio
                )
                print(f"\n[Resample @ step {step}] {n_resampled} / {d_lat} neurons resampled ({n_resampled/d_lat:.2%})")
            activation_counts.zero_()
        elif args.resample_every > 0 and step > 0 and step % args.resample_every == 0:
            # 非 dead_window 边界时的 resample
            window = min(dead_neuron_window, step)
            tmp_is_dead = activation_counts < (0.001 * args.batch_size * window)
            n_resampled = resample_dead_neurons(
                model, optimizer, tmp_is_dead, activation_counts,
                noise_scale=args.resample_noise_scale,
                max_ratio=args.max_resample_ratio
            )
            print(f"\n[Resample @ step {step}] {n_resampled} / {d_lat} neurons resampled ({n_resampled/d_lat:.2%})")
            activation_counts.zero_()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            path = os.path.join(
                save_dir,
                f"sae_stage{args.stage_idx}_din{d_in}_exp{args.expansion_rate}_k{args.k}_step_{step+1}.pt"
            )
            save_checkpoint(
                path, model.state_dict(), vars(args), args.stage_idx,
                d_in, d_lat, step + 1, global_mean, global_std
            )

    print("✅ Training Finished!")
    csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_idx", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--d_in", type=int, default=None)
    parser.add_argument("--expansion_rate", type=int, default=32)

    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux_coeff", type=float, default=0.1, help="Base aux_coeff; actual value follows staged schedule")
    parser.add_argument("--dead_window", type=int, default=2500)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--no_wandb", action="store_true", help="Kept for backward compat; wandb is now disabled by default")
    parser.add_argument("--stats_batches", type=int, default=None, help="Batches used for global normalization stats. Default: 200")
    parser.add_argument("--files_per_batch", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=4)
    parser.add_argument("--hot_file_pool_size", type=int, default=16)
    parser.add_argument("--pool_refresh_every", type=int, default=100)

    parser.add_argument("--resample_every", type=int, default=5000, help="Resample dead neurons every N steps. 0 to disable.")
    parser.add_argument("--resample_noise_scale", type=float, default=0.20)
    parser.add_argument("--max_resample_ratio", type=float, default=0.50, help="Max fraction of dead neurons to resample at once")

    args = parser.parse_args()
    train(args)
