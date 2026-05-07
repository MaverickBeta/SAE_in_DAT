import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
CHECKPOINT_ROOT = "checkpoints/stage3"
SAVE_DIRS = []  # populated automatically from subdirs containing training_log.csv
STEPS_PER_EPOCH = 1749.95556640625  # computed from dataset: 146282 images * 49 vectors / 4096 bs

# Stage boundaries for background shading (based on 50k steps default)
WARMUP_END = 2000
FINETUNE_START = 40000  # 0.8 * 50000

# Colors and styles for each experiment
STYLE_MAP = {
    "k64_exp16":  {"color": "#1f77b4", "linestyle": "-",  "label": "k64_exp16"},
    "k64_exp32":  {"color": "#d62728", "linestyle": "--", "label": "k64_exp32 (crashed)"},
    "k256_exp8":  {"color": "#2ca02c", "linestyle": "-",  "label": "k256_exp8"},
}

def add_epoch_axis(ax, steps_per_epoch):
    """Add a secondary x-axis on top showing approximate epoch."""
    secax = ax.secondary_xaxis('top', functions=(lambda s: s / steps_per_epoch, lambda e: e * steps_per_epoch))
    secax.set_xlabel('Approx. Epoch')
    secax.tick_params(axis='x')
    return secax

def add_stage_background(ax, max_step):
    """Add light background shading for warmup / main / finetune phases."""
    finetune_start = int(max_step * 0.8)
    ax.axvspan(0, WARMUP_END, color='#FFF3CD', alpha=0.4, label='_nolegend_')
    ax.axvspan(WARMUP_END, finetune_start, color='#D1ECF1', alpha=0.3, label='_nolegend_')
    ax.axvspan(finetune_start, max_step, color='#D4EDDA', alpha=0.4, label='_nolegend_')
    # Text annotations removed
    pass

def add_resample_vlines(ax, max_step):
    """Add vertical dashed lines at resample intervals (every 5000 steps)."""
    for step in range(5000, int(max_step) + 1, 5000):
        ax.axvline(step, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='_nolegend_')

def plot_all(experiments, save_paths):
    """
    experiments: dict {name: df}
    save_paths: list of directories to save the figures into
    """
    max_step = max(df['step'].max() for df in experiments.values())
    
    # Determine global y-limits for loss (clip crazy values)
    loss_vals = []
    for name, df in experiments.items():
        if name == "k64_exp32":
            # For crashed run, only take values before obvious explosion for y-limit calc
            loss_vals.extend(df.loc[df['step'] <= 5000, 'loss'].tolist())
        else:
            loss_vals.extend(df['loss'].tolist())
    loss_ymax = min(np.percentile(loss_vals, 99) * 1.2, 2.0) if loss_vals else 1.0
    loss_ymax = max(loss_ymax, 1.0)

    # ---------- Figure 1: Loss ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    add_stage_background(ax, max_step)
    
    for name, df in experiments.items():
        style = STYLE_MAP.get(name, {"color": "black", "linestyle": "-", "label": name})
        ax.plot(df['step'], df['loss'], **style, linewidth=1.5, alpha=0.9)
        
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, loss_ymax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (total = MSE + aux_coeff × aux)')
    ax.set_title('SAE Training Loss vs Step')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    add_epoch_axis(ax, STEPS_PER_EPOCH)
    plt.tight_layout()
    for sp in save_paths:
        fig.savefig(os.path.join(sp, 'training_loss.png'), dpi=200)
    plt.close(fig)

    # ---------- Figure 2: Explained Variance ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    add_stage_background(ax, max_step)
    
    for name, df in experiments.items():
        style = STYLE_MAP.get(name, {"color": "black", "linestyle": "-", "label": name})
        # Clip negative EV to 0 for visual clarity on the main plot
        ev = df['explained_variance'].clip(lower=0)
        ax.plot(df['step'], ev, **style, linewidth=1.5, alpha=0.9)
        
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Explained Variance')
    ax.set_title('SAE Explained Variance vs Step')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    add_epoch_axis(ax, STEPS_PER_EPOCH)
    plt.tight_layout()
    for sp in save_paths:
        fig.savefig(os.path.join(sp, 'training_ev.png'), dpi=200)
    plt.close(fig)

    # ---------- Figure 3: Dead Fraction ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    add_stage_background(ax, max_step)
    add_resample_vlines(ax, max_step)
    
    for name, df in experiments.items():
        style = STYLE_MAP.get(name, {"color": "black", "linestyle": "-", "label": name})
        ax.plot(df['step'], df['dead_fraction'], **style, linewidth=1.5, alpha=0.9)
    
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Step')
    ax.set_ylabel('Dead Fraction')
    ax.set_title('SAE Dead Fraction vs Step (dotted lines = resample events)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    add_epoch_axis(ax, STEPS_PER_EPOCH)
    plt.tight_layout()
    for sp in save_paths:
        fig.savefig(os.path.join(sp, 'training_dead_fraction.png'), dpi=200)
    plt.close(fig)

    print(f"✅ Figures saved to: {', '.join(save_paths)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, CHECKPOINT_ROOT)
    
    # Find all subdirectories with training_log.csv
    subdirs = sorted(glob.glob(os.path.join(root, "*")))
    experiments = {}
    save_paths = []
    
    for d in subdirs:
        if os.path.isdir(d):
            csv_path = os.path.join(d, "training_log.csv")
            if os.path.exists(csv_path):
                name = os.path.basename(d)
                df = pd.read_csv(csv_path)
                experiments[name] = df
                save_paths.append(d)
                print(f"Loaded {name}: {len(df)} rows, steps {df['step'].min():.0f}-{df['step'].max():.0f}")
    
    if not experiments:
        print("No training_log.csv files found.")
        return
    
    # Also save to the stage3 root directory
    save_paths.append(root)
    
    plot_all(experiments, save_paths)

if __name__ == "__main__":
    main()
