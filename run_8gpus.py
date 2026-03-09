#!/usr/bin/env python3
"""
Launch 8 concurrent AutoAttack evaluation jobs on 8 GPUs without DataParallel bottleneck.
Usage: python run_8gpus.py model_configs/imagenet-dat-ConvNeXtLarge-convst-256x256.yaml -bs 32
"""
import argparse
import subprocess
import os
import sys
import re
from pathlib import Path

# Add DAT dir to path so we can import eval_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import load_yaml_config

def main():
    parser = argparse.ArgumentParser(description="Launch multi-GPU adversarial evaluation safely")
    parser.add_argument("config_file", help="Path to config")
    parser.add_argument("-bs", "--batch-size", default=32, type=int, 
                        help="Batch size PER GPU (Wait! AutoAttack stores adversarial tensors in VRAM. Keep this around 32-64 for 24G cards like RTX 3090!)")
    parser.add_argument("-g", "--gpus", default=8, type=int, help="Number of GPUs to use")
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    
    # Format the model type.
    model_type_mapping = {
        "ResNet50ImageNet": "resnet50",
        "WideResNet50x4ImageNet": "wide_resnet50_4",
        "convnext_large": "convnext_large",
    }
    architecture = model_type_mapping.get(config["model_type"], config["model_type"])

    threat_model = config.get("threat_model", "L2")
    pgd_epsilon = str(config["pgd_epsilon"])
    
    cmd_base = [
        sys.executable, "-u", "evaluate_imagenet_robustbench.py",
        "--checkpoint", config["checkpoint"],
        "--data_dir", "./data/ImageNet",
        "--threat_model", threat_model,
        "--eps", pgd_epsilon,
        "--n_examples", "5000",
        "--batch_size", str(args.batch_size),
        "--architecture", architecture
    ]
    
    if "image_size" in config:
        cmd_base.extend(["--img_size", str(config["image_size"])])
        
    log_dir = "slurm_log/eval_acc_8gpu"
    os.makedirs(log_dir, exist_ok=True)
    basename = Path(args.config_file).stem
    
    processes = []
    
    print(f"==================================================")
    print(f"Starting {args.gpus}-GPU parallel data splitting...")
    print(f"Model: {architecture} | Batch Size per GPU: {args.batch_size}")
    print(f"Please check individual progress at: {log_dir}")
    print(f"==================================================\n")

    for rank in range(args.gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(args.gpus)
        
        log_file = os.path.join(log_dir, f"{basename}_gpu{rank}.log")
        # Ensure we write out log locally
        f_out = open(log_file, "w")
        
        print(f"--> Launching chunk {rank + 1} / {args.gpus} on GPU_ID={rank} | Logging to: {log_file}")
        p = subprocess.Popen(cmd_base, env=env, stdout=f_out, stderr=subprocess.STDOUT)
        processes.append((p, f_out, rank, log_file))
        
    print(f"\n[INFO] All {args.gpus} independent processes have been dispatched into the background!")
    print(f"[INFO] 1. Your 'nvidia-smi' will slowly climb to 100% per card and 350W power draw.")
    print(f"[INFO] 2. I am waiting here to automatically aggregate the final accuracy results...\n")
    
    all_ok = True
    for p, f_out, rank, log_file in processes:
        p.wait()
        f_out.close()
        if p.returncode != 0:
            print(f"[ERROR] Process on GPU {rank} failed with code {p.returncode}! Check log: {log_file}")
            all_ok = False
        else:
            print(f"[SUCCESS] GPU {rank} completed its chunk successfully.")
            
    if not all_ok:
        print("\nSome evaluation jobs failed. I cannot aggregate accurately. Please inspect the logs above.")
        return
        
    # Aggregate accurate results mathematically across logs
    total_robust_acc = 0.0
    parsed_count = 0
    
    print("\nParsing logs to yield final single accuracy metric...")
    for p, f_out, rank, log_file in processes:
        with open(log_file, "r") as f:
            content = f.read()
            # Try to grab robust accuracy printed
            matches = re.findall(r"Robust accuracy.*?:\s+([0-9.]+)", content)
            if matches:
                acc = float(matches[-1])  # the last printed percentage / 100.0 from that chunk
                print(f"  - Chunk {rank} robust accuracy: {acc:.4f}")
                total_robust_acc += acc
                parsed_count += 1
            else:
                print(f"  [Warning] Could not locate 'Robust accuracy' in log output from GPU {rank}")
                
    if parsed_count == args.gpus:
        final_acc = total_robust_acc / args.gpus
        print("\n" + "=" * 60)
        print(f"🔥 FINAL {args.gpus}-GPU COMBINED ROBUST ACCURACY: {final_acc:.4f} ({final_acc:.2%}) 🔥")
        print("=" * 60)
        
        # Save sum
        final_log = os.path.join(log_dir, f"{basename}_FINAL_{args.gpus}GPU_COMBINED.log")
        with open(final_log, "w") as f:
            f.write(f"FINAL {args.gpus}-GPU COMBINED ROBUST ACCURACY: {final_acc:.4f} ({final_acc:.2%})\n")
        print(f"Saved into: {final_log}")
    else:
        print(f"\n[Warning] Could not parse all {args.gpus} logs successfully, bypassing average calculation.")

if __name__ == "__main__":
    main()
