#!/usr/bin/env python3
"""
Launch concurrent AutoAttack evaluation jobs on multiple GPUs for the SAE-wrapped ConvNeXt.
Usage: python run_multigpus_sae.py --sae_ckpt project/checkpoints/k64/sae_64k_k64_step_50000.pt --batch_size 32
"""
import argparse
import subprocess
import os
import sys
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DAT_ROOT = SCRIPT_DIR
while DAT_ROOT.name != "DAT" and DAT_ROOT.parent != DAT_ROOT:
    DAT_ROOT = DAT_ROOT.parent
if DAT_ROOT.name != "DAT":
    raise RuntimeError(f"Unable to locate DAT root from {SCRIPT_DIR}")

RUN_SCRIPT = SCRIPT_DIR / "run_acc_eval_sae.py"
DEFAULT_CHECKPOINT = DAT_ROOT / "checkpoints" / "model_bestfid.pth"

def main():
    parser = argparse.ArgumentParser(description="Launch multi-GPU parallel SAE evaluation safely")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to base checkpoint")
    parser.add_argument("--sae_ckpt", type=str, default="none", help="Path to SAE. 'none' for vanilla.")
    parser.add_argument("--data_dir", type=str, default="../data/ImageNet", help="Path to ImageNet val dataset")
    parser.add_argument("--threat_model", type=str, default="L2", choices=["L2", "Linf", "corruptions"])
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--n_examples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=256)
    # 因为您说自己是用 1,2,3,4卡，这里我们让他默认映射到这四张卡
    parser.add_argument("--gpu_list", type=str, default="4,5,6,7", help="Comma separated list of GPU IDs to use")
    
    args = parser.parse_args()

    # 获取要用的真实物理卡号列表
    gpus_to_use = [g.strip() for g in args.gpu_list.split(',') if g.strip()]
    if not gpus_to_use:
        raise ValueError("--gpu_list is empty. Please provide at least one GPU id.")
    world_size = len(gpus_to_use)

    cmd_base = [
        sys.executable, "-u", str(RUN_SCRIPT),
        # "-u"：强制让 Python 的输出不缓存（Unbuffered）
        "--checkpoint", args.checkpoint,
        "--sae_ckpt", args.sae_ckpt,
        "--data_dir", args.data_dir,
        "--threat_model", args.threat_model,
        "--eps", str(args.eps),
        "--n_examples", str(args.n_examples),
        "--batch_size", str(args.batch_size),
        "--img_size", str(args.img_size)
    ]
    
    log_dir = SCRIPT_DIR / "slurm_log" / "eval_acc_multigpus_sae"
    os.makedirs(log_dir, exist_ok=True)
    
    # choose a run name based on whether using SAE or not. This will help us identify logs later.
    run_name = Path(args.sae_ckpt if args.sae_ckpt != "none" else args.checkpoint).stem
    
    processes = []
    
    print(f"==================================================")
    print(f"Starting {world_size}-GPU parallel data splitting for SAE Eval...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"SAE: {args.sae_ckpt}")
    print(f"Physical GPUs mapped: {gpus_to_use}")
    print(f"Please check individual progress logs at: {log_dir}")
    print(f"==================================================\n")

    for rank, gpu_id in enumerate(gpus_to_use):
        # 深拷贝独立副本，每个GPU进程有自己的配置，互不干扰
        env = os.environ.copy()
        # 限制底层的线程，防止切片过多导致死锁
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)
        
        log_file = log_dir / f"{run_name}_gpu{gpu_id}_rank{rank}.log"
        f_out = open(log_file, "w", encoding="utf-8")
        
        print(f"--> [Rank {rank}] Dispatching to GPU {gpu_id} | Logging to: {log_file}")
        # 3. 启动子进程，输出重定向到文件 cmd_base是用的上面的基础命令，env是为每个进程定制的环境变量，f_out是日志文件
        p = subprocess.Popen(cmd_base, env=env, cwd=str(SCRIPT_DIR), stdout=f_out, stderr=subprocess.STDOUT)
        # 4. 保存进程信息，方便后续管理（等待结束、关闭文件等）
        processes.append((p, f_out, rank, log_file))
        
    print(f"\n[INFO] All {world_size} independent processes dispatched successfully.")
    print(f"[INFO] Evaluating {world_size} splits in parallel. Awaiting completion to aggregate final results...\n")
    
    all_ok = True
    for p, f_out, rank, log_file in processes:
        p.wait()
        f_out.close()
        if p.returncode != 0:
            print(f"[ERROR] Process Rank {rank} failed with code {p.returncode}! Please check its log: {log_file}")
            all_ok = False
        else:
            print(f"[SUCCESS] Rank {rank} completed its dataset chunk successfully.")
            
    if not all_ok:
        print("\n[WARNING] Some evaluations failed. Aggregation will be skipped. Please inspect the logs above.")
        return
        
    weighted_clean_sum = 0.0
    weighted_robust_sum = 0.0
    total_samples = 0
    parsed_count = 0
    
    print("\n📊 Parsing logs and aggregating final accuracy numbers...")
    for p, f_out, rank, log_file in processes:
        with open(log_file, "r") as f:
            content = f.read()
            clean_match = re.search(r"Clean accuracy:\s+([0-9.]+)", content)
            robust_match = re.search(r"Robust accuracy.*?:\s+([0-9.]+)", content)
            samples_match = re.search(r"Samples evaluated:\s+([0-9]+)", content)
            
            if clean_match and robust_match and samples_match:
                c_acc = float(clean_match.group(1))
                r_acc = float(robust_match.group(1))
                n = int(samples_match.group(1))
                print(f"  - Chunk {rank} | Clean: {c_acc:.4f} | Robust: {r_acc:.4f} | Samples: {n}")
                weighted_clean_sum += c_acc * n
                weighted_robust_sum += r_acc * n
                total_samples += n
                parsed_count += 1
            else:
                print(f"  [Warning] Missing accuracy outputs in Rank {rank} log (need Clean/Robust/Samples).")
                
    if parsed_count == world_size and total_samples > 0:
        final_clean = weighted_clean_sum / total_samples
        final_robust = weighted_robust_sum / total_samples
        print("\n" + "=" * 60)
        print(f" FINAL {world_size}-GPU COMBINED CLEAN ACCURACY:  {final_clean:.4f} ({final_clean:.2%})")
        print(f" FINAL {world_size}-GPU COMBINED ROBUST ACCURACY: {final_robust:.4f} ({final_robust:.2%})")
        print(f" TOTAL SAMPLES (weighted): {total_samples}")
        if total_samples > args.n_examples:
            print(" [Notice] total samples exceed --n_examples; this usually means different ranks evaluated overlapping samples.")
        print("=" * 60)
        
        final_log = log_dir / f"{run_name}_FINAL_{world_size}GPU.log"
        with open(final_log, "w", encoding="utf-8") as f:
            f.write(f"FINAL {world_size}-GPU CLEAN ACCURACY:  {final_clean:.4f}\n")
            f.write(f"FINAL {world_size}-GPU ROBUST ACCURACY: {final_robust:.4f}\n")
            f.write(f"TOTAL SAMPLES (weighted): {total_samples}\n")
    else:
        print(f"\n[Warning] Could not parse all logs successfully.")

if __name__ == "__main__":
    main()