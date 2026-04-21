#!/usr/bin/env python3
"""Run class-wise SAE steering pipeline on sampled ImageNet classes.

Pipeline per class:
1) gen_evaluate_imagenet.py
2) sae_clean_adv_compare_npz.py
3) steering.py

This orchestrator parallelizes by GPU: one worker per GPU, each worker runs classes
sequentially on its assigned GPU to avoid memory contention.
"""
 
import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DAT_ROOT = SCRIPT_DIR.parents[1]


@dataclass
class ClassResult:
    wnid: str
    gpu_id: str
    ok: bool
    stage: str
    message: str
    run_dir: str
    compare_dir: str
    steering_json: str


def resolve_imagenet_paths(path_str: str) -> Tuple[Path, Path]:
    p = Path(path_str).resolve()
    if p.name == "val":
        val_dir = p
        data_dir = p.parent
    else:
        data_dir = p
        val_dir = p / "val"

    if not val_dir.exists() or not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")

    return data_dir, val_dir


def list_class_wnids(val_dir: Path) -> List[str]:
    return sorted([x.name for x in val_dir.iterdir() if x.is_dir()])


def resolve_existing_path(path_str: str, name: str) -> str:
    """Resolve a file path robustly and require it to exist.

    Search order:
    1) as provided (absolute or relative to current working directory)
    2) relative to DAT root
    3) relative to current script directory
    """
    p = Path(path_str)

    candidates = []
    candidates.append(p if p.is_absolute() else (Path.cwd() / p))
    if not p.is_absolute():
        candidates.append(DAT_ROOT / p)
        candidates.append(SCRIPT_DIR / p)

    for c in candidates:
        c_resolved = c.resolve()
        if c_resolved.exists():
            return str(c_resolved)

    tried = "\n  - " + "\n  - ".join(str(x.resolve()) for x in candidates)
    raise FileNotFoundError(f"{name} not found: {path_str}\nTried:{tried}")


def build_env(gpu_id: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def get_gpu_uuid_map() -> Dict[str, str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi query-gpu failed: {proc.stderr.strip()}")

    out: Dict[str, str] = {}
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def get_busy_pids_by_uuid() -> Dict[str, List[int]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}

    out: Dict[str, List[int]] = {}
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_uuid = parts[0]
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        out.setdefault(gpu_uuid, []).append(pid)
    return out


def wait_round_gpus_free(gpu_ids: List[str], poll_sec: int) -> None:
    uuid_map = get_gpu_uuid_map()
    target = []
    for gid in gpu_ids:
        if gid not in uuid_map:
            raise ValueError(f"GPU id {gid} not found in nvidia-smi")
        target.append((gid, uuid_map[gid]))

    reported = False
    while True:
        busy = get_busy_pids_by_uuid()
        blockers = {}
        for gid, uuid in target:
            pids = busy.get(uuid, [])
            if pids:
                blockers[gid] = pids

        if not blockers:
            if reported:
                print("[Round] all target GPUs are free, start this round")
            return

        if not reported:
            print(f"[Round] waiting GPUs to become free: {blockers}")
            reported = True
        time.sleep(max(1, poll_sec))


def run_cmd(cmd: List[str], env: Dict[str, str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 80 + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def is_pipeline_done(steering_json: Path, log_file: Path) -> bool:
    """A class is done only when output exists and final save marker appears in log."""
    if not steering_json.exists() or not log_file.exists():
        return False

    marker = f"Saved steering results: {steering_json}"
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return marker in text
    except Exception:
        return False


def maybe_add_sae_stage(cmd: List[str], sae_stage: Optional[int]) -> List[str]:
    if sae_stage is not None:
        cmd.extend(["--sae-stage", str(sae_stage)])
    return cmd


def run_one_class(
    wnid: str,
    gpu_id: str,
    round_idx: int,
    args,
    data_dir: Path,
    output_root: Path,
) -> ClassResult:
    class_dir = output_root / wnid
    run_dir = class_dir / "run"
    compare_dir = class_dir / "compare"
    summary_json = compare_dir / "sae_clean_adv_summary.json"
    steering_json = class_dir / "steering_search_results.json"
    log_file = class_dir / "pipeline.log"

    env = build_env(gpu_id)

    try:
        if not args.force and is_pipeline_done(steering_json=steering_json, log_file=log_file):
            return ClassResult(
                wnid=wnid,
                gpu_id=str(gpu_id),
                ok=True,
                stage="skipped_done",
                message="already completed (json + final log marker)",
                run_dir=str(run_dir),
                compare_dir=str(compare_dir),
                steering_json=str(steering_json),
            )

        if not args.append_logs:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# New run\n# round={round_idx} class={wnid} gpu={gpu_id}\n"
                )

        if args.force or not (run_dir / "rank000" / "metrics.json").exists():
            gen_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "gen_evaluate_imagenet.py"),
                "--checkpoint",
                args.checkpoint,
                "--data_dir",
                str(data_dir),
                "--threat_model",
                args.threat_model,
                "--eps",
                str(args.eps),
                "--n_examples",
                str(args.n_examples),
                "--batch_size",
                str(args.gen_batch_size),
                "--architecture",
                args.architecture,
                "--img_size",
                str(args.img_size),
                "--save_adv_dir",
                str(run_dir),
                "--source_wnid",
                wnid,
                "--save_format",
                args.save_format,
                "--save_shard_size",
                str(args.save_shard_size),
                "--sae_ckpt",
                args.sae_ckpt,
                "--sae_recon_mode",
                args.sae_recon_mode,
            ]
            gen_cmd = maybe_add_sae_stage(gen_cmd, args.sae_stage)
            run_cmd(gen_cmd, env=env, log_file=log_file)

        if args.force or not summary_json.exists():
            compare_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "sae_clean_adv_compare_npz.py"),
                "--run-dir",
                str(run_dir),
                "--checkpoint",
                args.checkpoint,
                "--sae-ckpt",
                args.sae_ckpt,
                "--batch-size",
                str(args.compare_batch_size),
                "--topk",
                str(args.compare_topk),
                "--max-samples",
                str(args.max_samples),
                "--output-dir",
                str(compare_dir),
            ]
            compare_cmd = maybe_add_sae_stage(compare_cmd, args.sae_stage)
            run_cmd(compare_cmd, env=env, log_file=log_file)

        if args.force or not steering_json.exists():
            steering_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "steering.py"),
                "--run-dir",
                str(run_dir),
                "--summary-json",
                str(summary_json),
                "--checkpoint",
                args.checkpoint,
                "--sae-ckpt",
                args.sae_ckpt,
                "--batch-size",
                str(args.steering_batch_size),
                "--max-samples",
                str(args.max_samples),
                "--max-pool-size",
                str(args.max_pool_size),
                "--max-features",
                str(args.max_features),
                "--beam-width",
                str(args.beam_width),
                "--w-top1",
                str(args.w_top1),
                "--w-top5",
                str(args.w_top5),
                "--w-trueprob",
                str(args.w_trueprob),
                "--output-json",
                str(steering_json),
                "--alphas",
            ]
            steering_cmd.extend([str(a) for a in args.alphas])
            steering_cmd = maybe_add_sae_stage(steering_cmd, args.sae_stage)
            run_cmd(steering_cmd, env=env, log_file=log_file)

        return ClassResult(
            wnid=wnid,
            gpu_id=str(gpu_id),
            ok=True,
            stage="done",
            message="ok",
            run_dir=str(run_dir),
            compare_dir=str(compare_dir),
            steering_json=str(steering_json),
        )
    except Exception as e:
        return ClassResult(
            wnid=wnid,
            gpu_id=str(gpu_id),
            ok=False,
            stage="failed",
            message=str(e),
            run_dir=str(run_dir),
            compare_dir=str(compare_dir),
            steering_json=str(steering_json),
        )


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def parse_args():
    p = argparse.ArgumentParser(description="Run 50-class SAE steering pipeline on GPUs 1,2,3,4")
    p.add_argument("--imagenet", type=str, default="/Data_share/hongyi/DAT/data/ImageNet/val")
    p.add_argument("--num-classes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-ids", type=str, default="1,2,3,4", help="Physical GPU ids, comma-separated")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(SCRIPT_DIR / "runs" / "steering_50classes"),
    )
    p.add_argument("--force", action="store_true", default=False, help="Re-run even if outputs already exist")
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of classes per round (default: 4). Next round starts after current round fully finishes.",
    )
    p.add_argument(
        "--round-gpu-poll-sec",
        type=int,
        default=15,
        help="Polling interval when waiting for all round GPUs to be free",
    )
    p.add_argument(
        "--append-logs",
        action="store_true",
        default=False,
        help="Append to existing class pipeline.log instead of overwriting each new attempt",
    )

    # Shared model/SAE settings
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--sae-ckpt",
        type=str,
        default="/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
    )
    p.add_argument("--sae-stage", type=int, default=None)

    # gen_evaluate_imagenet.py settings
    p.add_argument("--architecture", type=str, default="convnext_large")
    p.add_argument("--threat-model", type=str, default="L2", choices=["L2", "Linf"])
    p.add_argument("--eps", type=float, default=3.0)
    p.add_argument("--n-examples", type=int, default=0, help="Per class; 0 means all images in class folder")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--save-format", type=str, default="npz", choices=["npz", "pt"])
    p.add_argument("--save-shard-size", type=int, default=256)
    p.add_argument("--sae-recon-mode", type=str, default="raw", choices=["raw", "norm"])

    # compare and steering settings
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--compare-batch-size", type=int, default=32)
    p.add_argument("--compare-topk", type=int, default=300)
    p.add_argument("--steering-batch-size", type=int, default=8)
    p.add_argument("--max-pool-size", type=int, default=20)
    p.add_argument("--max-features", type=int, default=3)
    p.add_argument("--beam-width", type=int, default=12)
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 2.5, 3.0, 4.0, 5.0])
    p.add_argument("--w-top1", type=float, default=1.0)
    p.add_argument("--w-top5", type=float, default=0.25)
    p.add_argument("--w-trueprob", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()

    args.checkpoint = resolve_existing_path(args.checkpoint, "checkpoint")
    if args.sae_ckpt and args.sae_ckpt.strip():
        args.sae_ckpt = resolve_existing_path(args.sae_ckpt, "sae_ckpt")

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided")

    data_dir, val_dir = resolve_imagenet_paths(args.imagenet)
    all_wnids = list_class_wnids(val_dir)
    if args.num_classes > len(all_wnids):
        raise ValueError(f"Requested num-classes={args.num_classes}, but only {len(all_wnids)} found")

    rng = random.Random(args.seed)
    sampled = rng.sample(all_wnids, args.num_classes)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "sampled_classes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "imagenet_data_dir": str(data_dir),
                "imagenet_val_dir": str(val_dir),
                "num_classes": args.num_classes,
                "seed": args.seed,
                "gpu_ids": gpu_ids,
                "classes": sampled,
            },
            f,
            indent=2,
        )

    groups = chunk_list(sampled, args.group_size)
    print(f"Total classes: {len(sampled)}")
    print(f"Group size: {args.group_size}")
    print(f"Total rounds: {len(groups)}")
    print(f"GPUs used: {gpu_ids}")
    print(f"Resolved checkpoint: {args.checkpoint}")
    if args.sae_ckpt and args.sae_ckpt.strip():
        print(f"Resolved SAE ckpt: {args.sae_ckpt}")

    results: List[ClassResult] = []

    for round_idx, group in enumerate(groups, start=1):
        print(f"\n[Round {round_idx}/{len(groups)}] classes={group}")

        # Ensure round GPUs are not occupied by external leftovers before launching this round.
        round_gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(len(group))]
        wait_round_gpus_free(round_gpu_ids, args.round_gpu_poll_sec)

        futures = []
        with ThreadPoolExecutor(max_workers=min(len(group), len(gpu_ids))) as ex:
            for slot_idx, wnid in enumerate(group):
                gpu_id = gpu_ids[slot_idx % len(gpu_ids)]
                print(f"[Round {round_idx}] start {wnid} on GPU {gpu_id}")
                futures.append(
                    ex.submit(
                        run_one_class,
                        wnid,
                        gpu_id,
                        round_idx,
                        args,
                        data_dir,
                        output_root,
                    )
                )

            round_results = [fu.result() for fu in futures]

        for r in round_results:
            results.append(r)
            state = "OK" if r.ok else "FAIL"
            print(f"[Round {round_idx}] {r.wnid} (GPU {r.gpu_id}) -> {state}")
            if not r.ok:
                print(f"  reason: {r.message}")

        ok_in_round = sum(1 for r in round_results if r.ok)
        print(f"[Round {round_idx}] done: ok={ok_in_round} fail={len(round_results) - ok_in_round}")

    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]

    summary = {
        "total": len(results),
        "ok": len(ok),
        "fail": len(fail),
        "results": [asdict(r) for r in results],
    }

    with open(output_root / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPipeline finished")
    print(f"  total={len(results)} ok={len(ok)} fail={len(fail)}")
    print(f"  sampled classes: {output_root / 'sampled_classes.json'}")
    print(f"  run summary: {output_root / 'pipeline_results.json'}")


if __name__ == "__main__":
    main()
