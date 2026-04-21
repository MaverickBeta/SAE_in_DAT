import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, "/Data_share/hongyi/DAT/pytorch-image-models")
sys.path.insert(0, "/Data_share/hongyi/DAT")
sys.path.insert(0, "/Data_share/hongyi/DAT/SAE/project")

from rebm.training.modeling import load_checkpoint
from rebm.training.utils_architecture import create_convnext_model
from sae_core.model import TopKAutoencoder


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STAGE_DIN_MAP = {0: 192, 1: 384, 2: 768, 3: 1536}
DIN_STAGE_MAP = {v: k for k, v in STAGE_DIN_MAP.items()}


class FileListDataset(Dataset):
	def __init__(self, image_paths: List[str], transform=None):
		self.image_paths = image_paths
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		path = self.image_paths[idx]
		image = Image.open(path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		return image, os.path.basename(path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Inspect ConvNeXt stage representations via one SAE checkpoint at a time.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--base-ckpt",
		type=str,
		default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth",
		help="Path to ConvNeXt checkpoint.",
	)
	parser.add_argument(
		"--sae-ckpt",
		type=str,
		required=True,
		help="Path to one SAE checkpoint (single SAE per run).",
	)
	parser.add_argument(
		"--stage",
		type=int,
		choices=[0, 1, 2, 3],
		default=None,
		help="ConvNeXt stage index. If omitted, inferred from SAE d_in.",
	)
	parser.add_argument(
		"--imagenet-val-dir",
		type=str,
		default="/Data_share/hongyi/DAT/data/ImageNet/val",
		help="ImageNet val root directory.",
	)
	parser.add_argument(
		"--source-wnid",
		type=str,
		default="n02077923",
		help="Class folder to analyze under ImageNet val.",
	)
	parser.add_argument(
		"--source-dir",
		type=str,
		default="",
		help="Optional override for source image directory.",
	)
	parser.add_argument(
		"--max-images",
		type=int,
		default=100,
		help="Use first N images after sorting. 0 means use all.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		help="DataLoader batch size.",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=0,
		help="DataLoader workers.",
	)
	parser.add_argument(
		"--topk-features",
		type=int,
		default=300,
		help="Top-K features to plot by mean activation.",
	)
	parser.add_argument(
		"--act-threshold",
		type=float,
		default=0.4,
		help="Threshold for activation-rate statistics.",
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default="/Data_share/hongyi/DAT/SAE/results_representation",
		help="Output root directory.",
	)
	parser.add_argument(
		"--run-name",
		type=str,
		default="",
		help="Optional fixed run name. If empty, auto-generated.",
	)
	return parser.parse_args()


def get_transform():
	# Match ImageNet eval preprocessing pipeline: Resize256 (bicubic) + CenterCrop224 + ToTensor.
	return transforms.Compose([
		transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
	])


def list_images_in_dir(folder: str) -> List[str]:
	all_files = sorted(glob.glob(os.path.join(folder, "*")))
	return [p for p in all_files if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def resolve_source_dir(args: argparse.Namespace) -> str:
	if args.source_dir:
		return args.source_dir
	return os.path.join(args.imagenet_val_dir, args.source_wnid)


def validate_inputs(args: argparse.Namespace, source_dir: str):
	if not os.path.exists(args.base_ckpt):
		raise FileNotFoundError(f"Base checkpoint not found: {args.base_ckpt}")
	if not os.path.exists(args.sae_ckpt):
		raise FileNotFoundError(f"SAE checkpoint not found: {args.sae_ckpt}")
	if not os.path.isdir(source_dir):
		raise FileNotFoundError(f"Source directory not found: {source_dir}")


def build_model_from_ckpt(device: torch.device, base_ckpt_path: str):
	model = create_convnext_model(
		model_type="convnext_large",
		num_classes=1000,
		normalize_input=False,
		use_layernorm=True,
		use_convstem=True,
	)
	load_checkpoint(model, base_ckpt_path)
	model = model.to(device)
	model.eval()
	return model


def infer_stage_from_din(d_in: int) -> Optional[int]:
	return DIN_STAGE_MAP.get(int(d_in))


def infer_k_tag(config_k: int, sae_ckpt_path: str) -> str:
	if int(config_k) == 32:
		return "k32"
	if int(config_k) == 64:
		return "k64"
	stem = os.path.basename(sae_ckpt_path).lower()
	if "_k32" in stem or "k32" in stem:
		return "k32"
	if "_k64" in stem or "k64" in stem:
		return "k64"
	return f"k{int(config_k)}"


def build_sae(device: torch.device, sae_ckpt_path: str):
	ckpt = torch.load(sae_ckpt_path, map_location=device)
	config = ckpt.get("config", {})

	d_in = ckpt.get("d_in", None)
	if d_in is None:
		d_in = config.get("d_in", None)

	d_lat = ckpt.get("d_lat", None)
	if d_lat is None:
		d_lat = config.get("d_lat", None)

	k = config.get("k", ckpt.get("k", None))

	if d_in is None:
		raise KeyError("Cannot resolve d_in from checkpoint (top-level d_in and config['d_in'] are missing)")
	if k is None:
		raise KeyError("Cannot resolve k from checkpoint (config['k'] and top-level k are missing)")

	if d_lat is None:
		expansion_rate = config.get("expansion_rate", None)
		if expansion_rate is not None:
			d_lat = int(d_in) * int(expansion_rate)
		else:
			raise KeyError(
				"Cannot resolve d_lat from checkpoint (top-level d_lat/config['d_lat'] missing and no expansion_rate)"
			)

	sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
	sae.load_state_dict(ckpt["model_state_dict"])
	sae = sae.to(device)
	sae.eval()

	norm_mean = ckpt["norm_mean"].to(device)
	norm_std = ckpt["norm_std"].to(device)

	resolved_cfg = dict(config)
	resolved_cfg["d_in"] = int(d_in)
	resolved_cfg["d_lat"] = int(d_lat)
	resolved_cfg["k"] = int(k)
	return sae, norm_mean, norm_std, resolved_cfg


def analyze_with_sae(
	image_paths: List[str],
	base_model,
	sae_model,
	norm_mean: torch.Tensor,
	norm_std: torch.Tensor,
	stage_idx: int,
	batch_size: int,
	num_workers: int,
	act_threshold: float,
	device: torch.device,
) -> Dict:
	dataset = FileListDataset(image_paths, transform=get_transform())
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=False,
	)

	captured = {}

	def hook_fn(_, __, output):
		captured["stage"] = output.detach()

	handle = base_model.stages[stage_idx].register_forward_hook(hook_fn)

	total_acts = None
	total_acts_sq = None
	total_tokens = 0
	total_images = 0
	total_token_l0 = 0.0
	total_token_l0_sq = 0.0
	active_counts = None

	with torch.no_grad():
		for images, _ in tqdm(loader, desc=f"Extract stage{stage_idx} SAE", leave=False):
			images = images.to(device)
			_ = base_model(images)

			stage_out = captured["stage"]
			bsz, channels, h, w = stage_out.shape
			flat = stage_out.permute(0, 2, 3, 1).reshape(-1, channels)
			flat = (flat - norm_mean) / norm_std
			z = sae_model.encode(flat)

			z_cpu = z.detach().cpu()
			zsum = z_cpu.sum(dim=0)
			zsum_sq = (z_cpu * z_cpu).sum(dim=0)
			is_active = z_cpu > 0

			if total_acts is None:
				total_acts = zsum
				total_acts_sq = zsum_sq
				active_counts = is_active.sum(dim=0)
			else:
				total_acts += zsum
				total_acts_sq += zsum_sq
				active_counts += is_active.sum(dim=0)

			token_l0 = is_active.sum(dim=1).numpy().astype(np.float64)
			total_token_l0 += float(token_l0.sum())
			total_token_l0_sq += float((token_l0 * token_l0).sum())

			total_tokens += z_cpu.shape[0]
			total_images += bsz

	handle.remove()

	if total_tokens == 0:
		raise RuntimeError("No tokens extracted. Check source images and dataloader settings.")

	mean_act = (total_acts / total_tokens).numpy()
	var_act = (total_acts_sq / total_tokens).numpy() - np.square(mean_act)
	var_act = np.maximum(var_act, 0.0)
	std_act = np.sqrt(var_act)

	active_rate = (active_counts / total_tokens).numpy()
	threshold_rate = float((mean_act >= act_threshold).sum() / mean_act.shape[0])

	token_l0_mean = total_token_l0 / total_tokens
	token_l0_var = max(total_token_l0_sq / total_tokens - token_l0_mean * token_l0_mean, 0.0)
	token_l0_std = float(np.sqrt(token_l0_var))

	return {
		"mean_act": mean_act,
		"std_act": std_act,
		"active_rate": active_rate,
		"summary": {
			"total_images": int(total_images),
			"total_tokens": int(total_tokens),
			"token_l0_mean": float(token_l0_mean),
			"token_l0_std": token_l0_std,
			"features_mean_ge_threshold_rate": threshold_rate,
			"act_threshold": float(act_threshold),
		},
	}


def plot_topk_mean_activation(mean_act: np.ndarray, topk: int, out_png: str, title: str):
	k = min(topk, mean_act.shape[0])
	idx = np.argsort(mean_act)[::-1][:k]
	vals = mean_act[idx]

	fig, ax = plt.subplots(figsize=(22, 8))
	x = np.arange(k)
	ax.bar(x, vals, width=0.8, color="#1f77b4", zorder=3)
	ax.set_axisbelow(True)
	ax.grid(axis="y", linestyle="--", alpha=0.35)
	ax.set_title(title, fontsize=16)
	ax.set_xlabel("Feature index (sorted by mean activation)", fontsize=12)
	ax.set_ylabel("Mean SAE Activation", fontsize=12)

	tick_n = min(80, k)
	tick_pos = np.linspace(0, k - 1, num=tick_n, dtype=int)
	ax.set_xticks(tick_pos)
	ax.set_xticklabels(idx[tick_pos], rotation=90, fontsize=8)

	plt.tight_layout()
	plt.savefig(out_png, dpi=300)
	plt.close(fig)


def build_output_dir(
	results_dir: str,
	k_tag: str,
	stage_idx: int,
	source_wnid: str,
	sae_ckpt_path: str,
	run_name: str,
) -> str:
	os.makedirs(results_dir, exist_ok=True)

	if run_name:
		leaf = run_name
	else:
		ckpt_stem = os.path.splitext(os.path.basename(sae_ckpt_path))[0]
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		leaf = f"{source_wnid}_{ckpt_stem}_{ts}"

	out_dir = os.path.join(results_dir, k_tag, f"stage{stage_idx}", leaf)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def main():
	args = parse_args()

	source_dir = resolve_source_dir(args)
	validate_inputs(args, source_dir)

	image_paths = list_images_in_dir(source_dir)
	if len(image_paths) == 0:
		raise RuntimeError(f"No images found in source directory: {source_dir}")
	if args.max_images > 0:
		image_paths = image_paths[: args.max_images]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Source dir: {source_dir}")
	print(f"Images used: {len(image_paths)}")
	print(f"SAE checkpoint: {args.sae_ckpt}")

	base_model = build_model_from_ckpt(device, args.base_ckpt)
	sae_model, norm_mean, norm_std, sae_cfg = build_sae(device, args.sae_ckpt)

	inferred_stage = infer_stage_from_din(int(sae_cfg["d_in"]))
	if args.stage is None:
		if inferred_stage is None:
			raise ValueError(
				f"Cannot infer stage from SAE d_in={sae_cfg['d_in']}. Please provide --stage explicitly."
			)
		stage_idx = inferred_stage
	else:
		stage_idx = int(args.stage)

	expected_din = STAGE_DIN_MAP[stage_idx]
	if int(sae_cfg["d_in"]) != expected_din:
		raise ValueError(
			f"Stage/SAE mismatch: stage{stage_idx} expects d_in={expected_din}, "
			f"but SAE checkpoint has d_in={sae_cfg['d_in']}"
		)

	k_tag = infer_k_tag(int(sae_cfg["k"]), args.sae_ckpt)
	out_dir = build_output_dir(
		results_dir=args.results_dir,
		k_tag=k_tag,
		stage_idx=stage_idx,
		source_wnid=args.source_wnid,
		sae_ckpt_path=args.sae_ckpt,
		run_name=args.run_name,
	)

	print(f"Stage: {stage_idx}")
	print(f"k-tag: {k_tag}")
	print(f"Output dir: {out_dir}")

	stats = analyze_with_sae(
		image_paths=image_paths,
		base_model=base_model,
		sae_model=sae_model,
		norm_mean=norm_mean,
		norm_std=norm_std,
		stage_idx=stage_idx,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		act_threshold=args.act_threshold,
		device=device,
	)

	mean_act = stats["mean_act"]
	std_act = stats["std_act"]
	active_rate = stats["active_rate"]

	np.save(os.path.join(out_dir, "mean_activation.npy"), mean_act)
	np.save(os.path.join(out_dir, "std_activation.npy"), std_act)
	np.save(os.path.join(out_dir, "active_rate.npy"), active_rate)

	topk = min(args.topk_features, mean_act.shape[0])
	top_idx = np.argsort(mean_act)[::-1][:topk]

	plot_topk_mean_activation(
		mean_act,
		topk=args.topk_features,
		out_png=os.path.join(out_dir, "topk_mean_activation.png"),
		title=(
			f"SAE stage{stage_idx} ({k_tag}) | {args.source_wnid} | "
			f"Top-{topk} Mean Activation"
		),
	)
	summary = {
		"source_wnid": args.source_wnid,
		"source_dir": source_dir,
		"num_images_used": int(len(image_paths)),
		"base_checkpoint": args.base_ckpt,
		"sae_checkpoint": args.sae_ckpt,
		"sae_config": {
			"d_in": int(sae_cfg["d_in"]),
			"d_lat": int(sae_cfg["d_lat"]),
			"k": int(sae_cfg["k"]),
		},
		"stage": int(stage_idx),
		"k_tag": k_tag,
		"batch_size": int(args.batch_size),
		"num_workers": int(args.num_workers),
		"preprocess": "Resize(256, bicubic) -> CenterCrop(224) -> ToTensor",
		"summary": stats["summary"],
		"topk_features": [
			{
				"feature_idx": int(i),
				"mean_activation": float(mean_act[i]),
				"std_activation": float(std_act[i]),
				"active_rate": float(active_rate[i]),
			}
			for i in top_idx
		],
		"files": {
			"mean_activation_npy": os.path.join(out_dir, "mean_activation.npy"),
			"std_activation_npy": os.path.join(out_dir, "std_activation.npy"),
			"active_rate_npy": os.path.join(out_dir, "active_rate.npy"),
			"topk_plot": os.path.join(out_dir, "topk_mean_activation.png"),
		},
	}

	summary_path = os.path.join(out_dir, "summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("Done.")
	print(f"Summary: {summary_path}")


if __name__ == "__main__":
	main()
