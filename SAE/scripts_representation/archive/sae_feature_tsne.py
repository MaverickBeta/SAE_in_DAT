import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


class ImageRecordDataset(Dataset):
	def __init__(self, records: List[Dict], transform=None):
		self.records = records
		self.transform = transform

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx: int):
		rec = self.records[idx]
		image = Image.open(rec["path"]).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, rec["class_idx"], rec["wnid"], os.path.basename(rec["path"])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="t-SNE on ConvNeXt-large stage3 outputs with fixed class/image sampling.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--imagenet-val-dir",
		type=str,
		default="/Data_share/hongyi/DAT/data/ImageNet/val",
		help="ImageNet val root (class folders).",
	)
	parser.add_argument(
		"--base-ckpt",
		type=str,
		default="/Data_share/hongyi/DAT/checkpoints/model_bestfid.pth",
		help="ConvNeXt-large checkpoint path.",
	)
	parser.add_argument(
		"--sae-ckpt",
		type=str,
		default="/Data_share/hongyi/DAT/SAE/project/checkpoints/k64/sae_64k_k64_step_50000.pt",
		help="SAE checkpoint used to encode stage features.",
	)
	parser.add_argument(
		"--num-classes",
		type=int,
		default=100,
		help="Randomly sample this many classes.",
	)
	parser.add_argument(
		"--images-per-class",
		type=int,
		default=50,
		help="Use this many images per sampled class.",
	)
	parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=3, help="ConvNeXt stage to hook.")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
	parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
	parser.add_argument(
		"--tsne-max-points",
		type=int,
		default=3000,
		help="Maximum token points sampled for each t-SNE branch (pre/post SAE). Dense 65536-d latent is memory-heavy.",
	)
	parser.add_argument(
		"--latent-topk",
		type=int,
		default=64,
		help="Unused for post t-SNE now; kept for compatibility with previous runs.",
	)
	parser.add_argument(
		"--post-active-dim-cap",
		type=int,
		default=12000,
		help="After dense-z zero-column pruning, cap active dims by non-zero count for faster PCA+t-SNE (0 disables cap).",
	)
	parser.add_argument(
		"--tsne-perplexity",
		type=float,
		default=30.0,
		help="t-SNE perplexity. Auto-clipped by sample size.",
	)
	parser.add_argument("--tsne-random-state", type=int, default=42, help="Random seed for t-SNE.")
	parser.add_argument(
		"--pca-dim",
		type=int,
		default=50,
		help="Apply PCA to this dimension before t-SNE (0 to disable PCA).",
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default="/Data_share/hongyi/DAT/SAE/results_representation",
		help="Root output directory.",
	)
	parser.add_argument("--run-name", type=str, default="", help="Optional fixed run name.")
	return parser.parse_args()


def get_transform():
	return transforms.Compose(
		[
			transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
		]
	)


def list_images_in_dir(folder: str) -> List[str]:
	paths = sorted(glob.glob(os.path.join(folder, "*")))
	return [p for p in paths if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)]


def validate_paths(args: argparse.Namespace):
	if not os.path.isdir(args.imagenet_val_dir):
		raise FileNotFoundError(f"ImageNet val directory not found: {args.imagenet_val_dir}")
	if not os.path.exists(args.base_ckpt):
		raise FileNotFoundError(f"ConvNeXt checkpoint not found: {args.base_ckpt}")
	if not os.path.exists(args.sae_ckpt):
		raise FileNotFoundError(f"SAE checkpoint not found: {args.sae_ckpt}")


def sample_records(imagenet_val_dir: str, num_classes: int, images_per_class: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
	random.seed(seed)
	class_dirs = [d for d in sorted(glob.glob(os.path.join(imagenet_val_dir, "*"))) if os.path.isdir(d)]

	eligible = []
	for d in class_dirs:
		wnid = os.path.basename(d)
		images = list_images_in_dir(d)
		if len(images) >= images_per_class:
			eligible.append((wnid, d, images))

	if len(eligible) < num_classes:
		raise RuntimeError(
			f"Not enough eligible classes: need {num_classes}, found {len(eligible)} with >= {images_per_class} images."
		)

	selected = random.sample(eligible, k=num_classes)
	class_meta = []
	records = []

	for class_idx, (wnid, class_dir, images) in enumerate(selected):
		chosen = images[:images_per_class]
		class_meta.append(
			{
				"class_idx": int(class_idx),
				"wnid": wnid,
				"class_dir": class_dir,
				"num_images": int(len(chosen)),
			}
		)
		for p in chosen:
			records.append({"path": p, "class_idx": int(class_idx), "wnid": wnid})

	return records, class_meta


def build_model_from_ckpt(device: torch.device, ckpt_path: str):
	model = create_convnext_model(
		model_type="convnext_large",
		num_classes=1000,
		normalize_input=False,
		use_layernorm=True,
		use_convstem=True,
	)
	load_checkpoint(model, ckpt_path)
	model = model.to(device)
	model.eval()
	return model


def build_sae(device: torch.device, sae_ckpt_path: str):
	ckpt = torch.load(sae_ckpt_path, map_location=device)
	config = ckpt.get("config", {})

	d_in = ckpt.get("d_in", config.get("d_in", None))
	d_lat = ckpt.get("d_lat", config.get("d_lat", None))
	k = ckpt.get("k", config.get("k", None))

	if d_in is None:
		raise KeyError("Cannot resolve SAE d_in from checkpoint")
	if d_lat is None:
		expansion_rate = config.get("expansion_rate", None)
		if expansion_rate is None:
			raise KeyError("Cannot resolve SAE d_lat from checkpoint")
		d_lat = int(d_in) * int(expansion_rate)
	if k is None:
		raise KeyError("Cannot resolve SAE k from checkpoint")

	sae = TopKAutoencoder(d_in=int(d_in), d_lat=int(d_lat), k=int(k))
	sae.load_state_dict(ckpt["model_state_dict"])
	sae = sae.to(device)
	sae.eval()

	norm_mean = ckpt["norm_mean"].to(device)
	norm_std = ckpt["norm_std"].to(device)

	return sae, norm_mean, norm_std, {"d_in": int(d_in), "d_lat": int(d_lat), "k": int(k)}


def check_sae_stage_alignment(sae_ckpt_path: str, target_stage: int) -> Dict:
	ckpt = torch.load(sae_ckpt_path, map_location="cpu")
	config = ckpt.get("config", {})
	d_in = ckpt.get("d_in", config.get("d_in", None))
	if d_in is None:
		raise KeyError("Cannot resolve SAE d_in from checkpoint.")

	inferred_stage = DIN_STAGE_MAP.get(int(d_in), None)
	is_aligned = inferred_stage == int(target_stage)
	expected_din = STAGE_DIN_MAP[int(target_stage)]

	return {
		"sae_ckpt": sae_ckpt_path,
		"sae_d_in": int(d_in),
		"inferred_stage": None if inferred_stage is None else int(inferred_stage),
		"target_stage": int(target_stage),
		"target_stage_expected_d_in": int(expected_din),
		"is_aligned": bool(is_aligned),
	}


def reservoir_update(
	rng: np.random.Generator,
	res_pre: List[np.ndarray],
	res_post: List[np.ndarray],
	res_label: List[int],
	batch_pre: np.ndarray,
	batch_post: np.ndarray,
	batch_label: np.ndarray,
	seen_count: int,
	max_points: int,
) -> int:
	for i in range(batch_pre.shape[0]):
		seen_count += 1
		if len(res_pre) < max_points:
			res_pre.append(batch_pre[i])
			res_post.append(batch_post[i])
			res_label.append(int(batch_label[i]))
		else:
			j = int(rng.integers(0, seen_count))
			if j < max_points:
				res_pre[j] = batch_pre[i]
				res_post[j] = batch_post[i]
				res_label[j] = int(batch_label[i])
	return seen_count


def extract_stage_features(
	model,
	sae,
	norm_mean: torch.Tensor,
	norm_std: torch.Tensor,
	records: List[Dict],
	stage_idx: int,
	batch_size: int,
	num_workers: int,
	tsne_max_points: int,
	seed: int,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], Dict]:
	dataset = ImageRecordDataset(records=records, transform=get_transform())
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

	handle = model.stages[stage_idx].register_forward_hook(hook_fn)

	rng = np.random.default_rng(seed)
	res_pre: List[np.ndarray] = []
	res_post: List[np.ndarray] = []
	res_label: List[int] = []
	seen_tokens = 0
	total_tokens = 0
	item_meta = []
	latent_dim = None

	with torch.no_grad():
		for images, class_idx, wnids, basenames in tqdm(loader, desc=f"Extract stage{stage_idx}", leave=False):
			images = images.to(device)
			_ = model(images)

			stage_out = captured["stage"]
			bsz, channels, h, w = stage_out.shape
			flat = stage_out.permute(0, 2, 3, 1).reshape(-1, channels)
			flat_norm = (flat - norm_mean) / norm_std
			z = sae.encode(flat_norm)

			latent_dim = int(z.shape[1])

			labels_token = class_idx.to(device).view(bsz, 1).repeat(1, h * w).reshape(-1)

			batch_pre = flat_norm.detach().cpu().numpy().astype(np.float32)
			# Keep true SAE latent z (65536-d). Use float16 buffer to reduce host memory pressure.
			batch_post = z.detach().cpu().to(torch.float16).numpy()
			batch_label = labels_token.detach().cpu().numpy().astype(np.int32)

			total_tokens += int(batch_pre.shape[0])
			seen_tokens = reservoir_update(
				rng=rng,
				res_pre=res_pre,
				res_post=res_post,
				res_label=res_label,
				batch_pre=batch_pre,
				batch_post=batch_post,
				batch_label=batch_label,
				seen_count=seen_tokens,
				max_points=int(tsne_max_points),
			)

			for i in range(len(basenames)):
				item_meta.append(
					{
						"wnid": str(wnids[i]),
						"basename": str(basenames[i]),
						"class_idx": int(class_idx[i].item()),
					}
				)

	handle.remove()

	if len(res_pre) == 0:
		raise RuntimeError("No token samples collected for t-SNE.")

	pre = np.stack(res_pre, axis=0)
	post = np.stack(res_post, axis=0)
	labels = np.asarray(res_label, dtype=np.int32)

	meta = {
		"total_tokens_seen": int(total_tokens),
		"sampled_tokens": int(pre.shape[0]),
		"latent_dim": int(latent_dim) if latent_dim is not None else None,
		"post_repr_dim": int(post.shape[1]),
	}
	return pre, post, labels, item_meta, meta


def prepare_post_latent_for_tsne(post_feats: np.ndarray, post_active_dim_cap: int) -> Tuple[np.ndarray, Dict]:
	# 1) Remove columns that are always zero across sampled tokens (exact information-preserving step).
	active_cols = np.where(np.any(post_feats != 0, axis=0))[0]
	if active_cols.size == 0:
		raise RuntimeError("All sampled SAE latent values are zero; cannot run post-SAE t-SNE.")

	post_active = post_feats[:, active_cols]

	selected_cols = active_cols
	if int(post_active_dim_cap) > 0 and post_active.shape[1] > int(post_active_dim_cap):
		# 2) Cap dimensions by non-zero frequency to keep runtime bounded on large servers.
		nnz = np.count_nonzero(post_active, axis=0)
		keep_local = np.argsort(nnz)[::-1][: int(post_active_dim_cap)]
		post_active = post_active[:, keep_local]
		selected_cols = active_cols[keep_local]

	return post_active.astype(np.float32), {
		"latent_total_dim": int(post_feats.shape[1]),
		"latent_nonzero_dim": int(active_cols.size),
		"latent_selected_dim": int(post_active.shape[1]),
		"post_active_dim_cap": int(post_active_dim_cap),
		"selected_latent_indices_npy_note": "Saved in summary only as counts to avoid huge json.",
	}


def run_tsne(features: np.ndarray, perplexity: float, random_state: int, pca_dim: int) -> Tuple[np.ndarray, Dict]:
	if features.shape[0] < 3:
		raise ValueError("Need at least 3 samples for t-SNE.")

	try:
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE
	except Exception as e:
		raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

	x = features
	used_pca_dim: Optional[int] = None
	if pca_dim > 0:
		used_pca_dim = min(int(pca_dim), int(features.shape[1]), int(features.shape[0] - 1))
		if used_pca_dim >= 2:
			x = PCA(n_components=used_pca_dim, random_state=int(random_state)).fit_transform(features)

	actual_perplexity = min(float(perplexity), float(x.shape[0] - 1))
	actual_perplexity = max(actual_perplexity, 2.0)

	tsne = TSNE(
		n_components=2,
		perplexity=actual_perplexity,
		init="pca",
		learning_rate="auto",
		random_state=int(random_state),
	)
	emb = tsne.fit_transform(x)
	return emb, {
		"used_pca_dim": used_pca_dim,
		"actual_perplexity": float(actual_perplexity),
	}


def plot_tsne(emb: np.ndarray, labels: np.ndarray, out_png: str, title: str):
	fig, ax = plt.subplots(figsize=(12, 10))
	sc = ax.scatter(
		emb[:, 0],
		emb[:, 1],
		c=labels,
		cmap="nipy_spectral",
		s=10,
		alpha=0.8,
		edgecolors="none",
	)
	ax.set_title(title, fontsize=14)
	ax.set_xlabel("t-SNE dim 1")
	ax.set_ylabel("t-SNE dim 2")
	ax.grid(alpha=0.25, linestyle="--")

	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("Sampled class index")

	plt.tight_layout()
	plt.savefig(out_png, dpi=300)
	plt.close(fig)


def plot_tsne_compare(pre_emb: np.ndarray, post_emb: np.ndarray, labels: np.ndarray, out_png: str, title: str):
	fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

	sc1 = axes[0].scatter(
		pre_emb[:, 0],
		pre_emb[:, 1],
		c=labels,
		cmap="nipy_spectral",
		s=10,
		alpha=0.8,
		edgecolors="none",
	)
	axes[0].set_title("Before SAE (input tensor)")
	axes[0].set_xlabel("t-SNE dim 1")
	axes[0].set_ylabel("t-SNE dim 2")
	axes[0].grid(alpha=0.25, linestyle="--")

	axes[1].scatter(
		post_emb[:, 0],
		post_emb[:, 1],
		c=labels,
		cmap="nipy_spectral",
		s=10,
		alpha=0.8,
		edgecolors="none",
	)
	axes[1].set_title("After SAE (dense latent z)")
	axes[1].set_xlabel("t-SNE dim 1")
	axes[1].set_ylabel("t-SNE dim 2")
	axes[1].grid(alpha=0.25, linestyle="--")

	cbar = fig.colorbar(sc1, ax=axes, fraction=0.02, pad=0.02)
	cbar.set_label("Sampled class index")

	fig.suptitle(title, fontsize=14)
	plt.savefig(out_png, dpi=300)
	plt.close(fig)


def make_out_dir(results_dir: str, run_name: str) -> str:
	os.makedirs(results_dir, exist_ok=True)
	if run_name:
		leaf = run_name
	else:
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		leaf = f"stage3_tsne_{ts}"
	out_dir = os.path.join(results_dir, "tsne_stage3", leaf)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def main():
	args = parse_args()
	validate_paths(args)

	out_dir = make_out_dir(args.results_dir, args.run_name)

	records, class_meta = sample_records(
		imagenet_val_dir=args.imagenet_val_dir,
		num_classes=args.num_classes,
		images_per_class=args.images_per_class,
		seed=args.tsne_random_state,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	print(f"Sampled classes: {len(class_meta)}")
	print(f"Total sampled images: {len(records)}")
	print(f"Output dir: {out_dir}")

	model = build_model_from_ckpt(device=device, ckpt_path=args.base_ckpt)
	sae, norm_mean, norm_std, sae_cfg = build_sae(device=device, sae_ckpt_path=args.sae_ckpt)

	sae_check = check_sae_stage_alignment(sae_ckpt_path=args.sae_ckpt, target_stage=args.stage)
	print(
		"SAE stage alignment | "
		f"d_in={sae_check['sae_d_in']}, inferred_stage={sae_check['inferred_stage']}, "
		f"target_stage={sae_check['target_stage']}, aligned={sae_check['is_aligned']}"
	)
	if not sae_check["is_aligned"]:
		raise ValueError("SAE and selected stage are not aligned. Please use a matching SAE checkpoint.")

	pre_feats, post_feats, labels, item_meta, sampling_meta = extract_stage_features(
		model=model,
		sae=sae,
		norm_mean=norm_mean,
		norm_std=norm_std,
		records=records,
		stage_idx=args.stage,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		tsne_max_points=args.tsne_max_points,
		seed=args.tsne_random_state,
		device=device,
	)

	pre_emb, pre_tsne_meta = run_tsne(
		features=pre_feats,
		perplexity=args.tsne_perplexity,
		random_state=args.tsne_random_state,
		pca_dim=args.pca_dim,
	)
	post_tsne_input, post_prepare_meta = prepare_post_latent_for_tsne(
		post_feats=post_feats,
		post_active_dim_cap=args.post_active_dim_cap,
	)
	post_emb, post_tsne_meta = run_tsne(
		features=post_tsne_input,
		perplexity=args.tsne_perplexity,
		random_state=args.tsne_random_state,
		pca_dim=args.pca_dim,
	)

	pre_png = os.path.join(out_dir, "tsne_pre_sae.png")
	post_png = os.path.join(out_dir, "tsne_post_sae.png")
	cmp_png = os.path.join(out_dir, "tsne_pre_post_compare.png")
	pre_emb_npy = os.path.join(out_dir, "tsne_pre_embedding.npy")
	post_emb_npy = os.path.join(out_dir, "tsne_post_embedding.npy")
	pre_feat_npy = os.path.join(out_dir, "pre_sae_tensor_samples.npy")
	post_feat_npy = os.path.join(out_dir, "post_sae_latent_samples.npy")
	post_tsne_input_npy = os.path.join(out_dir, "post_sae_tsne_input.npy")
	label_npy = os.path.join(out_dir, "labels.npy")
	class_json = os.path.join(out_dir, "sampled_classes.json")
	item_json = os.path.join(out_dir, "sampled_items.json")

	plot_tsne(
		emb=pre_emb,
		labels=labels,
		out_png=pre_png,
		title=f"Before SAE | stage{args.stage} tensor t-SNE ({len(class_meta)} classes x {args.images_per_class} images)",
	)
	plot_tsne(
		emb=post_emb,
		labels=labels,
		out_png=post_png,
		title=f"After SAE | dense latent z t-SNE ({len(class_meta)} classes x {args.images_per_class} images)",
	)
	plot_tsne_compare(
		pre_emb=pre_emb,
		post_emb=post_emb,
		labels=labels,
		out_png=cmp_png,
		title=f"Before vs After SAE t-SNE | stage{args.stage}",
	)

	np.save(pre_emb_npy, pre_emb)
	np.save(post_emb_npy, post_emb)
	np.save(pre_feat_npy, pre_feats)
	np.save(post_feat_npy, post_feats)
	np.save(post_tsne_input_npy, post_tsne_input)
	np.save(label_npy, labels)

	with open(class_json, "w", encoding="utf-8") as f:
		json.dump(class_meta, f, indent=2)

	with open(item_json, "w", encoding="utf-8") as f:
		json.dump(item_meta, f, indent=2)

	summary = {
		"imagenet_val_dir": args.imagenet_val_dir,
		"base_ckpt": args.base_ckpt,
		"sae_ckpt": args.sae_ckpt,
		"sae_config": sae_cfg,
		"stage": int(args.stage),
		"num_classes": int(args.num_classes),
		"images_per_class": int(args.images_per_class),
		"num_samples": int(len(records)),
		"token_sampling": sampling_meta,
		"pre_feature_shape": list(pre_feats.shape),
		"post_feature_shape": list(post_feats.shape),
		"post_tsne_input_shape": list(post_tsne_input.shape),
		"pre_tsne_shape": list(pre_emb.shape),
		"post_tsne_shape": list(post_emb.shape),
		"post_prepare": post_prepare_meta,
		"tsne": {
			"requested_perplexity": float(args.tsne_perplexity),
			"random_state": int(args.tsne_random_state),
			"pre": pre_tsne_meta,
			"post": post_tsne_meta,
		},
		"sae_stage_check": sae_check,
		"files": {
			"pre_tsne_plot": pre_png,
			"post_tsne_plot": post_png,
			"compare_plot": cmp_png,
			"pre_tsne_embedding_npy": pre_emb_npy,
			"post_tsne_embedding_npy": post_emb_npy,
			"pre_sae_tensor_samples_npy": pre_feat_npy,
			"post_sae_latent_samples_npy": post_feat_npy,
			"post_sae_tsne_input_npy": post_tsne_input_npy,
			"labels_npy": label_npy,
			"sampled_classes_json": class_json,
			"sampled_items_json": item_json,
		},
	}

	summary_path = os.path.join(out_dir, "summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("Done.")
	print(f"Summary: {summary_path}")


if __name__ == "__main__":
	main()
