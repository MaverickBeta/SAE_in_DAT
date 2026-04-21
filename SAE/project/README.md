# ImageNet Small Top-K Sparse Autoencoder (SAE)

This project trains a 64k dimension Top-K SAE on ConvNeXt-Large image features.

## Setup
We use the pre-extracted `imagenet_small_features_merged.pt` tensor for training, which allows O(1) random batch sampling from 7.16M total feature vectors.

## How to train:
Simply run the train script specifying the GPU you want to use. You can tweak parameters such as `--k`, `--lr`, and `--steps`.
It relies on `wandb` for logging by default. Use `--no_wandb` if you wish to run it offline.

```bash
conda activate rebm
cd /Data_share/hongyi/DAT/SAE/project
python3 train.py --gpu_id 2 --k 32 --d_lat 65536 --steps 50000
```
Checkpoints will be saved automatically into the `checkpoints/` folder.
