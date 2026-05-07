# SAE Inspection for WRN34x10-CIFAR10

可解释性研究：分析 SAE 学到的特征与 CIFAR-10 语义之间的关系。

## 主版本 SAE

- **Checkpoint**: `../checkpoints/k64_exp32_no_resample_ep10/epoch_8.pt`
- **Config**: k=64, expansion=32, d_lat=20480, 10 epochs, no_resample
- **EV**: 0.992, Loss: 0.0023
- **Active features**: ~1600 (dead_fraction ≈ 92%, structural)

## 目录结构

```
inspection/
├── README.md              # 本文档
├── sae_feature_inspect.py # Max activation 可视化
├── steering_experiment.py # 特征 steering 实验
├── feature_stats.py       # 特征统计与分类关联
└── outputs/               # 输出图表与结果
```

## 待做分析

1. **Max Activation 可视化** — 对每个 active feature，找出使其激活最强的输入图像
2. **Feature-Label 关联** — 统计每个 feature 在哪些类别上激活最强
3. **Steering 实验** — 沿特定特征方向修改激活，观察分类结果变化
4. **对抗样本分析** — 检查对抗扰动是否对应特定 SAE 特征的变化
