import sys
import torch
sys.path.insert(0, './pytorch-image-models')
from rebm.training.utils_architecture import create_convnext_model
from rebm.training.modeling import load_checkpoint
ckpt_path = "checkpoints/convnext_large_cvst_robust.pt"
model = create_convnext_model(model_type="convnext_large", num_classes=1000, normalize_input=False, use_layernorm=True, use_convstem=True)
load_checkpoint(model, ckpt_path)
print("SUCCESS!")
