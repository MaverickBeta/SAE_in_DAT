import torch
import sys
sys.path.insert(0, './pytorch-image-models')
from rebm.training.utils_architecture import create_convnext_model
m = create_convnext_model('convnext_large', 1000, False, True, True)
x = torch.randn(1, 3, 224, 224)
acts = []
def hook(m, i, o):
    acts.append(o.shape)
m.stages[2].blocks[26].register_forward_hook(hook)
m(x)
print(acts[0])
