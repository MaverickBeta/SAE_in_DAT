import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKAutoencoder(nn.Module):
    def __init__(self, d_in, d_lat, k):
        super().__init__()
        self.d_in = d_in
        self.d_lat = d_lat
        self.k = k

        # --- 改进 1: 更好的初始化 ---
        # 1. Decoder 初始化为单位向量
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_lat, d_in)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.set_decoder_norm_to_unit_norm()

        # 2. Encoder 初始化为 Decoder 的转置 (Tied Init)，这对 TopK 收敛很有帮助
        self.W_enc = nn.Parameter(self.W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_lat))

    def set_decoder_norm_to_unit_norm(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

    def encode(self, x, return_pre_acts=False):
        # 1. Pre-activation
        pre_acts = (x @ self.W_enc) + self.b_enc
        
        # 2. ReLU (保证特征非负)
        post_relu = F.relu(pre_acts)
        
        # 3. TopK Selection
        topk_values, topk_indices = torch.topk(post_relu, k=self.k, dim=-1)
        
        # 4. Construct sparse z
        z = torch.zeros_like(post_relu)
        z.scatter_(-1, topk_indices, topk_values)
        
        if return_pre_acts:
            return z, post_relu # 返回 pre_relu 用于计算 Aux Loss
        return z

    def forward(self, x):
        # 训练时我们需要 post_relu 来计算 Aux Loss
        z, post_relu_acts = self.encode(x, return_pre_acts=True)
        x_reconstruct = (z @ self.W_dec) + self.b_dec
        return x_reconstruct, z, post_relu_acts