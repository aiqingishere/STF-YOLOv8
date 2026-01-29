import torch
import torch.nn as nn
import torch.nn.functional as F

class TextResidualFusion(torch.nn.Module):
    def __init__(self, text_dim, feat_dim, scale=1.0):
        super().__init__()
        self.text_proj = torch.nn.Linear(text_dim, feat_dim)  # ⚡ 保证输出通道=feat通道
        self.scale = scale

    def forward(self, feat, text_feat):
        if text_feat is None:
            return feat

        # ⚡ device 保证
        text_feat = text_feat.to(feat.device)

        # 映射到 feat 通道
        t = self.text_proj(text_feat)  # [B, feat_dim]
        gamma = torch.sigmoid(t).view(t.shape[0], t.shape[1], 1, 1)  # [B, C, 1, 1]
        out = feat + self.scale * gamma * feat

        print(f"🔥 TextResidualFusion called: feat {feat.shape}, text_feat {text_feat.shape}, gamma {gamma.shape}")
        return out
