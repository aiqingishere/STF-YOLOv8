import torch
import torch.nn as nn
import torch.nn.functional as F


class TextResidualFusion(nn.Module):
    """
    CLIP Text → CNN Feature 的残差调制模块
    - 不改变 feature map 尺寸
    - channel-wise modulation
    """

    def __init__(self, text_dim: int, feat_dim: int):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # ⚠️ 关键：不要初始化为 0
        self.scale = nn.Parameter(torch.tensor(0.1))  # 或 0.05
                # ⭐⭐⭐ 关键：补 stride，欺骗 YOLO 的统一逻辑
        self.stride = torch.tensor([1])

    def forward(self, feat: torch.Tensor, text_feat: torch.Tensor | None):
        """
        feat: [B, C, H, W]
        text_feat: [B, text_dim]

        """

        #raise RuntimeError("TEXT FUSION IS CALLED")
        print("🔥 TEXT FUSION EXECUTED")


        
        # CLIP 标准操作
        text_feat = F.normalize(text_feat, dim=-1)

        # [B, C]
        t = self.text_proj(text_feat)

        # channel-wise gate
        gamma = torch.sigmoid(t).unsqueeze(-1).unsqueeze(-1)

        # residual modulation
        return feat * (1.0 + self.scale * gamma)


# 兼容旧引用
TextFusion = TextResidualFusion
