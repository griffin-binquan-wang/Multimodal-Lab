import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()

        # 定义一个简单的两层 MLP
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        # x 是来自 CLIP 的图像特征
        return self.projector(x)