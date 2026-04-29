import torch
import torch.nn as nn
from MHA import MultiHeadAttention

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        # 使用一个卷积核大小为 patch_size，步长也为 patch_size 的卷积层
        # 它一步完成了切块、展平和投影三个操作
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x) # 卷积后变成 (B, 768, 14, 14)
        x = x.flatten(2) # 展平后两个维度变为 (B, 768, 196)
        x = x.transpose(1,2) # 最后交换维度变为 (B, 196, 768)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 num_layers=12, num_heads=12, d_ff=3072, num_classes=1000, dropout=0.1):
        super().__init__()

        # 实例化刚才写的切块层
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3,embed_dim)
        # 计算总共有多少个块 (224/16 * 224/16 = 196)
        num_patches = (img_size // patch_size) ** 2

        # 定义 [CLS] token：这是一个可学习的向量，形状为 (1, 1, 768)
        # 我们用 nn.Parameter 包装，这样它就会在反向传播中被更新
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 定义位置编码：对应 196个块 + 1个 cls_token = 197
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 堆叠 Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最后的归一化和分类头 (Head)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # 第一阶段：Embedding
        # 1. 基础切块投影 (B, 3, 224, 224) -> (B, 196, 768)
        x = self.patch_embed(x)
        B = x.shape[0]
        # 2. 准备 cls_token：将 (1, 1, 768) 扩展成 (B, 1, 768)
        # 这样才能和 Batch 里的每一条数据对齐
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 3. 拼接：把 cls_token 放在 196 个 patch 的最前面
        # 结果维度: (B, 197, 768)
        x = torch.cat((cls_tokens, x), dim=1)
        # 4. 叠加位置编码：(B, 197, 768) + (1, 197, 768)
        # PyTorch 的广播机制会自动处理 Batch 维度的对齐
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 第二阶段：Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 第三阶段：提取 [CLS] token 的输出进行分类
        # x[:, 0] 取出序列中的第一个位置
        cls_token_out = x[:, 0]
        out = self.head(cls_token_out)

        return out
    
class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Pre-Norm: 先归一化，再进 Attention，最后残差连接
        # 注意：残差连接是连在“进入 Norm 之前”的 x 上
        x = x + self.dropout1(self.mha(self.norm1(x), self.norm1(x), self.norm1(x))[0])

        # 2. Pre-Norm: 先归一化，再进 Feed Forward，最后残差连接
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x
    
# if __name__ == "__main__":
#     # 设定参数 (参考 ViT-Base 规格)
#     img_size = 224
#     patch_size = 16
#     embed_dim = 768
#     num_classes = 10  # 假设我们要分10类，比如 CIFAR-10
    
#     # 模拟输入：1张 224x224 的 RGB 图片
#     # 维度: (Batch, Channels, Height, Width)
#     dummy_input = torch.randn(1, 3, img_size, img_size)
    
#     # 实例化模型
#     model = VisionTransformer(
#         img_size=img_size, 
#         patch_size=patch_size, 
#         embed_dim=embed_dim, 
#         num_layers=12, 
#         num_heads=12, 
#         d_ff=3072, 
#         num_classes=num_classes, 
#         dropout=0.1
#     )
    
#     # 前向传播
#     output = model(dummy_input)
    
#     print(f"输入形状: {dummy_input.shape}")
#     print(f"输出形状: {output.shape}") # 应该是 (1, 10)
    
#     # 计算总参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"模型总参数量: {total_params / 1e6:.2f}M")