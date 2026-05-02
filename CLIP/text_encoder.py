import torch.nn as nn
import torch.nn.functional as F
from ViT.MHA import MultiHeadAttention, PositionalEncoding

class EncoderLayer(nn.Module):
    """
    单个编码器层 (Single Encoder Layer)
    包含: 多头自注意力 (Self-Attention) + 前馈网络 (Feed Forward)
    """
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Linear(d_ff,d_model)
        )

        # 层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        # 1. 自注意力 + 残差连接 + 归一化 (Self-Attn + Residual + Norm)
        norm_x = self.norm1(x)
        attn_output, _ = self.mha(norm_x,norm_x,norm_x,mask)
        x = x + self.dropout1(attn_output)

        # 2. 前馈网络 + 残差连接 + 归一化 (Feed Forward + Residual + Norm)
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)
        return x

class TransformerEncoder(nn.Module):
    """
    编码器整体 (Full Transformer Encoder)
    负责将源序列转换为连续的表示向量
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 堆叠 num_layers 个编码器层 (Stack num_layers of EncoderLayers)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,x,mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x,mask=mask)
        return x
    