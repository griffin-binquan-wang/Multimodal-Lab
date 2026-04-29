import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    点积缩放注意力机制 (Scaled Dot-Product Attention)
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self):
        super().__init__()

    def forward(self,q,k,v,mask=None):
        # d_k: 特征维度，用于缩放以防止梯度消失
        # d_k: Dimension of keys, used for scaling to prevent vanishing gradients
        d_k = q.size(-1)

        # 计算注意力分数: QK^T / sqrt(d_k)
        # Compute scores: Q multiplied by K transposed, divided by scaling factor
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
        
        # 如果存在掩码，将掩码为0的位置填充为极小值，使其在Softmax后权重接近0
        # If mask exists, fill 0 positions with a very large negative value
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 对最后一个维度做 Softmax，得到权重分布
        # Softmax on the last dimension to get attention weight distribution
        attn = F.softmax(scores,dim=-1)

        # 将权重应用于数值矩阵 V
        # Apply weights to value matrix V
        output = torch.matmul(attn,v)
        return output,attn
    
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    允许模型在不同子空间同时关注来自不同位置的信息
    """
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出 O 的线性映射层
        # Define linear layers for Q, K, V and final output O
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
     
    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0)

        # 1. 线性投影并拆分为多头结构
        # 1. Linear projection and split into multi-heads
        q = self.W_q(q).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        k = self.W_k(k).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        v = self.W_v(v).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        # 2. 调用点积缩放注意力
        # 2. Apply Scaled Dot-Product Attention
        attn_output, attn_weights = ScaledDotProductAttention()(q,k,v, mask=mask)
        
        # 3. 拼接所有头的输出 (Concatenate heads)
        # Shape: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.d_k)
        
        # 4. 最后的线性层输出
        # 4. Final linear layer output
        output = self.W_o(attn_output)
        return output, attn_weights