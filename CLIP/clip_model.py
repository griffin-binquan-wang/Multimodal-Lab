import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_projection = nn.Linear(768, embed_dim)
        self.text_projection = nn.Linear(512, embed_dim)

        # 温度系数：初始化为 ln(1/0.07)，也就是约 2.65
        # 使用 nn.Parameter 确保它在训练时会被更新
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, image, text):
        # 1. 进入视觉分支
        # 得到输出形状: (Batch, Num_Classes) 或者 (Batch, Embed_Dim) 
        # 注意：如果你之前 ViT 最后带了 head，输出就是分类结果。
        # 在 CLIP 中，我们通常要的是 head 之前的那个特征。
        image_features = self.image_encoder(image, return_features=True)

        # 2. 进入文本分支
        # 得到输出形状: (Batch, Seq_Len, d_model)
        text_outputs = self.text_encoder(text)

        # 3. 提取文本的特征 (取 [CLS] 位置，即序列第一个位置)
        # 形状变成: (Batch, 512)
        text_features = text_outputs[:, 0, :]

        # 4. 投影到统一维度 (256)
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)

        # 5. 归一化 (L2 Normalization)
        # 这是为了让向量长度为 1，这样后面做矩阵乘法得到的就是“余弦相似度”
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        #  Batch
        batch_size = image.shape[0]

        # 1. 构造目标：[0, 1, 2, ..., B-1]
        # 表示第 i 个图像应该对应第 i 个文本
        labels = torch.arange(batch_size, device=image.device)

        # 2. 计算横向损失：图像找文本
        # 3. 计算纵向损失：文本找图像
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_image.t(), labels)

        return (loss_i + loss_t) / 2
    
if __name__ == "__main__":
    # 模拟参数
    batch_size = 2
    vocab_size = 1000
    
    # 1. 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 找到上一层目录
    parent_dir = os.path.dirname(current_dir)
    # 3. 把上一层目录加入到搜索路径中
    sys.path.append(parent_dir)

    # 实例化你的组件
    # 注意：这里需要确保你导入了之前的类
    from ViT.model import VisionTransformer
    from text_encoder import TransformerEncoder #[cite: 1]
    
    img_enc = VisionTransformer(img_size=224, patch_size=16, embed_dim=768)
    txt_enc = TransformerEncoder(vocab_size=vocab_size, d_model=512, num_layers=4, num_heads=8, d_ff=1024) #[cite: 1]
    
    # 构建 CLIP
    clip_model = CLIP(img_enc, txt_enc)
    
    # 伪造数据
    fake_img = torch.randn(batch_size, 3, 224, 224)
    fake_txt = torch.randint(0, vocab_size, (batch_size, 20))
    
    # 跑一下 forward
    loss = clip_model(fake_img, fake_txt)
    print(f"CLIP 训练损失: {loss.item():.4f}")
    
    # 反向传播测试
    loss.backward()
    print("反向传播成功！参数已更新。")
