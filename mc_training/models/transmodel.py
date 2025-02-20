import torch
import torch.nn as nn
import torch.nn.functional as F
from mc_training.models.layers.column_normalization import PerFeatureNormalization
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建位置向量，保持为一维张量 [max_len]
        position = torch.arange(0, max_len, dtype=torch.float)

        # 创建分母项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 现在明确地处理维度
        for i in range(0, d_model, 2):
            # 使用外积计算方式，确保维度正确
            pe[:, i] = position * div_term[i // 2]  # 这会自动进行正确的广播
            pe[:, i] = torch.sin(pe[:, i])

            if i + 1 < d_model:
                pe[:, i + 1] = position * div_term[i // 2]
                pe[:, i + 1] = torch.cos(pe[:, i + 1])

        # 添加batch维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # 注册为buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            带位置编码的张量 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=30):  # 这里 max_len 设为 30
#         super(PositionalEncoding, self).init()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1, 30, d_model)
#         self.register_buffer('pe', pe)  # 作为 buffer，避免训练时被更新
#
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]

class TransformerEncoderDualInput(nn.Module):
    def __init__(self, class_num=2, feature_num=95, window_size=1, seq_length=30, d_model=128, nhead=16, num_layers=6, dim_feedforward=256, dropout=0.15):

        super(TransformerEncoderDualInput, self).__init__()

        # Input embedding layer
        self.input_proj = nn.Linear(feature_num, d_model)

        # Positional embedding
        self.position_encoding = PositionalEncoding(d_model, max_len=30)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, class_num)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_length, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.pfn = PerFeatureNormalization(method="min_max")



    def forward(self, x, img):

        # Normalize x
        x = self.pfn(x)  # [batch_size, seq_length, feature_num]

        x = x.squeeze(1)

        # Project input
        x = self.input_proj(x)  # [batch_size, seq_length, d_model]

        # Add positional embedding
        x = x + self.position_encoding(x)

        x = self.transformer_encoder(x)  # [batch_size, seq_length, d_model]

        output = self.output_proj(x[:, 0, :])  # [batch_size, class_num]

        return output