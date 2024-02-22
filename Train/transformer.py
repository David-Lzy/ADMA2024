# 还没测试

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)[:,:(d_model+1)//2]
        pe[:, 1::2] = torch.cos(position * div_term)[:,:(d_model)//2]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假设 x 的形状是 [batch, seq, feature]
        x = x + self.pe[:, :x.size(1), :]
        return x



# 定义Transformer模型
class ClassifierTransformer(nn.Module):
    def __init__(
        self,
        input_shape,
        nb_classes,
        num_layers=2,
        dim_feedforward=128,
        dropout_rate=0.25,
        *wargs,
        **kwargs,
    ):
        super(ClassifierTransformer, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        channels = input_shape[1]
        seq_length = input_shape[2]

        # 使用单头注意力
        nhead = kwargs.get("nhead", 1)

        self.positional_encoding = PositionalEncoding(d_model=input_shape[2])
        self.dropout = nn.Dropout(p=dropout_rate)

        # batch_first (bool) – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=seq_length,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(seq_length, nb_classes)

    def forward(self, x):
        # print(1, x.shape)
        #x = x.permute(0, 2, 1)
        # 重新排列为 (batch, seq_length, features)
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

        # print(2, x.shape)
        x = self.positional_encoding(x)
        # print(3, x.shape)
        x = self.dropout(x)  # Dropout after positional encoding
        # print(4, x.shape)
        x = self.transformer_encoder(x)
        # print(5, x.shape)
        x = self.relu(x)
        x = x.mean(dim=1)  # Global average pooling
        # print(6, x.shape)
        x = self.fc(x)
        return x
