import torch.nn as nn
from torch import torch
import math

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim=5, model_dim=128, num_heads=8, num_layers=10, pred_length=5, dropout=0.1):
        super(SpatioTemporalTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.pred_length = pred_length
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        
        # 地理位置编码
        self.geo_embedding = nn.Linear(2, model_dim // 4)  # 单独处理经纬度
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, 1)  # 预测震级
        )
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_dim)
        
        # 分离地理特征和其他特征
        geo_features = x[:, :, 1:3]  # 纬度和经度
        other_features = torch.cat([x[:, :, :1], x[:, :, 3:]], dim=-1)  # 震级和其他特征
        
        # 嵌入主要特征
        x_embed = self.input_embedding(other_features)
        x_embed = self.input_norm(x_embed)
        
        # 嵌入地理特征
        geo_embed = self.geo_embedding(geo_features)
        
        # 合并特征
        x_combined = torch.cat([x_embed, geo_embed], dim=-1)
        
        # 添加位置编码
        x_combined = self.positional_encoding(x_combined)
        
        # Transformer处理
        encoded = self.transformer_encoder(x_combined)
        
        # 预测未来地震
        predictions = self.prediction_head(encoded[:, -self.pred_length:, :])
        
        return predictions.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        x = self.norm(x)
        return self.dropout(x)