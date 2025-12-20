"""
动态多塔 Transformer 架构

支持任意频率组合，无硬编码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class FrequencyTower(nn.Module):
    """
    单频率塔
    
    对单一频率的时序数据进行编码，并支持重构（AutoEncoder）
    """
    
    def __init__(self, 
                 input_dim: int,
                 seq_len: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 use_ae: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            seq_len: 序列长度
            d_model: Transformer隐藏维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: FFN维度
            dropout: Dropout率
            use_ae: 是否使用自编码器重构任务
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_ae = use_ae
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Decoder (AutoEncoder)
        if self.use_ae:
            decoder_hidden = d_model * 2
            self.decoder = nn.Sequential(
                nn.Linear(d_model, decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, seq_len * input_dim),
                nn.Unflatten(1, (seq_len, input_dim))
            )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            encoded: (batch, d_model)
            reconstructed: (batch, seq_len, input_dim) if use_ae else None
        """
        # 1. Encoding
        # 投影到d_model维度
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        h = self.pos_encoder(h)
        
        # Transformer编码
        h = self.transformer_encoder(h)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步作为编码向量
        encoded = h[:, -1, :]  # (batch, d_model)
        
        # Layer Norm
        encoded = self.layer_norm(encoded)

        # 2. Decoding (Reconstruction)
        reconstructed = None
        if self.use_ae:
            reconstructed = self.decoder(encoded)
        
        return encoded, reconstructed


class MultiTowerTransformer(nn.Module):
    """
    动态多塔 Transformer
    
    根据传入的频率信息动态构建塔，支持 AutoEncoder 辅助任务
    """
    
    def __init__(self,
                 freq_info: Dict[str, Dict],
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 fusion_method: str = 'concat',
                 fusion_hidden_dims: List[int] = [256, 128],
                 freq_configs: Optional[Dict[str, Dict]] = None,
                 use_ae: bool = True):
        """
        Args:
            freq_info: 频率信息字典
                格式: {
                    'monthly': {'seq_len': 6, 'input_dim': 50},
                    'weekly': {'seq_len': 24, 'input_dim': 50},
                }
            d_model: 默认隐藏维度
            nhead: 默认注意力头数
            num_layers: 默认Transformer层数
            dim_feedforward: 默认FFN维度
            dropout: 默认Dropout率
            fusion_method: 融合方法，'concat' 或 'weighted'
            fusion_hidden_dims: 融合MLP的隐藏层维度
            freq_configs: 各频率的特定配置，可覆盖默认值
            use_ae: 是否启用 AutoEncoder 重构辅助任务
        """
        super().__init__()
        
        self.freq_info = freq_info
        self.freq_names = sorted(freq_info.keys())  # 保证顺序一致性
        self.d_model = d_model
        self.fusion_method = fusion_method
        self.freq_configs = freq_configs or {}
        self.use_ae = use_ae
        
        # 动态构建频率塔
        self.towers = nn.ModuleDict()
        tower_output_dims = []
        
        for freq_name in self.freq_names:
            info = freq_info[freq_name]
            # 获取该频率的特定配置，如果没有则使用全局默认值
            cfg = self.freq_configs.get(freq_name, {})
            
            f_d_model = cfg.get('d_model', d_model)
            f_nhead = cfg.get('nhead', nhead)
            f_num_layers = cfg.get('num_layers', num_layers)
            f_dim_feedforward = cfg.get('dim_feedforward', dim_feedforward)
            f_dropout = cfg.get('dropout', dropout)
            
            self.towers[freq_name] = FrequencyTower(
                input_dim=info['input_dim'],
                seq_len=info['seq_len'],
                d_model=f_d_model,
                nhead=f_nhead,
                num_layers=f_num_layers,
                dim_feedforward=f_dim_feedforward,
                dropout=f_dropout,
                use_ae=use_ae
            )
            tower_output_dims.append(f_d_model)
        
        # 融合层
        if fusion_method == 'concat':
            fusion_input_dim = sum(tower_output_dims)
        elif fusion_method == 'weighted':
            # 加权求和要求所有塔的输出维度一致
            if len(set(tower_output_dims)) > 1:
                raise ValueError("Weighted fusion requires all frequency towers to have the same d_model. "
                                 f"Got output dims: {tower_output_dims}")
            fusion_input_dim = tower_output_dims[0]
            # 可学习权重
            self.fusion_weights = nn.Parameter(torch.ones(len(self.freq_names)))
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # 融合MLP
        mlp_layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x_dict: {freq_name: tensor of shape (batch, seq_len, input_dim)}
        
        Returns:
            predictions: (batch, 1)
            reconstructions_dict: {freq_name: reconstructed_tensor}
        """
        # 验证输入
        if set(x_dict.keys()) != set(self.freq_names):
            raise ValueError(
                f"Input frequencies {set(x_dict.keys())} do not match "
                f"model frequencies {set(self.freq_names)}"
            )
        
        # 每个塔独立编码
        tower_outputs = []
        reconstructions_dict = {}
        for freq_name in self.freq_names:  # 使用排序后的顺序
            x = x_dict[freq_name]
            tower_out, reconstructed = self.towers[freq_name](x)  # (batch, d_model), (batch, seq_len, input_dim)
            tower_outputs.append(tower_out)
            if reconstructed is not None:
                reconstructions_dict[freq_name] = reconstructed
        
        # 融合
        if self.fusion_method == 'concat':
            # 直接拼接
            fused = torch.cat(tower_outputs, dim=1)  # (batch, d_model * n_freq)
        elif self.fusion_method == 'weighted':
            # 加权求和
            weights = torch.softmax(self.fusion_weights, dim=0)
            fused = sum(w * out for w, out in zip(weights, tower_outputs))
        
        # MLP预测
        predictions = self.fusion_mlp(fused)  # (batch, 1)
        
        return predictions, reconstructions_dict
    
    def get_num_parameters(self) -> int:
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        n_params = self.get_num_parameters()
        return (
            f"MultiTowerTransformer(\n"
            f"  Frequencies: {self.freq_names}\n"
            f"  Towers: {len(self.towers)}\n"
            f"  d_model: {self.d_model}\n"
            f"  Fusion: {self.fusion_method}\n"
            f"  Parameters: {n_params:,}\n"
            f")"
        )

    @staticmethod
    def get_standardized_config(config: dict) -> dict:
        """标准化配置字典，用于生成唯一ID"""
        if not config:
            return {}
            
        # 提取并标准化 global 配置
        raw_global = config.get("global", {})
        std_global = {
            "dropout": float(raw_global.get("dropout", 0.1)),
            "fusion_method": str(raw_global.get("fusion_method", "concat")).lower(),
            "use_ae": bool(raw_global.get("use_ae", True)),
            "ae_alpha": float(raw_global.get("ae_alpha", 0.1))
        }
        
        # 提取 freq_specific 配置
        std_config = {
            "global": std_global,
            "freq_specific": config.get("freq_specific", {})
        }
        return std_config


def create_model(freq_info: Dict[str, Dict], **kwargs) -> MultiTowerTransformer:
    """
    工厂函数：创建模型
    
    Args:
        freq_info: 频率信息字典
        **kwargs: 其他模型参数
    
    Returns:
        MultiTowerTransformer实例
    """
    return MultiTowerTransformer(freq_info=freq_info, **kwargs)
