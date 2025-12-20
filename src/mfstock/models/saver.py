"""
模型保存器：配置哈希与智能路径管理
"""

import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import pandas as pd


def generate_config_hash(config: Dict[str, Any]) -> Tuple[Dict, str]:
    """
    生成配置哈希
    
    根据关键配置参数生成唯一的MD5哈希值
    
    Args:
        config: 配置字典，应包含：
            - frequencies: 使用的频率列表（已排序）
            - lookback_windows: 回看窗口配置
            - train_window: 训练窗口
            - val_window: 验证窗口
            - test_window: 测试窗口
            - rebalance_freq: 调仓频率
            - test_start_time: 测试开始时间
            其他影响数据加载的关键参数
    
    Returns:
        (config_dict, config_id)
    """
    # 提取关键配置
    key_config = {
        'frequencies': sorted(config.get('frequencies', [])),
        'lookback_windows': config.get('lookback_windows', {}),
        'train_window': str(config.get('train_window', '')),
        'val_window': str(config.get('val_window', '')),
        'test_window': str(config.get('test_window', '')),
        'rebalance_freq': str(config.get('rebalance_freq', '')),
        'test_start_time': str(config.get('test_start_time', '')),
    }
    
    # 序列化为JSON字符串（确保顺序一致）
    config_str = json.dumps(key_config, sort_keys=True)
    
    # 生成MD5哈希
    config_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    return key_config, config_id


class ModelSaver:
    """
    模型保存器
    
    职责：
    1. 管理实验结果目录结构
    2. 保存/加载模型和配置
    3. 判断模型是否已存在
    """
    
    def __init__(self, 
                 base_dir: str = "output",
                 config: Optional[Dict] = None):
        """
        Args:
            base_dir: 实验结果根目录
            config: 配置字典（可选，如果提供则自动生成config_id）
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if config is not None:
            self.config, self.config_id = generate_config_hash(config)
            self.output_dir = self.base_dir / self.config_id
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存配置文件
            self._save_config()
        else:
            self.config = None
            self.config_id = None
            self.output_dir = None
    
    def set_config(self, config: Dict):
        """
        设置配置并初始化目录
        
        Args:
            config: 配置字典
        """
        self.config, self.config_id = generate_config_hash(config)
        self.output_dir = self.base_dir / self.config_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()
    
    def _save_config(self):
        """保存配置到YAML文件"""
        if self.config is None:
            return
        
        config_file = self.output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Config saved to {config_file}")
    
    def get_window_dir(self, window_idx: int) -> Path:
        """
        获取窗口目录
        
        Args:
            window_idx: 窗口索引
        
        Returns:
            窗口目录路径
        """
        if self.output_dir is None:
            raise ValueError("Config not set. Call set_config() first.")
        
        window_dir = self.output_dir / f"window_{window_idx}"
        window_dir.mkdir(parents=True, exist_ok=True)
        return window_dir
    
    def save_model(self, 
                   model: torch.nn.Module,
                   window_idx: int,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: Optional[int] = None,
                   metrics: Optional[Dict] = None):
        """
        保存模型
        
        Args:
            model: 模型
            window_idx: 窗口索引
            optimizer: 优化器（可选）
            epoch: 当前epoch（可选）
            metrics: 指标字典（可选）
        """
        window_dir = self.get_window_dir(window_idx)
        model_path = window_dir / "best_model.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'window_idx': window_idx,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self,
                   model: torch.nn.Module,
                   window_idx: int,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu') -> Dict:
        """
        加载模型
        
        Args:
            model: 模型实例
            window_idx: 窗口索引
            optimizer: 优化器（可选）
            device: 设备
        
        Returns:
            checkpoint字典
        """
        window_dir = self.get_window_dir(window_idx)
        model_path = window_dir / "best_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {model_path}")
        return checkpoint
    
    def model_exists(self, window_idx: int) -> bool:
        """
        检查模型是否存在
        
        Args:
            window_idx: 窗口索引
        
        Returns:
            是否存在
        """
        if self.output_dir is None:
            return False
        
        window_dir = self.output_dir / f"window_{window_idx}"
        model_path = window_dir / "best_model.pt"
        return model_path.exists()
    
    def save_window_predictions(self,
                                predictions_df,
                                window_idx: int):
        """
        保存单个窗口的预测结果（临时文件，防止中断丢失）
        文件名格式: pred_{config_id}_{window_idx}.parquet
        列名格式: TradingDate, Stkcd, pred_{config_id}_{window_idx}
        
        Args:
            predictions_df: 预测结果DataFrame (必须包含 'time', 'stock', 'prediction' 列)
            window_idx: 窗口索引
        """
        import pandas as pd
        window_dir = self.get_window_dir(window_idx)
        
        # 生成标准文件名和列名
        config_id = self.get_config_id()
        pred_col_name = f"pred_{config_id}_{window_idx}"
        filename = f"{pred_col_name}.parquet"
        save_path = window_dir / filename
        
        # 创建标准格式的DataFrame：TradingDate, Stkcd, pred_{config_id}_{window_idx}
        result_df = pd.DataFrame()
        
        # 确保time列是pd.Timestamp格式并重命名为TradingDate
        if 'time' in predictions_df.columns:
            if predictions_df['time'].dtype == 'int64':
                result_df['TradingDate'] = pd.to_datetime(predictions_df['time'])
            else:
                result_df['TradingDate'] = predictions_df['time']
        else:
            raise ValueError("predictions_df must contain 'time' column")
        
        # 重命名stock列为Stkcd
        if 'stock' in predictions_df.columns:
            result_df['Stkcd'] = predictions_df['stock']
        else:
            raise ValueError("predictions_df must contain 'stock' column")
        
        # 重命名prediction列为pred_{config_id}_{window_idx}
        if 'prediction' in predictions_df.columns:
            result_df[pred_col_name] = predictions_df['prediction']
        else:
            raise ValueError("predictions_df must contain 'prediction' column")
        
        result_df.to_parquet(save_path, index=False)
        print(f"Window {window_idx} predictions saved to {save_path}")
        print(f"  Columns: {list(result_df.columns)}")
    
    def save_predictions(self, predictions_df: pd.DataFrame):
        """
        保存完整的预测结果DataFrame
        
        Args:
            predictions_df: 包含 [time, stock, prediction, window_idx] 的DataFrame
        """
        if self.output_dir is None:
            raise ValueError("Config not set. Call set_config() first.")
        
        config_id = self.get_config_id()
        
        # 重命名列
        result_df = pd.DataFrame()
        
        # 重命名time列为TradingDate
        if 'time' in predictions_df.columns:
            result_df['TradingDate'] = predictions_df['time']
        else:
            raise ValueError("predictions_df must contain 'time' column")
        
        # 重命名stock列为Stkcd
        if 'stock' in predictions_df.columns:
            result_df['Stkcd'] = predictions_df['stock']
        else:
            raise ValueError("predictions_df must contain 'stock' column")
        
        # 重命名prediction列为pred_{config_id}
        if 'prediction' in predictions_df.columns:
            result_df[f'pred_{config_id}'] = predictions_df['prediction']
        else:
            raise ValueError("predictions_df must contain 'prediction' column")
        
        # 保存
        filename = f"pred_{config_id}.parquet"
        save_path = self.output_dir / filename
        result_df.to_parquet(save_path, index=False)
        
        print(f"✓ All predictions saved to {save_path}")
        print(f"  Total samples: {len(result_df)}")
        print(f"  Columns: {list(result_df.columns)}")
        if 'TradingDate' in result_df.columns:
            print(f"  Time range: {result_df['TradingDate'].min()} to {result_df['TradingDate'].max()}")
    
    def merge_all_predictions(self):
        """
        合并所有窗口的预测结果为单个文件
        文件名格式: pred_{config_id}.parquet
        输出格式: TradingDate, Stkcd, pred_{config_id}
        
        Returns:
            合并后的DataFrame
        """
        if self.output_dir is None:
            raise ValueError("Config not set. Call set_config() first.")
        
        import pandas as pd
        config_id = self.get_config_id()
        
        # 遍历所有window目录，按顺序收集预测文件
        window_dfs = []
        
        for window_dir in sorted(self.output_dir.glob("window_*")):
            window_idx = int(window_dir.name.split('_')[1])
            pred_col_name = f"pred_{config_id}_{window_idx}"
            pred_file = window_dir / f"{pred_col_name}.parquet"
            
            if pred_file.exists():
                df = pd.read_parquet(pred_file)
                # 重命名预测列为统一的列名
                df.rename(columns={pred_col_name: f'pred_{config_id}'}, inplace=True)
                window_dfs.append(df)
        
        if len(window_dfs) == 0:
            print("Warning: No window predictions found to merge!")
            return None
        
        # 直接纵向拼接所有窗口的预测（时间段不重合）
        merged_df = pd.concat(window_dfs, ignore_index=True)
        
        # 按TradingDate和Stkcd排序
        merged_df = merged_df.sort_values(['TradingDate', 'Stkcd'])
        
        # 保存合并结果
        filename = f"pred_{config_id}.parquet"
        save_path = self.output_dir / filename
        merged_df.to_parquet(save_path, index=False)
        
        print(f"\n✓ Merged {len(window_dfs)} windows into {save_path}")
        print(f"  Total samples: {len(merged_df)}")
        print(f"  Columns: {list(merged_df.columns)}")
        if 'TradingDate' in merged_df.columns:
            print(f"  Time range: {merged_df['TradingDate'].min()} to {merged_df['TradingDate'].max()}")
        
        return merged_df
    
    
    def get_config_id(self) -> str:
        """返回当前配置ID"""
        if self.config_id is None:
            raise ValueError("Config not set")
        return self.config_id
    
    def __repr__(self):
        if self.config_id is None:
            return "ModelSaver(config not set)"
        return f"ModelSaver(config_id={self.config_id}, exp_dir={self.output_dir})"
