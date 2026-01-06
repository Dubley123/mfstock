"""
多频率数据加载器（高性能版本）
使用字典遍历和二分查找，避免DataFrame操作和groupby
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Tuple
from mfstock.dataset.dataset import FeatureDataset, TargetDataset
from mfstock.dataset.utils import window_to_periods, generate_time_points


class MultiFreqDataLoader(Dataset):
    """
    多频率数据加载器（高性能版本）
    
    职责：
    1. 接收 feature_datasets + target_dataset + 时间范围 + lookback_windows + rebalance_freq
    2. 根据 rebalance_freq 在时间范围内生成样本时间点
    3. 对每个时间点和每个股票，生成一个样本
    4. 从 feature_datasets 中按 lookback 提取 X，从 target_dataset 提取 y
    
    性能优化：
    - 字典遍历：直接遍历 target_dataset.stock_data，O(1) 访问
    - 向量化二分查找：使用 np.searchsorted 批量定位时间点
    - 零拷贝转换：torch.from_numpy() 替代 torch.FloatTensor()
    - 整数存储：samples 存储纳秒级 int64，避免反复转换
    
    关键理解：
    - pred_start 和 pred_end 定义预测值的时间范围（不包括lookback）
    - rebalance_freq 定义在这个范围内按什么频率生成样本
    - 例如：[2020-01, 2020-06), rebalance_freq=1m → 在2020-01, 2020-02, ..., 2020-05各生成样本
    """
    
    def __init__(self,
                 feature_datasets: Dict[str, FeatureDataset],
                 target_dataset: TargetDataset,
                 pred_start: pd.Timestamp,
                 pred_end: pd.Timestamp,
                 lookback_windows: Dict[str, str],
                 rebalance_freq: str):
        """
        初始化数据加载器
        
        Args:
            feature_datasets: 特征数据集字典，{freq_name: FeatureDataset}
            target_dataset: 目标数据集
            pred_start: 预测时间范围起始（包含）
            pred_end: 预测时间范围结束（不包含）
            lookback_windows: 回看窗口字典，{freq_name: "6m"}
            rebalance_freq: 调仓频率，如"1m"
        """
        self.feature_datasets = feature_datasets
        self.target_dataset = target_dataset
        self.pred_start = pred_start
        self.pred_end = pred_end
        self.lookback_windows = lookback_windows
        self.rebalance_freq = rebalance_freq
        
        self.time_col = target_dataset.time_col
        self.stock_col = target_dataset.stock_col
        
        # 将lookback_window转换为各频率需要的记录数
        self.lookback_periods = {}
        for freq_name, window in lookback_windows.items():
            dataset = feature_datasets[freq_name]
            n_periods = window_to_periods(window, dataset.frequency)
            self.lookback_periods[freq_name] = n_periods
        
        # 生成样本列表：[(time, stock_code), ...]
        self.samples = self._generate_samples()
        
        print(f"DataLoader initialized: {len(self.samples)} samples "
              f"in [{pred_start.date()}, {pred_end.date()})")
    
    def _generate_samples(self) -> List[Tuple[int, str]]:
        """
        根据 rebalance_freq 生成样本列表（高性能版本）
        
        核心优化：
        1. 直接遍历 target_dataset.stock_data 字典，避免 DataFrame 操作
        2. 使用 np.searchsorted 向量化二分查找，替代循环+groupby
        3. 样本存储为 (time_int64, stock_code)，避免 __getitem__ 中反复转换
        
        Returns:
            [(time_int64, stock_code), ...] 样本列表
        """
        # 1. 生成理论时间点并转换为 int64
        time_points = generate_time_points(self.pred_start, self.pred_end, self.rebalance_freq)
        time_points_int = np.array([pd.Timestamp(tp).value for tp in time_points], dtype=np.int64)
        
        start_int = pd.Timestamp(self.pred_start).value
        end_int = pd.Timestamp(self.pred_end).value
        
        # 2. 遍历每只股票的索引数据
        samples = []
        
        for stock_code, stock_info in self.target_dataset.stock_data.items():
            times = stock_info['times']  # 已经是 int64 数组
            
            # 筛选在 [pred_start, pred_end) 范围内的时间点
            mask = (times >= start_int) & (times < end_int)
            valid_times = times[mask]
            
            if len(valid_times) == 0:
                continue
            
            # 3. 向量化二分查找：对每个调仓时间点，找到该股票最接近的实际时间
            for tp_int in time_points_int:
                # 找到 <= tp_int 的最大索引
                idx = np.searchsorted(valid_times, tp_int, side='right') - 1
                
                if idx >= 0:
                    # 存在 <= tp_int 的数据点
                    actual_time = valid_times[idx]
                    samples.append((actual_time, stock_code))
        
        if len(samples) == 0:
            print(f"Warning: No samples generated for time range "
                  f"[{self.pred_start.date()}, {self.pred_end.date()})")
        
        return samples
    
    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, dict]:
        """
        获取单个样本（零拷贝优化版本）
        
        从各个频率的 feature_datasets 中提取 X，
        从 target_dataset 中提取 y
        
        性能优化：
        - 使用 torch.from_numpy() 实现零拷贝转换
        - 时间已存储为 int64，避免重复转换
        
        Args:
            idx: 样本索引
        
        Returns:
            (X_dict, y, metadata)
            - X_dict: {freq_name: tensor of shape (seq_len, n_features)}
            - y: tensor of shape (1,)
            - metadata: {'time': timestamp, 'stock': code}
        """
        time_int, stock = self.samples[idx]
        
        # 1. 从 feature_datasets 中提取特征（零拷贝）
        X_dict = {}
        for freq_name, dataset in self.feature_datasets.items():
            n_periods = self.lookback_periods[freq_name]
            
            # get_lookback_data 返回 np.float32 数组
            features = dataset.get_lookback_data(stock, time_int, n_periods)
            
            # 零拷贝转换：from_numpy 与原数组共享内存
            X_dict[freq_name] = torch.from_numpy(features)
        
        # 2. 从 target_dataset 中提取目标值
        y_value = self.target_dataset.get_target(stock, time_int)
        y = torch.tensor([y_value], dtype=torch.float32)
        
        # 3. 元数据（转回 pd.Timestamp 供用户使用）
        metadata = {
            'time': pd.Timestamp(time_int),
            'stock': stock
        }
        
        return X_dict, y, metadata


def collate_fn(batch: List[Tuple]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[dict]]:
    """
    批处理函数
    
    由于lookback_periods固定，所有样本的序列长度相同，可以直接stack
    
    Args:
        batch: [(X_dict, y, metadata), ...]
    
    Returns:
        (X_batch, y_batch, metadata_list)
        - X_batch: {freq_name: tensor of shape (batch, seq_len, n_features)}
        - y_batch: tensor of shape (batch, 1)
        - metadata_list: [metadata, ...]
    """
    X_list, y_list, metadata_list = zip(*batch)
    
    # 合并各频率特征
    X_batch = {}
    freq_names = X_list[0].keys()
    
    for freq_name in freq_names:
        freq_tensors = [x[freq_name] for x in X_list]
        # 直接stack，因为序列长度固定
        X_batch[freq_name] = torch.stack(freq_tensors, dim=0)
    
    # 合并目标
    y_batch = torch.stack(y_list, dim=0)
    
    return X_batch, y_batch, list(metadata_list)


def create_dataloader(dataset: MultiFreqDataLoader,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0) -> TorchDataLoader:
    """
    创建PyTorch DataLoader
    
    Args:
        dataset: MultiFreqDataLoader实例
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作线程数
    
    Returns:
        DataLoader
    """
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
