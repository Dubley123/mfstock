"""
滚动窗口管理器
"""

import pandas as pd
from typing import Dict, Union, Optional, Tuple
from mfstock.dataset.dataset import FeatureDataset, TargetDataset
from mfstock.dataset.dataloader import MultiFreqDataLoader, create_dataloader
from mfstock.dataset.utils import parse_window_str
from torch.utils.data import DataLoader


class RollingWindow:
    """
    滚动窗口管理器
    
    职责：
    1. 接收 feature_datasets + target_dataset
    2. 根据 train/val/test_window 定义数据集的预测时间范围
    3. 根据 rebalance_freq 控制窗口内的样本生成频率
    4. 迭代生成 (train_loader, val_loader, test_loader) 三元组
    
    不负责：具体的数据提取和样本构造（交给DataLoader）
    """
    
    def __init__(self,
                 feature_datasets: Dict[str, FeatureDataset],
                 target_dataset: TargetDataset,
                 lookback_windows: Dict[str, str],
                 train_window: Union[str, pd.DateOffset],
                 val_window: Union[str, pd.DateOffset],
                 test_window: Union[str, pd.DateOffset],
                 test_start_time: Union[str, pd.Timestamp],
                 rebalance_freq: Union[str, pd.DateOffset] = '1m',
                 max_iterations: Optional[int] = None,
                 batch_size: int = 1024,
                 shuffle_train: bool = True,
                 num_workers: int = 0):
        """
        初始化滚动窗口
        
        Args:
            feature_datasets: 特征数据集字典，{freq_name: FeatureDataset}
            target_dataset: 目标数据集
            lookback_windows: 回看窗口，{freq_name: "6m"}
            train_window: 训练窗口大小，如"2y"
            val_window: 验证窗口大小，如"6m"
            test_window: 测试窗口大小，如"6m"
            test_start_time: 测试开始时间
            rebalance_freq: 调仓频率，控制窗口内样本生成频率，如"1m"
            max_iterations: 最大滚动次数
            batch_size: 批大小
            shuffle_train: 是否打乱训练集
            num_workers: 数据加载线程数
        """
        self.feature_datasets = feature_datasets
        self.target_dataset = target_dataset
        self.lookback_windows = lookback_windows
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        
        # 解析时间窗口
        if isinstance(train_window, str):
            train_window = parse_window_str(train_window)
        self.train_window = train_window
        
        if isinstance(val_window, str):
            val_window = parse_window_str(val_window)
        self.val_window = val_window
        
        if isinstance(test_window, str):
            test_window = parse_window_str(test_window)
        self.test_window = test_window
        
        if isinstance(test_start_time, str):
            test_start_time = pd.Timestamp(test_start_time)
        self.test_start_time = test_start_time
        
        if isinstance(rebalance_freq, str):
            rebalance_freq = parse_window_str(rebalance_freq)
        self.rebalance_freq = rebalance_freq
        
        # 滚动步长 = test_window（窗口不重叠）
        self.rolling_step = test_window
        
        # 获取数据时间范围
        self.data_end_time = target_dataset.get_time_range()[1]
        
        # 预计算所有窗口
        self.windows = self._precompute_windows()
        
        # 迭代器状态
        self.iteration_count = 0
        
        print(f"\nRollingWindow initialized:")
        print(f"  Train/Val/Test windows: {train_window}, {val_window}, {test_window}")
        print(f"  Test start: {test_start_time.date()}")
        print(f"  Rebalance frequency: {rebalance_freq}")
        print(f"  Lookback windows: {lookback_windows}")
        print(f"  Total windows: {len(self.windows)}")

    @staticmethod
    def get_standardized_config(config: dict) -> dict:
        """标准化配置字典，用于生成唯一ID"""
        if not config:
            return {}
            
        # 剔除 batch_size/shuffle_train/num_workers
        # 标准化 test_start_time 为 YYYY-MM-DD
        test_start = config.get("test_start_time", "")
        if test_start:
            try:
                # 尝试解析并格式化为 YYYY-MM-DD
                test_start_std = pd.Timestamp(test_start).strftime("%Y-%m-%d")
            except:
                test_start_std = str(test_start)
        else:
            test_start_std = ""

        std_config = {
            "lookback_windows": config.get("lookback_windows", {}),
            "train_window": str(config.get("train_window", "")),
            "val_window": str(config.get("val_window", "")),
            "test_window": str(config.get("test_window", "")),
            "test_start_time": test_start_std,
            "rebalance_freq": str(config.get("rebalance_freq", "")),
        }
        
        return std_config

    def _precompute_windows(self) -> list:
        """预计算所有有效的滚动窗口"""
        windows = []
        current_test_start = self.test_start_time
        
        while True:
            test_end = current_test_start + self.test_window
            
            # 如果 test_start 已经超过了数据截止日期，停止
            if current_test_start >= self.data_end_time:
                break
                
            # 如果 test_end 超过了数据截止日期，截断到数据截止日期
            if test_end > self.data_end_time:
                test_end = self.data_end_time
            
            # 计算 val 和 train 的时间范围
            val_end = current_test_start
            val_start = val_end - self.val_window
            
            train_end = val_start
            train_start = train_end - self.train_window
            
            # 验证时间范围是否有效（至少要有训练集和验证集）
            if train_start >= train_end or val_start >= val_end or current_test_start >= test_end:
                break
                
            windows.append({
                'train': (train_start, train_end),
                'val': (val_start, val_end),
                'test': (current_test_start, test_end)
            })
            
            # 步进到下一个窗口
            current_test_start = current_test_start + self.rolling_step
            
            # 检查最大迭代次数
            if self.max_iterations and len(windows) >= self.max_iterations:
                break
                
        return windows

    def __len__(self):
        """返回总窗口数"""
        return len(self.windows)
    
    def __iter__(self):
        """重置迭代器"""
        self.iteration_count = 0
        return self
    
    def __next__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        生成下一个滚动窗口的DataLoader三元组
        
        Returns:
            (train_loader, val_loader, test_loader)
        
        Raises:
            StopIteration: 当没有更多窗口时
        """
        if self.iteration_count >= len(self.windows):
            raise StopIteration
            
        win = self.windows[self.iteration_count]
        train_start, train_end = win['train']
        val_start, val_end = win['val']
        test_start, test_end = win['test']
        
        print(f"\n{'='*60}")
        print(f"Rolling Window {self.iteration_count + 1} / {len(self.windows)}")
        print(f"  Train: [{train_start.date()}, {train_end.date()})")
        print(f"  Val:   [{val_start.date()}, {val_end.date()})")
        print(f"  Test:  [{test_start.date()}, {test_end.date()})")
        print(f"{'='*60}")
        
        # 更新计数器
        self.iteration_count += 1
        
        # 创建三个DataLoader
        train_loader = self._create_dataloader(
            train_start, train_end, 
            shuffle=self.shuffle_train, 
            split_name="Train"
        )
        val_loader = self._create_dataloader(
            val_start, val_end, 
            shuffle=False, 
            split_name="Val"
        )
        test_loader = self._create_dataloader(
            test_start, test_end, 
            shuffle=False, 
            split_name="Test"
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_dataloader(self, 
                          pred_start: pd.Timestamp, 
                          pred_end: pd.Timestamp,
                          shuffle: bool,
                          split_name: str) -> DataLoader:
        """
        创建指定时间范围的DataLoader
        
        将时间范围、lookback_windows、rebalance_freq传给MultiFreqDataLoader
        
        Args:
            pred_start: 预测起始时间
            pred_end: 预测结束时间
            shuffle: 是否打乱
            split_name: 数据集名称（用于日志）
        
        Returns:
            DataLoader
        """
        # 创建MultiFreqDataLoader
        dataset = MultiFreqDataLoader(
            feature_datasets=self.feature_datasets,
            target_dataset=self.target_dataset,
            pred_start=pred_start,
            pred_end=pred_end,
            lookback_windows=self.lookback_windows,
            rebalance_freq=self.rebalance_freq
        )
        
        print(f"  {split_name}: {len(dataset)} samples")
        
        # 创建PyTorch DataLoader
        loader = create_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        
        return loader
    
    def get_info(self) -> dict:
        """获取当前配置信息"""
        return {
            'train_window': str(self.train_window),
            'val_window': str(self.val_window),
            'test_window': str(self.test_window),
            'test_start_time': self.test_start_time,
            'rebalance_freq': str(self.rebalance_freq),
            'batch_size': self.batch_size,
            'current_iteration': self.iteration_count,
            'current_test_start': self.current_test_start
        }
