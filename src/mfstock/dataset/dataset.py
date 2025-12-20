"""
单频率时序数据集（高性能版本）
使用预分组存储和二分查找，避免DataFrame线性扫描
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict
from mfstock.dataset.utils import normalize_frequency
from mfstock.dataset.processor import FeatureProcessor


class FeatureDataset:
    """
    特征数据集（高性能版本）
    
    职责：
    1. 管理单一频率的特征数据
    2. 提供按时间点+回看条数提取数据的接口
    3. 只包含特征列，不包含目标列
    
    性能优化：
    - 预分组存储：按股票分组为字典，O(1)访问
    - 二分查找：使用np.searchsorted，O(log N)查找
    - NumPy存储：时间戳为int64，特征为float32
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 frequency: str,
                 time_col: str = 'TradingDate',
                 stock_col: str = 'Stkcd',
                 feature_cols: Optional[List[str]] = None,
                 processor: Optional[FeatureProcessor] = None):
        """
        初始化特征数据集
        
        Args:
            df: DataFrame，已包含 [time_col, stock_col, feature_cols]
            frequency: 数据频率标识，如"monthly", "weekly"
            time_col: 时间列名
            stock_col: 股票列名
            feature_cols: 特征列名列表，如不指定则自动推断
        """
        self.frequency = normalize_frequency(frequency)
        self.time_col = time_col
        self.stock_col = stock_col
        
        # 确保时间列是datetime类型，并排序
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([stock_col, time_col]).reset_index(drop=True)
        
        # 确定特征列
        if feature_cols is None:
            exclude = {time_col, stock_col}
            self.feature_cols = [
                col for col in df.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            self.feature_cols = feature_cols
        
        if len(self.feature_cols) == 0:
            raise ValueError("No feature columns found")

        # 可选：特征预处理（在构建索引存储之前执行）
        if processor is not None:
            processed = processor.fit_transform(df, self.feature_cols)
            # 仅回写特征列，保留其它列原样
            df[self.feature_cols] = processed[self.feature_cols]
        
        n_records = len(df)
        n_stocks = df[stock_col].nunique()
        n_features = len(self.feature_cols)
        
        print(f"FeatureDataset: {self.frequency}, "
              f"{n_records} records, {n_stocks} stocks, {n_features} features")
        print("  Building indexed storage...")
        
        # 预分组存储：按股票分组
        self.stock_data: Dict[str, Dict[str, np.ndarray]] = {}
        
        for stock_code, group in df.groupby(stock_col):
            # 时间戳转为int64（纳秒）
            times = group[time_col].values.astype('datetime64[ns]').astype(np.int64)
            
            # 特征转为float32矩阵
            features = group[self.feature_cols].values.astype(np.float32)
            
            self.stock_data[stock_code] = {
                'times': times,
                'features': features
            }
        
        # 释放原始DataFrame
        del df
        
        print(f"  ✓ Indexed {len(self.stock_data)} stocks")
    
    @classmethod
    def from_file(cls,
                  file_path: Union[str, Path],
                  frequency: str,
                  time_col: str = 'TradingDate',
                  stock_col: str = 'Stkcd',
                  exclude_cols: Optional[List[str]] = None,
                  processor: Optional[FeatureProcessor] = None):
        """
        从文件加载特征数据集
        
        Args:
            file_path: 数据文件路径
            frequency: 数据频率
            time_col: 时间列名
            stock_col: 股票列名
            exclude_cols: 要排除的列，如['r_shift']
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 加载数据
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 排除指定列
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        
        return cls(df, frequency, time_col, stock_col, None, processor)
    
    def get_lookback_data(self, 
                         stock_code: str, 
                         end_time: pd.Timestamp,
                         n_periods: int) -> np.ndarray:
        """
        获取指定股票在指定时间点之前的n条记录（高性能版本）
        
        核心优化：
        - O(1) 字典查找股票数据
        - O(log N) 二分查找时间点
        - NumPy切片提取数据
        
        Args:
            stock_code: 股票代码
            end_time: 结束时间点（不包含）
            n_periods: 需要的记录数
        
        Returns:
            形状为 (n_periods, n_features) 的数组
        """
        # O(1) 从字典获取股票数据
        if stock_code not in self.stock_data:
            # 股票不存在，返回全零矩阵
            return np.zeros((n_periods, len(self.feature_cols)), dtype=np.float32)
        
        stock_info = self.stock_data[stock_code]
        times = stock_info['times']
        features = stock_info['features']
        
        # 转换end_time为int64纳秒时间戳
        end_time_int = pd.Timestamp(end_time).value
        
        # O(log N) 二分查找：找到第一个 >= end_time 的索引
        idx = np.searchsorted(times, end_time_int, side='left')
        
        # 提取 [idx-n_periods : idx] 的数据（严格 < end_time）
        start_idx = max(0, idx - n_periods)
        available_data = features[start_idx:idx]
        
        n_available = len(available_data)
        
        if n_available >= n_periods:
            # 数据足够，直接返回最后n_periods条
            result = available_data[-n_periods:]
        else:
            # 数据不足，前面补零
            n_pad = n_periods - n_available
            padding = np.zeros((n_pad, len(self.feature_cols)), dtype=np.float32)
            result = np.vstack([padding, available_data])
        
        return result
    
    def get_time_range(self) -> tuple:
        """
        返回数据的时间范围
        
        Returns:
            (最早时间, 最晚时间)
        """
        all_times = []
        for stock_info in self.stock_data.values():
            all_times.extend(stock_info['times'])
        
        if len(all_times) == 0:
            raise ValueError("No data available")
        
        all_times = np.array(all_times)
        min_time = pd.Timestamp(all_times.min())
        max_time = pd.Timestamp(all_times.max())
        
        return (min_time, max_time)
    
    def __repr__(self):
        t_min, t_max = self.get_time_range()
        return (f"FeatureDataset(freq={self.frequency}, "
                f"stocks={len(self.stock_data)}, "
                f"features={len(self.feature_cols)}, "
                f"time=[{t_min.date()} to {t_max.date()}])")


class TargetDataset:
    """
    目标数据集（高性能版本）
    
    职责：
    1. 管理目标变量数据
    2. 提供按时间点提取目标值的接口
    3. 只包含目标列，不包含特征列
    
    性能优化：
    - 预分组存储：按股票分组为字典
    - 哈希查找：使用字典映射 (stock, time) -> value
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 time_col: str = 'TradingDate',
                 stock_col: str = 'Stkcd'):
        """
        初始化目标数据集
        
        Args:
            df: DataFrame，包含 [time_col, stock_col, target_col]
            target_col: 目标列名
            time_col: 时间列名
            stock_col: 股票列名
        """
        self.target_col = target_col
        self.time_col = time_col
        self.stock_col = stock_col
        
        # 确保时间列是datetime类型，并排序
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([stock_col, time_col]).reset_index(drop=True)
        
        # 只保留需要的列
        df = df[[time_col, stock_col, target_col]]
        
        n_records = len(df)
        n_stocks = df[stock_col].nunique()
        
        print(f"TargetDataset: {n_records} records, {n_stocks} stocks")
        print("  Building indexed storage...")
        
        # 预分组存储：按股票分组
        self.stock_data: Dict[str, Dict[str, np.ndarray]] = {}
        
        for stock_code, group in df.groupby(stock_col):
            # 时间戳转为int64（纳秒）
            times = group[time_col].values.astype('datetime64[ns]').astype(np.int64)
            
            # 目标值转为float32
            targets = group[target_col].values.astype(np.float32)
            
            self.stock_data[stock_code] = {
                'times': times,
                'targets': targets
            }
        
        # 释放原始DataFrame
        del df
        
        print(f"  ✓ Indexed {len(self.stock_data)} stocks")
    
    @classmethod
    def from_file(cls,
                  file_path: Union[str, Path],
                  target_col: Optional[str] = None,
                  time_col: str = 'TradingDate',
                  stock_col: str = 'Stkcd'):
        """
        从文件加载目标数据集
        
        Args:
            file_path: 数据文件路径
            target_col: 目标列名。如果为None，则自动推断（排除time_col和stock_col后剩下的一列）
            time_col: 时间列名
            stock_col: 股票列名
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 加载数据
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 自动推断 target_col
        if target_col is None:
            exclude = {time_col, stock_col}
            remaining_cols = [col for col in df.columns if col not in exclude]
            
            if len(remaining_cols) == 0:
                raise ValueError(f"No target column found in {file_path} after excluding {exclude}")
            elif len(remaining_cols) > 1:
                raise ValueError(f"Multiple potential target columns found in {file_path}: {remaining_cols}. "
                                 f"Please specify target_col explicitly or ensure only one target column exists.")
            
            target_col = remaining_cols[0]
            print(f"Inferred target_col: '{target_col}' from {file_path.name}")
        
        return cls(df, target_col, time_col, stock_col)
    
    def get_target(self, stock_code: str, time: pd.Timestamp) -> float:
        """
        获取指定股票在指定时间的目标值（高性能版本）
        
        核心优化：
        - O(1) 字典查找股票数据
        - O(log N) 二分查找精确时间点
        
        Args:
            stock_code: 股票代码
            time: 时间点
        
        Returns:
            目标值
        """
        # O(1) 从字典获取股票数据
        if stock_code not in self.stock_data:
            raise ValueError(f"Stock {stock_code} not found in target dataset")
        
        stock_info = self.stock_data[stock_code]
        times = stock_info['times']
        targets = stock_info['targets']
        
        # 转换time为int64纳秒时间戳
        time_int = pd.Timestamp(time).value
        
        # O(log N) 二分查找精确匹配
        idx = np.searchsorted(times, time_int, side='left')
        
        # 检查是否精确匹配
        if idx < len(times) and times[idx] == time_int:
            return float(targets[idx])
        else:
            raise ValueError(f"No target value for stock={stock_code}, time={time}")
    
    def filter_by_time(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        按时间范围筛选数据（返回DataFrame供兼容性）
        
        Args:
            start: 起始时间（包含）
            end: 结束时间（不包含）
        
        Returns:
            筛选后的DataFrame
        """
        start_int = pd.Timestamp(start).value
        end_int = pd.Timestamp(end).value
        
        records = []
        for stock_code, stock_info in self.stock_data.items():
            times = stock_info['times']
            targets = stock_info['targets']
            
            # 筛选时间范围
            mask = (times >= start_int) & (times < end_int)
            filtered_times = times[mask]
            filtered_targets = targets[mask]
            
            for t, tgt in zip(filtered_times, filtered_targets):
                records.append({
                    self.time_col: pd.Timestamp(t),
                    self.stock_col: stock_code,
                    self.target_col: tgt
                })
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values([self.stock_col, self.time_col]).reset_index(drop=True)
        
        return df
    
    def get_time_range(self) -> tuple:
        """
        返回数据的时间范围
        
        Returns:
            (最早时间, 最晚时间)
        """
        all_times = []
        for stock_info in self.stock_data.values():
            all_times.extend(stock_info['times'])
        
        if len(all_times) == 0:
            raise ValueError("No data available")
        
        all_times = np.array(all_times)
        min_time = pd.Timestamp(all_times.min())
        max_time = pd.Timestamp(all_times.max())
        
        return (min_time, max_time)
    
    def __repr__(self):
        t_min, t_max = self.get_time_range()
        return (f"TargetDataset(col={self.target_col}, "
                f"stocks={len(self.stock_data)}, "
                f"time=[{t_min.date()} to {t_max.date()}])")
