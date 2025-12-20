"""
特征预处理器：标准化流水线

处理顺序：
1) Inf -> NaN
2) 缺失值处理（zero/median）
3) 去极值（winsorize/mad）
4) 横截面归一化（rank/zscore）
5) 最终兜底 fillna(0)

要求：
- 使用 Pandas 向量化和 groupby/transform，保证高性能
- 输出转为 np.float32
"""

from typing import List

import numpy as np
import pandas as pd


class FeatureProcessor:
    def __init__(
        self,
        nan_handler: str = "median",
        outlier_handler: str = "mad",
        norm_handler: str = "zscore",
    ) -> None:
        valid_nan = {"zero", "median"}
        valid_outlier = {"winsorize", "mad"}
        valid_norm = {"rank", "zscore"}

        if nan_handler not in valid_nan:
            raise ValueError(f"Invalid nan_handler: {nan_handler}")
        if outlier_handler not in valid_outlier:
            raise ValueError(f"Invalid outlier_handler: {outlier_handler}")
        if norm_handler not in valid_norm:
            raise ValueError(f"Invalid norm_handler: {norm_handler}")

        self.nan_handler = nan_handler
        self.outlier_handler = outlier_handler
        self.norm_handler = norm_handler

    @staticmethod
    def get_standardized_config(config: dict) -> dict:
        """标准化配置字典，用于生成唯一ID"""
        if not config:
            return {}
        
        std_config = {
            "nan_handler": str(config.get("nan_handler", "")).lower(),
            "outlier_handler": str(config.get("outlier_handler", "")).lower(),
            "norm_handler": str(config.get("norm_handler", "")).lower(),
        }
        
        return std_config

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        按指定顺序对特征列进行预处理（按 TradingDate 横截面处理）。

        注意：该方法不会修改 ID/时间/目标列，只对 feature_cols 生效。
        """
        if len(feature_cols) == 0:
            return df

        if "TradingDate" not in df.columns:
            raise KeyError("Column 'TradingDate' not found in DataFrame for cross-section ops")

        result = df.copy()

        # 1) Inf -> NaN
        result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan)

        # 2) 缺失值处理
        if self.nan_handler == "median":
            med = result.groupby("TradingDate")[feature_cols].transform("median")
            result[feature_cols] = result[feature_cols].fillna(med)
        elif self.nan_handler == "zero":
            result[feature_cols] = result[feature_cols].fillna(0.0)

        # 3) 去极值
        if self.outlier_handler == "winsorize":
            q_low = result.groupby("TradingDate")[feature_cols].transform(lambda x: x.quantile(0.01))
            q_hi = result.groupby("TradingDate")[feature_cols].transform(lambda x: x.quantile(0.99))
            # 对齐索引和列后进行裁剪
            result[feature_cols] = result[feature_cols].clip(lower=q_low, upper=q_hi)
        elif self.outlier_handler == "mad":
            med = result.groupby("TradingDate")[feature_cols].transform("median")
            mad = result.groupby("TradingDate")[feature_cols].transform(lambda x: (x - x.median()).abs().median())
            # 防止 MAD 为 0 导致阈值退化
            mad_scaled = 1.4826 * mad
            lower = med - 3.0 * mad_scaled
            upper = med + 3.0 * mad_scaled
            result[feature_cols] = result[feature_cols].clip(lower=lower, upper=upper)

        # 4) 横截面归一化
        if self.norm_handler == "rank":
            ranked = result.groupby("TradingDate")[feature_cols].rank(pct=True)
            result[feature_cols] = ranked - 0.5
        elif self.norm_handler == "zscore":
            grp = result.groupby("TradingDate")[feature_cols]
            mean = grp.transform("mean")
            std = grp.transform(lambda x: x.std(ddof=0))
            result[feature_cols] = (result[feature_cols] - mean) / (std + 1e-6)

        # 5) 兜底填充 & 类型转换
        result[feature_cols] = result[feature_cols].fillna(0.0).astype(np.float32)

        return result
