"""
时间与频率工具模块
"""

import pandas as pd
import re
from typing import Union


def parse_window_str(window: str) -> pd.DateOffset:
    """
    解析时间窗口字符串
    
    Args:
        window: 时间窗口，如"1y", "6m", "3w", "20d"
    
    Returns:
        pd.DateOffset对象
    """
    window = window.strip().lower()
    match = re.match(r'^(\d+)([ymwd])$', window)
    if not match:
        raise ValueError(f"Invalid window format: {window}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    unit_map = {
        'y': lambda v: pd.DateOffset(years=v),
        'm': lambda v: pd.DateOffset(months=v),
        'w': lambda v: pd.DateOffset(weeks=v),
        'd': lambda v: pd.DateOffset(days=v)
    }
    
    return unit_map[unit](value)


def normalize_frequency(freq: str) -> str:
    """
    规范化频率字符串
    
    Args:
        freq: 频率字符串
    
    Returns:
        规范化后的频率："daily", "weekly", "monthly", "yearly"
    """
    freq = freq.strip().lower()
    
    freq_map = {
        'daily': 'daily', 'day': 'daily', 'd': 'daily',
        'weekly': 'weekly', 'week': 'weekly', 'w': 'weekly',
        'monthly': 'monthly', 'month': 'monthly', 'm': 'monthly',
        'yearly': 'yearly', 'year': 'yearly', 'y': 'yearly',
    }
    
    if freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    return freq_map[freq]


def window_to_periods(window: Union[str, pd.DateOffset], frequency: str) -> int:
    """
    将时间窗口转换为固定的记录数量
    
    这是核心转换函数，将"lookback 6个月"转换为"往回看6条记录"（如果数据是月频）
    
    Args:
        window: 时间窗口，如"6m", "1y"
        frequency: 数据频率，如"monthly", "weekly"
    
    Returns:
        需要回看的记录数量
    
    Examples:
        >>> window_to_periods("6m", "monthly")
        6
        >>> window_to_periods("1y", "weekly")  
        52
        >>> window_to_periods("1y", "monthly")
        12
    """
    # 解析window
    if isinstance(window, str):
        window_str = window.strip().lower()
        match = re.match(r'^(\d+)([ymwd])$', window_str)
        if not match:
            raise ValueError(f"Invalid window format: {window}")
        
        value = int(match.group(1))
        unit = match.group(2)
    else:
        # 从DateOffset提取
        if hasattr(window, 'years') and window.years > 0:
            value, unit = window.years, 'y'
        elif hasattr(window, 'months') and window.months > 0:
            value, unit = window.months, 'm'
        elif hasattr(window, 'weeks') and window.weeks > 0:
            value, unit = window.weeks, 'w'
        elif hasattr(window, 'days') and window.days > 0:
            value, unit = window.days, 'd'
        else:
            raise ValueError(f"Cannot parse DateOffset: {window}")
    
    # 规范化数据频率
    freq = normalize_frequency(frequency)
    
    # 转换表：(window单位, 数据频率) -> 每个window单位包含多少条数据
    conversion = {
        ('y', 'yearly'): 1,
        ('y', 'monthly'): 12,
        ('y', 'weekly'): 52,
        ('y', 'daily'): 252,
        
        ('m', 'monthly'): 1,
        ('m', 'weekly'): 4,
        ('m', 'daily'): 21,
        
        ('w', 'weekly'): 1,
        ('w', 'daily'): 5,
        
        ('d', 'daily'): 1,
    }
    
    key = (unit, freq)
    if key not in conversion:
        raise ValueError(
            f"Cannot convert window unit '{unit}' to data frequency '{freq}'. "
            f"Window frequency must be >= data frequency."
        )
    
    return value * conversion[key]


def generate_time_points(start: pd.Timestamp, end: pd.Timestamp, 
                        freq: Union[str, pd.DateOffset]) -> list:
    """
    生成时间点序列
    
    Args:
        start: 起始时间（包含）
        end: 结束时间（不包含）
        freq: 频率
    
    Returns:
        时间点列表
    """
    if isinstance(freq, str):
        freq = parse_window_str(freq)
    
    times = []
    current = start
    while current < end:
        times.append(current)
        current += freq
    
    return times
