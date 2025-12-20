import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.dataset import FeatureDataset, TargetDataset, RollingWindow
from src.dataset.processor import FeatureProcessor


def test_basic_flow():
    """测试基本流程"""
    print("\n" + "="*80)
    print("dataset测试")
    print("="*80)
    
    # 数据路径
    data_dir = project_root / "data"
    
    # 1. 加载数据集
    print("\n[1] Loading datasets...")
    
    # 首先加载一个数据集用于构造 feature 和 target
    monthly_file = data_dir / "data_in_sample_monthly.parquet"
    if not monthly_file.exists():
        print(f"ERROR: {monthly_file} not found")
        return
    
    # 读取原始数据
    monthly_df = pd.read_parquet(monthly_file)
    
    # 分离特征和目标
    feature_datasets = {}
    
    # Monthly features (排除 r_shift)
    processor = FeatureProcessor(nan_handler='median', outlier_handler='winsorize', norm_handler='zscore')
    feature_datasets['monthly'] = FeatureDataset.from_file(
        file_path=monthly_file,
        frequency='monthly',
        exclude_cols=['r_shift'],
        processor=processor
    )
    
    # 加载其他频率（如果存在）
    weekly_file = data_dir / "data_in_sample_weekly.parquet"
    if weekly_file.exists():
        feature_datasets['weekly'] = FeatureDataset.from_file(
            file_path=weekly_file,
            frequency='weekly',
            exclude_cols=['r_shift'],
            processor=processor
        )
    
    # Target dataset（只包含 r_shift）
    target_dataset = TargetDataset.from_file(
        file_path=monthly_file,
        target_col='r_shift'
    )
    
    # 2. 创建 lookback_windows
    lookback_windows = {}
    for freq in feature_datasets.keys():
        if freq == 'monthly':
            lookback_windows[freq] = '6m'
        elif freq == 'weekly':
            lookback_windows[freq] = '6m'
    
    print(f"\nLookback windows: {lookback_windows}")
    
    # 3. 创建RollingWindow
    print("\n[2] Creating RollingWindow...")
    
    rolling_window = RollingWindow(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        lookback_windows=lookback_windows,
        train_window='2y',
        val_window='6m',
        test_window='6m',
        test_start_time='2018-01-01',
        rebalance_freq='1m',
        batch_size=32,
        shuffle_train=True,
        max_iterations=2  # 只测试2个窗口
    )
    
    # 4. 迭代滚动窗口
    print("\n[3] Iterating through rolling windows...")
    
    for i, (train_loader, val_loader, test_loader) in enumerate(rolling_window):
        print(f"\n--- Window {i+1} ---")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # 测试一个batch
        print("\nTesting one batch from train_loader...")
        for X_batch, y_batch, metadata in train_loader:
            print(f"\nBatch shapes:")
            for freq_name, X_freq in X_batch.items():
                print(f"  {freq_name}: {X_freq.shape}")
            print(f"  y: {y_batch.shape}")
            print(f"  metadata: {len(metadata)} samples")
            
            # 显示第一个样本的元数据
            print(f"\nFirst sample metadata:")
            print(f"  Time: {metadata[0]['time']}")
            print(f"  Stock: {metadata[0]['stock']}")
            
            break  # 只测试一个batch
        
        if i >= 1:  # 只测试前2个窗口
            break
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


def test_single_sample():
    """测试单样本提取"""
    print("\n" + "="*80)
    print("测试单样本提取逻辑")
    print("="*80)
    
    data_dir = project_root / "data"
    monthly_file = data_dir / "data_in_sample_monthly.parquet"
    
    if not monthly_file.exists():
        print(f"ERROR: {monthly_file} not found")
        return
    
    # 加载特征数据集
    processor = FeatureProcessor(nan_handler='median', outlier_handler='winsorize', norm_handler='zscore')
    feature_dataset = FeatureDataset.from_file(
        file_path=monthly_file,
        frequency='monthly',
        exclude_cols=['r_shift'],
        processor=processor
    )
    
    print(f"\nFeatureDataset: {feature_dataset}")
    print(f"Feature columns: {feature_dataset.feature_cols[:5]}...")
    
    # 测试get_lookback_data - 从索引存储中获取测试样本
    # 随机选择一个股票和时间点
    test_stock = list(feature_dataset.stock_data.keys())[0]
    stock_info = feature_dataset.stock_data[test_stock]
    
    # 选择该股票的第10个时间点（如果存在）
    if len(stock_info['times']) > 10:
        test_time = pd.Timestamp(stock_info['times'][10])
    else:
        test_time = pd.Timestamp(stock_info['times'][-1])
    
    print(f"\nTesting get_lookback_data:")
    print(f"  Stock: {test_stock}")
    print(f"  Time: {test_time}")
    print(f"  Lookback: 6 periods")
    
    data = feature_dataset.get_lookback_data(test_stock, test_time, n_periods=6)
    print(f"\nExtracted data shape: {data.shape}")
    print(f"Expected shape: (6, {len(feature_dataset.feature_cols)})")
    
    if data.shape == (6, len(feature_dataset.feature_cols)):
        print("✓ Shape correct!")
    else:
        print("✗ Shape mismatch!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        test_single_sample()
        test_basic_flow()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
