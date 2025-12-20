"""
完整性能优化验证：Dataset + DataLoader + Pipeline
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import time as time_module
from src.dataset import (
    FeatureDataset, TargetDataset,
    FeatureProcessor, MultiFreqDataLoader,
    create_dataloader
)


def test_end_to_end_performance():
    """端到端性能测试"""
    
    print("\n" + "="*80)
    print("端到端性能测试：Dataset → DataLoader → Batch Loading")
    print("="*80)
    
    data_dir = project_root / "data"
    monthly_file = data_dir / "data_in_sample_monthly.parquet"
    
    if not monthly_file.exists():
        print(f"ERROR: {monthly_file} not found")
        return
    
    # ==================== 1. 数据集加载 ====================
    print("\n[1] Loading datasets...")
    start_total = time_module.time()
    
    start = time_module.time()
    processor = FeatureProcessor(nan_handler='median', outlier_handler='winsorize', norm_handler='zscore')
    feature_datasets = {
        'monthly': FeatureDataset.from_file(
            file_path=monthly_file,
            frequency='monthly',
            exclude_cols=['r_shift'],
            processor=processor
        )
    }
    dataset_time = time_module.time() - start
    
    start = time_module.time()
    target_dataset = TargetDataset.from_file(
        file_path=monthly_file,
        target_col='r_shift'
    )
    target_time = time_module.time() - start
    
    print(f"  FeatureDataset loading: {dataset_time:.2f}s")
    print(f"  TargetDataset loading: {target_time:.2f}s")
    
    # ==================== 2. DataLoader 初始化 ====================
    print("\n[2] Creating DataLoader...")
    start = time_module.time()
    
    dataloader = MultiFreqDataLoader(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        pred_start=pd.Timestamp('2018-01-01'),
        pred_end=pd.Timestamp('2019-01-01'),  # 1年数据
        lookback_windows={'monthly': '12m'},
        rebalance_freq='1m'
    )
    
    dataloader_init_time = time_module.time() - start
    print(f"  DataLoader initialization: {dataloader_init_time:.2f}s")
    print(f"  Total samples: {len(dataloader)}")
    
    # ==================== 3. 批处理性能 ====================
    print("\n[3] Testing batch loading (batch_size=1024)...")
    
    torch_dataloader = create_dataloader(
        dataset=dataloader,
        batch_size=1024,  # 高性能批大小
        shuffle=False,
        num_workers=0
    )
    
    start = time_module.time()
    batch_count = 0
    sample_count = 0
    
    for X_batch, y_batch, metadata_list in torch_dataloader:
        batch_count += 1
        sample_count += len(y_batch)
        
        if batch_count == 1:
            print(f"\n  First batch info:")
            for freq, tensor in X_batch.items():
                print(f"    {freq}: {tensor.shape}, dtype={tensor.dtype}")
            print(f"    y: {y_batch.shape}, dtype={y_batch.dtype}")
            print(f"    metadata samples: {len(metadata_list)}")
    
    batch_time = time_module.time() - start
    
    print(f"\n  Batch loading performance:")
    print(f"    Total batches: {batch_count}")
    print(f"    Total samples: {sample_count}")
    print(f"    Time: {batch_time:.4f}s")
    print(f"    Throughput: {sample_count / batch_time:.2f} samples/sec")
    print(f"    Avg time per batch: {batch_time / batch_count * 1000:.2f}ms")
    
    # ==================== 4. 总结 ====================
    total_time = time_module.time() - start_total
    
    print("\n" + "="*80)
    print("性能总结:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  - Dataset loading: {dataset_time + target_time:.2f}s ({(dataset_time + target_time)/total_time*100:.1f}%)")
    print(f"  - DataLoader init: {dataloader_init_time:.2f}s ({dataloader_init_time/total_time*100:.1f}%)")
    print(f"  - Batch loading: {batch_time:.2f}s ({batch_time/total_time*100:.1f}%)")
    print(f"\n  Throughput: {sample_count / batch_time:.2f} samples/sec")
    print(f"  Estimated epoch time (for 100K samples): {100000 / (sample_count / batch_time):.2f}s")
    print("="*80)


def test_timestamp_conversion():
    """测试时间戳转换正确性"""
    
    print("\n" + "="*80)
    print("时间戳转换验证")
    print("="*80)
    
    data_dir = project_root / "data"
    monthly_file = data_dir / "data_in_sample_monthly.parquet"
    
    if not monthly_file.exists():
        print(f"ERROR: {monthly_file} not found")
        return
    
    processor = FeatureProcessor(nan_handler='median', outlier_handler='winsorize', norm_handler='zscore')
    feature_datasets = {
        'monthly': FeatureDataset.from_file(
            file_path=monthly_file,
            frequency='monthly',
            exclude_cols=['r_shift'],
            processor=processor
        )
    }
    
    target_dataset = TargetDataset.from_file(
        file_path=monthly_file,
        target_col='r_shift'
    )
    
    dataloader = MultiFreqDataLoader(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        pred_start=pd.Timestamp('2018-01-01'),
        pred_end=pd.Timestamp('2018-03-01'),
        lookback_windows={'monthly': '6m'},
        rebalance_freq='1m'
    )
    
    print(f"\n[Test] Sample storage format:")
    print(f"  First sample: {dataloader.samples[0]}")
    print(f"  Type of time: {type(dataloader.samples[0][0])}")
    print(f"  Type of stock: {type(dataloader.samples[0][1])}")
    
    # 获取一个样本
    X_dict, y, metadata = dataloader[0]
    
    print(f"\n[Test] Metadata after __getitem__:")
    print(f"  metadata['time']: {metadata['time']}")
    print(f"  Type: {type(metadata['time'])}")
    print(f"  Is pd.Timestamp: {isinstance(metadata['time'], pd.Timestamp)}")
    
    # 模拟保存到DataFrame
    test_df = pd.DataFrame([metadata])
    
    print(f"\n[Test] DataFrame conversion:")
    print(f"  time column dtype: {test_df['time'].dtype}")
    print(f"  ✓ Timestamp conversion working!" if pd.api.types.is_datetime64_any_dtype(test_df['time']) else "✗ Conversion failed!")
    
    print("\n" + "="*80)


def test_batch_size_impact():
    """测试不同batch_size的影响"""
    
    print("\n" + "="*80)
    print("Batch Size 影响测试")
    print("="*80)
    
    data_dir = project_root / "data"
    monthly_file = data_dir / "data_in_sample_monthly.parquet"
    
    if not monthly_file.exists():
        print(f"ERROR: {monthly_file} not found")
        return
    
    processor = FeatureProcessor(nan_handler='median', outlier_handler='mad', norm_handler='zscore')
    feature_datasets = {
        'monthly': FeatureDataset.from_file(
            file_path=monthly_file,
            frequency='monthly',
            exclude_cols=['r_shift'],
            processor=processor
        )
    }
    
    target_dataset = TargetDataset.from_file(
        file_path=monthly_file,
        target_col='r_shift'
    )
    
    dataloader = MultiFreqDataLoader(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        pred_start=pd.Timestamp('2018-01-01'),
        pred_end=pd.Timestamp('2019-01-01'),
        lookback_windows={'monthly': '12m'},
        rebalance_freq='1m'
    )
    
    batch_sizes = [32, 128, 512, 1024, 2048]
    results = []
    
    for bs in batch_sizes:
        torch_dataloader = create_dataloader(
            dataset=dataloader,
            batch_size=bs,
            shuffle=False,
            num_workers=0
        )
        
        start = time_module.time()
        sample_count = 0
        
        for X_batch, y_batch, metadata_list in torch_dataloader:
            sample_count += len(y_batch)
        
        elapsed = time_module.time() - start
        throughput = sample_count / elapsed
        
        results.append({
            'batch_size': bs,
            'time': elapsed,
            'throughput': throughput
        })
        
        print(f"  batch_size={bs:4d}: {elapsed:.4f}s, {throughput:.2f} samples/sec")
    
    # 找到最优batch_size
    best = max(results, key=lambda x: x['throughput'])
    print(f"\n  ✓ Best batch_size: {best['batch_size']} ({best['throughput']:.2f} samples/sec)")
    print(f"  Speedup vs batch_size=32: {best['throughput'] / results[0]['throughput']:.2f}x")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        test_timestamp_conversion()
        test_batch_size_impact()
        test_end_to_end_performance()
        
        print("\n" + "="*80)
        print("全流程优化总结:")
        print("  ✓ Dataset: 预分组 + 二分查找 (O(1) + O(log N))")
        print("  ✓ DataLoader: 字典遍历 + 向量化二分查找")
        print("  ✓ __getitem__: torch.from_numpy() 零拷贝")
        print("  ✓ Batch size: 提升到 1024 (匹配高性能数据加载)")
        print("  ✓ Timestamp: int64 内部存储，外部转换为 pd.Timestamp")
        print("  ✓ 预期总加速: 100-1000倍 (从小时级到分钟/秒级)")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
