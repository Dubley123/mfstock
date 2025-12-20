"""
调试NaN损失问题：检查数据和模型输出
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from mfs.dataset import FeatureDataset, TargetDataset, RollingWindow, FeatureProcessor
from mfs.models import create_model


def check_tensor_health(tensor, name):
    """检查张量是否健康"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    status = "✓" if not (has_nan or has_inf) else "✗"
    
    print(f"  {status} {name}:")
    print(f"      Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"      Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print(f"      Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    print(f"      Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"      ⚠ PROBLEM DETECTED!")
        if has_nan:
            print(f"         NaN count: {torch.isnan(tensor).sum().item()}")
        if has_inf:
            print(f"         Inf count: {torch.isinf(tensor).sum().item()}")
    
    return not (has_nan or has_inf)


def debug_single_window():
    """调试单个window的前几个样本"""
    
    print("="*80)
    print("NaN Loss 调试工具")
    print("="*80)
    
    # ==================== 1. 加载数据 ====================
    print("\n[1] Loading datasets...")
    data_dir = project_root / "data"
    
    processor = FeatureProcessor(nan_handler='median', outlier_handler='mad', norm_handler='zscore')
    
    feature_datasets = {
        'monthly': FeatureDataset.from_file(
            file_path=data_dir / "data_in_sample_monthly.parquet",
            frequency='monthly',
            exclude_cols=['r_shift'],
            processor=processor
        )
    }
    
    target_dataset = TargetDataset.from_file(
        file_path=data_dir / "data_in_sample_monthly.parquet",
        target_col='r_shift'
    )
    
    # ==================== 2. 创建单个Window ====================
    print("\n[2] Creating single window...")
    
    lookback_windows = {'monthly': '12m'}
    
    rolling_window = RollingWindow(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        lookback_windows=lookback_windows,
        train_window='5y',
        val_window='2y',
        test_window='1y',
        test_start_time='2018-01-01',
        rebalance_freq='1m',
        batch_size=64,
        shuffle_train=False,  # 不打乱，方便调试
        max_iterations=1  # 只取第一个window
    )
    
    # 获取第一个window
    for train_loader, val_loader, test_loader in rolling_window:
        break
    
    print(f"Train samples: {len(train_loader.dataset)}")
    
    # ==================== 3. 检查前N个样本 ====================
    print("\n[3] Inspecting first 5 training samples...")
    
    n_samples = min(5, len(train_loader.dataset))
    all_healthy = True
    
    for i in range(n_samples):
        print(f"\n{'─'*80}")
        print(f"Sample {i}:")
        print(f"{'─'*80}")
        
        X_dict, y, metadata = train_loader.dataset[i]
        
        print(f"  Metadata:")
        print(f"    Time: {metadata['time']}")
        print(f"    Stock: {metadata['stock']}")
        
        # 检查特征
        print(f"\n  Features:")
        for freq_name, X_tensor in X_dict.items():
            is_healthy = check_tensor_health(X_tensor, f"{freq_name}")
            all_healthy = all_healthy and is_healthy
        
        # 检查目标
        print(f"\n  Target:")
        is_healthy = check_tensor_health(y, "y")
        all_healthy = all_healthy and is_healthy
        
        print(f"\n  Raw values:")
        print(f"    y value: {y.item():.6f}")
        
        # 如果有问题，提前退出
        if not all_healthy:
            print(f"\n⚠ Found problematic sample at index {i}!")
            break
    
    if not all_healthy:
        print("\n" + "="*80)
        print("⚠ DATA QUALITY ISSUE DETECTED - Fix data before proceeding!")
        print("="*80)
        return
    
    # ==================== 4. 创建模型并检查输出 ====================
    print("\n[4] Creating model and testing forward pass...")
    
    # 构建freq_info
    freq_info = {}
    for freq_name, dataset in feature_datasets.items():
        from src.dataset.utils import window_to_periods
        n_periods = window_to_periods(lookback_windows[freq_name], dataset.frequency)
        freq_info[freq_name] = {
            'seq_len': n_periods,
            'input_dim': len(dataset.feature_cols)
        }
    
    print(f"  Freq info: {freq_info}")
    
    # 创建模型
    model = create_model(
        freq_info=freq_info,
        d_model=128,
        nhead=4,
        num_layers=2,
        fusion_method='concat'
    )
    
    model.eval()  # 评估模式
    
    print(f"\n  Model created: {model.__class__.__name__}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 5. 测试前5个样本的前向传播 ====================
    print("\n[5] Testing forward pass on first 5 samples...")
    
    for i in range(n_samples):
        print(f"\n{'─'*80}")
        print(f"Forward Pass - Sample {i}:")
        print(f"{'─'*80}")
        
        X_dict, y_true, metadata = train_loader.dataset[i]
        
        # 添加batch维度
        X_batch = {freq: X_tensor.unsqueeze(0) for freq, X_tensor in X_dict.items()}
        
        # 前向传播
        with torch.no_grad():
            y_pred = model(X_batch)
        
        print(f"  Input shapes:")
        for freq, tensor in X_batch.items():
            print(f"    {freq}: {tensor.shape}")
        
        print(f"\n  Model output:")
        is_healthy = check_tensor_health(y_pred, "y_pred")
        
        print(f"\n  Comparison:")
        print(f"    Predicted: {y_pred.item():.6f}")
        print(f"    True:      {y_true.item():.6f}")
        print(f"    Diff:      {abs(y_pred.item() - y_true.item()):.6f}")
        
        if not is_healthy:
            print(f"\n⚠ Model output is unhealthy at sample {i}!")
            print(f"\n  Checking intermediate outputs...")
            
            # 逐层检查
            model.eval()
            for name, module in model.named_children():
                print(f"\n  Checking module: {name}")
                # 这需要更详细的实现
            
            break
    
    # ==================== 6. 测试一个小批次 ====================
    print("\n[6] Testing batch forward pass...")
    
    batch_iter = iter(train_loader)
    X_batch, y_batch, metadata_list = next(batch_iter)
    
    print(f"\n  Batch info:")
    print(f"    Batch size: {len(y_batch)}")
    for freq, tensor in X_batch.items():
        print(f"    {freq}: {tensor.shape}")
    
    print(f"\n  Input batch health:")
    batch_healthy = True
    for freq, tensor in X_batch.items():
        is_healthy = check_tensor_health(tensor, f"{freq}")
        batch_healthy = batch_healthy and is_healthy
    
    is_healthy = check_tensor_health(y_batch, "y_batch")
    batch_healthy = batch_healthy and is_healthy
    
    if not batch_healthy:
        print(f"\n⚠ Batch data is unhealthy!")
        return
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y_pred_batch = model(X_batch)
    
    print(f"\n  Batch output:")
    is_healthy = check_tensor_health(y_pred_batch, "y_pred_batch")
    
    if not is_healthy:
        print(f"\n⚠ Batch output is unhealthy!")
        return
    
    # 计算损失
    criterion = torch.nn.MSELoss()
    loss = criterion(y_pred_batch, y_batch)
    
    print(f"\n  Loss:")
    print(f"    Value: {loss.item():.6f}")
    print(f"    Is NaN: {torch.isnan(loss).item()}")
    print(f"    Is Inf: {torch.isinf(loss).item()}")
    
    # ==================== 7. 测试训练模式 ====================
    print("\n[7] Testing training mode (with gradient)...")
    
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 前向传播
    y_pred_train = model(X_batch)
    
    print(f"\n  Training mode output:")
    is_healthy = check_tensor_health(y_pred_train, "y_pred_train")
    
    # 计算损失
    loss = criterion(y_pred_train, y_batch)
    
    print(f"\n  Training loss:")
    print(f"    Value: {loss.item():.6f}")
    print(f"    Is NaN: {torch.isnan(loss).item()}")
    print(f"    Requires grad: {loss.requires_grad}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    print(f"\n  Gradient health:")
    grad_healthy = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            
            if has_nan or has_inf:
                print(f"    ✗ {name}: Has NaN: {has_nan}, Has Inf: {has_inf}")
                grad_healthy = False
            else:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1000:
                    print(f"    ⚠ {name}: grad_norm = {grad_norm:.2f} (very large!)")
    
    if grad_healthy:
        print(f"    ✓ All gradients are healthy!")
    else:
        print(f"    ✗ Some gradients are unhealthy!")
    
    # ==================== 总结 ====================
    print("\n" + "="*80)
    print("诊断总结:")
    print("="*80)
    
    if all_healthy and batch_healthy and is_healthy and grad_healthy:
        print("✓ 所有检查通过！数据和模型看起来正常。")
        print("  如果训练时仍出现NaN，可能是学习率过大或训练过程中的数值不稳定。")
        print("  建议：")
        print("    1. 降低学习率 (如 1e-5)")
        print("    2. 添加梯度裁剪")
        print("    3. 检查数据标准化")
    else:
        print("✗ 发现问题！")
        if not all_healthy:
            print("  - 样本数据存在 NaN/Inf")
        if not batch_healthy:
            print("  - 批次数据存在 NaN/Inf")
        if not is_healthy:
            print("  - 模型输出存在 NaN/Inf")
        if not grad_healthy:
            print("  - 梯度存在 NaN/Inf")
    
    print("="*80)


if __name__ == "__main__":
    try:
        debug_single_window()
    except Exception as e:
        print(f"\n✗ Debug script failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
