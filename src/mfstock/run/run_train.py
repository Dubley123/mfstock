"""
完整训练流程示例
"""

import pandas as pd
import yaml
from pathlib import Path
from mfstock.dataset import FeatureDataset, TargetDataset, RollingWindow, FeatureProcessor
from mfstock.models import create_model, ModelSaver, create_engine
from mfstock.utils.misc import get_project_root

PROJECT_ROOT = get_project_root()

def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "training_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main(config_path: str = None):
    """主训练流程"""
    
    print("="*80)
    print("Multi-Frequency Based Transformer Training")
    print("="*80)
    
    # ==================== 0. 加载配置 ====================
    print("\n[0] Loading configuration...")
    config = load_config(config_path)
    print(f"Config loaded from: {config_path or 'configs/training_config.yaml'}")
    
    # 过滤频率
    selected_freqs = config.get('selected_freqs', None)
    if selected_freqs:
        print(f"Filtering frequencies: {selected_freqs}")
        # 过滤 feature_files
        if 'feature_files' in config['dataset']:
            config['dataset']['feature_files'] = {
                k: v for k, v in config['dataset']['feature_files'].items() if k in selected_freqs
            }
        # 过滤 target_files
        if 'target_files' in config['dataset']:
            config['dataset']['target_files'] = {
                k: v for k, v in config['dataset']['target_files'].items() if k in selected_freqs
            }
        # 过滤 lookback_windows
        if 'lookback_windows' in config['rolling_window']:
            config['rolling_window']['lookback_windows'] = {
                k: v for k, v in config['rolling_window']['lookback_windows'].items() if k in selected_freqs
            }
        # 过滤 freq_specific
        if 'model' in config and 'freq_specific' in config['model']:
            config['model']['freq_specific'] = {
                k: v for k, v in config['model']['freq_specific'].items() if k in selected_freqs
            }
    
    # ==================== 1. 数据加载 ====================
    print("\n[1] Loading datasets...")
    data_dir = PROJECT_ROOT / config['paths']['data_dir']
    time_col = config['dataset'].get('time_col', 'TradingDate')
    stkcd_col = config['dataset'].get('stkcd_col', 'Stkcd')
    
    # 创建特征处理器
    proc_cfg = config['dataset']['processor']
    processor = FeatureProcessor(
        nan_handler=proc_cfg['nan_handler'],
        outlier_handler=proc_cfg['outlier_handler'],
        norm_handler=proc_cfg['norm_handler']
    )
    
    # 加载特征数据集
    feature_datasets = {}
    feature_files = config['dataset'].get('feature_files', {})
    
    for freq, rel_path in feature_files.items():
        f_path = data_dir / rel_path
        if f_path.exists():
            print(f"  Loading {freq} features from {rel_path}...")
            feature_datasets[freq] = FeatureDataset.from_file(
                file_path=f_path,
                frequency=freq,
                time_col=time_col,
                stock_col=stkcd_col,
                processor=processor
            )
        else:
            print(f"  WARNING: Feature file not found: {f_path}")
    
    if len(feature_datasets) == 0:
        print("ERROR: No feature datasets loaded!")
        return
    
    # 加载目标数据集
    target_files = config['dataset'].get('target_files', {})
    
    # 目前逻辑只支持一个主目标数据集，通常取最高频率或月频
    # 这里我们取第一个可用的目标文件
    target_dataset = None
    for freq, rel_path in target_files.items():
        t_path = data_dir / rel_path
        if t_path.exists():
            print(f"  Loading target from {rel_path}...")
            target_dataset = TargetDataset.from_file(
                file_path=t_path,
                time_col=time_col,
                stock_col=stkcd_col
            )
            break # 只取一个
    
    if target_dataset is None:
        print("ERROR: No target datasets loaded!")
        return
    
    # ==================== 2. 配置 ====================
    lookback_windows = config['rolling_window']['lookback_windows']
    
    # ==================== 3. 初始化Saver ====================
    print("\n[2] Initializing ModelSaver...")
    saver = ModelSaver(base_dir=config['paths']['output_dir'], config=config)
    print(f"Config ID: {saver.config_id}")
    print(f"Experiment directory: {saver.output_dir}")
    
    # ==================== 4. 创建滚动窗口 ====================
    print("\n[3] Creating RollingWindow...")
    rw_cfg = config['rolling_window']
    rolling_window = RollingWindow(
        feature_datasets=feature_datasets,
        target_dataset=target_dataset,
        lookback_windows=lookback_windows,
        train_window=rw_cfg['train_window'],
        val_window=rw_cfg['val_window'],
        test_window=rw_cfg['test_window'],
        test_start_time=rw_cfg['test_start_time'],
        rebalance_freq=rw_cfg['rebalance_freq'],
        batch_size=rw_cfg['batch_size'],
        shuffle_train=rw_cfg['shuffle_train'],
        num_workers=rw_cfg.get('num_workers', 0)
    )
    
    # ==================== 5. 构建freq_info ====================
    # 从第一个窗口的数据推断
    print("\n[4] Building freq_info...")
    freq_info = {}
    for freq_name, dataset in feature_datasets.items():
        from mfstock.dataset.utils import window_to_periods
        n_periods = window_to_periods(lookback_windows[freq_name], dataset.frequency)
        freq_info[freq_name] = {
            'seq_len': n_periods,
            'input_dim': len(dataset.feature_cols)
        }
    
    print(f"Frequency info: {freq_info}")
    
    # ==================== 6. 滚动窗口训练 ====================
    print("\n[5] Starting rolling window training...")
    
    all_predictions = []
    all_test_losses = []
    
    for window_idx, (train_loader, val_loader, test_loader) in enumerate(rolling_window):
        # 使用 1-indexed 的窗口 ID
        display_idx = window_idx + 1
        
        # 检查是否已有训练好的模型
        model_exists = saver.model_exists(window_idx)
        
        # 创建模型实例
        model = create_model(
            freq_info=freq_info,
            dropout=config['model']['global'].get('dropout', 0.1),
            fusion_method=config['model']['global']['fusion_method'],
            use_ae=config['model']['global'].get('use_ae', False),
            freq_configs=config['model'].get('freq_specific', {})
        )
        
        # 创建引擎
        engine = create_engine(
            model=model,
            learning_rate=config['training']['learning_rate'],
            patience=config['training']['patience'],
            verbose=config['training']['verbose'],
            ae_alpha=config['model']['global'].get('ae_alpha', 0.1)
        )
        
        if model_exists:
            print(f"\n✓ Found existing model for window {display_idx}")
            saver.load_model(model, window_idx, device=engine.device)
            
            # 后期追加训练 (Refine)
            refine_epochs = config['training'].get('refine_epochs', 0)
            if refine_epochs > 0:
                print(f"\nRefining model for {refine_epochs} additional epochs...")
                history = engine.train_one_window(
                    train_loader, val_loader,
                    epochs=refine_epochs,
                    window_idx=window_idx
                )
                
                # 保存微调后的模型
                saver.save_model(
                    model=engine.model,
                    window_idx=window_idx,
                    optimizer=engine.optimizer,
                    metrics={
                        'val_loss': history['best_val_loss'],
                        'val_ic': history['best_val_ic']
                    }
                )
                
                # 保存训练历史
                saver.save_training_history(history, window_idx)
        else:
            print(f"\n✗ No existing model found for window {display_idx}")
            
            # 增量训练逻辑
            if window_idx > 0:
                # 尝试加载上一个窗口的模型
                prev_window_idx = window_idx - 1
                if saver.model_exists(prev_window_idx):
                    print(f"  Inheriting weights from window {prev_window_idx + 1}...")
                    saver.load_model(model, prev_window_idx, device=engine.device)
                    train_epochs = config['training'].get('rolling_epochs', 5)
                    print(f"  Incremental training for {train_epochs} epochs...")
                else:
                    print(f"  WARNING: Previous window model not found. Training from scratch.")
                    train_epochs = config['training'].get('base_epochs', 50)
            else:
                # 第一个窗口从头训练
                print(f"  Base window training from scratch...")
                train_epochs = config['training'].get('base_epochs', 50)
            
            # 执行训练
            history = engine.train_one_window(
                train_loader, val_loader,
                epochs=train_epochs,
                window_idx=window_idx
            )
            
            # 保存模型
            saver.save_model(
                model=engine.model,
                window_idx=window_idx,
                optimizer=engine.optimizer,
                epoch=history['best_epoch'],
                metrics={
                    'val_loss': history['best_val_loss'],
                    'val_ic': history['best_val_ic']
                }
            )
            
            # 保存训练历史
            saver.save_training_history(history, window_idx)
        
        # 计算测试集Loss和推理
        if len(test_loader) > 0:
            print(f"Calculating test loss for window {display_idx}...")
            test_total, test_pred, test_ae = engine.test_loss(test_loader)
            all_test_losses.append({
                'window_idx': window_idx,
                'total_loss': test_total,
                'pred_loss': test_pred,
                'ae_loss': test_ae
            })
            
            # 推理
            print(f"\nPredicting on test set...")
            predictions_df = engine.predict_one_window(test_loader)
            
            if not predictions_df.empty:
                predictions_df['window_idx'] = window_idx
                
                # 确保time列是pd.Timestamp格式（从int64转换）
                if 'time' in predictions_df.columns and predictions_df['time'].dtype == 'int64':
                    predictions_df['time'] = pd.to_datetime(predictions_df['time'])
                
                # 实时保存窗口预测（防止中断丢失）
                saver.save_window_predictions(predictions_df, window_idx)
                
                # 汇总到内存
                all_predictions.append(predictions_df)
                
                print(f"\nWindow {display_idx} completed!")
                print(f"Predictions: {len(predictions_df)} samples")
            else:
                print(f"\nWindow {display_idx} has no predictions (empty test set).")
        else:
            print(f"\nWindow {display_idx} has no test data, skipping test phase.")
    
    # 保存所有窗口的测试集Loss汇总
    if all_test_losses:
        saver.save_test_losses(all_test_losses)
    
    # ==================== 7. 汇总并保存所有预测 ====================
    print("\n[6] Aggregating and saving all predictions...")
    
    # 方案1: 从内存中合并（如果训练完整运行）
    if len(all_predictions) > 0:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.sort_values(['time', 'stock'])
        
        # 确保time列格式正确
        if final_predictions['time'].dtype == 'int64':
            final_predictions['time'] = pd.to_datetime(final_predictions['time'])
        
        # 保存完整历史因子文件（parquet格式）
        saver.save_predictions(final_predictions)
        
        print(f"\n✓ All predictions saved (from memory)!")
        print(f"Total samples: {len(final_predictions)}")
        print(f"Time range: {final_predictions['time'].min()} to {final_predictions['time'].max()}")
        print(f"Unique stocks: {final_predictions['stock'].nunique()}")
        print(f"Windows: {final_predictions['window_idx'].nunique()}")
    else:
        print("\n⚠ No predictions in memory, attempting to merge from saved files...")
    
    # 方案2: 从磁盘合并（用于中断恢复或二次运行）
    try:
        merged_predictions = saver.merge_all_predictions()
        if merged_predictions is not None:
            print(f"✓ Successfully merged predictions from disk!")
    except Exception as e:
        print(f"Warning: Failed to merge predictions from disk: {e}")
    
    print("\n" + "="*80)
    print("Training pipeline completed!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Pipeline failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
