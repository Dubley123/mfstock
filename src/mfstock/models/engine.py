"""
训练引擎：增量训练与推理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm


class Engine:
    """
    训练引擎
    
    职责：
    1. 训练单个窗口
    2. 验证集评估
    3. 推理并返回预测结果
    4. Early Stopping
    """
    
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 patience: int = 10,
                 verbose: bool = True):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            patience: Early Stopping耐心值
            verbose: 是否打印详细信息
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.verbose = verbose
    
    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", disable=not self.verbose)
        for X_batch, y_batch, _ in pbar:
            # 转移到设备
            X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证集评估
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            (avg_loss, ic)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                # 转移到设备
                X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
                
                # 收集预测和真实值
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        # 计算IC (Information Coefficient)
        ic = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 0 else 0.0
        
        return avg_loss, ic
    
    def train_one_window(self,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        epochs: int = 50,
                        window_idx: Optional[int] = None) -> Dict:
        """
        训练单个窗口
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            window_idx: 窗口索引（用于日志）
        
        Returns:
            训练历史字典
        """
        best_val_loss = float('inf')
        best_val_ic = -float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_ic': [],
            'best_epoch': 0
        }
        
        window_str = f"Window {window_idx}" if window_idx is not None else "Training"
        
        for epoch in range(epochs):
            if self.verbose:
                print(f"\n{window_str} - Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_one_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_ic'].append(val_ic)
            
            if self.verbose:
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val IC: {val_ic:.4f}")
            
            # Early Stopping（以val_loss为准）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ic = val_ic
                history['best_epoch'] = epoch
                patience_counter = 0
                
                if self.verbose:
                    print(f"✓ New best model! (Val Loss: {best_val_loss:.6f}, Val IC: {best_val_ic:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        history['best_val_loss'] = best_val_loss
        history['best_val_ic'] = best_val_ic
        
        return history
    
    def predict_one_window(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        推理单个窗口
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            包含 [time, stock, prediction] 的DataFrame
        """
        self.model.eval()
        
        results = []
        
        with torch.no_grad():
            for X_batch, y_batch, metadata_list in tqdm(test_loader, desc="Predicting", disable=not self.verbose):
                # 转移到设备
                X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
                
                # 前向传播
                predictions = self.model(X_batch)
                
                # 提取结果
                preds = predictions.cpu().numpy().flatten()
                
                for i, meta in enumerate(metadata_list):
                    results.append({
                        'time': meta['time'],
                        'stock': meta['stock'],
                        'prediction': preds[i]
                    })
        
        df = pd.DataFrame(results)
        return df


def create_engine(model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 patience: int = 10,
                 verbose: bool = True) -> Engine:
    """
    创建训练引擎
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        patience: Early Stopping耐心值
        verbose: 是否打印详细信息
    
    Returns:
        Engine实例
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return Engine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=patience,
        verbose=verbose
    )
