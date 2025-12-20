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
                 verbose: bool = True,
                 ae_alpha: float = 0.1):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            patience: Early Stopping耐心值
            verbose: 是否打印详细信息
            ae_alpha: 重构损失权重
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.ae_criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.verbose = verbose
        self.ae_alpha = ae_alpha
    
    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            (avg_total_loss, avg_pred_loss, avg_ae_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_ae_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", disable=not self.verbose)
        for X_batch, y_batch, _ in pbar:
            # 转移到设备
            X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions, reconstructions_dict = self.model(X_batch)
            
            # 1. 预测损失
            pred_loss = self.criterion(predictions, y_batch)
            
            # 2. 重构损失 (AutoEncoder)
            ae_loss = torch.tensor(0.0, device=self.device)
            if reconstructions_dict:
                for freq_name, reconstructed in reconstructions_dict.items():
                    # 原始输入
                    original = X_batch[freq_name]
                    ae_loss += self.ae_criterion(reconstructed, original)
                ae_loss = ae_loss / len(reconstructions_dict)
            
            # 3. 合并损失
            loss = pred_loss + self.ae_alpha * ae_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_ae_loss += ae_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}',
                'ae': f'{ae_loss.item():.4f}'
            })
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_pred_loss = total_pred_loss / n_batches if n_batches > 0 else 0.0
        avg_ae_loss = total_ae_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss, avg_pred_loss, avg_ae_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """
        验证集评估
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            (avg_total_loss, avg_pred_loss, avg_ae_loss, ic)
        """
        self.model.eval()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_ae_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                # 转移到设备
                X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                predictions, reconstructions_dict = self.model(X_batch)
                
                # 1. 预测损失
                pred_loss = self.criterion(predictions, y_batch)
                
                # 2. 重构损失
                ae_loss = torch.tensor(0.0, device=self.device)
                if reconstructions_dict:
                    for freq_name, reconstructed in reconstructions_dict.items():
                        original = X_batch[freq_name]
                        ae_loss += self.ae_criterion(reconstructed, original)
                    ae_loss = ae_loss / len(reconstructions_dict)
                
                # 3. 合并损失
                loss = pred_loss + self.ae_alpha * ae_loss
                
                total_loss += loss.item()
                total_pred_loss += pred_loss.item()
                total_ae_loss += ae_loss.item()
                n_batches += 1
                
                # 收集预测和真实值
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_pred_loss = total_pred_loss / n_batches if n_batches > 0 else 0.0
        avg_ae_loss = total_ae_loss / n_batches if n_batches > 0 else 0.0
        
        # 计算IC (Information Coefficient)
        ic = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 0 else 0.0
        
        return avg_loss, avg_pred_loss, avg_ae_loss, ic
    
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
            'train_pred_loss': [],
            'train_ae_loss': [],
            'val_loss': [],
            'val_pred_loss': [],
            'val_ae_loss': [],
            'val_ic': [],
            'best_epoch': 0
        }
        
        window_str = f"Window {window_idx}" if window_idx is not None else "Training"
        
        for epoch in range(epochs):
            if self.verbose:
                print(f"\n{window_str} - Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss, train_pred, train_ae = self.train_one_epoch(train_loader)
            
            # 验证
            val_loss, val_pred, val_ae, val_ic = self.validate(val_loader)
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_pred_loss'].append(train_pred)
            history['train_ae_loss'].append(train_ae)
            history['val_loss'].append(val_loss)
            history['val_pred_loss'].append(val_pred)
            history['val_ae_loss'].append(val_ae)
            history['val_ic'].append(val_ic)
            
            if self.verbose:
                print(f"Train Loss: {train_loss:.6f} (Pred: {train_pred:.6f}, AE: {train_ae:.6f})")
                print(f"Val Loss: {val_loss:.6f} (Pred: {val_pred:.6f}, AE: {val_ae:.6f}), Val IC: {val_ic:.4f}")
            
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
                outputs = self.model(X_batch)
                
                # 处理多输出（AE任务）
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
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
                 verbose: bool = True,
                 ae_alpha: float = 0.1) -> Engine:
    """
    创建训练引擎
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        patience: Early Stopping耐心值
        verbose: 是否打印详细信息
        ae_alpha: AutoEncoder 损失权重
    
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
        verbose=verbose,
        ae_alpha=ae_alpha
    )
