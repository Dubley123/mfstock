"""
测试模型架构
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import MultiTowerTransformer, create_model


def test_dynamic_model():
    """测试动态模型构建"""
    
    print("\n" + "="*80)
    print("测试动态多塔Transformer")
    print("="*80)
    
    # 测试不同的频率组合
    test_cases = [
        # Case 1: 仅月度数据
        {
            'monthly': {'seq_len': 6, 'input_dim': 50}
        },
        # Case 2: 月度+周度
        {
            'monthly': {'seq_len': 6, 'input_dim': 50},
            'weekly': {'seq_len': 24, 'input_dim': 50}
        },
        # Case 3: 月度+周度+日度
        {
            'monthly': {'seq_len': 6, 'input_dim': 50},
            'weekly': {'seq_len': 24, 'input_dim': 50},
            'daily': {'seq_len': 60, 'input_dim': 50}
        },
    ]
    
    for i, freq_info in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {list(freq_info.keys())}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_model(
            freq_info=freq_info,
            d_model=128,
            nhead=4,
            num_layers=2,
            fusion_method='concat'
        )
        
        print(f"\n{model}")
        
        # 构造输入
        batch_size = 16
        x_dict = {}
        for freq_name, info in freq_info.items():
            x_dict[freq_name] = torch.randn(
                batch_size,
                info['seq_len'],
                info['input_dim']
            )
        
        # 前向传播
        with torch.no_grad():
            output = model(x_dict)
        
        print(f"\nInput shapes:")
        for freq_name, x in x_dict.items():
            print(f"  {freq_name}: {x.shape}")
        
        print(f"\nOutput shape: {output.shape}")
        
        assert output.shape == (batch_size, 1), f"Expected (16, 1), got {output.shape}"
        print("✓ Forward pass successful!")
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


def test_fusion_methods():
    """测试不同的融合方法"""
    
    print("\n" + "="*80)
    print("测试融合方法")
    print("="*80)
    
    freq_info = {
        'monthly': {'seq_len': 6, 'input_dim': 50},
        'weekly': {'seq_len': 24, 'input_dim': 50}
    }
    
    fusion_methods = ['concat', 'weighted']
    
    for method in fusion_methods:
        print(f"\n{'='*60}")
        print(f"Fusion Method: {method}")
        print(f"{'='*60}")
        
        model = create_model(
            freq_info=freq_info,
            d_model=128,
            fusion_method=method
        )
        
        print(f"\n{model}")
        
        # 测试前向传播
        batch_size = 8
        x_dict = {
            'monthly': torch.randn(batch_size, 6, 50),
            'weekly': torch.randn(batch_size, 24, 50)
        }
        
        with torch.no_grad():
            output = model(x_dict)
        
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, 1)
        print("✓ Test passed!")
    
    print("\n" + "="*80)
    print("All fusion methods work correctly!")
    print("="*80)


if __name__ == "__main__":
    try:
        test_dynamic_model()
        test_fusion_methods()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
