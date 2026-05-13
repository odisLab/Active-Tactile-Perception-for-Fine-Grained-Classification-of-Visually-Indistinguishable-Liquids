"""
Step4: 数据维度验证（支持单分支和双分支）
检查Dataset输出维度是否符合模型输入要求

运行: python scripts/step4_check.py
参数修改: configs/config.py中修改DATA_ROOT或DUAL_BRANCH_ROOT
"""

import sys
import os

# 动态获取项目根目录并添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs import config


def validate_single_branch():
    """验证单分支数据集"""
    from src.data.dataset import STFTDataset
    
    print("=" * 70)
    print("Step4: 数据维度验证（单分支模式）")
    print("=" * 70)
    print(f"项目根目录: {project_root}")
    print(f"数据路径: {config.DATA_ROOT}")
    print(f"预期STFT shape: ({config.NUM_CHANNELS}, 1, {config.STFT_SHAPE[0]}, {config.STFT_SHAPE[1]})")
    print(f"预期邻接矩阵 shape: ({config.NUM_CHANNELS}, {config.NUM_CHANNELS})")
    print("=" * 70)
    
    try:
        dataset = STFTDataset(root_dir=config.DATA_ROOT, config=config)
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        return False
    
    if len(dataset) == 0:
        print("\n❌ 数据集为空，请检查路径")
        return False
    
    # 获取第一个样本
    stft_batch, adj, label = dataset[0]
    
    print(f"\n✅ 数据集加载成功")
    print(f"   样本数: {len(dataset)}")
    print(f"\n📊 第一个样本维度:")
    print(f"   STFT shape: {stft_batch.shape}")
    print(f"   邻接矩阵 shape: {adj.shape}")
    print(f"   标签: {label.item()}")
    
    # 验证维度
    expected_stft = (config.NUM_CHANNELS, 1, config.STFT_SHAPE[0], config.STFT_SHAPE[1])
    expected_adj = (config.NUM_CHANNELS, config.NUM_CHANNELS)
    
    if stft_batch.shape == expected_stft and adj.shape == expected_adj:
        print("\n" + "="*70)
        print("✅ 单分支维度验证通过，可正常训练")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("❌ 维度不匹配，请检查配置")
        print(f"   预期STFT: {expected_stft}")
        print(f"   实际STFT: {stft_batch.shape}")
        print(f"   预期邻接矩阵: {expected_adj}")
        print(f"   实际邻接矩阵: {adj.shape}")
        print("="*70)
        return False


def validate_dual_branch():
    """验证双分支数据集"""
    from src.data.dual_branch_dataset import DualBranchSTFTDataset
    
    print("=" * 70)
    print("Step4: 数据维度验证（双分支模式）")
    print("=" * 70)
    print(f"项目根目录: {project_root}")
    print(f"双分支数据根目录: {config.DUAL_BRANCH_ROOT}")
    print(f"   Phase E 路径: {config.PHASE_E_DIR}")
    print(f"   Phase H 路径: {config.PHASE_H_DIR}")
    print(f"\n预期维度:")
    print(f"   Phase E: ({config.NUM_CHANNELS}, 1, {config.DUAL_FREQ_BINS}, {config.E_TARGET_TIME_BINS})")
    print(f"   Phase H: ({config.NUM_CHANNELS}, 1, {config.DUAL_FREQ_BINS}, {config.H_TARGET_TIME_BINS})")
    print(f"   邻接矩阵: ({config.NUM_CHANNELS}, {config.NUM_CHANNELS})")
    print("=" * 70)
    
    # 验证双分支配置
    try:
        config.validate_dual_branch_config()
    except ValueError as e:
        print(f"\n❌ 双分支配置验证失败:\n{e}")
        return False
    
    # 加载数据集
    try:
        dataset = DualBranchSTFTDataset(config=config, mode='train')
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        return False
    
    if len(dataset) == 0:
        print("\n❌ 数据集为空，请检查路径")
        return False
    
    # 获取第一个样本
    X_e, X_h, adj, label, metadata = dataset[0]
    
    print(f"\n✅ 数据集加载成功")
    print(f"   配对样本数: {len(dataset)}")
    print(f"   Phase E 策略: {config.E_STRATEGY}")
    
    print(f"\n📊 第一个样本维度:")
    print(f"   Phase E shape: {X_e.shape}")
    print(f"   Phase H shape: {X_h.shape}")
    print(f"   邻接矩阵 shape: {adj.shape}")
    print(f"   标签: {label.item()}")
    
    print(f"\n📝 元数据:")
    print(f"   类别名称: {metadata['class_name']}")
    print(f"   Phase E 文件: {os.path.basename(metadata['phase_e_path'])}")
    print(f"   Phase H 文件: {os.path.basename(metadata['phase_h_path'])}")
    
    # 验证维度
    expected_e = (config.NUM_CHANNELS, 1, config.DUAL_FREQ_BINS, config.E_TARGET_TIME_BINS)
    expected_h = (config.NUM_CHANNELS, 1, config.DUAL_FREQ_BINS, config.H_TARGET_TIME_BINS)
    expected_adj = (config.NUM_CHANNELS, config.NUM_CHANNELS)
    
    errors = []
    if X_e.shape != expected_e:
        errors.append(f"Phase E 维度不匹配: 预期 {expected_e}, 实际 {X_e.shape}")
    if X_h.shape != expected_h:
        errors.append(f"Phase H 维度不匹配: 预期 {expected_h}, 实际 {X_h.shape}")
    if adj.shape != expected_adj:
        errors.append(f"邻接矩阵维度不匹配: 预期 {expected_adj}, 实际 {adj.shape}")
    
    if errors:
        print("\n" + "="*70)
        print("❌ 维度验证失败:")
        for error in errors:
            print(f"   - {error}")
        print("="*70)
        return False
    else:
        print("\n" + "="*70)
        print("✅ 双分支维度验证通过，可正常训练")
        print("="*70)
        return True


if __name__ == "__main__":
    # 根据 config.USE_DUAL_BRANCH 自动选择验证模式
    if config.USE_DUAL_BRANCH:
        success = validate_dual_branch()
    else:
        success = validate_single_branch()
    
    # 返回退出码（用于脚本自动化）
    sys.exit(0 if success else 1)
