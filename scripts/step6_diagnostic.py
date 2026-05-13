"""
Step6: 深度诊断工具（支持单分支+双分支）

运行: python scripts/step6_diagnostic.py
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

from configs import config


# ==================== 数据划分（兼容单分支和双分支）====================

def stratified_split_fixed_per_class(dataset, test_samples_per_class, random_seed=42, is_dual_branch=False):
    """
    每个类别固定测试集数量的分层划分
    
    Args:
        dataset: 数据集
        test_samples_per_class: 每类测试样本数
        random_seed: 随机种子
        is_dual_branch: 是否为双分支模式
    """
    all_labels = []
    for i in range(len(dataset)):
        if is_dual_branch:
            _, _, _, label, _ = dataset[i]  # 双分支: X_e, X_h, adj, label, metadata
        else:
            _, _, label = dataset[i]  # 单分支: X, adj, label
        all_labels.append(label.item())

    label_to_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    np.random.seed(random_seed)
    train_indices = []
    test_indices = []

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total = len(indices)

        if total < test_samples_per_class:
            train_indices.extend(indices)
            continue

        np.random.shuffle(indices)
        test_idx = indices[:test_samples_per_class]
        train_idx = indices[test_samples_per_class:]

        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


def split_dataset(dataset, config, is_dual_branch=False):
    """
    数据集划分（兼容单分支和双分支）
    
    Args:
        dataset: 数据集
        config: 配置对象
        is_dual_branch: 是否为双分支模式
    """
    split_strategy = getattr(config, 'SPLIT_STRATEGY', 'stratified_ratio')

    if split_strategy == "random":
        from torch.utils.data import random_split
        train_size = int((1 - config.VALID_RATIO) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )

    elif split_strategy == "stratified_ratio":
        all_labels = []
        for i in range(len(dataset)):
            if is_dual_branch:
                _, _, _, label, _ = dataset[i]
            else:
                _, _, label = dataset[i]
            all_labels.append(label.item())

        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            test_size=config.VALID_RATIO,
            stratify=all_labels,
            random_state=config.RANDOM_SEED
        )

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

    elif split_strategy == "stratified_fixed":
        test_samples_per_class = getattr(config, 'TEST_SAMPLES_PER_CLASS', 20)
        train_dataset, test_dataset = stratified_split_fixed_per_class(
            dataset,
            test_samples_per_class=test_samples_per_class,
            random_seed=config.RANDOM_SEED,
            is_dual_branch=is_dual_branch  # ✅ 传递标志
        )

    else:
        raise ValueError(f"未知的划分策略: {split_strategy}")

    return train_dataset, test_dataset


def detect_num_classes_from_checkpoint(checkpoint_path, device='cpu'):
    """从 checkpoint 自动检测类别数"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 方法1：从保存的 config_params 读取
        if 'config_params' in checkpoint:
            if 'NUM_CLASSES' in checkpoint['config_params']:
                return checkpoint['config_params']['NUM_CLASSES']
        
        # 方法2：从旧版 config 对象读取
        if 'config' in checkpoint:
            if hasattr(checkpoint['config'], 'NUM_CLASSES'):
                return checkpoint['config'].NUM_CLASSES

        # 方法3：从模型权重推断
        state_dict = checkpoint['model_state_dict']
        for key in ['classifier.8.weight', 'classifier.6.weight', 'fc.weight']:
            if key in state_dict:
                return state_dict[key].shape[0]

        return None
    except:
        return None


def main():
    print("="*70)
    print("🔬 Step6: 深度诊断工具 - 完整版（支持单分支+双分支）")
    print("="*70)
    
    # 更新实验目录名称（与 Step5 保持一致）
    config.DATASET_NAME = config.get_dataset_name()
    config.EVAL_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "evaluation_results", config.DATASET_NAME)
    os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
    
    # 打印横幅
    if hasattr(config, "print_project_banner"):
        config.print_project_banner()
    elif hasattr(config, "print_eval_paths_banner"):
        config.print_eval_paths_banner()
    
    # 检查开关
    if not getattr(config, 'DIAG_ENABLE', True):
        print("⚠️ 诊断模块未启用")
        return
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n📱 设备: {device}")
    
    # ==================== 🔥 步骤 1: 检测模型类型 ====================
    print("\n🔧 加载模型...")
    checkpoint_input = input("\n请输入 checkpoint 路径: ").strip().strip('"').strip("'")
    checkpoint_path_obj = Path(checkpoint_input)
    
    # 处理文件夹/文件输入
    if checkpoint_path_obj.is_dir():
        pth_files = sorted(list(checkpoint_path_obj.glob("*.pth")))
        if not pth_files:
            print("❌ 未找到模型文件")
            return
        print(f"\n📂 找到 {len(pth_files)} 个模型:")
        for i, p in enumerate(pth_files, 1):
            file_size = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p.name:<60s} {file_size:>6.2f} MB")
        choice = input(f"\n选择 [1-{len(pth_files)}] (默认=1): ").strip() or "1"
        checkpoint_path = str(pth_files[int(choice) - 1])
    else:
        checkpoint_path = str(checkpoint_path_obj)
    
    print(f"\n✅ checkpoint: {Path(checkpoint_path).name}")
    
    # ==================== 🔥 检测是否为双分支模型 ====================
    is_dual_branch = False
    
    # 方法1：从文件名判断
    if "dual" in checkpoint_path.lower() or "dual_branch" in checkpoint_path.lower():
        is_dual_branch = True
        print("🔍 从文件名检测到双分支模型")
    
    # 方法2：从当前 config 判断
    if config.USE_DUAL_BRANCH:
        is_dual_branch = True
        print("🔍 从 config.USE_DUAL_BRANCH 检测到双分支模式")
    
    # 方法3：从 checkpoint 内容判断
    try:
        temp_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config_params' in temp_checkpoint:
            if temp_checkpoint['config_params'].get('USE_DUAL_BRANCH', False):
                is_dual_branch = True
                print("🔍 从 checkpoint 检测到双分支模型")
        elif 'config' in temp_checkpoint:
            if hasattr(temp_checkpoint['config'], 'USE_DUAL_BRANCH'):
                is_dual_branch = temp_checkpoint['config'].USE_DUAL_BRANCH
                print("🔍 从 checkpoint.config 检测到双分支模型")
    except Exception as e:
        print(f"⚠️ 读取 checkpoint 配置失败: {e}")
    
    # ==================== 🔥 根据检测结果导入对应模块 ====================
    if is_dual_branch:
        print("\n✅ 确认为双分支模型，验证配置...")
        try:
            config.validate_dual_branch_config()
        except ValueError as e:
            print(f"\n❌ 双分支配置错误:\n{e}\n")
            print("提示：请检查 configs/config.py 中的双分支配置")
            return
        
        # 导入双分支模块
        from src.data.dual_branch_dataset import DualBranchSTFTDataset as DatasetClass
        from src.models.dual_branch_gnn_transformer import DualBranchGNNTransformer as ModelClass
        
        print(f"   数据路径: {config.DUAL_BRANCH_ROOT}")
        print(f"   Phase E: {config.PHASE_E_DIR}")
        print(f"   Phase H: {config.PHASE_H_DIR}")
        data_root = config.DUAL_BRANCH_ROOT
        
    else:
        print("\n✅ 确认为单分支模型")
        from src.data.dataset import STFTDataset as DatasetClass
        from src.models.gnn_transformer import GNN_Transformer_Fusion as ModelClass
        
        print(f"   数据路径: {config.DATA_ROOT}")
        data_root = config.DATA_ROOT
    
    print("=" * 70)
    # ==============================================================
    
    # ==================== 步骤 2: 匹配评估文件夹 ====================
    checkpoint_parent = Path(checkpoint_path).parent.name
    
    print(f"\n🔍 从 checkpoint 提取标识: {checkpoint_parent}")
    
    eval_root = Path(config.EVAL_OUTPUT_DIR).parent  # ✅ 使用父目录进行搜索
    
    if not eval_root.exists():
        print(f"⚠️  评估根目录不存在: {eval_root}")
        print("   将使用默认配置创建新文件夹")
        selected_folder_name = None
    else:
        exact_match = eval_root / checkpoint_parent
        base_name = '_'.join(checkpoint_parent.split('_')[:-1])
        prefix_matches = sorted([
            d for d in eval_root.iterdir() 
            if d.is_dir() and d.name.startswith(base_name)
        ])
        
        selected_folder_name = None
        
        if exact_match.exists():
            diag_folder = exact_match / "diagnostics"
            has_diag = diag_folder.exists()
            
            if has_diag:
                diag_count = len(list(diag_folder.glob("*.csv")))
                print(f"\n✅ 找到精确匹配: {checkpoint_parent}")
                print(f"   状态: 已有诊断 ({diag_count} 文件)")
                
                overwrite = input(f"\n是否覆盖现有诊断结果? [y/N]: ").strip().lower()
                if overwrite != 'y':
                    print("❌ 取消运行")
                    return
            else:
                print(f"\n✅ 找到精确匹配: {checkpoint_parent}")
                print(f"   状态: 未诊断")
            
            selected_folder_name = checkpoint_parent
        
        elif prefix_matches:
            print(f"\n📂 未找到精确匹配，但发现 {len(prefix_matches)} 个相关文件夹:")
            
            for i, folder in enumerate(prefix_matches, 1):
                folder_name = folder.name
                csv_files = list(folder.glob("*.csv"))
                diag_folder = folder / "diagnostics"
                has_diag = diag_folder.exists()
                
                if has_diag:
                    diag_count = len(list(diag_folder.glob("*.csv")))
                    status = f"✅ 已有诊断 ({diag_count}文件)"
                else:
                    status = "🆕 未诊断"
                
                print(f"  [{i:2d}] {folder_name:<40s} | {len(csv_files)} CSV | {status}")
            
            print(f"  [ 0] 创建新文件夹")
            
            folder_choice = input(f"\n请选择 [0-{len(prefix_matches)}] (默认=1): ").strip() or "1"
            
            if folder_choice.isdigit():
                idx = int(folder_choice)
                
                if idx == 0:
                    print("\n✅ 将创建新文件夹")
                    selected_folder_name = None
                elif 1 <= idx <= len(prefix_matches):
                    selected = prefix_matches[idx - 1]
                    selected_folder_name = selected.name
                    
                    diag_folder = selected / "diagnostics"
                    if diag_folder.exists() and any(diag_folder.glob("*.csv")):
                        overwrite = input(f"\n⚠️  已有诊断结果，是否覆盖? [y/N]: ").strip().lower()
                        if overwrite != 'y':
                            print("❌ 取消运行")
                            return
                    
                    print(f"\n✅ 使用文件夹: {selected_folder_name}")
        
        else:
            print(f"\n⚠️  未找到匹配的评估文件夹")
            print(f"   将使用默认配置创建新文件夹")
            selected_folder_name = None
    
    # ==================== 关键：先应用选择，再加载数据 ====================
    if selected_folder_name:
        config.DATASET_NAME = selected_folder_name
        config.EVAL_OUTPUT_DIR = str(eval_root / selected_folder_name)
        print(f"   诊断结果将保存到: {config.EVAL_OUTPUT_DIR}/diagnostics")
    
    # ==================== 步骤 3: 加载对应的数据集 ====================
    print(f"\n📊 加载数据集...")
    print(f"   模式: {'双分支' if is_dual_branch else '单分支'}")
    print(f"   路径: {data_root}")
    
    if is_dual_branch:
        # ✅ 双分支数据集
        full_dataset = DatasetClass(config=config, mode='train')
    else:
        # ✅ 单分支数据集
        full_dataset = DatasetClass(root_dir=data_root, config=config)
    
    # ✅ 传递 is_dual_branch 标志
    _, test_dataset = split_dataset(full_dataset, config, is_dual_branch=is_dual_branch)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  ✅ 测试集: {len(test_dataset)} 样本")
    
    # ==================== 步骤 4: 检测类别数并加载模型 ====================
    print("\n🔍 自动检测类别数...")
    num_classes = detect_num_classes_from_checkpoint(checkpoint_path, device)
    
    if num_classes is None:
        num_classes = config.NUM_CLASSES
        print(f"  ⚠️ 使用 config 默认值: num_classes={num_classes}")
    else:
        print(f"✅ 检测到: num_classes={num_classes}")
    
    # 初始化模型
    original_num_classes = config.NUM_CLASSES
    config.NUM_CLASSES = num_classes
    model = ModelClass(config).to(device)  # ✅ 使用动态导入的 ModelClass
    config.NUM_CLASSES = original_num_classes
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ 权重加载成功")
    
    if 'epoch' in checkpoint:
        print(f"     - Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"     - Val Acc: {checkpoint['val_acc']:.4f}")
    
    # ==================== 步骤 5: 运行诊断 ====================
    # ✅ 根据模式选择诊断管道
    if is_dual_branch:
        try:
            from src.diagnostics.dual_branch_diagnostic_pipeline import DualBranchDiagnosticPipeline
            pipeline = DualBranchDiagnosticPipeline(model, config, device)
            print("\n✅ 使用双分支诊断管道")
        except ImportError:
            print("\n⚠️ 双分支诊断管道未找到，使用标准诊断管道（可能不完全适配）")
            from src.diagnostics.diagnostic_pipeline import DiagnosticPipeline
            pipeline = DiagnosticPipeline(model, config, device)
    else:
        from src.diagnostics.diagnostic_pipeline import DiagnosticPipeline
        pipeline = DiagnosticPipeline(model, config, device)
        print("\n✅ 使用单分支诊断管道")
    

    results = pipeline.run(test_loader)

    # ==================== Step6 可选：双分支随机多通道消融 ====================
    if getattr(config, 'DIAG_MULTI_CHANNEL_ABLATION_ENABLE', False):
        try:
            if hasattr(pipeline, 'sensitivity_tester') and hasattr(pipeline.sensitivity_tester, 'test_random_multi_channel_ablation'):
                print("\n📊 追加诊断: 随机多通道消融（k-list 多次采样）")
                pipeline.sensitivity_tester.test_random_multi_channel_ablation(test_loader, pipeline.output_dir)
            else:
                print("\n⚠️ 未找到 test_random_multi_channel_ablation，跳过随机多通道消融")
        except Exception as e:
            print(f"\n⚠️ 随机多通道消融执行失败: {e}")

    print("\n✅ 诊断完成！")

    print(f"   结果保存在: {config.EVAL_OUTPUT_DIR}/diagnostics/")


if __name__ == "__main__":
    main()
