"""
Step5: 模型训练 + 评估
- 自动生成实验唯一ID
- 保存完整配置快照
- 记录超参数到CSV
- Checkpoint关联实验ID
- 生成实验元数据
运行: python scripts/step5_train.py
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from configs import config
from src.data.dataset import STFTDataset
from src.models.gnn_transformer import GNN_Transformer_Fusion
from src.utils.evaluator import evaluate_model
from src.utils.experiment_tracker import ExperimentTracker  

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter
import numpy as np

def calculate_accuracy(outputs, labels):
    """计算分类准确率"""
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / labels.size(0)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
    for stft_batch, adj_matrix, labels in pbar:
        stft_batch = stft_batch.to(device)
        adj_matrix = adj_matrix.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(stft_batch, adj_matrix)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_acc = calculate_accuracy(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += batch_acc * batch_size
        total_samples += batch_size

        pbar.set_postfix({'loss': f"{total_loss/total_samples:.4f}", 'acc': f"{total_acc/total_samples:.4f}"})

    return total_loss / total_samples, total_acc / total_samples

# ==================== 新增：自动检测类别数 ====================
def auto_detect_and_update_num_classes(data_root):
    """在训练前自动检测并更新类别数"""
    if not os.path.exists(data_root):
        print(f"⚠️  数据路径不存在: {data_root}")
        return
    
    class_folders = [
        f for f in os.listdir(data_root) 
        if os.path.isdir(os.path.join(data_root, f))
    ]
    class_folders = sorted(class_folders)
    num_classes_detected = len(class_folders)
    
    if num_classes_detected != config.NUM_CLASSES:
        print(f"⚠️  类别数不匹配:")
        print(f"   config.NUM_CLASSES = {config.NUM_CLASSES}")
        print(f"   实际检测到 = {num_classes_detected} 个类别")
        print(f"   类别列表: {class_folders}")
        print(f"✅ 自动更新 NUM_CLASSES -> {num_classes_detected}")
        config.NUM_CLASSES = num_classes_detected
    else:
        print(f"✅ 类别数一致: {num_classes_detected} 个类别")
        print(f"   类别列表: {class_folders}")
# =============================================================


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for stft_batch, adj_matrix, labels in pbar:
            stft_batch = stft_batch.to(device)
            adj_matrix = adj_matrix.to(device)
            labels = labels.to(device)

            logits = model(stft_batch, adj_matrix)
            loss = criterion(logits, labels)
            batch_acc = calculate_accuracy(logits, labels)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_acc += batch_acc * batch_size
            total_samples += batch_size

            pbar.set_postfix({'loss': f"{total_loss/total_samples:.4f}", 'acc': f"{total_acc/total_samples:.4f}"})

    return total_loss / total_samples, total_acc / total_samples

def train_for_learning_curve(model, train_loader, config, device, epochs):
    """学习曲线专用训练函数（简化版）"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(epochs):
        model.train()
        for stft_batch, adj_matrix, labels in train_loader:
            stft_batch = stft_batch.to(device)
            adj_matrix = adj_matrix.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(stft_batch, adj_matrix)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def stratified_split_fixed_per_class(dataset, test_samples_per_class, random_seed=42):
    """
    每个类别固定测试集数量的分层划分

    Args:
        dataset: 完整数据集
        test_samples_per_class: 每个类别的测试集样本数
        random_seed: 随机种子

    Returns:
        train_dataset, test_dataset
    """
    # 1. 提取所有标签
    all_labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        all_labels.append(label.item())

    # 2. 按类别组织索引
    label_to_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # 3. 每类随机选择固定数量作为测试集
    np.random.seed(random_seed)
    train_indices = []
    test_indices = []

    print("\n" + "="*70)
    print(f"固定每类测试集数量: {test_samples_per_class}")
    print("="*70)
    print(f"{'类别':<8} {'总样本':<10} {'训练集':<10} {'测试集':<10}")
    print("-"*70)

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total = len(indices)

        # 检查是否有足够样本
        if total < test_samples_per_class:
            print(f"⚠️  类别 {label} 样本不足: {total} < {test_samples_per_class}")
            train_indices.extend(indices)
            continue

        # 随机打乱并划分
        np.random.shuffle(indices)
        test_idx = indices[:test_samples_per_class]
        train_idx = indices[test_samples_per_class:]

        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

        print(f"{label:<8} {total:<10} {len(train_idx):<10} {len(test_idx):<10}")

    print("="*70)
    print(f"总计: 训练集 {len(train_indices)}, 测试集 {len(test_indices)}")
    print("="*70)

    # 4. 创建子集
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

def print_split_stats(train_labels, test_labels):
    """打印划分统计"""
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("\n" + "="*70)
    print("数据集划分统计")
    print("="*70)
    print(f"训练集: {len(train_labels)}, 测试集: {len(test_labels)}\n")
    print(f"{'类别':<8} {'训练集':<10} {'测试集':<10} {'测试占比':<12}")
    print("-"*70)

    for label in sorted(train_counts.keys()):
        train_count = train_counts[label]
        test_count = test_counts[label]
        test_ratio = test_count / (train_count + test_count) * 100
        print(f"{label:<8} {train_count:<10} {test_count:<10} {test_ratio:.1f}%")

    print("="*70)

def split_dataset(dataset, config):
    """
    根据配置选择数据集划分策略

    Args:
        dataset: 完整数据集
        config: 配置对象

    Returns:
        train_dataset, test_dataset
    """
    split_strategy = getattr(config, 'SPLIT_STRATEGY', 'stratified_ratio')

    if split_strategy == "random":
        # 方案1: 随机划分
        from torch.utils.data import random_split
        train_size = int((1 - config.VALID_RATIO) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )
        print("✅ 使用随机划分")

    elif split_strategy == "stratified_ratio":
        # 方案2: 按比例分层（推荐）
        all_labels = []
        for i in range(len(dataset)):
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

        print("✅ 使用按比例分层划分")

        train_labels = [all_labels[i] for i in train_indices]
        test_labels = [all_labels[i] for i in test_indices]
        print_split_stats(train_labels, test_labels)

    elif split_strategy == "stratified_fixed":
        # 方案3: 固定每类测试数
        test_samples_per_class = getattr(config, 'TEST_SAMPLES_PER_CLASS', 20)
        train_dataset, test_dataset = stratified_split_fixed_per_class(
            dataset,
            test_samples_per_class=test_samples_per_class,
            random_seed=config.RANDOM_SEED
        )
        print(f"✅ 使用固定每类测试数: {test_samples_per_class}")

    else:
        raise ValueError(f"未知的划分策略: {split_strategy}")

    return train_dataset, test_dataset

def main():
    # 更新实验目录名称（参数变化时自动递增）
    from datetime import datetime
    config.DATASET_NAME = config.get_dataset_name()
    config.EVAL_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "evaluation_results", config.DATASET_NAME)
    config.CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "checkpoints", config.DATASET_NAME)
    os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    # 目录与实验名确定后，再打印一次（只在 Step5）
    if hasattr(config, "print_paths_banner"):
        config.print_paths_banner()
        config.print_project_banner()

    # cuDNN配置
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # ===== 新增：自动检测类别数 =====
    print("\n" + "="*70)
    print("自动检测数据集类别数...")
    print("="*70)
    auto_detect_and_update_num_classes(config.DATA_ROOT)
    # ===============================
    
    # 实验追踪器
    print("\n" + "="*70)
    print("Step5: 模型训练 + 评估（增强可追溯性版）")
    print("="*70)
    
    tracker = ExperimentTracker(config)
    # ... 后续代码保持不变

    print("=" * 70)
    print("Step5: 模型训练 + 评估（增强可追溯性版）")
    print("=" * 70)

    # ==================== 新增：初始化实验追踪器 ====================
    tracker = ExperimentTracker(config)
    print(f"\n🆔 实验ID: {tracker.experiment_id}")
    print(f"📁 实验目录: {tracker.exp_dir}")

    # 保存配置快照
    tracker.save_config_snapshot()
    # ==================== 实验追踪器初始化结束 ====================

    print(f"\n项目根目录: {project_root}")
    print(f"数据路径: {config.DATA_ROOT}")
    print(f"设备: {config.DEVICE}")
    print(f"Batch: {config.BATCH_SIZE}, Epochs: {config.EPOCHS}, LR: {config.LEARNING_RATE}")
    print(f"模型: {config.NUM_CHANNELS}通道 -> {config.NUM_CLASSES}类")

    print("\n评估配置:")
    print(f"  - 标准指标: {config.EVAL_ENABLE_METRICS}")
    print(f"  - 混淆矩阵: {config.EVAL_ENABLE_CONFUSION_MATRIX}")
    print(f"  - 学习曲线: {config.EVAL_ENABLE_LEARNING_CURVE}")
    print("=" * 70)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    full_dataset = STFTDataset(root_dir=config.DATA_ROOT, config=config)
    train_dataset, test_dataset = split_dataset(full_dataset, config)

    print(f"\n训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    # ==================== 新增：记录数据集信息 ====================
    tracker.record_data_info(len(train_dataset), len(test_dataset))
    # 提取超参数和模型架构
    tracker.extract_hyperparameters()
    tracker.extract_model_architecture()
    # ==================== 数据信息记录结束 ====================

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    model = GNN_Transformer_Fusion(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    print("\n开始训练...")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            # ==================== 修改：使用实验ID命名checkpoint ====================
            best_model_path = tracker.get_checkpoint_path(val_acc)
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'val_acc': val_acc}, best_model_path)
            print(f"✅ 保存最佳模型: {best_model_path}")
            # ==================== Checkpoint命名修改结束 ====================

    print(f"\n✅ 训练完成 | 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # ==================== 新增：记录训练信息 ====================
    tracker.record_training_info(config.EPOCHS, best_epoch, best_val_acc)
    # ==================== 训练信息记录结束 ====================

    # 加载最佳模型
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n加载最佳模型进行最终评估: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # ==================== 修改：使用实验ID作为run_name ====================
    run_name = tracker.experiment_id
    # ==================== run_name修改结束 ====================

    # 调用评估模块
    evaluate_model(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        run_name=run_name,
        train_dataset=train_dataset if config.EVAL_ENABLE_LEARNING_CURVE else None,
        model_class=GNN_Transformer_Fusion if config.EVAL_ENABLE_LEARNING_CURVE else None,
        train_fn=train_for_learning_curve if config.EVAL_ENABLE_LEARNING_CURVE else None,
        experiment_tracker=tracker  # 传递tracker给evaluator
    )

    # ==================== 新增：生成实验摘要 ====================
    tracker.save_metadata()
    tracker.generate_summary_report()
    # ==================== 实验摘要生成结束 ====================

    print("\n" + "="*70)
    print(f"🎯 实验完成！实验ID: {tracker.experiment_id}")
    print(f"📂 所有文件保存在: {tracker.exp_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
