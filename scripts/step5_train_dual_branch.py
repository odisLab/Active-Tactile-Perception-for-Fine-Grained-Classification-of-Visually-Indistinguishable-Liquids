"""
Step5: 双分支模型训练脚本（完整功能版）
- 保留原有所有评估和追踪功能
- 支持两种 Phase E 策略切换
- 完整的实验追踪和诊断支持
- 修复 pickle 序列化问题
- 增强：可选保存 Train/Val 每epoch指标到 CSV/JSON（config.TRAIN_LOG_ENABLE 控制）
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from configs import config
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import csv
import json
import pickle
from src.data.dual_branch_dataset import DualBranchSTFTDataset
from src.models.dual_branch_gnn_transformer import DualBranchGNNTransformer
from src.utils.evaluator import evaluate_model
from src.utils.evaluator import compute_metrics_from_collections
from src.utils.experiment_tracker import ExperimentTracker
def get_serializable_config(config_obj):
    """提取可序列化的配置参数"""
    serializable_params = {}
    for key, value in vars(config_obj).items():
        if key.startswith('_') or callable(value):
            continue
        if isinstance(value, (int, float, str, bool, type(None))):
            serializable_params[key] = value
        elif isinstance(value, (list, tuple)):
            try:
                pickle.dumps(value)
                serializable_params[key] = value
            except Exception:
                continue
        elif isinstance(value, dict):
            try:
                pickle.dumps(value)
                serializable_params[key] = value
            except Exception:
                continue
        elif hasattr(value, '__module__') and 'numpy' in value.__module__:
            try:
                if hasattr(value, 'tolist'):
                    serializable_params[key] = value.tolist()
                else:
                    serializable_params[key] = str(value)
            except Exception:
                continue
        else:
            try:
                pickle.dumps(value)
                serializable_params[key] = value
            except Exception:
                continue
    return serializable_params
def calculate_accuracy(outputs, labels):
    """计算分类准确率"""
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / labels.size(0)
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    collect_metrics = getattr(config, 'TRAIN_LOG_ENABLE', False)
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    pbar = tqdm(loader, desc="Epoch {}/{} [Train]".format(epoch + 1, config.EPOCHS))
    for X_e, X_h, adj, labels, metadata in pbar:
        X_e = X_e.to(device)
        X_h = X_h.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(X_e, X_h, adj)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_acc = calculate_accuracy(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += batch_acc * batch_size
        total_samples += batch_size
        if collect_metrics:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                y_true_list.extend(labels.detach().cpu().numpy().tolist())
                y_pred_list.extend(preds.detach().cpu().numpy().tolist())
                y_prob_list.extend(probs.detach().cpu().numpy().tolist())
        if total_samples > 0:
            pbar.set_postfix({'loss': "{:.4f}".format(total_loss / total_samples), 'acc': "{:.4f}".format(total_acc / total_samples)})
    metrics = None
    if collect_metrics and total_samples > 0:
        metrics = compute_metrics_from_collections(y_true_list, y_pred_list, y_prob_list, config)
    if total_samples == 0:
        return 0.0, 0.0, 0, metrics
    return total_loss / total_samples, total_acc / total_samples, total_samples, metrics
def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    collect_metrics = getattr(config, 'TRAIN_LOG_ENABLE', False)
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for X_e, X_h, adj, labels, metadata in pbar:
            X_e = X_e.to(device)
            X_h = X_h.to(device)
            adj = adj.to(device)
            labels = labels.to(device)
            logits, _ = model(X_e, X_h, adj)
            loss = criterion(logits, labels)
            batch_acc = calculate_accuracy(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += batch_acc * batch_size
            total_samples += batch_size
            if collect_metrics:
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                y_true_list.extend(labels.detach().cpu().numpy().tolist())
                y_pred_list.extend(preds.detach().cpu().numpy().tolist())
                y_prob_list.extend(probs.detach().cpu().numpy().tolist())
            if total_samples > 0:
                pbar.set_postfix({'loss': "{:.4f}".format(total_loss / total_samples), 'acc': "{:.4f}".format(total_acc / total_samples)})
    metrics = None
    if collect_metrics and total_samples > 0:
        metrics = compute_metrics_from_collections(y_true_list, y_pred_list, y_prob_list, config)
    if total_samples == 0:
        return 0.0, 0.0, 0, metrics
    return total_loss / total_samples, total_acc / total_samples, total_samples, metrics
def stratified_split_fixed_per_class(dataset, test_samples_per_class, random_seed=42):
    """每个类别固定测试集数量的分层划分"""
    all_labels = []
    for i in range(len(dataset)):
        _, _, _, label, _ = dataset[i]
        all_labels.append(label.item())
    label_to_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    np.random.seed(random_seed)
    train_indices = []
    test_indices = []
    print("\n" + "=" * 70)
    print("固定每类测试集数量: {}".format(test_samples_per_class))
    print("=" * 70)
    print("{:<8} {:<10} {:<10} {:<10}".format("类别", "总样本", "训练集", "测试集"))
    print("-" * 70)
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total = len(indices)
        if total < test_samples_per_class:
            print("⚠️ 类别 {} 样本不足: {} < {}".format(label, total, test_samples_per_class))
            train_indices.extend(indices)
            continue
        np.random.shuffle(indices)
        test_idx = indices[:test_samples_per_class]
        train_idx = indices[test_samples_per_class:]
        test_indices.extend(test_idx)
        train_indices.extend(train_idx)
        print("{:<8} {:<10} {:<10} {:<10}".format(label, total, len(train_idx), len(test_idx)))
    print("=" * 70)
    print("总计: 训练集 {}, 测试集 {}".format(len(train_indices), len(test_indices)))
    print("=" * 70)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset
def print_split_stats(train_labels, test_labels):
    """打印划分统计"""
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    print("\n" + "=" * 70)
    print("数据集划分统计")
    print("=" * 70)
    print("训练集: {}, 测试集: {}\n".format(len(train_labels), len(test_labels)))
    print("{:<8} {:<10} {:<10} {:<12}".format("类别", "训练集", "测试集", "测试占比"))
    print("-" * 70)
    for label in sorted(train_counts.keys()):
        train_count = train_counts[label]
        test_count = test_counts[label]
        test_ratio = test_count / (train_count + test_count) * 100
        print("{:<8} {:<10} {:<10} {:.1f}%".format(label, train_count, test_count, test_ratio))
    print("=" * 70)
def split_dataset(dataset, config_obj):
    """根据配置选择数据集划分策略"""
    split_strategy = getattr(config_obj, 'SPLIT_STRATEGY', 'stratified_ratio')
    if split_strategy == "random":
        from torch.utils.data import random_split
        train_size = int((1 - config_obj.VALID_RATIO) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(config_obj.RANDOM_SEED))
        print("✅ 使用随机划分")
    elif split_strategy == "stratified_ratio":
        all_labels = []
        for i in range(len(dataset)):
            _, _, _, label, _ = dataset[i]
            all_labels.append(label.item())
        train_indices, test_indices = train_test_split(range(len(dataset)), test_size=config_obj.VALID_RATIO, stratify=all_labels, random_state=config_obj.RANDOM_SEED)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        print("✅ 使用按比例分层划分")
        train_labels = [all_labels[i] for i in train_indices]
        test_labels = [all_labels[i] for i in test_indices]
        print_split_stats(train_labels, test_labels)
    elif split_strategy == "stratified_fixed":
        test_samples_per_class = getattr(config_obj, 'TEST_SAMPLES_PER_CLASS', 20)
        train_dataset, test_dataset = stratified_split_fixed_per_class(dataset, test_samples_per_class=test_samples_per_class, random_seed=config_obj.RANDOM_SEED)
        print("✅ 使用固定每类测试数: {}".format(test_samples_per_class))
    else:
        raise ValueError("未知的划分策略: {}".format(split_strategy))
    return train_dataset, test_dataset
def auto_detect_and_update_num_classes(dataset):
    """从数据集自动检测并更新类别数"""
    if hasattr(dataset, 'samples'):
        all_labels = set(sample['class_idx'] for sample in dataset.samples)
        num_classes_detected = len(all_labels)
        print("📊 类别统计:")
        class_counts = {}
        for sample in dataset.samples:
            class_idx = sample['class_idx']
            class_name = sample['class_name']
            if class_idx not in class_counts:
                class_counts[class_idx] = {'name': class_name, 'count': 0}
            class_counts[class_idx]['count'] += 1
        for class_idx in sorted(class_counts.keys()):
            info = class_counts[class_idx]
            print("   类别 {} ({}): {} 个样本".format(class_idx, info['name'], info['count']))
    else:
        print("⚠️ 无法直接访问 samples，扫描所有数据...")
        all_labels = set()
        for i in range(len(dataset)):
            _, _, _, label, _ = dataset[i]
            all_labels.add(label.item())
        num_classes_detected = len(all_labels)
    if num_classes_detected != config.NUM_CLASSES:
        print("\n⚠️ 类别数不匹配:")
        print("   config.NUM_CLASSES = {}".format(config.NUM_CLASSES))
        print("   实际检测到 = {} 个类别".format(num_classes_detected))
        print("   类别标签: {}".format(sorted(list(all_labels))))
        print("✅ 自动更新 NUM_CLASSES -> {}".format(num_classes_detected))
        config.NUM_CLASSES = num_classes_detected
    else:
        print("\n✅ 类别数一致: {} 个类别".format(num_classes_detected))
        print("   类别标签: {}".format(sorted(list(all_labels))))
def train_for_learning_curve(model, train_loader, config_obj, device, epochs):
    """学习曲线专用训练函数（简化版）"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config_obj.LEARNING_RATE, weight_decay=config_obj.WEIGHT_DECAY)
    for epoch in range(epochs):
        model.train()
        for X_e, X_h, adj, labels, metadata in train_loader:
            X_e = X_e.to(device)
            X_h = X_h.to(device)
            adj = adj.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_e, X_h, adj)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
def main():
    print("=" * 70)
    print("🔥 Step5: 双分支 Phase-Aware 模型训练（完整功能版）")
    print("=" * 70)
    try:
        config.validate_dual_branch_config()
    except ValueError as e:
        print("\n❌ 双分支配置错误:\n{}\n".format(e))
        print("提示：请检查 configs/config.py 中的双分支配置")
        return
    print("\n📊 双分支配置:")
    print("   Phase E 策略: {}".format(config.E_STRATEGY))
    if config.E_STRATEGY == "fixed_quota":
        print("     - 固定时长: {}s".format(config.E_FIXED_QUOTA_SEC))
        print("     - 对齐方式: {}".format(config.E_FIXED_ALIGN))
        print("     - 填充模式: {}".format(config.E_FIXED_PADDING_MODE))
    elif config.E_STRATEGY == "multi_window":
        print("     - 窗口数: {}".format(config.E_MULTI_NUM_WINDOWS))
        print("     - 窗口时长: {}s".format(config.E_MULTI_WINDOW_SEC))
        print("     - 采样方式: {}".format(config.E_MULTI_SAMPLING))
        print("     - 聚合方式: {}".format(config.E_MULTI_AGGREGATION))
    print("   Phase H 时长: {}s".format(config.PHASE_H_FIXED_SEC))
    print("   分支融合: {}".format(config.BRANCH_FUSION_MODE))
    print("   权重共享: {}".format(config.BRANCH_WEIGHT_SHARING))
    print("=" * 70)
    config.DATASET_NAME = config.get_dataset_name()
    config.EVAL_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "evaluation_results", config.DATASET_NAME)
    config.CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "checkpoints", config.DATASET_NAME)
    os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("\n🖥️ 设备: {}".format(device))
    tracker = ExperimentTracker(config)
    print("\n🆔 实验ID: {}".format(tracker.experiment_id))
    print("📁 实验目录: {}".format(tracker.exp_dir))
    tracker.save_config_snapshot()
    print("\n📂 双分支数据路径:")
    print("   Phase E: {}".format(config.PHASE_E_DIR))
    print("   Phase H: {}".format(config.PHASE_H_DIR))
    full_dataset = DualBranchSTFTDataset(config=config, mode='train')
    print("\n" + "=" * 70)
    print("自动检测数据集类别数...")
    print("=" * 70)
    auto_detect_and_update_num_classes(full_dataset)
    train_dataset, test_dataset = split_dataset(full_dataset, config)
    print("\n✅ 训练集: {}, 测试集: {}".format(len(train_dataset), len(test_dataset)))
    tracker.record_data_info(len(train_dataset), len(test_dataset))
    tracker.extract_hyperparameters()
    tracker.extract_model_architecture()
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    model = DualBranchGNNTransformer(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    print("\n🚀 开始训练...")
    print("   Batch Size: {}".format(config.BATCH_SIZE))
    print("   Epochs: {}".format(config.EPOCHS))
    print("   Learning Rate: {}".format(config.LEARNING_RATE))
    train_log_enable = getattr(config, 'TRAIN_LOG_ENABLE', False)
    training_history_rows = []
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None
    for epoch in range(config.EPOCHS):
        lr_this_epoch = optimizer.param_groups[0]['lr']
        train_loss, train_acc, num_train_samples, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, num_val_samples, val_metrics = validate(model, test_loader, criterion, device)
        scheduler.step()
        print("\n📊 Epoch {}/{}".format(epoch + 1, config.EPOCHS))
        print("   Train - Loss: {:.4f}, Acc: {:.4f}".format(train_loss, train_acc))
        print("   Val   - Loss: {:.4f}, Acc: {:.4f}".format(val_loss, val_acc))
        if train_log_enable:
            row = {'epoch': int(epoch + 1), 'train_loss': float(train_loss), 'val_loss': float(val_loss), 'train_accuracy': float(train_metrics['accuracy']) if train_metrics else float(train_acc), 'val_accuracy': float(val_metrics['accuracy']) if val_metrics else float(val_acc), 'train_precision': float(train_metrics.get('precision', 0.0)) if train_metrics else 0.0, 'val_precision': float(val_metrics.get('precision', 0.0)) if val_metrics else 0.0, 'train_recall': float(train_metrics.get('recall', 0.0)) if train_metrics else 0.0, 'val_recall': float(val_metrics.get('recall', 0.0)) if val_metrics else 0.0, 'train_f1': float(train_metrics.get('f1_score', 0.0)) if train_metrics else 0.0, 'val_f1': float(val_metrics.get('f1_score', 0.0)) if val_metrics else 0.0, 'lr': float(lr_this_epoch), 'num_train_samples': int(num_train_samples), 'num_val_samples': int(num_val_samples)}
            include_mse = getattr(config, 'TRAIN_LOG_INCLUDE_MSE', True) and getattr(config, 'EVAL_ENABLE_MSE_RMSE', False)
            if include_mse and train_metrics and val_metrics:
                row['train_mse'] = float(train_metrics.get('mse', 0.0))
                row['val_mse'] = float(val_metrics.get('mse', 0.0))
                row['train_rmse'] = float(train_metrics.get('rmse', 0.0))
                row['val_rmse'] = float(val_metrics.get('rmse', 0.0))
            else:
                row['train_mse'] = ''
                row['val_mse'] = ''
                row['train_rmse'] = ''
                row['val_rmse'] = ''
            training_history_rows.append(row)
            if getattr(config, 'TRAIN_LOG_PRINT_EVERY_EPOCH', True):
                print("   [TrainLog] train_acc={:.4f} val_acc={:.4f} train_f1={:.4f} val_f1={:.4f} lr={:.6f}".format(row['train_accuracy'], row['val_accuracy'], row['train_f1'], row['val_f1'], row['lr']))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = tracker.get_checkpoint_path(val_acc)
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc, 'val_loss': val_loss, 'config_params': get_serializable_config(config)}, best_model_path)
            print("   ✅ 保存最佳模型: {}".format(os.path.basename(best_model_path)))
    print("\n🎯 训练完成!")
    print("   最佳验证准确率: {:.4f} (Epoch {})".format(best_val_acc, best_epoch))
    if train_log_enable and len(training_history_rows) > 0:
        try:
            log_csv_name = getattr(config, 'TRAIN_LOG_CSV_NAME', 'training_history.csv')
            log_csv_path = os.path.join(tracker.exp_dir, log_csv_name)
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_precision', 'val_precision', 'train_recall', 'val_recall', 'train_f1', 'val_f1', 'train_mse', 'val_mse', 'train_rmse', 'val_rmse', 'lr', 'num_train_samples', 'num_val_samples']
            with open(log_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in training_history_rows:
                    writer.writerow(r)
            print("\n📝 已保存训练日志 CSV: {}".format(log_csv_path))
            best_row = max(training_history_rows, key=lambda x: x.get('val_accuracy', 0.0))
            summary = {'best_epoch': int(best_row['epoch']), 'best_val_accuracy': float(best_row.get('val_accuracy', 0.0)), 'best_metrics': best_row}
            log_json_name = getattr(config, 'TRAIN_LOG_JSON_NAME', 'training_summary.json')
            log_json_path = os.path.join(tracker.exp_dir, log_json_name)
            with open(log_json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print("📝 已保存训练摘要 JSON: {}".format(log_json_path))
        except Exception as e:
            print("⚠️ 保存训练日志失败: {}".format(e))
    tracker.record_training_info(config.EPOCHS, best_epoch, best_val_acc)
    if best_model_path and os.path.exists(best_model_path):
        print("\n📈 加载最佳模型进行完整评估...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'config_params' in checkpoint:
            saved_params = checkpoint['config_params']
            print("✅ 模型训练信息:")
            print("   Epoch: {}".format(checkpoint.get('epoch', 'NA')))
            print("   Val Acc: {:.4f}".format(checkpoint.get('val_acc', 0.0)))
            print("   保存了 {} 个配置参数".format(len(saved_params)))
        run_name = tracker.experiment_id
        try:
            from src.utils.dual_branch_evaluator import evaluate_dual_branch_model
            evaluate_dual_branch_model(model=model, test_loader=test_loader, config=config, device=device, run_name=run_name, train_dataset=train_dataset if config.EVAL_ENABLE_LEARNING_CURVE else None, model_class=DualBranchGNNTransformer if config.EVAL_ENABLE_LEARNING_CURVE else None, train_fn=train_for_learning_curve if config.EVAL_ENABLE_LEARNING_CURVE else None, experiment_tracker=tracker)
        except ImportError:
            print("⚠️ 双分支评估模块未找到，使用标准评估")
            from src.utils.evaluator import evaluate_model as _evaluate_model_fallback
            _evaluate_model_fallback(model=model, test_loader=test_loader, config=config, device=device, run_name=run_name)
    tracker.save_metadata()
    tracker.generate_summary_report()
    print("\n✅ 实验完成! 实验ID: {}".format(tracker.experiment_id))
    print("   所有文件保存在: {}".format(tracker.exp_dir))
    print("=" * 70)
if __name__ == "__main__":
    main()