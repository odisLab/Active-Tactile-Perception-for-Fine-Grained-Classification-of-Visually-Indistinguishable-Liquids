"""
模型评估模块 - 增强版（支持实验追踪）
支持保存完整超参数到CSV，便于实验对比
"""
import os
import json
import csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)

# ==================== Module A: 标准分类指标（增强版） ====================



def compute_metrics_from_collections(y_true_list, y_pred_list, y_prob_list, config):
    """从逐 batch 收集的列表中计算分类指标（可用于 Train/Val 每个 epoch 记录）"""
    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_pred = np.asarray(y_pred_list, dtype=np.int64)
    y_prob = np.asarray(y_prob_list, dtype=np.float32)
    return compute_classification_metrics(y_true, y_pred, y_prob, config)

def compute_classification_metrics(y_true, y_pred, y_pred_probs, config):
    """计算分类指标（测试集专用）"""
    avg = config.EVAL_METRICS_AVERAGE

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # MSE/RMSE
    if config.EVAL_ENABLE_MSE_RMSE:
        num_classes = y_pred_probs.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        mse = mean_squared_error(y_true_onehot, y_pred_probs)
        rmse = np.sqrt(mse)
        metrics['mse'] = mse
        metrics['rmse'] = rmse

    return metrics


def print_metrics(metrics, config):
    """格式化打印指标到终端"""
    if not config.EVAL_PRINT_TO_CONSOLE:
        return

    print("\n" + "="*60)
    print("📊 测试集评估结果（Test Set Evaluation）")
    print("="*60)

    for key, value in metrics.items():
        print(f"  {key:15s}: {value:.4f}")

    print("="*60)


def save_metrics_to_csv(metrics, config, run_name, experiment_tracker=None):
    """
    保存指标到CSV（增强版：支持超参数记录）

    Args:
        metrics: 评估指标字典
        config: 配置对象
        run_name: 运行名称
        experiment_tracker: 实验追踪器对象（可选）
    """
    if not config.EVAL_SAVE_TO_CSV:
        return

    # 确定CSV路径
    if experiment_tracker:
        # 使用实验专属目录
        csv_path = experiment_tracker.get_csv_path('eval')
    else:
        # 使用传统路径（兼容性）
        csv_path = os.path.join(config.EVAL_OUTPUT_DIR, f"{run_name}_{config.EVAL_CSV_SUFFIX_MAIN}.csv")

    # 准备数据行
    row = {}

    # 添加基本信息
    if experiment_tracker:
        # 增强版：包含完整超参数
        row = experiment_tracker.get_full_info_for_csv()
    else:
        # 传统版：仅包含timestamp和run_name
        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'run_name': run_name
        }

    # 添加评估指标
    row.update(metrics)

    # 定义列顺序
    if experiment_tracker:
        # 增强版列顺序：experiment_id, timestamp, 超参数, 模型参数, 数据信息, 评估指标
        fieldnames = ['experiment_id', 'timestamp', 
                      'batch_size', 'learning_rate', 'weight_decay', 'epochs', 
                      'lr_step_size', 'lr_gamma', 'valid_ratio', 'random_seed',
                      'gnn_hidden', 'transformer_layers',
                      'train_size', 'test_size', 'best_epoch', 'duration_min',
                      'accuracy', 'precision', 'recall', 'f1_score']

        if config.EVAL_ENABLE_MSE_RMSE:
            fieldnames.extend(['mse', 'rmse'])
    else:
        # 传统版列顺序
        fieldnames = ['timestamp', 'run_name', 'accuracy', 'precision', 'recall', 'f1_score']
        if config.EVAL_ENABLE_MSE_RMSE:
            fieldnames.extend(['mse', 'rmse'])

    # 追加模式写入
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n✅ 评估指标已保存: {csv_path}")


# ==================== Module B: 混淆矩阵 ====================

def compute_and_save_confusion_matrix(y_true, y_pred, config, run_name, experiment_tracker=None):
    """计算混淆矩阵并保存"""
    if not config.EVAL_ENABLE_CONFUSION_MATRIX:
        return

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 可选归一化
    cm_to_display = cm
    if config.EVAL_CONFUSION_NORMALIZE:
        cm_to_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Console输出
    if config.EVAL_CONFUSION_TO_CONSOLE:
        print("\n" + "="*60)
        print("🔢 混淆矩阵（Confusion Matrix - Test Set）")
        print("="*60)
        print(f"形状: {cm.shape}")
        if cm.shape[0] <= 10:
            np.set_printoptions(precision=2, suppress=True)
            print(cm_to_display)
        else:
            print("矩阵过大，仅保存到CSV")
        print("="*60)

    # CSV保存
    if config.EVAL_CONFUSION_TO_CSV:
        if experiment_tracker:
            csv_path = experiment_tracker.get_csv_path('confusion_matrix')
        else:
            csv_path = os.path.join(config.EVAL_OUTPUT_DIR, f"{run_name}_confusion_matrix.csv")

        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'run_name': run_name if not experiment_tracker else experiment_tracker.experiment_id,
            'matrix_shape': str(list(cm.shape)),
            'normalized': config.EVAL_CONFUSION_NORMALIZE,
            'matrix': json.dumps(cm_to_display.tolist())
        }

        fieldnames = ['timestamp', 'run_name', 'matrix_shape', 'normalized', 'matrix']
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"✅ 混淆矩阵已保存: {csv_path}")


# ==================== Module C: 学习曲线 ====================

def generate_learning_curve_data(train_dataset, test_loader, model_class, config, train_fn, device, run_name, experiment_tracker=None):
    """生成学习曲线数据"""
    if not config.EVAL_ENABLE_LEARNING_CURVE:
        return

    print("\n" + "="*60)
    print("📈 生成学习曲线数据（Learning Curve - 多次重训练）")
    print("="*60)

    total_train_size = len(train_dataset)
    fractions = config.EVAL_LEARNING_CURVE_FRACS
    repeats = config.EVAL_LEARNING_CURVE_REPEATS
    metric_name = config.EVAL_LEARNING_CURVE_METRIC

    results = []

    for frac in fractions:
        subset_size = int(total_train_size * frac)
        if subset_size < 1:
            continue

        for repeat_idx in range(repeats):
            print(f"\n训练集比例: {int(frac*100)}% ({subset_size}/{total_train_size} 样本), 重复 {repeat_idx+1}/{repeats}")

            # 随机采样子集
            indices = np.random.choice(total_train_size, subset_size, replace=False)
            subset = Subset(train_dataset, indices)
            subset_loader = DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

            # 重新初始化模型
            model = model_class(config).to(device)

            # 简化训练
            epochs = min(config.EPOCHS, 20)
            train_fn(model, subset_loader, config, device, epochs)

            # 在测试集上评估
            model.eval()
            y_true_list = []
            y_pred_list = []

            with torch.no_grad():
                for stft_batch, adj_matrix, labels in test_loader:
                    stft_batch = stft_batch.to(device)
                    adj_matrix = adj_matrix.to(device)
                    labels = labels.to(device)

                    logits = model(stft_batch, adj_matrix)
                    preds = torch.argmax(logits, dim=1)

                    y_true_list.extend(labels.cpu().numpy())
                    y_pred_list.extend(preds.cpu().numpy())

            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)

            # 计算指标
            if metric_name == "accuracy":
                metric_value = accuracy_score(y_true, y_pred)
            elif metric_name == "f1":
                metric_value = f1_score(y_true, y_pred, average=config.EVAL_METRICS_AVERAGE, zero_division=0)
            else:
                metric_value = accuracy_score(y_true, y_pred)

            print(f"  {metric_name}: {metric_value:.4f}")

            results.append({
                'train_fraction': frac,
                'train_size': subset_size,
                'repeat_index': repeat_idx,
                'metric_name': metric_name,
                'metric_value': metric_value
            })

    # 保存到CSV
    if experiment_tracker:
        csv_path = experiment_tracker.get_csv_path('learning_curve')
    else:
        csv_path = os.path.join(config.EVAL_OUTPUT_DIR, f"{run_name}_{config.EVAL_LEARNING_CURVE_CSV_SUFFIX}.csv")

    fieldnames = ['train_fraction', 'train_size', 'repeat_index', 'metric_name', 'metric_value']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ 学习曲线数据已保存: {csv_path}")
    print("="*60)


# ==================== 综合评估接口（增强版） ====================

def evaluate_model(model, test_loader, config, device, run_name,
                   train_dataset=None, model_class=None, train_fn=None,
                   experiment_tracker=None):  # 新增参数
    """
    综合评估入口（增强版）

    新增参数：
        experiment_tracker: ExperimentTracker对象，用于记录超参数
    """
    print("\n" + "="*60)
    print("🔍 开始模型评估（仅测试集）")
    print("="*60)
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 确保模型处于评估模式
    model.eval()

    y_true_list = []
    y_pred_list = []
    y_pred_probs_list = []

    # 收集测试集预测结果
    with torch.no_grad():
        for stft_batch, adj_matrix, labels in test_loader:
            stft_batch = stft_batch.to(device)
            adj_matrix = adj_matrix.to(device)
            labels = labels.to(device)

            logits = model(stft_batch, adj_matrix)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_pred_probs_list.extend(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_probs = np.array(y_pred_probs_list)

    # Module A: 标准指标
    if config.EVAL_ENABLE_METRICS:
        metrics = compute_classification_metrics(y_true, y_pred, y_pred_probs, config)
        print_metrics(metrics, config)
        save_metrics_to_csv(metrics, config, run_name, experiment_tracker)  # 传递tracker

        # 如果有tracker，记录结果到元数据
        if experiment_tracker:
            experiment_tracker.record_results(metrics)

    # Module B: 混淆矩阵
    if config.EVAL_ENABLE_CONFUSION_MATRIX:
        compute_and_save_confusion_matrix(y_true, y_pred, config, run_name, experiment_tracker)

    # Module C: 学习曲线
    if config.EVAL_ENABLE_LEARNING_CURVE:
        if train_dataset is None or model_class is None or train_fn is None:
            print("\n⚠️  学习曲线功能启用，但缺少必要参数，已跳过")
        else:
            generate_learning_curve_data(train_dataset, test_loader, model_class, config, train_fn, device, run_name, experiment_tracker)

    print("\n" + "="*60)
    print("✅ 评估完成")
    print("="*60)
