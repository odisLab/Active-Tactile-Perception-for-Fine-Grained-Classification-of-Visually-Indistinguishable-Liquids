"""
双分支模型评估器
- 适配双分支模型的 5 元组输出
- 保留所有原有评估功能
- 支持中间特征导出
- 完整学习曲线支持
"""

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from collections import Counter


def evaluate_dual_branch_model(
    model,
    test_loader,
    config,
    device,
    run_name=None,
    train_dataset=None,
    model_class=None,
    train_fn=None,
    experiment_tracker=None
):
    """
    双分支模型完整评估函数
    
    Args:
        model: 训练好的双分支模型
        test_loader: 测试集 DataLoader
        config: 配置对象
        device: 设备
        run_name: 运行名称（用于文件命名）
        train_dataset: 训练集（学习曲线用）
        model_class: 模型类（学习曲线用）
        train_fn: 训练函数（学习曲线用）
        experiment_tracker: 实验追踪器
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    # 中间特征收集
    embeddings_dict = {
        'node_feat_e': [],
        'node_feat_h': [],
        'node_feat_fused': [],
        'transformer': []
    }
    
    # 元数据收集
    metadata_list = []
    
    print("\n" + "="*70)
    print("🔬 双分支模型评估中...")
    print("="*70)
    
    with torch.no_grad():
        for batch_idx, (X_e, X_h, adj, labels, metadata) in enumerate(tqdm(test_loader, desc="评估中")):
            X_e = X_e.to(device)
            X_h = X_h.to(device)
            adj = adj.to(device)
            labels = labels.to(device)
            
            # 前向传播（双分支返回 logits 和 embeddings）
            model_output = model(X_e, X_h, adj)
            
            # 解包模型输出
            if isinstance(model_output, tuple):
                logits, embeddings = model_output
            else:
                logits = model_output
                embeddings = {}
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 收集中间特征
            if getattr(config, 'DUAL_DIAG_ENABLE_EMBEDDING_EXPORT', False):
                for key in embeddings_dict.keys():
                    if key in embeddings:
                        feat = embeddings[key].detach().cpu().numpy()
                        embeddings_dict[key].append(feat)
            
            # 收集元数据
            for i in range(len(labels)):
                meta_entry = {
                    'sample_id': batch_idx * config.BATCH_SIZE + i,
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item()
                }
                
                # 添加 Phase E/H 路径（如果存在）
                if isinstance(metadata, dict):
                    if 'phase_e_path' in metadata:
                        meta_entry['phase_e_path'] = metadata['phase_e_path'][i]
                    if 'phase_h_path' in metadata:
                        meta_entry['phase_h_path'] = metadata['phase_h_path'][i]
                    if 'class_name' in metadata:
                        meta_entry['class_name'] = metadata['class_name'][i]
                
                metadata_list.append(meta_entry)
    
    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ==================== Module A: 标准分类指标 ====================
    results = {}
    
    if getattr(config, 'EVAL_ENABLE_METRICS', True):
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=config.EVAL_METRICS_AVERAGE, zero_division=0
        )
        
        results.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # 计算 MSE/RMSE（基于概率）
        if getattr(config, 'EVAL_ENABLE_MSE_RMSE', True):
            # One-hot 编码真实标签
            num_classes = len(set(all_labels))
            y_true_onehot = np.zeros((len(all_labels), num_classes))
            for i, label in enumerate(all_labels):
                y_true_onehot[i, label] = 1.0
            
            mse = mean_squared_error(y_true_onehot, all_probs)
            rmse = np.sqrt(mse)
            results['mse'] = mse
            results['rmse'] = rmse
        
        # Console 输出
        if getattr(config, 'EVAL_PRINT_TO_CONSOLE', True):
            print(f"\n{'='*70}")
            print(f"📊 评估结果 ({run_name})")
            print(f"{'='*70}")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            if getattr(config, 'EVAL_ENABLE_MSE_RMSE', True):
                print(f"   MSE:       {mse:.4f}")
                print(f"   RMSE:      {rmse:.4f}")
            print(f"{'='*70}\n")
        
        # CSV 输出
        if getattr(config, 'EVAL_SAVE_TO_CSV', True):
            df_results = pd.DataFrame([results])
            csv_path = os.path.join(
                config.EVAL_OUTPUT_DIR,
                f"{run_name}_eval_metrics.csv"
            )
            df_results.to_csv(csv_path, index=False)
            print(f"✅ 评估指标已保存: {csv_path}")
    
    # ==================== Module B: 混淆矩阵 ====================
    if getattr(config, 'EVAL_ENABLE_CONFUSION_MATRIX', True):
        cm = confusion_matrix(all_labels, all_preds)
        
        if getattr(config, 'EVAL_CONFUSION_NORMALIZE', True):
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm_normalized = cm
        
        # Console 输出
        if getattr(config, 'EVAL_CONFUSION_TO_CONSOLE', True):
            print("\n混淆矩阵（归一化）:")
            print(cm_normalized)
        
        # CSV 输出
        if getattr(config, 'EVAL_CONFUSION_TO_CSV', True):
            df_cm = pd.DataFrame(
                cm_normalized,
                columns=[f"Pred_{i}" for i in range(cm_normalized.shape[1])],
                index=[f"True_{i}" for i in range(cm_normalized.shape[0])]
            )
            cm_path = os.path.join(
                config.EVAL_OUTPUT_DIR,
                f"{run_name}_confusion_matrix.csv"
            )
            df_cm.to_csv(cm_path)
            print(f"✅ 混淆矩阵已保存: {cm_path}")
    
    # ==================== Module C: 学习曲线 ====================
    if getattr(config, 'EVAL_ENABLE_LEARNING_CURVE', False):
        if train_dataset is None or model_class is None or train_fn is None:
            print("\n⚠️ 学习曲线需要 train_dataset, model_class, train_fn 参数")
        else:
            print("\n📈 生成学习曲线...")
            learning_curve_results = _generate_dual_branch_learning_curve(
                train_dataset=train_dataset,
                test_loader=test_loader,
                model_class=model_class,
                train_fn=train_fn,
                config=config,
                device=device,
                run_name=run_name
            )
            results['learning_curve'] = learning_curve_results
    
    # ==================== 🔥 双分支特有：中间特征导出 ====================
    if getattr(config, 'DUAL_DIAG_ENABLE_EMBEDDING_EXPORT', False):
        diag_dir = os.path.join(config.EVAL_OUTPUT_DIR, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        
        csv_prefix = getattr(config, 'DUAL_DIAG_CSV_PREFIX', 'dual_diag')
        
        for layer_name, embeddings_list in embeddings_dict.items():
            if embeddings_list:
                embeddings_array = np.concatenate(embeddings_list, axis=0)
                
                # 展平为 2D（如果是高维特征）
                if embeddings_array.ndim > 2:
                    embeddings_flat = embeddings_array.reshape(embeddings_array.shape[0], -1)
                else:
                    embeddings_flat = embeddings_array
                
                # 保存为 CSV
                df_emb = pd.DataFrame(embeddings_flat)
                csv_path = os.path.join(
                    diag_dir,
                    f"{csv_prefix}_{layer_name}_embeddings.csv"
                )
                df_emb.to_csv(csv_path, index=False)
                print(f"✅ 特征已导出: {csv_path} (shape: {embeddings_flat.shape})")
        
        # 保存元数据
        df_metadata = pd.DataFrame(metadata_list)
        metadata_path = os.path.join(diag_dir, f"{csv_prefix}_metadata.csv")
        df_metadata.to_csv(metadata_path, index=False)
        print(f"✅ 元数据已保存: {metadata_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ 双分支模型评估完成！")
    print(f"{'='*70}\n")
    
    return results


def _generate_dual_branch_learning_curve(
    train_dataset,
    test_loader,
    model_class,
    train_fn,
    config,
    device,
    run_name
):
    """
    生成学习曲线（双分支版本）
    
    Args:
        train_dataset: 完整训练集
        test_loader: 测试集 DataLoader
        model_class: 模型类（如 DualBranchGNNTransformer）
        train_fn: 训练函数
        config: 配置对象
        device: 设备
        run_name: 运行名称
    
    Returns:
        dict: 学习曲线结果
    """
    # 学习曲线配置
    train_fracs = getattr(config, 'EVAL_LEARNING_CURVE_FRACS', [0.1, 0.25, 0.5, 0.75, 1.0])
    num_repeats = getattr(config, 'EVAL_LEARNING_CURVE_REPEATS', 3)
    metric = getattr(config, 'EVAL_LEARNING_CURVE_METRIC', 'accuracy')
    
    print(f"   训练集大小比例: {train_fracs}")
    print(f"   每个比例重复: {num_repeats} 次")
    print(f"   评估指标: {metric}")
    
    results = []
    
    # 获取训练集所有标签（用于分层采样）
    all_train_labels = []
    for i in range(len(train_dataset)):
        _, _, _, label, _ = train_dataset[i]  # 双分支数据集返回 5 个值
        all_train_labels.append(label.item())
    all_train_labels = np.array(all_train_labels)
    
    for frac in tqdm(train_fracs, desc="   学习曲线"):
        for repeat in range(num_repeats):
            # 分层采样训练子集
            subset_size = int(frac * len(train_dataset))
            
            if frac < 1.0:
                # 使用分层采样
                train_indices, _ = train_test_split(
                    range(len(train_dataset)),
                    train_size=subset_size,
                    stratify=all_train_labels,
                    random_state=config.RANDOM_SEED + repeat
                )
                subset_train = Subset(train_dataset, train_indices)
            else:
                # 使用全部训练集
                subset_train = train_dataset
            
            # 创建 DataLoader
            subset_loader = DataLoader(
                subset_train,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            
            # 初始化新模型
            temp_model = model_class(config).to(device)
            
            # 训练模型（使用提供的训练函数）
            epochs = getattr(config, 'EVAL_LEARNING_CURVE_EPOCHS', 10)
            train_fn(temp_model, subset_loader, config, device, epochs)
            
            # 评估模型
            temp_model.eval()
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for X_e, X_h, adj, labels, metadata in test_loader:
                    X_e = X_e.to(device)
                    X_h = X_h.to(device)
                    adj = adj.to(device)
                    
                    model_output = temp_model(X_e, X_h, adj)
                    logits = model_output[0] if isinstance(model_output, tuple) else model_output
                    
                    preds = torch.argmax(logits, dim=1)
                    
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.numpy())
            
            # 计算指标
            if metric == 'accuracy':
                score = accuracy_score(test_labels, test_preds)
            elif metric == 'f1':
                _, _, f1, _ = precision_recall_fscore_support(
                    test_labels, test_preds, average='macro', zero_division=0
                )
                score = f1
            else:
                score = accuracy_score(test_labels, test_preds)
            
            results.append({
                'train_fraction': frac,
                'train_size': subset_size,
                'repeat': repeat,
                'metric': metric,
                'score': score
            })
            
            # 清理内存
            del temp_model
            torch.cuda.empty_cache()
    
    # 保存学习曲线结果
    df_lc = pd.DataFrame(results)
    
    # 计算平均值和标准差
    df_summary = df_lc.groupby('train_fraction')['score'].agg(['mean', 'std']).reset_index()
    df_summary.columns = ['train_fraction', 'mean_score', 'std_score']
    
    # 合并
    df_lc_with_summary = df_lc.merge(df_summary, on='train_fraction')
    
    # 保存
    csv_suffix = getattr(config, 'EVAL_LEARNING_CURVE_CSV_SUFFIX', 'learning_curve')
    lc_path = os.path.join(
        config.EVAL_OUTPUT_DIR,
        f"{run_name}_{csv_suffix}.csv"
    )
    df_lc_with_summary.to_csv(lc_path, index=False)
    print(f"\n✅ 学习曲线已保存: {lc_path}")
    
    # 打印摘要
    print("\n学习曲线摘要:")
    print(df_summary.to_string(index=False))
    
    return {
        'results': results,
        'summary': df_summary.to_dict('records')
    }
