"""
双分支诊断管线 - 支持 Phase E/H 分别分析
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .diagnostic_pipeline import DiagnosticPipeline


class DualBranchDiagnosticPipeline(DiagnosticPipeline):
    """
    双分支诊断管线（扩展版）
    
    新增功能：
    1. 分别提取 Phase E 和 Phase H 的特征
    2. 分析 Phase E 和 Phase H 的贡献度
    3. 测试单独屏蔽 Phase E 或 Phase H 的影响
    """
    
    def __init__(self, model, config, device):
        # 先调用父类初始化（会创建 self.sensitivity_tester）
        super().__init__(model, config, device)
        
        # ✅ 替换为双分支版本
        from .dual_branch_sensitivity_tester import DualBranchSensitivityTester
        self.sensitivity_tester = DualBranchSensitivityTester(model, config, device)
        
        # 双分支特有的特征提取器
        from .dual_branch_feature_extractor import DualBranchFeatureExtractor
        self.dual_feature_extractor = DualBranchFeatureExtractor(model, config, device)
        
        print(f"  ✅ 双分支诊断管线已初始化")

    
    # ==================== 重写核心方法（适配双分支数据格式）====================
    
    def _step1_extract_features(self, data_loader):
        """步骤 1: 提取双分支三层表征并评估基线（重写）"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            # ✅ 修改：适配双分支数据格式
            for batch_idx, (X_e, X_h, adj_matrix, labels, metadata) in enumerate(tqdm(data_loader, desc="  提取特征")):
                # 数据移到 GPU
                X_e_device = X_e.to(self.device)
                X_h_device = X_h.to(self.device)
                adj_matrix_device = adj_matrix.to(self.device)
                
                # ✅ 修复：正确解包模型返回值
                model_output = self.model(X_e_device, X_h_device, adj_matrix_device)
                
                # 判断返回值类型
                if isinstance(model_output, tuple):
                    outputs, _ = model_output  # (logits, embeddings)
                else:
                    outputs = model_output  # 只返回 logits
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # ✅ 修改：双分支特征提取
                self.dual_feature_extractor.extract_batch(
                    X_e_device, X_h_device,  # 两个 Phase
                    adj_matrix_device,
                    labels,
                    metadata,  # 包含 phase_e_path, phase_h_path
                    batch_idx
                )
                
                # 收集预测结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 保存表征
        self.dual_feature_extractor.save_features(self.output_dir)

        # 基线指标
        baseline_metrics = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'accuracy': (np.array(all_preds) == np.array(all_labels)).mean()
        }

        print(f"  ✅ 基线准确率: {baseline_metrics['accuracy']:.4f}")

        # 加载保存的 embeddings
        embeddings_dict = self._load_embeddings()

        return baseline_metrics, embeddings_dict
    
    def _step1_baseline_only(self, data_loader):
        """仅评估基线（重写，适配双分支）"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            # ✅ 修改：适配双分支数据格式
            for X_e, X_h, adj_matrix, labels, metadata in tqdm(data_loader, desc="  评估基线"):
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)

                # ✅ 修复：正确解包模型返回值
                model_output = self.model(X_e, X_h, adj_matrix)
                
                if isinstance(model_output, tuple):
                    outputs, _ = model_output
                else:
                    outputs = model_output
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        baseline_metrics = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'accuracy': (np.array(all_preds) == np.array(all_labels)).mean()
        }

        print(f"  ✅ 基线准确率: {baseline_metrics['accuracy']:.4f}")

        return baseline_metrics, {}
    
    # ==================== 新增：双分支特有的分析功能 ====================
    
    def analyze_phase_contribution(self, test_loader):
        """
        新增功能：分析 Phase E 和 Phase H 的贡献度
        
        测试策略：
        1. 只使用 Phase E（Phase H 全零）
        2. 只使用 Phase H（Phase E 全零）
        3. 对比完整双分支的性能
        
        Returns:
            dict: 包含各种配置下的准确率
        """
        print("\n📊 分析 Phase E/H 贡献度...")
        self.model.eval()
        
        results = {
            'full': [],        # 完整双分支
            'phase_e_only': [],  # 只用 Phase E
            'phase_h_only': [],  # 只用 Phase H
        }
        
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in tqdm(test_loader, desc="  测试Phase贡献"):
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                labels = labels.to(self.device)
                
                # ✅ 修复：正确解包模型返回值
                # 1. 完整双分支
                model_output_full = self.model(X_e, X_h, adj_matrix)
                outputs_full = model_output_full[0] if isinstance(model_output_full, tuple) else model_output_full
                _, preds_full = torch.max(outputs_full, 1)
                results['full'].extend((preds_full == labels).cpu().numpy())
                
                # 2. 只用 Phase E（Phase H 置零）
                X_h_zero = torch.zeros_like(X_h)
                model_output_e = self.model(X_e, X_h_zero, adj_matrix)
                outputs_e = model_output_e[0] if isinstance(model_output_e, tuple) else model_output_e
                _, preds_e = torch.max(outputs_e, 1)
                results['phase_e_only'].extend((preds_e == labels).cpu().numpy())
                
                # 3. 只用 Phase H（Phase E 置零）
                X_e_zero = torch.zeros_like(X_e)
                model_output_h = self.model(X_e_zero, X_h, adj_matrix)
                outputs_h = model_output_h[0] if isinstance(model_output_h, tuple) else model_output_h
                _, preds_h = torch.max(outputs_h, 1)
                results['phase_h_only'].extend((preds_h == labels).cpu().numpy())
        
        # 计算准确率
        acc_summary = {
            'full': np.mean(results['full']),
            'phase_e_only': np.mean(results['phase_e_only']),
            'phase_h_only': np.mean(results['phase_h_only']),
        }
        
        # 计算贡献度（相对于完整模型的性能下降）
        acc_summary['phase_e_contribution'] = acc_summary['full'] - acc_summary['phase_h_only']
        acc_summary['phase_h_contribution'] = acc_summary['full'] - acc_summary['phase_e_only']
        
        # 保存结果
        self._save_phase_contribution(acc_summary)
        
        print(f"\n  ✅ Phase 贡献度分析:")
        print(f"     完整双分支:       {acc_summary['full']:.4f}")
        print(f"     仅 Phase E:       {acc_summary['phase_e_only']:.4f}")
        print(f"     仅 Phase H:       {acc_summary['phase_h_only']:.4f}")
        print(f"     Phase E 贡献:     {acc_summary['phase_e_contribution']:.4f}")
        print(f"     Phase H 贡献:     {acc_summary['phase_h_contribution']:.4f}")
        
        return acc_summary
    
    def _save_phase_contribution(self, acc_summary):
        """保存 Phase 贡献度分析结果"""
        df = pd.DataFrame([acc_summary])
        output_path = self.output_dir / "phase_contribution_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"  💾 保存到: {output_path}")
    
    def run(self, test_loader):
        """
        运行完整诊断流程（扩展版）
        
        在父类基础上新增：
        - Phase E/H 贡献度分析
        """
        # 调用父类的 run 方法（执行标准诊断）
        results = super().run(test_loader)
        
        # ==================== 新增：Phase 贡献度分析 ====================
        print("\n" + "="*70)
        print("🆕 双分支特有分析")
        print("="*70)
        
        phase_contrib = self.analyze_phase_contribution(test_loader)
        results['phase_contribution'] = phase_contrib
        
        # 清理
        self.dual_feature_extractor.cleanup()
        
        return results
    
    def _load_embeddings(self):
        """加载双分支特征（重写，支持 Phase E/H 分别的特征）"""
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        
        # ✅ 双分支特有的层名称
        layer_names = getattr(self.config, 'DUAL_DIAG_EXPORT_LAYERS', [
            'node_feat_e',      # Phase E GNN 输出
            'node_feat_h',      # Phase H GNN 输出
            'node_feat_fused',  # 融合后特征
            'transformer'       # Transformer 输出
        ])

        embeddings_dict = {}

        for layer_name in layer_names:
            csv_path = self.output_dir / f"{dataset_name}_{layer_name}.csv"

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            # 分离元信息和特征
            meta_cols = ['sample_id', 'phase_e_path', 'phase_h_path', 'class_label', 'split']
            # ✅ 兼容缺失的列
            meta_cols = [col for col in meta_cols if col in df.columns]
            feat_cols = [c for c in df.columns if c not in meta_cols]

            embeddings_dict[layer_name] = {
                'embeddings': df[feat_cols].values,
                'labels': df['class_label'].values if 'class_label' in df.columns else None,
                'metadata': df[meta_cols] if meta_cols else None
            }

        return embeddings_dict
