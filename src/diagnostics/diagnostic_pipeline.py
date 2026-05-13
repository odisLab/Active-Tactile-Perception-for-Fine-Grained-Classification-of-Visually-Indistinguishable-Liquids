"""
完整诊断管线 - 整合所有模块
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .three_layer_feature_extractor import ThreeLayerFeatureExtractor
from .dimensionality_reducer import DimensionalityReducer
from .separability_analyzer import SeparabilityAnalyzer
from .misclassification_analyzer import MisclassificationAnalyzer
from .sensitivity_tester import SensitivityTester

class DiagnosticPipeline:
    """
    完整诊断管线 - 严格按照10项功能要求
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # 输出目录
        if hasattr(config, 'DIAG_OUTPUT_DIR') and config.DIAG_OUTPUT_DIR:
            self.output_dir = Path(config.DIAG_OUTPUT_DIR)
        else:
            eval_dir = Path(getattr(config, 'EVAL_OUTPUT_DIR', 'outputs'))
            # ✅ 如果 eval_dir 已经包含 dataset_name，不要重复添加
            if eval_dir.name == getattr(config, 'DATASET_NAME', ''):
                self.output_dir = eval_dir  # 直接使用
            else:
                dataset_name = getattr(config, 'DATASET_NAME', 'default')
                self.output_dir = eval_dir / dataset_name

        # 初始化各模块
        self.feature_extractor = ThreeLayerFeatureExtractor(model, config, device)
        self.dim_reducer = DimensionalityReducer(config)
        self.sep_analyzer = SeparabilityAnalyzer(config)
        self.misclass_analyzer = MisclassificationAnalyzer(config)
        self.sensitivity_tester = SensitivityTester(model, config, device)

        print(f"\n  📁 输出目录: {self.output_dir}")

    def run(self, test_loader):
        """
        运行完整诊断流程
        """
        print("\n" + "="*70)
        print("🔬 开始运行完整诊断管线")
        print("="*70)

        results = {}

        # ==================== 步骤 1: 基线评估 + 三层表征提取 ====================
        if getattr(self.config, 'DIAG_ENABLE_EMBEDDING_EXPORT', True):
            print("\n📊 步骤 1: 基线评估 + 三层表征提取")
            baseline_metrics, embeddings_dict = self._step1_extract_features(test_loader)
            results['baseline'] = baseline_metrics
            results['embeddings'] = embeddings_dict
        else:
            print("\n⏭️  步骤 1: 跳过特征提取")
            baseline_metrics, embeddings_dict = self._step1_baseline_only(test_loader)
            results['baseline'] = baseline_metrics

        # ==================== 步骤 2-4: 表征分析（降维 + 可分性） ====================
        if getattr(self.config, 'DIAG_ENABLE_DIMENSIONALITY_REDUCTION', True) and embeddings_dict:
            print("\n📊 步骤 2-4: 表征降维与可分性分析")
            self._step2_analyze_embeddings(embeddings_dict)
        else:
            print("\n⏭️  步骤 2-4: 跳过降维分析")

        # ==================== 步骤 5: 错分样本分析 ====================
        if getattr(self.config, 'DIAG_ENABLE_MISCLASSIFICATION_ANALYSIS', True):
            print("\n📊 步骤 5: 错分样本分析")
            misclass_results = self._step5_misclassification_analysis(baseline_metrics)
            results['misclassification'] = misclass_results
        else:
            print("\n⏭️  步骤 5: 跳过错分样本分析")

        # ==================== 步骤 6-8: 敏感性测试 ====================
        if getattr(self.config, 'DIAG_ENABLE_FREQUENCY_SENSITIVITY', False):
            print("\n📊 步骤 6: 频段敏感性测试")
            freq_sens = self.sensitivity_tester.test_frequency_sensitivity(test_loader, self.output_dir)
            results['frequency_sensitivity'] = freq_sens

        if getattr(self.config, 'DIAG_ENABLE_TIME_SENSITIVITY', False):
            print("\n📊 步骤 6: 时间窗敏感性测试")
            time_sens = self.sensitivity_tester.test_time_sensitivity(test_loader, self.output_dir)
            results['time_sensitivity'] = time_sens

        if getattr(self.config, 'DIAG_ENABLE_AUTO_SCAN', False):
            print("\n📊 步骤 7: 自动扫描敏感区域")
            auto_scan = self.sensitivity_tester.auto_scan_sensitive_regions(test_loader, self.output_dir)
            results['auto_scan'] = auto_scan

        if getattr(self.config, 'DIAG_ENABLE_CHANNEL_ABLATION', False):
            print("\n📊 步骤 8: 通道消融测试")
            channel_abl = self.sensitivity_tester.test_channel_ablation(test_loader, self.output_dir)
            results['channel_ablation'] = channel_abl

        # 清理
        self.feature_extractor.cleanup()

        print("\n" + "="*70)
        print(f"✅ 诊断完成！所有结果已保存到: {self.output_dir}")
        print("="*70)

        return results

    def _step1_extract_features(self, data_loader):
        """步骤 1: 提取三层表征并评估基线"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, (stft_batch, adj_matrix, labels) in enumerate(tqdm(data_loader, desc="  提取特征")):
                # ===== 关键修复：分别处理 CPU/GPU =====
                # stft_batch 和 adj_matrix 移到 GPU
                stft_batch_device = stft_batch.to(self.device)
                adj_matrix_device = adj_matrix.to(self.device)
                
                # 前向传播
                outputs = self.model(stft_batch_device, adj_matrix_device)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # 提取三层表征（传入 GPU tensor 和 labels）
                self.feature_extractor.extract_batch(
                    stft_batch_device,  # GPU tensor
                    adj_matrix_device,  # GPU tensor
                    labels,             # CPU tensor
                    batch_idx
                )
        
        # 收集预测结果
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # 保存表征
        self.feature_extractor.save_features(self.output_dir)

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
        """仅评估基线（不提取特征）"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for stft_batch, adj_matrix, labels in tqdm(data_loader, desc="  评估基线"):
                stft_batch = stft_batch.to(self.device)
                adj_matrix = adj_matrix.to(self.device)

                outputs = self.model(stft_batch, adj_matrix)
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

    def _load_embeddings(self):
        """从保存的 CSV 加载 embeddings"""
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        layer_names = getattr(self.config, 'DIAG_EXPORT_LAYERS', ['raw_stft', 'node_feat', 'transformer'])

        embeddings_dict = {}

        for layer_name in layer_names:
            csv_path = self.output_dir / f"{dataset_name}_{layer_name}.csv"

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            # 分离元信息和特征
            meta_cols = ['sample_id', 'source_file', 'class_label', 'split']
            feat_cols = [c for c in df.columns if c not in meta_cols]

            embeddings_dict[layer_name] = {
                'embeddings': df[feat_cols].values,
                'labels': df['class_label'].values,
                'metadata': df[meta_cols]
            }

        return embeddings_dict

    def _step2_analyze_embeddings(self, embeddings_dict):
        """步骤 2-4: 降维 + 可分性分析"""
        for layer_name, data in embeddings_dict.items():
            embeddings = data['embeddings']
            labels = data['labels']
            metadata = data['metadata']

            # 降维
            if getattr(self.config, 'DIAG_ENABLE_DIMENSIONALITY_REDUCTION', True):
                self.dim_reducer.reduce_and_save(embeddings, metadata, layer_name, self.output_dir)

            # 可分性
            if getattr(self.config, 'DIAG_ENABLE_SEPARABILITY_METRICS', True):
                self.sep_analyzer.analyze(embeddings, labels, layer_name, self.output_dir)

    def _step5_misclassification_analysis(self, baseline_metrics):
        """步骤 5: 错分样本分析"""
        # 构建元信息
        metadata = pd.DataFrame({
            'sample_id': range(len(baseline_metrics['labels'])),
            'source_file': [f"sample_{i}" for i in range(len(baseline_metrics['labels']))],
            'class_label': baseline_metrics['labels'],
            'split': ['test'] * len(baseline_metrics['labels'])
        })

        results = self.misclass_analyzer.analyze(
            baseline_metrics['labels'],
            baseline_metrics['predictions'],
            baseline_metrics['probabilities'],
            metadata,
            self.output_dir
        )

        return results
