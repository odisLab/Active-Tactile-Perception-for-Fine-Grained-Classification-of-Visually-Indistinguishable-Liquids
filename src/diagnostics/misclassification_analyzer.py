"""
错分样本分析器 - 功能要求 5
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

class MisclassificationAnalyzer:
    """错分样本分组统计"""

    def __init__(self, config):
        self.config = config
        self.confidence_bins = getattr(config, 'DIAG_CONFIDENCE_BINS', [0.0, 0.5, 0.7, 0.9, 0.95, 1.0])

    def analyze(self, true_labels, pred_labels, probabilities, metadata, output_dir):
        """分析错分样本"""
        output_dir = Path(output_dir)
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'

        print(f"\n  📊 错分样本分析")

        num_classes = probabilities.shape[1]
        max_probs = np.max(probabilities, axis=1)

        # 1. 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
        cm_df = pd.DataFrame(cm, 
                             index=[f"True_{i}" for i in range(num_classes)],
                             columns=[f"Pred_{i}" for i in range(num_classes)])
        cm_path = output_dir / f"{dataset_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print(f"     ✅ 保存: {cm_path.name}")

        # 2. Per-class metrics
        report_dict = classification_report(true_labels, pred_labels, 
                                   labels=list(range(num_classes)),
                                   output_dict=True, 
                                   zero_division=0)

        per_class_metrics = []
        for i in range(num_classes):
            key = str(i)
            if key in report_dict:
                per_class_metrics.append({
                    'class': i,
                    'precision': report_dict[key]['precision'],
                    'recall': report_dict[key]['recall'],
                    'f1': report_dict[key]['f1-score'],
                    'support': report_dict[key]['support']
                })

        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_path = output_dir / f"{dataset_name}_per_class_metrics.csv"
        per_class_df.to_csv(per_class_path, index=False)
        print(f"     ✅ 保存: {per_class_path.name}")

        # 打印
        print(f"\n     Per-Class Metrics:")
        for row in per_class_metrics:
            print(f"       Class {row['class']}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")

        # 3. 按置信度区间统计
        conf_stats = self._analyze_by_confidence(true_labels, pred_labels, max_probs)
        conf_path = output_dir / f"{dataset_name}_confidence_stats.csv"
        conf_stats.to_csv(conf_path, index=False)
        print(f"     ✅ 保存: {conf_path.name}")

        # 4. 错分样本详情
        misclass_df = self._extract_misclassified(true_labels, pred_labels, max_probs, metadata)
        if len(misclass_df) > 0:
            misclass_path = output_dir / f"{dataset_name}_misclassified_samples.csv"
            misclass_df.to_csv(misclass_path, index=False)
            print(f"     ✅ 保存: {misclass_path.name} ({len(misclass_df)} 个错分样本)")

        return {
            'confusion_matrix': cm,
            'per_class_metrics': per_class_df,
            'confidence_stats': conf_stats,
            'misclassified_samples': misclass_df
        }

    def _analyze_by_confidence(self, true_labels, pred_labels, max_probs):
        """按置信度区间统计"""
        stats = []
        correct = (true_labels == pred_labels)

        for i in range(len(self.confidence_bins) - 1):
            low = self.confidence_bins[i]
            high = self.confidence_bins[i+1]
            mask = (max_probs >= low) & (max_probs < high)

            if mask.sum() == 0:
                continue

            stats.append({
                'confidence_range': f"[{low:.2f}, {high:.2f})",
                'count': mask.sum(),
                'accuracy': correct[mask].mean()
            })

        return pd.DataFrame(stats)

    def _extract_misclassified(self, true_labels, pred_labels, max_probs, metadata):
        """提取错分样本"""
        misclass_mask = true_labels != pred_labels

        if misclass_mask.sum() == 0:
            return pd.DataFrame()

        df = metadata.copy()
        df['true_label'] = true_labels
        df['pred_label'] = pred_labels
        df['confidence'] = max_probs

        misclass_df = df[misclass_mask].sort_values('confidence', ascending=False).reset_index(drop=True)

        return misclass_df
