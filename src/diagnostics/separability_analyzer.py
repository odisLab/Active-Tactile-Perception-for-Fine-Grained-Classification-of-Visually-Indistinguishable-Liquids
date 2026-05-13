"""
可分性指标计算 - 功能要求 4
"""

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from pathlib import Path

class SeparabilityAnalyzer:
    """可分性指标计算"""

    def __init__(self, config):
        self.config = config
        self.test_ratio = getattr(config, 'DIAG_LINEAR_TEST_RATIO', 0.3)
        self.classifier_type = getattr(config, 'DIAG_LINEAR_CLASSIFIER', 'logistic')

    def analyze(self, embeddings, labels, layer_name, output_dir):
        """
        计算可分性指标

        Returns:
            dict: {
                'silhouette_score': float,
                'inter_intra_ratio': float,
                'linear_accuracy': float,
                'linear_f1': float
            }
        """
        print(f"\n  📊 可分性指标: {layer_name}")

        results = {}

        # 1. Silhouette Score
        try:
            sil_score = silhouette_score(embeddings, labels)
            results['silhouette_score'] = sil_score
            print(f"     - Silhouette Score: {sil_score:.4f}")
        except:
            results['silhouette_score'] = None

        # 2. Inter/Intra Distance Ratio
        inter_intra = self._compute_inter_intra_ratio(embeddings, labels)
        results['inter_intra_ratio'] = inter_intra
        print(f"     - Inter/Intra Ratio: {inter_intra:.4f}")

        # 3. 线性可分性
        lin_acc, lin_f1 = self._compute_linear_separability(embeddings, labels)
        results['linear_accuracy'] = lin_acc
        results['linear_f1'] = lin_f1
        print(f"     - Linear Accuracy: {lin_acc:.4f}")
        print(f"     - Linear F1: {lin_f1:.4f}")

        # 保存
        self._save_results(results, layer_name, output_dir)

        return results

    def _compute_inter_intra_ratio(self, embeddings, labels):
        """计算类间/类内距离比"""
        unique_labels = np.unique(labels)

        # 类内距离（每类内部样本的平均距离）
        intra_dists = []
        for label in unique_labels:
            mask = labels == label
            class_emb = embeddings[mask]
            if len(class_emb) > 1:
                centroid = class_emb.mean(axis=0)
                dists = np.linalg.norm(class_emb - centroid, axis=1)
                intra_dists.append(dists.mean())

        intra_dist = np.mean(intra_dists) if intra_dists else 1.0

        # 类间距离（类中心之间的距离）
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroids.append(embeddings[mask].mean(axis=0))

        centroids = np.array(centroids)
        inter_dists = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

        inter_dist = np.mean(inter_dists) if inter_dists else 0.0

        return inter_dist / (intra_dist + 1e-8)

    def _compute_linear_separability(self, embeddings, labels):
        """线性可分性：训练 LogisticRegression 并测试"""
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=self.test_ratio, stratify=labels, random_state=42
        )

        # 训练分类器
        if self.classifier_type == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = LinearSVC(max_iter=1000, random_state=42)

        clf.fit(X_train, y_train)

        # 测试
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        return acc, f1

    def _save_results(self, results, layer_name, output_dir):
        """保存结果到 CSV"""
        output_dir = Path(output_dir)
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'

        df = pd.DataFrame([results])
        df.insert(0, 'layer', layer_name)

        save_path = output_dir / f"{dataset_name}_{layer_name}_separability.csv"
        df.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")
