"""
表征降维与可分性分析模块
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP未安装，降维功能将跳过UMAP。安装: pip install umap-learn")

class EmbeddingAnalyzer:
    """
    表征分析器

    功能：
    1. 降维（PCA, t-SNE, UMAP）
    2. 可分性指标计算
    3. 类间/类内距离分析
    """

    def __init__(self, config):
        self.config = config

    def reduce_dimensions(self, embeddings, labels, method='pca', n_components=2):
        """
        降维

        Args:
            embeddings: (N, D) 特征矩阵
            labels: (N,) 标签
            method: 降维方法
            n_components: 目标维度

        Returns:
            reduced: (N, n_components) 降维后的坐标
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, n_jobs=-1)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                print(f"      ⚠️  跳过UMAP（未安装）")
                return None
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=self.config.DIAG_UMAP_N_NEIGHBORS,
                min_dist=self.config.DIAG_UMAP_MIN_DIST,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        reduced = reducer.fit_transform(embeddings)
        return reduced

    def compute_separability_metrics(self, embeddings, labels):
        """
        计算可分性指标

        Returns:
            dict: 各项指标
        """
        metrics = {}

        # 1. Silhouette Score
        if "silhouette" in self.config.DIAG_SEPARABILITY_METRICS:
            try:
                sil_score = silhouette_score(embeddings, labels)
                metrics['silhouette_score'] = float(sil_score)
            except Exception as e:
                metrics['silhouette_score'] = None
                print(f"      ⚠️  Silhouette计算失败: {e}")

        # 2. 类间/类内距离比
        if "inter_intra_ratio" in self.config.DIAG_SEPARABILITY_METRICS:
            try:
                inter_intra = self._compute_inter_intra_ratio(embeddings, labels)
                metrics.update(inter_intra)
            except Exception as e:
                print(f"      ⚠️  类间/类内距离比计算失败: {e}")

        # 3. 线性可分性
        if "linear_separability" in self.config.DIAG_SEPARABILITY_METRICS:
            try:
                linear_sep = self._compute_linear_separability(embeddings, labels)
                metrics.update(linear_sep)
            except Exception as e:
                print(f"      ⚠️  线性可分性计算失败: {e}")

        return metrics

    def _compute_inter_intra_ratio(self, embeddings, labels):
        """计算类间距离/类内距离比"""
        unique_labels = np.unique(labels)
        centroids = np.array([embeddings[labels == l].mean(axis=0) 
                             for l in unique_labels])

        # 类内距离（平均）
        intra_dists = []
        for l in unique_labels:
            class_samples = embeddings[labels == l]
            if len(class_samples) > 0:
                centroid = centroids[l]
                intra_dists.append(np.mean(cdist(class_samples, [centroid])))
        intra_dist = np.mean(intra_dists) if intra_dists else 0

        # 类间距离（平均）
        if len(unique_labels) > 1:
            inter_dist = np.mean(cdist(centroids, centroids)[np.triu_indices(len(unique_labels), k=1)])
        else:
            inter_dist = 0

        return {
            'inter_distance': float(inter_dist),
            'intra_distance': float(intra_dist),
            'inter_intra_ratio': float(inter_dist / (intra_dist + 1e-8))
        }

    def _compute_linear_separability(self, embeddings, labels):
        """计算线性可分性（用LR测试）"""
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        # 交叉验证
        cv_scores = cross_val_score(
            clf, embeddings, labels, 
            cv=self.config.DIAG_LINEAR_SEP_CV_FOLDS,
            scoring='accuracy',
            n_jobs=-1
        )

        return {
            'linear_sep_accuracy': float(cv_scores.mean()),
            'linear_sep_std': float(cv_scores.std())
        }

    def analyze_layer(self, embeddings, labels, metadata, layer_name, output_dir):
        """
        分析单个层的表征

        返回降维坐标CSV和可分性指标
        """
        output_dir = Path(output_dir)
        results = {}

        print(f"    📊 分析 {layer_name} 表征...")

        # 降维
        for method in self.config.DIAG_REDUCTION_METHODS:
            if method == 'pca':
                n_comp_list = self.config.DIAG_PCA_COMPONENTS
            elif method == 'umap':
                n_comp_list = self.config.DIAG_UMAP_COMPONENTS
            else:
                n_comp_list = [2]

            for n_comp in n_comp_list:
                reduced = self.reduce_dimensions(embeddings, labels, method, n_comp)

                if reduced is None:
                    continue

                # 保存降维坐标
                coord_cols = [f'{method}_{i+1}' for i in range(n_comp)]
                coord_df = pd.DataFrame(reduced, columns=coord_cols)
                df = pd.concat([metadata.reset_index(drop=True), coord_df], axis=1)

                csv_path = output_dir / f"{layer_name}_{method}_{n_comp}d.csv"
                df.to_csv(csv_path, index=False)
                print(f"      ✅ {method.upper()}-{n_comp}D: {csv_path.name}")

        # 可分性指标
        print(f"      🔍 计算可分性指标...")
        metrics = self.compute_separability_metrics(embeddings, labels)
        results['separability'] = metrics

        # 打印指标
        for key, value in metrics.items():
            if value is not None:
                print(f"        - {key}: {value:.4f}")

        return results
