"""
降维模块 - 功能要求 3: PCA/UMAP 导出坐标（不画图）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

class DimensionalityReducer:
    """降维并导出坐标到 CSV"""

    def __init__(self, config):
        self.config = config
        self.methods = getattr(config, 'DIAG_DIM_REDUCTION_METHODS', ['pca_2d'])

    def reduce_and_save(self, embeddings, metadata, layer_name, output_dir):
        """
        对 embeddings 做降维并保存坐标

        Args:
            embeddings: [N, D] numpy array
            metadata: DataFrame with ['sample_id', 'source_file', 'class_label', 'split']
            layer_name: 层名称
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'

        print(f"\n  🔍 降维: {layer_name} ({embeddings.shape[0]} 样本, {embeddings.shape[1]} 维)")

        results = metadata.copy()

        # PCA 2D
        if 'pca_2d' in self.methods:
            pca_2d = PCA(n_components=2).fit_transform(embeddings)
            results['pca_x'] = pca_2d[:, 0]
            results['pca_y'] = pca_2d[:, 1]
            print(f"     ✅ PCA 2D 完成")

        # PCA 3D
        if 'pca_3d' in self.methods:
            pca_3d = PCA(n_components=3).fit_transform(embeddings)
            results['pca_x_3d'] = pca_3d[:, 0]
            results['pca_y_3d'] = pca_3d[:, 1]
            results['pca_z_3d'] = pca_3d[:, 2]
            print(f"     ✅ PCA 3D 完成")

        # UMAP 2D
        if 'umap_2d' in self.methods:
            try:
                import umap
                n_neighbors = getattr(self.config, 'DIAG_UMAP_N_NEIGHBORS', 15)
                min_dist = getattr(self.config, 'DIAG_UMAP_MIN_DIST', 0.1)

                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                umap_2d = reducer.fit_transform(embeddings)
                results['umap_x'] = umap_2d[:, 0]
                results['umap_y'] = umap_2d[:, 1]
                print(f"     ✅ UMAP 2D 完成")
            except ImportError:
                print(f"     ⚠️  未安装 umap-learn，跳过 UMAP")

        # 保存
        save_path = output_dir / f"{dataset_name}_{layer_name}_dimreduction.csv"
        results.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")

        return results
