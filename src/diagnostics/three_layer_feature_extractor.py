"""
三层表征提取器 - 严格按照要求 1
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class ThreeLayerFeatureExtractor:
    """
    提取三层表征：
    1. raw_stft: 原始 STFT 统计特征
    2. node_feat: CNN 输出的每通道压缩向量
    3. transformer: Transformer 输入/输出的 embedding
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # 存储
        self.features = {
            'raw_stft': [],
            'node_feat': [],
            'transformer': []
        }

        self.metadata = {
            'sample_id': [],
            'source_file': [],
            'class_label': [],
            'split': []
        }

        # 注册钩子
        self.hooks = {}
        self.feature_cache = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子提取 node_feat 和 transformer"""

        def make_hook(name):
            def hook(module, input, output):
                self.feature_cache[name] = output.detach()
            return hook

        # node_feat: 假设在 stft_compress 或 cnn_layers 之后
        if hasattr(self.model, 'stft_compress'):
            self.hooks['node_feat'] = self.model.stft_compress.register_forward_hook(
                make_hook('node_feat')
            )
            print("  ✅ 注册钩子: node_feat (stft_compress)")

        # transformer: 假设在 transformer 层输出
        if hasattr(self.model, 'transformer'):
            self.hooks['transformer'] = self.model.transformer.register_forward_hook(
                make_hook('transformer')
            )
            print("  ✅ 注册钩子: transformer")

    def extract_batch(self, stft_batch, adj_matrix, labels, batch_idx, filenames=None):
        """
        提取一个 batch 的三层表征
        
        Args:
            stft_batch: [B, C, F, T] 或 [B, C, 1, F, T] (在 GPU 上)
            adj_matrix: [B, N, N] (在 GPU 上)
            labels: [B]
            batch_idx: batch 索引
            filenames: 文件名列表（可选）
        """
        # ===== 修复：处理 5D tensor [B, C, 1, F, T] -> [B, C, F, T] =====
        stft_batch_cpu = stft_batch.cpu()
        
        # 如果是 5D，squeeze 掉中间维度
        if len(stft_batch_cpu.shape) == 5:
            stft_batch_cpu = stft_batch_cpu.squeeze(2)  # [B, C, 1, F, T] -> [B, C, F, T]
        
        B = stft_batch_cpu.size(0)
        
        # 1. raw_stft 统计特征（在 CPU 上计算）
        raw_stft_feat = self._extract_raw_stft_stats(stft_batch_cpu)
        
        # 2. 前向传播触发钩子（在 GPU 上，使用原始 tensor）
        self.model.eval()
        with torch.no_grad():
            _ = self.model(stft_batch, adj_matrix)
        
        # 3. 提取 node_feat 和 transformer
        node_feat = self._extract_node_feat()
        transformer_feat = self._extract_transformer_feat()
        
        # 4. 保存
        self.features['raw_stft'].append(raw_stft_feat)
        self.features['node_feat'].append(node_feat)
        self.features['transformer'].append(transformer_feat)
        
        # 5. 元信息
        for i in range(B):
            sample_id = batch_idx * self.config.BATCH_SIZE + i
            source_file = filenames[i] if filenames else f"sample_{sample_id}"
            
            self.metadata['sample_id'].append(sample_id)
            self.metadata['source_file'].append(source_file)
            self.metadata['class_label'].append(labels[i].item())
            self.metadata['split'].append('test')
        
        # 清空缓存
        self.feature_cache.clear()



    def _extract_raw_stft_stats(self, stft_batch):
        """
        提取 raw_stft 统计特征
        [B, C, F, T] -> [B, C*5] (mean, std, max, min, median per channel)
        
        Args:
            stft_batch: CPU tensor [B, C, F, T]
        """
        # ===== 关键修复：确保是 4D tensor =====
        if len(stft_batch.shape) != 4:
            raise ValueError(f"stft_batch 应该是 4D [B,C,F,T]，实际: {stft_batch.shape}")
        
        B, C, F, T = stft_batch.shape
        stats = []
        
        for i in range(B):
            sample_stats = []
            for c in range(C):
                channel_data = stft_batch[i, c, :, :].flatten()
                sample_stats.extend([
                    channel_data.mean().item(),
                    channel_data.std().item(),
                    channel_data.max().item(),
                    channel_data.min().item(),
                    channel_data.median().item()
                ])
            stats.append(sample_stats)
        
        return np.array(stats)

    def _extract_node_feat(self):
        """提取 node_feat (CNN 输出)"""
        if 'node_feat' in self.feature_cache:
            feat = self.feature_cache['node_feat']
            # [B, C, ...] -> [B, C*D]
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            return feat
        else:
            return None

    def _extract_transformer_feat(self):
        """提取 transformer embedding"""
        if 'transformer' in self.feature_cache:
            feat = self.feature_cache['transformer']
            # [B, Seq, D] -> [B, D] (取 mean pool)
            if len(feat.shape) == 3:
                feat = feat.mean(dim=1)
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            return feat
        else:
            return None

    def save_features(self, output_dir):
        """
        保存三层表征到 CSV/NPY
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        export_format = getattr(self.config, 'DIAG_EXPORT_FORMAT', 'csv')

        for layer_name, feat_list in self.features.items():
            if not feat_list or feat_list[0] is None:
                print(f"  ⚠️  {layer_name} 为空，跳过")
                continue

            # 合并所有 batch
            feat_np = np.vstack([f for f in feat_list if f is not None])

            # 构建 DataFrame
            df = pd.DataFrame(feat_np, columns=[f"feat_{i}" for i in range(feat_np.shape[1])])

            # 添加元信息
            for col in ['sample_id', 'source_file', 'class_label', 'split']:
                df.insert(len(df.columns), col, self.metadata[col])

            # 保存 CSV
            if export_format in ['csv', 'both']:
                csv_path = output_dir / f"{dataset_name}_{layer_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"  ✅ 保存: {csv_path.name} ({feat_np.shape[0]} 样本, {feat_np.shape[1]} 维)")

            # 保存 NPY
            if export_format in ['npy', 'both']:
                npy_path = output_dir / f"{dataset_name}_{layer_name}.npy"
                np.save(npy_path, feat_np)
                print(f"  ✅ 保存: {npy_path.name}")

    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
