"""
双分支特征提取器
支持提取 Phase E、Phase H 和融合后的特征
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path


class DualBranchFeatureExtractor:
    """双分支三层特征提取器"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # 存储特征
        self.features = {
            'node_feat_e': [],      # Phase E GNN 输出
            'node_feat_h': [],      # Phase H GNN 输出
            'node_feat_fused': [],  # 融合后特征
            'transformer': []       # Transformer 输出
        }
        
        self.labels = []
        self.metadata = []
    
    def extract_batch(self, X_e, X_h, adj_matrix, labels, metadata, batch_idx):
        """
        提取一个 batch 的双分支特征
        
        Args:
            X_e: Phase E STFT (batch, C, F, T_e)
            X_h: Phase H STFT (batch, C, F, T_h)
            adj_matrix: 邻接矩阵
            labels: 标签
            meta 元信息（包含 phase_e_path, phase_h_path）
            batch_idx: batch 索引
        """
        # 钩子函数提取中间层特征
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 注册钩子（需要根据实际模型结构调整）
        hooks = []
        if hasattr(self.model, 'gnn_e'):
            hooks.append(self.model.gnn_e.register_forward_hook(get_activation('node_feat_e')))
        if hasattr(self.model, 'gnn_h'):
            hooks.append(self.model.gnn_h.register_forward_hook(get_activation('node_feat_h')))
        if hasattr(self.model, 'fusion'):
            hooks.append(self.model.fusion.register_forward_hook(get_activation('node_feat_fused')))
        if hasattr(self.model, 'transformer'):
            hooks.append(self.model.transformer.register_forward_hook(get_activation('transformer')))
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(X_e, X_h, adj_matrix)
        
        # 提取特征
        for key in self.features.keys():
            if key in activations:
                feat = activations[key].cpu().numpy()
                # 展平为 (batch, -1)
                feat_flat = feat.reshape(feat.shape[0], -1)
                self.features[key].append(feat_flat)
        
        # 保存标签和元信息
        self.labels.append(labels.cpu().numpy())
        self.metadata.extend([
            {
                'sample_id': batch_idx * len(labels) + i,
                'phase_e_path': metadata['phase_e_path'][i] if 'phase_e_path' in metadata else '',
                'phase_h_path': metadata['phase_h_path'][i] if 'phase_h_path' in metadata else '',
                'class_label': labels[i].item(),
                'split': 'test'
            }
            for i in range(len(labels))
        ])
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
    
    def save_features(self, output_dir):
        """保存所有特征到 CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        
        for layer_name, feats in self.features.items():
            if not feats:
                continue
            
            # 拼接所有 batch
            feats_array = np.vstack(feats)
            
            # 创建 DataFrame
            df_feat = pd.DataFrame(
                feats_array,
                columns=[f"{layer_name}_dim{i}" for i in range(feats_array.shape[1])]
            )
            
            df_meta = pd.DataFrame(self.metadata)
            df = pd.concat([df_meta, df_feat], axis=1)
            
            # 保存
            output_path = output_dir / f"{dataset_name}_{layer_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"  💾 {layer_name}: {output_path} ({feats_array.shape})")
    
    def cleanup(self):
        """清理缓存"""
        for key in self.features.keys():
            self.features[key].clear()
        self.labels.clear()
        self.metadata.clear()
