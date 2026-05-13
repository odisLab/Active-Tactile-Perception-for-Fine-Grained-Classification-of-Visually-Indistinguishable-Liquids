"""
多层特征提取模块
从模型的不同层抽取表征，保存为CSV/NPY
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

class FeatureExtractor:
    """
    多层特征提取器

    提取模型中间层表征：
    - STFT统计特征（均值、标准差、峰度、偏度等）
    - CNN节点特征（GNN输入）
    - Transformer序列特征
    - Transformer输出特征
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.features = {layer: [] for layer in config.DIAG_EXPORT_LAYERS}
        self.metadata = []

        # 存储中间层输出
        self.layer_outputs = {}

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册前向传播钩子以捕获中间层输出"""
        self.hooks = []

        # 注册CNN输出钩子（如果需要）
        if "node_feat" in self.config.DIAG_EXPORT_LAYERS:
            def hook_cnn(module, input, output):
                self.layer_outputs['node_feat'] = output.detach()

            # 根据模型结构调整钩子位置
            if hasattr(self.model, 'cnn_branch'):
                self.hooks.append(
                    self.model.cnn_branch.register_forward_hook(hook_cnn)
                )

        # 注册Transformer输入钩子
        if "seq_feat" in self.config.DIAG_EXPORT_LAYERS:
            def hook_seq(module, input, output):
                self.layer_outputs['seq_feat'] = input[0].detach() if isinstance(input, tuple) else input.detach()

            if hasattr(self.model, 'transformer'):
                self.hooks.append(
                    self.model.transformer.register_forward_hook(hook_seq)
                )

        # 注册Transformer输出钩子
        if "trans_out" in self.config.DIAG_EXPORT_LAYERS:
            def hook_trans_out(module, input, output):
                self.layer_outputs['trans_out'] = output.detach()

            if hasattr(self.model, 'transformer'):
                self.hooks.append(
                    self.model.transformer.register_forward_hook(hook_trans_out)
                )

    def extract_stft_stats(self, stft_data):
        """
        提取STFT统计特征

        Args:
            stft_data: (B, C, F, T) STFT幅度谱

        Returns:
            dict: 统计特征字典
        """
        B, C, F, T = stft_data.shape
        stats = {}

        # 按通道计算统计量
        for c in range(C):
            channel_data = stft_data[:, c, :, :].reshape(B, -1)
            stats[f'ch{c}_mean'] = channel_data.mean(dim=1).cpu().numpy()
            stats[f'ch{c}_std'] = channel_data.std(dim=1).cpu().numpy()
            stats[f'ch{c}_max'] = channel_data.max(dim=1)[0].cpu().numpy()
            stats[f'ch{c}_min'] = channel_data.min(dim=1)[0].cpu().numpy()

            # 频域统计 - 频率质心
            freq_mean = stft_data[:, c, :, :].mean(dim=2)  # (B, F)
            freq_range = torch.arange(F, device=self.device, dtype=torch.float32)
            stats[f'ch{c}_freq_centroid'] = (
                (freq_mean * freq_range).sum(dim=1) / 
                (freq_mean.sum(dim=1) + 1e-8)
            ).cpu().numpy()

            # 时域统计 - 时间质心
            time_mean = stft_data[:, c, :, :].mean(dim=1)  # (B, T)
            time_range = torch.arange(T, device=self.device, dtype=torch.float32)
            stats[f'ch{c}_time_centroid'] = (
                (time_mean * time_range).sum(dim=1) / 
                (time_mean.sum(dim=1) + 1e-8)
            ).cpu().numpy()

        return stats

    def extract_batch(self, batch, sample_ids, labels, filenames):
        """
        提取一个batch的所有层特征

        Args:
            batch: 输入数据 (B, C, F, T)
            sample_ids: 样本ID列表
            labels: 真实标签
            filenames: 源文件名
        """
        self.model.eval()
        with torch.no_grad():
            # 清空之前的输出
            self.layer_outputs = {}

            # 前向传播（触发钩子）
            outputs = self.model(batch)

            # STFT统计特征
            if "stft_stats" in self.config.DIAG_EXPORT_LAYERS:
                stft_stats = self.extract_stft_stats(batch)
                self.features["stft_stats"].append(stft_stats)

            # 其他层特征通过钩子获取
            for layer_name in ["node_feat", "seq_feat", "trans_out"]:
                if layer_name in self.config.DIAG_EXPORT_LAYERS:
                    if layer_name in self.layer_outputs:
                        feat = self.layer_outputs[layer_name]
                        # 如果是多维张量，展平或池化
                        if feat.dim() > 2:
                            feat = feat.reshape(feat.size(0), -1)  # (B, D)
                        self.features[layer_name].append(feat.cpu())

            # 记录元信息
            for i, (sid, label, fname) in enumerate(zip(sample_ids, labels, filenames)):
                self.metadata.append({
                    'sample_id': int(sid) if hasattr(sid, 'item') else sid,
                    'source_file': fname,
                    'class_label': int(label.item()) if hasattr(label, 'item') else int(label),
                    'split': 'test'
                })

    def save_features(self, output_dir):
        """
        保存所有提取的特征

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建元信息DataFrame
        metadata_df = pd.DataFrame(self.metadata)

        # 保存每一层特征
        for layer_name, features in self.features.items():
            if not features:
                continue

            # 合并所有batch
            if isinstance(features[0], dict):
                # 统计特征（字典格式）
                combined = {k: np.concatenate([f[k] for f in features]) 
                           for k in features[0].keys()}
                feat_df = pd.DataFrame(combined)
            else:
                # embedding特征（张量格式）
                combined = np.concatenate([f.numpy() for f in features])
                feat_df = pd.DataFrame(combined, 
                                      columns=[f'{layer_name}_dim{i}' 
                                              for i in range(combined.shape[1])])

            # 合并元信息和特征
            df = pd.concat([metadata_df.reset_index(drop=True), 
                           feat_df.reset_index(drop=True)], axis=1)

            # 保存CSV
            if self.config.DIAG_EXPORT_FORMAT in ["csv", "both"]:
                csv_path = output_dir / f"{self.config.DIAG_CSV_PREFIX}_{layer_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"    ✅ {layer_name} 特征: {csv_path.name}")

            # 保存NPY（只保存特征，不含元信息）
            if self.config.DIAG_EXPORT_FORMAT in ["npy", "both"]:
                if isinstance(features[0], dict):
                    combined_array = np.column_stack([combined[k] for k in sorted(combined.keys())])
                else:
                    combined_array = combined

                npy_path = output_dir / f"{self.config.DIAG_CSV_PREFIX}_{layer_name}.npy"
                np.save(npy_path, combined_array)
                print(f"    ✅ {layer_name} NPY: {npy_path.name}")

    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
