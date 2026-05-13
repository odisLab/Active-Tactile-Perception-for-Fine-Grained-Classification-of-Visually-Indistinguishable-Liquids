"""
通道消融分析模块
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class ChannelAblation:
    """
    通道消融分析器

    功能：
    1. Drop-one: 逐个通道置零，评估性能下降
    2. Keep-one: 只保留一个通道，评估单通道贡献
    3. 通道重要性排名
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.valid_channels = config.VALID_PHYSICAL_CHANNELS if hasattr(config, 'VALID_PHYSICAL_CHANNELS') else list(range(config.VALID_CHANNELS_NUM))

    def ablate_channel(self, data, channel_idx):
        """
        置零指定通道

        Args:
            data: (B, C, F, T) STFT数据
            channel_idx: 要置零的通道索引

        Returns:
            ablated_data: 通道置零后的数据
        """
        ablated = data.clone()
        ablated[:, channel_idx, :, :] = 0
        return ablated

    def keep_only_channel(self, data, channel_idx):
        """只保留指定通道"""
        kept = torch.zeros_like(data)
        kept[:, channel_idx, :, :] = data[:, channel_idx, :, :]
        return kept

    def evaluate_ablation(self, data_loader, ablation_func, channel_idx):
        """
        评估通道消融后的性能

        Args:
            data_loader: 数据加载器
            ablation_func: 消融函数
            channel_idx: 通道索引

        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 应用通道消融
                ablated_data = ablation_func(batch_data, channel_idx)

                # 前向传播
                outputs = self.model(ablated_data)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

        accuracy = correct / total
        return {'accuracy': accuracy}

    def drop_one_analysis(self, data_loader, baseline_metrics):
        """
        Drop-one分析：逐个通道置零

        Returns:
            results: DataFrame包含每个通道的重要性
        """
        print("\n    🔍 通道消融分析 (Drop-one)...")

        results = []

        for ch_idx in tqdm(range(self.config.VALID_CHANNELS_NUM), desc="      Drop-one"):
            # 获取物理通道编号
            physical_ch = self.valid_channels[ch_idx] if ch_idx < len(self.valid_channels) else ch_idx

            # 评估该通道置零后的性能
            metrics = self.evaluate_ablation(
                data_loader,
                self.ablate_channel,
                ch_idx
            )

            # 计算性能下降
            delta_acc = baseline_metrics['accuracy'] - metrics['accuracy']

            results.append({
                'channel_idx': ch_idx,
                'physical_channel': physical_ch,
                'accuracy': metrics['accuracy'],
                'delta_accuracy': delta_acc,
                'importance_score': delta_acc
            })

        df = pd.DataFrame(results)
        df = df.sort_values('importance_score', ascending=False)

        return df

    def keep_one_analysis(self, data_loader, baseline_metrics):
        """
        Keep-one分析：只保留一个通道

        Returns:
            results: DataFrame包含每个通道的单独贡献
        """
        print("\n    🔍 通道单独贡献分析 (Keep-one)...")

        results = []

        for ch_idx in tqdm(range(self.config.VALID_CHANNELS_NUM), desc="      Keep-one"):
            physical_ch = self.valid_channels[ch_idx] if ch_idx < len(self.valid_channels) else ch_idx

            # 评估只保留该通道时的性能
            metrics = self.evaluate_ablation(
                data_loader,
                self.keep_only_channel,
                ch_idx
            )

            results.append({
                'channel_idx': ch_idx,
                'physical_channel': physical_ch,
                'accuracy': metrics['accuracy'],
                'contribution_score': metrics['accuracy']
            })

        df = pd.DataFrame(results)
        df = df.sort_values('contribution_score', ascending=False)

        return df

    def run_full_ablation(self, data_loader, baseline_metrics, output_dir):
        """
        运行完整的通道消融分析

        Args:
            data_loader: 测试数据加载器
            baseline_metrics: 基线性能指标
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Drop-one分析
        if self.config.DIAG_ABLATION_MODE in ["drop_one", "both"]:
            drop_one_results = self.drop_one_analysis(data_loader, baseline_metrics)
            results['drop_one'] = drop_one_results

            path = output_dir / f"{self.config.DIAG_CSV_PREFIX}_channel_drop_one.csv"
            drop_one_results.to_csv(path, index=False)
            print(f"    ✅ Drop-one结果: {path.name}")

            print("\n    📊 Top-5 最重要通道 (Drop-one):")
            for _, row in drop_one_results.head(5).iterrows():
                print(f"      - Ch{row['physical_channel']}: Δacc = {row['delta_accuracy']:.4f}")

        # Keep-one分析
        if self.config.DIAG_ABLATION_MODE in ["keep_one", "both"]:
            keep_one_results = self.keep_one_analysis(data_loader, baseline_metrics)
            results['keep_one'] = keep_one_results

            path = output_dir / f"{self.config.DIAG_CSV_PREFIX}_channel_keep_one.csv"
            keep_one_results.to_csv(path, index=False)
            print(f"    ✅ Keep-one结果: {path.name}")

            print("\n    📊 Top-5 单通道贡献 (Keep-one):")
            for _, row in keep_one_results.head(5).iterrows():
                print(f"      - Ch{row['physical_channel']}: acc = {row['accuracy']:.4f}")

        return results
