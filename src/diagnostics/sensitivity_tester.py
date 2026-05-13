"""
敏感性测试模块 - 功能要求 6, 7, 8
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score

class SensitivityTester:
    """敏感性测试：频段/时间/通道消融"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def test_frequency_sensitivity(self, test_loader, output_dir):
        """
        功能要求 6: 频段敏感性测试
        """
        print(f"\n  🔍 频段敏感性测试")

        # 基线
        baseline_metrics = self._evaluate(test_loader)
        print(f"     - 基线准确率: {baseline_metrics['accuracy']:.4f}")

        # 逐个频段遮挡
        freq_bands = getattr(self.config, 'DIAG_FREQ_MASK_BANDS', [])

        results = []
        for f_low, f_high in freq_bands:
            metrics = self._evaluate(test_loader, mask_freq=(f_low, f_high))

            delta_acc = baseline_metrics['accuracy'] - metrics['accuracy']
            delta_f1 = baseline_metrics['f1'] - metrics['f1']

            results.append({
                'freq_band': f"{f_low}-{f_high} Hz",
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'delta_accuracy': delta_acc,
                'delta_f1': delta_f1
            })

            print(f"     - {f_low}-{f_high} Hz: Δacc={delta_acc:+.4f}, ΔF1={delta_f1:+.4f}")

        # 保存
        df = pd.DataFrame(results)
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        save_path = Path(output_dir) / f"{dataset_name}_freq_sensitivity.csv"
        df.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")

        return df

    def test_time_sensitivity(self, test_loader, output_dir):
        """
        功能要求 6: 时间窗敏感性测试
        """
        print(f"\n  🔍 时间窗敏感性测试")

        baseline_metrics = self._evaluate(test_loader)

        n_segments = getattr(self.config, 'DIAG_TIME_MASK_WINDOWS', 10)

        # 假设时间轴长度
        T = self.config.STFT_NPERSEG if hasattr(self.config, 'STFT_NPERSEG') else 193
        segment_len = T // n_segments

        results = []
        for i in range(n_segments):
            t0 = i * segment_len
            t1 = min((i+1) * segment_len, T)

            metrics = self._evaluate(test_loader, mask_time=(t0, t1))

            delta_acc = baseline_metrics['accuracy'] - metrics['accuracy']
            delta_f1 = baseline_metrics['f1'] - metrics['f1']

            results.append({
                'time_window': f"[{t0}, {t1})",
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'delta_accuracy': delta_acc,
                'delta_f1': delta_f1
            })

            print(f"     - [{t0}, {t1}): Δacc={delta_acc:+.4f}")

        # 保存
        df = pd.DataFrame(results)
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        save_path = Path(output_dir) / f"{dataset_name}_time_sensitivity.csv"
        df.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")

        return df

    def auto_scan_sensitive_regions(self, test_loader, output_dir):
        """
        功能要求 7: 自动扫描敏感频段/时间段
        """
        print(f"\n  🔍 自动扫描敏感区域")

        baseline_metrics = self._evaluate(test_loader)

        freq_segments = getattr(self.config, 'DIAG_AUTO_SCAN_FREQ_SEGMENTS', 10)
        time_segments = getattr(self.config, 'DIAG_AUTO_SCAN_TIME_SEGMENTS', 10)
        topk = getattr(self.config, 'DIAG_AUTO_SCAN_TOPK', 5)

        # 频率扫描
        F = self.config.STFT_NFFT // 2 if hasattr(self.config, 'STFT_NFFT') else 101
        freq_len = F // freq_segments

        freq_results = []
        for i in range(freq_segments):
            f_low = i * freq_len
            f_high = min((i+1) * freq_len, F)

            metrics = self._evaluate(test_loader, mask_freq=(f_low, f_high))
            delta = baseline_metrics['accuracy'] - metrics['accuracy']

            freq_results.append({
                'region': f"Freq [{f_low}, {f_high})",
                'delta_accuracy': delta
            })

        # 时间扫描
        T = self.config.STFT_NPERSEG if hasattr(self.config, 'STFT_NPERSEG') else 193
        time_len = T // time_segments

        time_results = []
        for i in range(time_segments):
            t0 = i * time_len
            t1 = min((i+1) * time_len, T)

            metrics = self._evaluate(test_loader, mask_time=(t0, t1))
            delta = baseline_metrics['accuracy'] - metrics['accuracy']

            time_results.append({
                'region': f"Time [{t0}, {t1})",
                'delta_accuracy': delta
            })

        # Top-K
        all_results = freq_results + time_results
        df = pd.DataFrame(all_results).sort_values('delta_accuracy', ascending=False)

        print(f"\n     Top-{topk} 最敏感区域:")
        for idx, row in df.head(topk).iterrows():
            print(f"       {row['region']}: Δacc={row['delta_accuracy']:+.4f}")

        # 保存
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        save_path = Path(output_dir) / f"{dataset_name}_auto_scan_topk.csv"
        df.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")

        return df

    def test_channel_ablation(self, test_loader, output_dir):
        """
        功能要求 8: 通道消融测试
        """
        print(f"\n  🔍 通道消融测试")

        baseline_metrics = self._evaluate(test_loader)

        num_channels = self.config.NUM_CHANNELS
        ablation_mode = getattr(self.config, 'DIAG_ABLATION_MODE', 'zero')

        results = []
        for ch_idx in range(num_channels):
            metrics = self._evaluate(test_loader, ablate_channel=ch_idx)

            delta_acc = baseline_metrics['accuracy'] - metrics['accuracy']

            # 对齐物理通道
            # 安全获取物理通道号
            if hasattr(self.config, 'VALID_PHYSICAL_CHANNELS'):
                if ch_idx < len(self.config.VALID_PHYSICAL_CHANNELS):
                    physical_ch = self.config.VALID_PHYSICAL_CHANNELS[ch_idx]
                else:
                    physical_ch = ch_idx  # 超出范围，使用索引本身
            else:
                physical_ch = ch_idx

            results.append({
                'channel_idx': ch_idx,
                'physical_channel': physical_ch,
                'accuracy': metrics['accuracy'],
                'delta_accuracy': delta_acc,
                'importance_rank': 0  # 后续排序
            })

            print(f"     - Channel {ch_idx} (物理: {physical_ch}): Δacc={delta_acc:+.4f}")

        # 排序
        df = pd.DataFrame(results).sort_values('delta_accuracy', ascending=False).reset_index(drop=True)
        df['importance_rank'] = range(1, len(df)+1)

        # 保存
        dataset_name = self.config.DATASET_NAME if hasattr(self.config, 'DATASET_NAME') else 'default'
        save_path = Path(output_dir) / f"{dataset_name}_channel_ablation.csv"
        df.to_csv(save_path, index=False)
        print(f"     ✅ 保存: {save_path.name}")

        return df

    def _evaluate(self, data_loader, mask_freq=None, mask_time=None, ablate_channel=None):
        """
        评估模型（可选扰动）
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for stft_batch, adj_matrix, labels in data_loader:
                stft_batch = stft_batch.to(self.device)
                adj_matrix = adj_matrix.to(self.device)

                # 应用扰动
                if mask_freq is not None:
                    stft_batch = self._mask_frequency_band(stft_batch, *mask_freq)

                if mask_time is not None:
                    stft_batch = self._mask_time_window(stft_batch, *mask_time)

                if ablate_channel is not None:
                    stft_batch = self._ablate_channel(stft_batch, ablate_channel)

                outputs = self.model(stft_batch, adj_matrix)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_per_class': recall_score(all_labels, all_preds, average=None, zero_division=0)
        }

    def _mask_frequency_band(self, stft_batch, f_low, f_high):
        """遮挡频段 [f_low, f_high)"""
        stft_batch = stft_batch.clone()
        stft_batch[:, :, f_low:f_high, :] = 0
        return stft_batch

    def _mask_time_window(self, stft_batch, t0, t1):
        """遮挡时间窗 [t0, t1)"""
        stft_batch = stft_batch.clone()
        stft_batch[:, :, :, t0:t1] = 0
        return stft_batch

    def _ablate_channel(self, stft_batch, channel_idx):
        """消融通道"""
        stft_batch = stft_batch.clone()
        stft_batch[:, channel_idx, :, :] = 0
        return stft_batch
