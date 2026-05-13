"""
双分支敏感性测试器
适配双分支数据格式，支持分别测试 Phase E 和 Phase H
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .sensitivity_tester import SensitivityTester
class DualBranchSensitivityTester(SensitivityTester):
    """
    双分支敏感性测试器（继承自 SensitivityTester）
    新增功能：
    - 分别测试 Phase E 和 Phase H 的频段/时间敏感性
    - 支持双分支数据格式 (X_e, X_h, adj, labels, metadata)
    - 支持随机多通道消融（config 控制）
    """
    def __init__(self, model, config, device):
        super(DualBranchSensitivityTester, self).__init__(model, config, device)
        print("  ✅ 双分支敏感性测试器已初始化")
    # ==================== 核心评估方法（双分支） ====================
    def _evaluate(self, data_loader, return_probs=False):
        """
        评估模型（适配双分支数据格式）
        Returns:
            dict: {'accuracy': float, 'predictions': np.array, 'labels': np.array, ('probabilities': np.array optional)}
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in data_loader:
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                model_output = self.model(X_e, X_h, adj_matrix)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
                if return_probs:
                    all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = float((all_preds == all_labels).mean()) if all_labels.size > 0 else 0.0
        result = {'accuracy': accuracy, 'predictions': all_preds, 'labels': all_labels}
        if return_probs:
            result['probabilities'] = np.array(all_probs)
        return result
    # ==================== 频段敏感性 ====================
    def test_frequency_sensitivity(self, test_loader, output_dir):
        """
        频段敏感性测试（双分支）
        1) 基线
        2) Mask Phase E
        3) Mask Phase H
        4) Mask 两个 Phase
        """
        print("\n  🔬 频段敏感性测试（双分支模式）")
        baseline = self._evaluate(test_loader, return_probs=False)
        baseline_acc = baseline['accuracy']
        print("     基线准确率: {:.4f}".format(baseline_acc))
        freq_bands = getattr(self.config, 'DIAG_FREQ_MASK_BANDS', [(0, 50), (50, 150), (150, 300), (300,500),(500, 1000), (1000, 2000)])
        results = []
        for freq_min, freq_max in tqdm(freq_bands, desc="     测试频段"):
            acc_e_masked = self._test_freq_mask_phase(test_loader, freq_min, freq_max, mask_phase='e', return_probs=False)
            acc_h_masked = self._test_freq_mask_phase(test_loader, freq_min, freq_max, mask_phase='h', return_probs=False)
            acc_both_masked = self._test_freq_mask_phase(test_loader, freq_min, freq_max, mask_phase='both', return_probs=False)
            results.append({
                'freq_min': freq_min,
                'freq_max': freq_max,
                'baseline_acc': baseline_acc,
                'phase_e_masked_acc': acc_e_masked,
                'phase_h_masked_acc': acc_h_masked,
                'both_masked_acc': acc_both_masked,
                'phase_e_drop': baseline_acc - acc_e_masked,
                'phase_h_drop': baseline_acc - acc_h_masked,
                'both_drop': baseline_acc - acc_both_masked,
            })
        df = pd.DataFrame(results)
        output_path = output_dir / "frequency_sensitivity_dual_branch.csv"
        df.to_csv(output_path, index=False)
        print("     ✅ 保存: {}".format(output_path.name))
        return results
    def _test_freq_mask_phase(self, test_loader, freq_min, freq_max, mask_phase='e', return_probs=False):
        """
        测试 Mask 特定 Phase 的某个频段后的性能
        mask_phase: 'e' / 'h' / 'both'
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        fmax = float(getattr(self.config, 'DUAL_STFT_FMAX', 1000.0))
        freq_bins = int(getattr(self.config, 'DUAL_FREQ_BINS', 103))
        freq_bin_min = int(float(freq_min) / fmax * freq_bins)
        freq_bin_max = int(float(freq_max) / fmax * freq_bins)
        freq_bin_min = max(0, min(freq_bins, freq_bin_min))
        freq_bin_max = max(0, min(freq_bins, freq_bin_max))
        if freq_bin_max <= freq_bin_min:
            freq_bin_max = min(freq_bins, freq_bin_min + 1)
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in test_loader:
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                labels_t = labels.to(self.device)
                X_e_mod = X_e
                X_h_mod = X_h
                if mask_phase in ['e', 'both']:
                    X_e_mod = X_e.clone()
                    # 形状一般为 (B, C, 1, F, T) 或 (B, C, F, T)，兼容两种
                    if X_e_mod.dim() == 5:
                        X_e_mod[:, :, :, freq_bin_min:freq_bin_max, :] = 0
                    else:
                        X_e_mod[:, :, freq_bin_min:freq_bin_max, :] = 0
                if mask_phase in ['h', 'both']:
                    X_h_mod = X_h.clone()
                    if X_h_mod.dim() == 5:
                        X_h_mod[:, :, :, freq_bin_min:freq_bin_max, :] = 0
                    else:
                        X_h_mod[:, :, freq_bin_min:freq_bin_max, :] = 0
                model_output = self.model(X_e_mod, X_h_mod, adj_matrix)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels_t.detach().cpu().numpy().tolist())
                if return_probs:
                    all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = float((all_preds == all_labels).mean()) if all_labels.size > 0 else 0.0
        if return_probs:
            return acc, np.array(all_probs), all_preds, all_labels
        return acc
    # ==================== 时间窗敏感性 ====================
    def test_time_sensitivity(self, test_loader, output_dir):
        """
        时间窗敏感性测试（双分支）
        1) Mask Phase E 的不同时间窗
        2) Mask Phase H 的不同时间窗
        """
        print("\n  🔬 时间窗敏感性测试（双分支模式）")
        baseline = self._evaluate(test_loader, return_probs=False)
        baseline_acc = baseline['accuracy']
        print("     基线准确率: {:.4f}".format(baseline_acc))
        num_windows = int(getattr(self.config, 'DIAG_TIME_MASK_WINDOWS', 10))
        results = []
        time_bins_e = int(getattr(self.config, 'E_TARGET_TIME_BINS', 39))
        window_size_e = max(1, time_bins_e // max(1, num_windows))
        for i in tqdm(range(num_windows), desc="     Phase E 时间窗"):
            start = i * window_size_e
            end = min((i + 1) * window_size_e, time_bins_e)
            acc = self._test_time_mask_phase(test_loader, start, end, mask_phase='e', return_probs=False)
            results.append({'phase': 'E', 'window_idx': i, 'time_start': start, 'time_end': end, 'baseline_acc': baseline_acc, 'masked_acc': acc, 'acc_drop': baseline_acc - acc})
        time_bins_h = int(getattr(self.config, 'H_TARGET_TIME_BINS', 78))
        window_size_h = max(1, time_bins_h // max(1, num_windows))
        for i in tqdm(range(num_windows), desc="     Phase H 时间窗"):
            start = i * window_size_h
            end = min((i + 1) * window_size_h, time_bins_h)
            acc = self._test_time_mask_phase(test_loader, start, end, mask_phase='h', return_probs=False)
            results.append({'phase': 'H', 'window_idx': i, 'time_start': start, 'time_end': end, 'baseline_acc': baseline_acc, 'masked_acc': acc, 'acc_drop': baseline_acc - acc})
        df = pd.DataFrame(results)
        output_path = output_dir / "time_sensitivity_dual_branch.csv"
        df.to_csv(output_path, index=False)
        print("     ✅ 保存: {}".format(output_path.name))
        return results
    def _test_time_mask_phase(self, data_loader, time_start, time_end, mask_phase='e', return_probs=False):
        """
        测试 Mask 特定 Phase 的某个时间窗后的性能
        mask_phase: 'e' / 'h'
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in data_loader:
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                labels_t = labels.to(self.device)
                X_e_mod = X_e
                X_h_mod = X_h
                if mask_phase == 'e':
                    X_e_mod = X_e.clone()
                    if X_e_mod.dim() == 5:
                        X_e_mod[:, :, :, :, time_start:time_end] = 0
                    else:
                        X_e_mod[:, :, :, time_start:time_end] = 0
                elif mask_phase == 'h':
                    X_h_mod = X_h.clone()
                    if X_h_mod.dim() == 5:
                        X_h_mod[:, :, :, :, time_start:time_end] = 0
                    else:
                        X_h_mod[:, :, :, time_start:time_end] = 0
                model_output = self.model(X_e_mod, X_h_mod, adj_matrix)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels_t.detach().cpu().numpy().tolist())
                if return_probs:
                    all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = float((all_preds == all_labels).mean()) if all_labels.size > 0 else 0.0
        if return_probs:
            return acc, np.array(all_probs), all_preds, all_labels
        return acc
    # ==================== 单通道消融 ====================
    def test_channel_ablation(self, test_loader, output_dir):
        """
        通道消融测试（双分支模式）
        同时对 Phase E 与 Phase H 的相同通道进行消融
        """
        print("\n  🔬 通道消融测试（双分支模式）")
        baseline = self._evaluate(test_loader, return_probs=False)
        baseline_acc = baseline['accuracy']
        print("     基线准确率: {:.4f}".format(baseline_acc))
        num_channels = int(getattr(self.config, 'NUM_CHANNELS', 12))
        ablation_mode = getattr(self.config, 'DIAG_ABLATION_MODE', 'zero')
        results = []
        for ch_idx in tqdm(range(num_channels), desc="     消融通道"):
            acc = self._test_channel_ablation_single(test_loader, ch_idx, mode=ablation_mode, return_probs=False)
            results.append({'channel_idx': ch_idx, 'ablation_mode': ablation_mode, 'baseline_acc': baseline_acc, 'ablated_acc': acc, 'acc_drop': baseline_acc - acc})
        df = pd.DataFrame(results)
        output_path = output_dir / "channel_ablation_dual_branch.csv"
        df.to_csv(output_path, index=False)
        print("     ✅ 保存: {}".format(output_path.name))
        return results
    def _test_channel_ablation_single(self, data_loader, channel_idx, mode='zero', return_probs=False):
        """
        测试消融单个通道后的性能（同时消融 Phase E 和 Phase H）
        mode: 目前仅实现 'zero'，其他模式退化为 zero
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in data_loader:
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                labels_t = labels.to(self.device)
                X_e_mod = X_e.clone()
                X_h_mod = X_h.clone()
                # 形状一般为 (B, C, 1, F, T) 或 (B, C, F, T)
                if mode == 'zero':
                    if X_e_mod.dim() == 5:
                        X_e_mod[:, channel_idx, :, :, :] = 0
                    else:
                        X_e_mod[:, channel_idx, :, :] = 0
                    if X_h_mod.dim() == 5:
                        X_h_mod[:, channel_idx, :, :, :] = 0
                    else:
                        X_h_mod[:, channel_idx, :, :] = 0
                else:
                    if X_e_mod.dim() == 5:
                        X_e_mod[:, channel_idx, :, :, :] = 0
                    else:
                        X_e_mod[:, channel_idx, :, :] = 0
                    if X_h_mod.dim() == 5:
                        X_h_mod[:, channel_idx, :, :, :] = 0
                    else:
                        X_h_mod[:, channel_idx, :, :] = 0
                model_output = self.model(X_e_mod, X_h_mod, adj_matrix)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels_t.detach().cpu().numpy().tolist())
                if return_probs:
                    all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = float((all_preds == all_labels).mean()) if all_labels.size > 0 else 0.0
        if return_probs:
            return acc, np.array(all_probs), all_preds, all_labels
        return acc
    # ==================== 随机多通道消融（k=2/4/6/8/10, >=5次） ====================
    def test_random_multi_channel_ablation(self, test_loader, output_dir):
        """
        随机多通道消融测试（双分支模式）
        - Phase E 与 Phase H 同时对相同通道集合进行消融
        - 输出:
            1) channel_ablation_dual_branch_random.csv
            2) channel_ablation_dual_branch_random_summary.csv
        """
        print("\n  🎲 随机多通道消融测试（双分支模式）")
        save_extra = bool(getattr(self.config, 'DIAG_MULTI_ABLATION_SAVE_EXTRA_METRICS', True))
        baseline = self._evaluate(test_loader, return_probs=save_extra)
        baseline_acc = baseline['accuracy']
        print("     基线准确率: {:.4f}".format(baseline_acc))
        num_channels = int(getattr(self.config, 'NUM_CHANNELS', 12))
        k_list = getattr(self.config, 'DIAG_MULTI_ABLATION_K_LIST', [2, 4, 6, 8, 10])
        min_trials = int(getattr(self.config, 'DIAG_MULTI_ABLATION_MIN_TRIALS_PER_K', 5))
        seed = int(getattr(self.config, 'DIAG_MULTI_ABLATION_RANDOM_SEED', 42))
        mode = getattr(self.config, 'DIAG_ABLATION_MODE', 'zero')
        rng = np.random.RandomState(seed)
        trial_rows = []
        summary_rows = []
        extra_baseline_metrics = None
        if save_extra and ('probabilities' in baseline):
            try:
                from src.utils.evaluator import compute_classification_metrics
                extra_baseline_metrics = compute_classification_metrics(baseline['labels'], baseline['predictions'], baseline['probabilities'], self.config)
            except Exception as e:
                print("     ⚠️ 计算基线额外指标失败: {}".format(e))
                extra_baseline_metrics = None
        for k in k_list:
            k = int(k)
            if num_channels < k:
                print("     ⚠️ NUM_CHANNELS={} < k={}，跳过该 k".format(num_channels, k))
                continue
            seen = set()
            combos = []
            max_attempts = 20000
            attempts = 0
            while (len(combos) < min_trials) and (attempts < max_attempts):
                attempts += 1
                combo = tuple(sorted(rng.choice(num_channels, size=k, replace=False).tolist()))
                if combo in seen:
                    continue
                seen.add(combo)
                combos.append(combo)
            if len(combos) < min_trials:
                print("     ⚠️ k={} 仅采样到 {} 组唯一组合（期望 ≥{}）".format(k, len(combos), min_trials))
            acc_drops = []
            worst_drop = None
            worst_combo = None
            for ti, combo in enumerate(combos):
                ablated_acc, ablated_metrics = self._test_multi_channel_ablation_single(test_loader, list(combo), mode=mode, return_extra=save_extra)
                acc_drop = baseline_acc - ablated_acc
                acc_drops.append(acc_drop)
                channels_str = "|".join([str(x) for x in combo])
                row = {'k': k, 'trial_index': ti + 1, 'channels': channels_str, 'baseline_acc': float(baseline_acc), 'ablated_acc': float(ablated_acc), 'acc_drop': float(acc_drop)}
                if save_extra and extra_baseline_metrics and ablated_metrics:
                    row['baseline_precision'] = float(extra_baseline_metrics.get('precision', 0.0))
                    row['baseline_recall'] = float(extra_baseline_metrics.get('recall', 0.0))
                    row['baseline_f1'] = float(extra_baseline_metrics.get('f1_score', 0.0))
                    row['ablated_precision'] = float(ablated_metrics.get('precision', 0.0))
                    row['ablated_recall'] = float(ablated_metrics.get('recall', 0.0))
                    row['ablated_f1'] = float(ablated_metrics.get('f1_score', 0.0))
                trial_rows.append(row)
                if (worst_drop is None) or (acc_drop > worst_drop):
                    worst_drop = acc_drop
                    worst_combo = combo
            if len(acc_drops) > 0:
                acc_drops_arr = np.asarray(acc_drops, dtype=np.float32)
                summary_rows.append({
                    'k': k,
                    'num_trials': int(len(acc_drops)),
                    'mean_acc_drop': float(acc_drops_arr.mean()),
                    'std_acc_drop': float(acc_drops_arr.std()),
                    'min_acc_drop': float(acc_drops_arr.min()),
                    'max_acc_drop': float(acc_drops_arr.max()),
                    'worst_channels': "|".join([str(x) for x in worst_combo]) if worst_combo else ""
                })
        try:
            df_trials = pd.DataFrame(trial_rows)
            out1 = output_dir / "channel_ablation_dual_branch_random.csv"
            df_trials.to_csv(out1, index=False)
            print("     ✅ 保存: {}".format(out1.name))
            df_summary = pd.DataFrame(summary_rows)
            out2 = output_dir / "channel_ablation_dual_branch_random_summary.csv"
            df_summary.to_csv(out2, index=False)
            print("     ✅ 保存: {}".format(out2.name))
        except Exception as e:
            print("     ❌ 保存随机多通道消融结果失败: {}".format(e))
        return {'trials': trial_rows, 'summary': summary_rows}
    def _test_multi_channel_ablation_single(self, data_loader, channel_indices, mode='zero', return_extra=False):
        """
        测试消融多个通道后的性能（同时消融 Phase E 和 Phase H）
        Returns:
            (ablated_acc, extra_metrics_or_None)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X_e, X_h, adj_matrix, labels, metadata in data_loader:
                X_e = X_e.to(self.device)
                X_h = X_h.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                labels_t = labels.to(self.device)
                X_e_mod = X_e.clone()
                X_h_mod = X_h.clone()
                for c in channel_indices:
                    if mode == 'zero':
                        if X_e_mod.dim() == 5:
                            X_e_mod[:, c, :, :, :] = 0
                        else:
                            X_e_mod[:, c, :, :] = 0
                        if X_h_mod.dim() == 5:
                            X_h_mod[:, c, :, :, :] = 0
                        else:
                            X_h_mod[:, c, :, :] = 0
                    else:
                        if X_e_mod.dim() == 5:
                            X_e_mod[:, c, :, :, :] = 0
                        else:
                            X_e_mod[:, c, :, :] = 0
                        if X_h_mod.dim() == 5:
                            X_h_mod[:, c, :, :, :] = 0
                        else:
                            X_h_mod[:, c, :, :] = 0
                model_output = self.model(X_e_mod, X_h_mod, adj_matrix)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels_t.detach().cpu().numpy().tolist())
                if return_extra:
                    all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        ablated_acc = float((all_preds == all_labels).mean()) if all_labels.size > 0 else 0.0
        extra_metrics = None
        if return_extra:
            try:
                all_probs = np.array(all_probs)
                from src.utils.evaluator import compute_classification_metrics
                extra_metrics = compute_classification_metrics(all_labels, all_preds, all_probs, self.config)
            except Exception as e:
                print("     ⚠️ 计算消融额外指标失败: {}".format(e))
                extra_metrics = None
        return ablated_acc, extra_metrics
    # ==================== 其他 ====================
    def auto_scan_sensitive_regions(self, test_loader, output_dir):
        print("\n  ⏭️  自动扫描功能（双分支版本开发中）")
        return {}