"""
双分支 STFT Dataset（自动配对 Hi/Lo）
支持 Phase E 多窗口采样 + Phase H 固定时长
支持缺失通道处理（零填充）
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import re


class DualBranchSTFTDataset(Dataset):
    """双分支 STFT Dataset（自动配对 Hi/Lo）"""
    
    def __init__(self, config, mode='train'):
        """
        Args:
            config: 配置对象
            mode: 'train' / 'test'
        """
        self.config = config
        self.mode = mode
        
        # 直接使用根目录
        self.dual_root = config.DUAL_BRANCH_ROOT
        
        # 验证根目录
        if not os.path.exists(self.dual_root):
            raise ValueError(f"双分支根目录不存在: {self.dual_root}")
        
        # 邻接矩阵
        self.adj_matrix = self._build_adjacency_matrix()
        
        # 扫描配对样本
        self.samples = self._scan_paired_samples()
        
        if len(self.samples) == 0:
            raise ValueError(
                f"未找到配对样本！\n"
                f"  Phase E: {self.dual_root}\n"
                f"  Phase H: {self.dual_root}\n"
                f"  请检查文件夹结构和文件命名"
            )
        
        # 打印初始化信息
        print(f"✅ DualBranchDataset 初始化完成:")
        print(f"   模式: {mode}")
        print(f"   配对样本数: {len(self.samples)}")
        print(f"   Phase E 策略: {config.E_STRATEGY}")
        print(f"   通道配置: {config.NUM_CHANNELS}通道（含{len(config.MISSING_CHANNELS)}个缺失通道）")
        
        # 计算预期输出维度
        if config.E_STRATEGY == "multi_window":
            e_time_bins = config.E_TARGET_TIME_BINS
        else:
            e_time_bins = config.E_TARGET_TIME_BINS
        
        h_time_bins = config.H_TARGET_TIME_BINS
        
        print(f"   输出形状:")
        print(f"     - Phase E: ({config.NUM_CHANNELS}, 1, {config.DUAL_FREQ_BINS}, {e_time_bins})")
        print(f"     - Phase H: ({config.NUM_CHANNELS}, 1, {config.DUAL_FREQ_BINS}, {h_time_bins})")
    
    def _build_adjacency_matrix(self):
        """构建邻接矩阵（基于欧式距离）"""
        coords = self.config.CHANNEL_COORDS
        num_nodes = coords.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    adj_matrix[i, j] = 1.0 / (dist + 1e-6)
        
        # 归一化
        row_sum = adj_matrix.sum(axis=1, keepdims=True)
        adj_matrix = adj_matrix / (row_sum + 1e-6)
        
        return adj_matrix
    
    def _scan_paired_samples(self):
        """扫描并自动配对 Hi/ 和 Lo/ 中的样本"""
        samples = []
        root_dir = self.dual_root
        
        if not os.path.exists(root_dir):
            raise ValueError(f"双分支根目录不存在: {root_dir}")
        
        # 获取所有类别文件夹
        class_folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        
        if not class_folders:
            raise ValueError(f"双分支根目录下未找到任何文件夹: {root_dir}")
        
        print(f"\n🔍 扫描双分支数据:")
        print(f"   根目录: {root_dir}")
        print(f"   检测到类别: {class_folders}")
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            
            # 在类别文件夹下寻找 Hi/ 和 Lo/
            e_class_dir = os.path.join(class_path, self.config.PHASE_E_SUBDIR)
            h_class_dir = os.path.join(class_path, self.config.PHASE_H_SUBDIR)
            
            # 检查 Hi/ 和 Lo/ 是否都存在
            if not os.path.exists(e_class_dir):
                print(f"   ⚠️ {class_name}: 缺少 {self.config.PHASE_E_SUBDIR}/ 文件夹，跳过")
                continue
            
            if not os.path.exists(h_class_dir):
                print(f"   ⚠️ {class_name}: 缺少 {self.config.PHASE_H_SUBDIR}/ 文件夹，跳过")
                continue
            
            # 扫描文件夹（不是文件）
            e_sample_dirs = sorted([
                d for d in os.listdir(e_class_dir)
                if os.path.isdir(os.path.join(e_class_dir, d))
            ])
            
            h_sample_dirs = sorted([
                d for d in os.listdir(h_class_dir)
                if os.path.isdir(os.path.join(h_class_dir, d))
            ])
            
            if len(e_sample_dirs) == 0:
                print(f"   ⚠️ {class_name}: {self.config.PHASE_E_SUBDIR}/ 下无样本文件夹，跳过")
                continue
            
            if len(h_sample_dirs) == 0:
                print(f"   ⚠️ {class_name}: {self.config.PHASE_H_SUBDIR}/ 下无样本文件夹，跳过")
                continue
            
            # 自动匹配 Phase E 和 Phase H 文件夹
            matched_count = 0
            for e_dir in e_sample_dirs:
                h_dir = self._find_matching_h_dir(e_dir, h_sample_dirs)
                
                if h_dir is None:
                    if self.config.DUAL_BRANCH_SKIP_INCOMPLETE:
                        continue
                    else:
                        print(f"   ⚠️ Phase E 无配对 Phase H: {e_dir}")
                        continue
                
                # 完整路径
                e_path = os.path.join(e_class_dir, e_dir)
                h_path = os.path.join(h_class_dir, h_dir)
                
                samples.append({
                    'phase_e_path': e_path,
                    'phase_h_path': h_path,
                    'class_idx': class_idx,
                    'class_name': class_name
                })
                matched_count += 1
            
            print(f"   ✅ {class_name}: 配对成功 {matched_count} 个样本")
        
        return samples
    
    def _find_matching_h_dir(self, e_dir_name, h_dir_list):
        """根据 Phase E 文件夹名，匹配 Phase H 文件夹"""
        # 方法1：简单替换 Hi → Lo
        h_candidate = e_dir_name.replace('_Hi_', '_Lo_').replace('_hi_', '_lo_')
        if h_candidate in h_dir_list:
            return h_candidate
        
        # 方法2：提取编号匹配
        e_match = re.search(r'_Hi_(\d+)_', e_dir_name, re.IGNORECASE)
        if e_match:
            sample_id = e_match.group(1)
            
            for h_dir in h_dir_list:
                h_match = re.search(r'_Lo_(\d+)_', h_dir, re.IGNORECASE)
                if h_match and h_match.group(1) == sample_id:
                    return h_dir
        
        # 方法3：最宽松匹配（基于编号）
        e_numbers = re.findall(r'\d+', e_dir_name)
        if e_numbers:
            for h_dir in h_dir_list:
                h_numbers = re.findall(r'\d+', h_dir)
                if e_numbers[-1] == h_numbers[-1]:
                    return h_dir
        
        return None
    
    def _load_stft_from_folder(self, folder_path):
        """
        从样本文件夹加载所有通道的 STFT 数据
        支持缺失通道处理（零填充）
        
        Returns:
            stft_ (C, F, T) 形状的 STFT 数据，C=NUM_CHANNELS（包括缺失通道的零填充）
        """
        # 获取所有 .npy 文件
        npy_files = sorted(glob.glob(os.path.join(folder_path, "Channel_*_STFT.npy")))
        
        if len(npy_files) == 0:
            raise ValueError(f"样本文件夹内无 .npy 文件: {folder_path}")
        
        # 加载第一个文件，获取维度
        try:
            first_data = np.load(npy_files[0], allow_pickle=True)
            
            # 解包 0 维数组
            if isinstance(first_data, np.ndarray) and first_data.ndim == 0:
                first_data = first_data.item()
            
            # 提取 model_input 字段
            if isinstance(first_data, dict):
                if 'model_input' in first_data:
                    model_input = first_data['model_input']  # 形状: (1, F, T)
                    
                    # 去掉第一个维度
                    if model_input.ndim == 3 and model_input.shape[0] == 1:
                        model_input = model_input[0]  # 形状: (F, T)
                    
                    F, T = model_input.shape
                else:
                    raise ValueError(
                        f"字典中缺少 'model_input' 字段\n"
                        f"  可用字段: {list(first_data.keys())}"
                    )
            else:
                raise ValueError(f"数据不是字典格式: {type(first_data)}")
        
        except Exception as e:
            raise ValueError(
                f"加载 .npy 文件失败\n"
                f"  文件: {npy_files[0]}\n"
                f"  错误: {str(e)}"
            )
        
        # ✅ 使用完整通道数（包括缺失通道）
        C_full = self.config.NUM_CHANNELS  # 12
        
        # 初始化完整数组（缺失通道自动为0）
        stft_data = np.zeros((C_full, F, T), dtype=np.float32)
        
        # ✅ 构建物理通道号 → 逻辑索引的映射
        missing_channels = self.config.MISSING_CHANNELS  # [4, 9]
        channel_to_idx = {}
        
        logical_idx = 0
        for physical_ch in range(1, C_full + 1):
            if physical_ch not in missing_channels:
                channel_to_idx[physical_ch] = logical_idx
            logical_idx += 1
        
        # 加载所有通道并填充到正确位置
        for npy_file in npy_files:
            try:
                # 从文件名提取物理通道号
                # Channel_10_STFT.npy → 10
                match = re.search(r'Channel_(\d+)_', os.path.basename(npy_file))
                if not match:
                    raise ValueError(f"无法从文件名提取通道号: {npy_file}")
                
                physical_ch = int(match.group(1))
                
                # 跳过缺失通道（理论上不应该存在这个文件）
                if physical_ch in missing_channels:
                    print(f"   ⚠️ 警告：发现缺失通道 {physical_ch} 的文件，已跳过")
                    continue
                
                # 映射到逻辑索引
                logical_idx = channel_to_idx.get(physical_ch)
                if logical_idx is None:
                    print(f"   ⚠️ 警告：通道 {physical_ch} 不在有效列表中，已跳过")
                    continue
                
                # 加载数据
                data = np.load(npy_file, allow_pickle=True)
                
                # 解包 0 维数组
                if isinstance(data, np.ndarray) and data.ndim == 0:
                    data = data.item()
                
                # 提取 model_input
                if isinstance(data, dict) and 'model_input' in data:
                    model_input = data['model_input']
                    
                    # 去掉第一个维度
                    if model_input.ndim == 3 and model_input.shape[0] == 1:
                        model_input = model_input[0]
                    
                    # ✅ 填充到对应的逻辑索引
                    stft_data[logical_idx] = model_input
                else:
                    raise ValueError(f"无法提取 model_input: {type(data)}")
            
            except Exception as e:
                raise ValueError(
                    f"加载通道失败\n"
                    f"  文件: {npy_file}\n"
                    f"  错误: {str(e)}"
                )
        
        # 缺失通道（索引3和8，对应通道4和9）自动保持为零
        
        return stft_data  # (12, F, T)
    
    def _apply_e_strategy(self, X_e):
        """应用 Phase E 策略（multi_window 或 fixed_quota）"""
        C, F, T = X_e.shape
        target_T = self.config.E_TARGET_TIME_BINS
        
        if self.config.E_STRATEGY == "multi_window":
            # 多窗口采样 + 聚合
            num_windows = self.config.E_MULTI_NUM_WINDOWS
            window_size = self.config.E_TARGET_TIME_BINS
            
            if self.config.E_MULTI_SAMPLING == "uniform":
                # 均匀采样
                if T < num_windows * window_size:
                    repeat_factor = int(np.ceil(num_windows * window_size / T))
                    X_e_extended = np.tile(X_e, (1, 1, repeat_factor))
                    T_extended = X_e_extended.shape[2]
                else:
                    X_e_extended = X_e
                    T_extended = T
                
                stride = (T_extended - window_size) // (num_windows - 1) if num_windows > 1 else 0
                windows = []
                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    if end > T_extended:
                        end = T_extended
                        start = end - window_size
                    windows.append(X_e_extended[:, :, start:end])
            
            else:  # random
                np.random.seed(self.config.E_MULTI_RANDOM_SEED)
                windows = []
                for _ in range(num_windows):
                    if T >= window_size:
                        start = np.random.randint(0, T - window_size + 1)
                    else:
                        start = 0
                    end = start + window_size
                    if end > T:
                        end = T
                    window = X_e[:, :, start:end]
                    if window.shape[2] < window_size:
                        pad_width = window_size - window.shape[2]
                        window = np.pad(window, ((0, 0), (0, 0), (0, pad_width)), mode='edge')
                    windows.append(window)
            
            # 聚合
            windows = np.stack(windows, axis=0)  # (N, C, F, T)
            if self.config.E_MULTI_AGGREGATION == "mean":
                X_e_processed = windows.mean(axis=0)
            elif self.config.E_MULTI_AGGREGATION == "max":
                X_e_processed = windows.max(axis=0)
            else:  # attention (简化为 mean)
                X_e_processed = windows.mean(axis=0)
        
        else:  # fixed_quota
            # 固定时长截取
            if self.config.E_FIXED_ALIGN == "end":
                if T >= target_T:
                    X_e_processed = X_e[:, :, -target_T:]
                else:
                    pad_width = target_T - T
                    if self.config.E_FIXED_PADDING_MODE == "edge":
                        X_e_processed = np.pad(X_e, ((0, 0), (0, 0), (pad_width, 0)), mode='edge')
                    else:
                        X_e_processed = np.pad(X_e, ((0, 0), (0, 0), (pad_width, 0)), mode='constant')
            else:  # start
                if T >= target_T:
                    X_e_processed = X_e[:, :, :target_T]
                else:
                    pad_width = target_T - T
                    if self.config.E_FIXED_PADDING_MODE == "edge":
                        X_e_processed = np.pad(X_e, ((0, 0), (0, 0), (0, pad_width)), mode='edge')
                    else:
                        X_e_processed = np.pad(X_e, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        
        return X_e_processed
    
    def _apply_h_strategy(self, X_h):
        """应用 Phase H 策略（固定时长）"""
        C, F, T = X_h.shape
        target_T = self.config.H_TARGET_TIME_BINS
        
        if self.config.PHASE_H_ALIGN == "start":
            if T >= target_T:
                X_h_processed = X_h[:, :, :target_T]
            else:
                pad_width = target_T - T
                if self.config.PHASE_H_PADDING_MODE == "edge":
                    X_h_processed = np.pad(X_h, ((0, 0), (0, 0), (0, pad_width)), mode='edge')
                else:
                    X_h_processed = np.pad(X_h, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:  # end
            if T >= target_T:
                X_h_processed = X_h[:, :, -target_T:]
            else:
                pad_width = target_T - T
                if self.config.PHASE_H_PADDING_MODE == "edge":
                    X_h_processed = np.pad(X_h, ((0, 0), (0, 0), (pad_width, 0)), mode='edge')
                else:
                    X_h_processed = np.pad(X_h, ((0, 0), (0, 0), (pad_width, 0)), mode='constant')
        
        return X_h_processed
    
    def _normalize(self, X):
        """标准化"""
        mean = X.mean()
        std = X.std()
        return (X - mean) / (std + 1e-6)
    
    def _augment(self, X):
        """数据增强（简化版）"""
        if np.random.rand() > self.config.AUG_PROB:
            return X
        
        # 添加噪声
        noise = torch.randn_like(X) * self.config.AUG_NOISE_STD
        X = X + noise
        
        return X
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回一个双分支样本
        
        Returns:
            X_e: Phase E 的 STFT 数据 (C, 1, F, T_e)
            X_h: Phase H 的 STFT 数据 (C, 1, F, T_h)
            adj_matrix: 邻接矩阵 (C, C)
            label: 类别标签 (标量)
            meta 元数据字典
        """
        sample = self.samples[idx]
        
        # 从文件夹加载 STFT（包括缺失通道的零填充）
        X_e = self._load_stft_from_folder(sample['phase_e_path'])  # (12, F, T)
        X_h = self._load_stft_from_folder(sample['phase_h_path'])  # (12, F, T)
        
        # 策略处理
        X_e = self._apply_e_strategy(X_e)  # (12, F, T_e)
        X_h = self._apply_h_strategy(X_h)  # (12, F, T_h)
        
        # 转换为张量并增加通道维度
        X_e = torch.from_numpy(X_e).float().unsqueeze(1)  # (12, 1, F, T_e)
        X_h = torch.from_numpy(X_h).float().unsqueeze(1)  # (12, 1, F, T_h)
        
        # 归一化和增强
        if self.config.DUAL_ENABLE_NORMALIZE:
            X_e = self._normalize(X_e)
            X_h = self._normalize(X_h)
        
        if self.mode == 'train' and self.config.DUAL_ENABLE_AUGMENT:
            X_e = self._augment(X_e)
            X_h = self._augment(X_h)
        
        # 邻接矩阵
        adj_matrix = torch.from_numpy(self.adj_matrix).float()
        
        # 标签
        label = torch.tensor(sample['class_idx'], dtype=torch.long)
        
        # 元数据
        metadata = {
            'class_name': sample['class_name'],
            'phase_e_path': sample['phase_e_path'],
            'phase_h_path': sample['phase_h_path']
        }
        
        return X_e, X_h, adj_matrix, label, metadata
