# =============================================================================
# dataset.py - 修复版（所有参数从config读取）
# =============================================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ===================== 简化版邻接矩阵（保证形状正确即可） =====================
def _build_adjacency_matrix(num_channels) -> torch.Tensor:
    """构建邻接矩阵，如需基于坐标的邻接矩阵，后续可在此扩展"""
    return torch.eye(num_channels, dtype=torch.float32)

# ===================== 工具：数据清洗/归一化/增强 =====================

def _sanitize_np(x: np.ndarray) -> np.ndarray:
    """NaN/Inf 清洗：避免进入模型后出现 nan loss。"""
    if x is None:
        return np.zeros((1, 1), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x

def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """每通道独立 min-max 归一化到 [0,1]。输入形状：(1,F,T)"""
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    if (x_max - x_min) < eps:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)

def _tf_augment(x: torch.Tensor, config) -> torch.Tensor:
    """轻量 STFT 时频增强（输入形状：(1,F,T)）。"""
    # ---- 1) 随机增益 ----
    if config.AUG_GAIN_RANGE is not None:
        g0, g1 = config.AUG_GAIN_RANGE
        gain = float(np.random.uniform(g0, g1))
        x = x * gain

    # ---- 2) 加性噪声 ----
    if config.AUG_NOISE_STD and config.AUG_NOISE_STD > 0:
        noise = torch.randn_like(x) * float(config.AUG_NOISE_STD)
        x = x + noise

    # ---- 3) 时间轴平移（循环移位，不丢信息）----
    if config.AUG_TIME_SHIFT and config.AUG_TIME_SHIFT > 0:
        shift = int(np.random.randint(-config.AUG_TIME_SHIFT, config.AUG_TIME_SHIFT + 1))
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=2)

    # ---- 4) 频率遮挡 ----
    if config.AUG_FREQ_MASK_MAX and config.AUG_FREQ_MASK_MAX > 0:
        f = int(np.random.randint(0, config.AUG_FREQ_MASK_MAX + 1))
        if f > 0 and x.shape[1] > f:
            f0 = int(np.random.randint(0, x.shape[1] - f + 1))
            x[:, f0:f0 + f, :] = 0.0

    # ---- 5) 时间遮挡 ----
    if config.AUG_TIME_MASK_MAX and config.AUG_TIME_MASK_MAX > 0:
        t = int(np.random.randint(0, config.AUG_TIME_MASK_MAX + 1))
        if t > 0 and x.shape[2] > t:
            t0 = int(np.random.randint(0, x.shape[2] - t + 1))
            x[:, :, t0:t0 + t] = 0.0

    return x

# ===================== Dataset =====================

class STFTDataset(Dataset):
    """最小可用 Dataset：读取 NPY -> 装配 (12,1,101,193) + adj(12,12) + label"""

    def __init__(self, root_dir, config):
        """
        Args:
            root_dir: STFT NPY数据根目录
            config: 配置对象，包含所有参数
        """
        self.root_dir = root_dir
        self.config = config

        # 从config读取参数
        self.num_channels = config.NUM_CHANNELS
        self.target_ft = config.STFT_SHAPE
        self.valid_npy_num = config.VALID_CHANNELS_NUM

        # 构建邻接矩阵
        self.adj_matrix = _build_adjacency_matrix(self.num_channels)

        # 加载样本路径
        self.samples = self._load_sample_paths()

        print(f"✅ 数据集加载完成：共 {len(self.samples)} 个有效样本")
        print(f"   - ENABLE_NORMALIZE = {config.ENABLE_NORMALIZE}")
        print(f"   - ENABLE_AUGMENT = {config.ENABLE_AUGMENT} (仅训练阶段生效)")

    def _load_sample_paths(self):
        """加载所有样本的 NPY 路径与对应标签（保持 label 的稳定性：按类名排序）。"""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root_dir not found: {self.root_dir}")

        samples = []
        class_folders = [
            f for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f))
        ]
        class_folders = sorted(class_folders)  # 保证 label 映射可复现

        for label, cls_name in enumerate(class_folders):
            cls_path = os.path.join(self.root_dir, cls_name)
            sample_folders = [
                f for f in os.listdir(cls_path)
                if os.path.isdir(os.path.join(cls_path, f))
            ]
            sample_folders = sorted(sample_folders)

            for sample_folder in sample_folders:
                sample_path = os.path.join(cls_path, sample_folder)
                npy_files = [f for f in os.listdir(sample_path) if f.endswith("_STFT.npy")]

                if len(npy_files) != self.valid_npy_num:
                    continue  # 样本不完整，跳过

                npy_files = sorted(npy_files)
                npy_paths = [os.path.join(sample_path, f) for f in npy_files]
                samples.append((npy_paths, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def _to_1xFT_tensor(self, raw_np: np.ndarray) -> torch.Tensor:
        """把任意维度的原始数据转成固定形状 (1,F,T)。"""
        raw_np = _sanitize_np(raw_np)

        # ---- 统一成 2D (F,T) 的 numpy ----
        if raw_np.ndim == 1:
            raw_2d = raw_np.reshape(-1, 1)
        elif raw_np.ndim == 2:
            raw_2d = raw_np
        else:
            # 兜底：取最后两个维度作为 (F,T)
            raw_2d = raw_np.reshape(raw_np.shape[-2], raw_np.shape[-1])

        # ---- 转成 torch，并补齐 channel 维度：(1,F,T) ----
        x = torch.from_numpy(raw_2d).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1,F,T)

        # ---- 统一 resize 到 TARGET_FT：保持"插值到固定尺寸"的思路 ----
        # F.interpolate 需要 4D: (N,C,H,W)，这里把 (1,F,T) 当作 (C,H,W)，N=1
        x4 = x.unsqueeze(0)  # (1,1,F,T)
        x4 = F.interpolate(x4, size=self.target_ft, mode="bilinear", align_corners=False)
        x = x4.squeeze(0)  # (1,101,193)

        # ---- 新增：仅训练阶段的时频增强（validate 时 torch.is_grad_enabled() == False）----
        if self.config.ENABLE_AUGMENT and torch.is_grad_enabled() and (np.random.rand() < self.config.AUG_PROB):
            x = _tf_augment(x, self.config)

        # ---- 新增：归一化（train/val 都生效）----
        if self.config.ENABLE_NORMALIZE:
            x = _minmax_norm(x)

        return x.contiguous()

    def __getitem__(self, idx):
        """
        返回：
            stft_output: (NUM_CHANNELS, 1, F, T)
            adj_matrix: (NUM_CHANNELS, NUM_CHANNELS)
            label_tensor: (1,)
        """
        # 输出：缺失通道用 0 填充，保证维度永远正确
        stft_output = torch.zeros(self.num_channels, 1, self.target_ft[0], self.target_ft[1], dtype=torch.float32)
        npy_paths, label = self.samples[idx]

        for npy_path in npy_paths:
            try:
                data_dict = np.load(npy_path, allow_pickle=True).item()
                raw_data = data_dict.get("model_input", None)
                physical_id = int(data_dict.get("physical_channel_id", -1))
                channel_id = physical_id - 1  # 1-based -> 0-based
            except Exception as e:
                print(f"⚠️ 加载失败：{npy_path} | {e}")
                continue

            # 通道合法性检查
            if channel_id < 0 or channel_id >= self.num_channels:
                continue

            x = self._to_1xFT_tensor(raw_data)  # (1,101,193)
            stft_output[channel_id] = x

        label_tensor = torch.tensor(label, dtype=torch.long)
        return stft_output, self.adj_matrix, label_tensor
