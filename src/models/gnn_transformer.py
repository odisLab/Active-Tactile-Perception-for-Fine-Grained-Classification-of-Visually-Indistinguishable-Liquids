import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 全局关闭无关警告
import warnings
warnings.filterwarnings('ignore')
torch.nn.Module.dump_patches = False

# ===================== 1. 纯Linear版GNN（修复维度匹配） =====================
class GNN_Spatial_Fusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_nodes=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.gnn1 = nn.Sequential(
            nn.Linear(in_dim + num_nodes, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gnn2 = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x, adj_matrix):
        batch_size = x.shape[0]

        # 统一adj_matrix为3维 (batch, 12, 12)
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        # 拼接节点特征+邻接向量
        x_fused = torch.cat([x, adj_matrix], dim=-1)

        # GNN前向
        x1 = self.gnn1(x_fused)
        x2 = self.gnn2(x1)

        return x2

# ===================== 2. Transformer模块（适配旧版PyTorch） =====================
class Transformer_Temporal_Fusion(nn.Module):
    def __init__(self, d_model, nhead=1, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()

        self.input_norm = nn.LayerNorm(d_model)

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        src = self.input_norm(src)
        output = self.transformer_encoder(src)
        return output

# ===================== 3. 整体模型（修复版：从config读取所有参数） =====================
class GNN_Transformer_Fusion(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有模型参数
        """
        super().__init__()

        # 从config读取参数
        stft_shape = config.STFT_SHAPE  # (F, T)
        self.F, self.T = stft_shape
        self.num_nodes = config.NUM_CHANNELS
        self.num_classes = config.NUM_CLASSES

        # 读取模型架构参数
        gnn_hidden_dim = config.GNN_HIDDEN_DIM
        transformer_nhead = config.TRANSFORMER_NHEAD
        transformer_layers = config.TRANSFORMER_LAYERS
        transformer_dim_feedforward = config.TRANSFORMER_DIM_FEEDFORWARD

        # STFT特征压缩
        self.stft_compress = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )

        # 预计算节点特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.F, self.T)
            self.node_dim = self.stft_compress(dummy).shape[1]

        # GNN模块
        self.gnn = GNN_Spatial_Fusion(
            in_dim=self.node_dim,
            hidden_dim=gnn_hidden_dim,
            num_nodes=self.num_nodes
        )

        # 维度映射
        self.seq_proj = nn.Linear(gnn_hidden_dim, self.T)
        self.freq_proj = nn.Linear(self.num_nodes, self.F)
        self.proj_norm = nn.LayerNorm(self.F)

        # Transformer模块
        self.transformer = Transformer_Temporal_Fusion(
            d_model=self.F,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=0.2
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.F * self.T, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, stft_batch, adj_matrix):
        batch_size = stft_batch.shape[0]

        # 提取每个通道的STFT特征
        node_feat = []
        for i in range(self.num_nodes):
            stft_ch = stft_batch[:, i, :, :, :]
            if stft_ch.dim() != 4:
                stft_ch = stft_ch.view(batch_size, 1, self.F, self.T)
            feat = self.stft_compress(stft_ch)
            node_feat.append(feat)

        node_feat = torch.stack(node_feat, dim=1)

        # GNN空间融合
        gnn_out = self.gnn(node_feat, adj_matrix)

        # 维度映射
        seq_feat = self.seq_proj(gnn_out)
        seq_feat = seq_feat.permute(0, 2, 1)
        seq_feat = self.freq_proj(seq_feat)
        seq_feat = self.proj_norm(seq_feat)

        # Transformer时序融合
        trans_out = self.transformer(seq_feat)

        # 分类
        trans_flat = trans_out.flatten(1)
        logits = self.classifier(trans_flat)

        return logits

# 测试模型
if __name__ == "__main__":
    # 模拟config对象
    class MockConfig:
        STFT_SHAPE = (101, 193)
        NUM_CHANNELS = 12
        NUM_CLASSES = 6
        GNN_HIDDEN_DIM = 64
        TRANSFORMER_NHEAD = 1
        TRANSFORMER_LAYERS = 2
        TRANSFORMER_DIM_FEEDFORWARD = 256

    config = MockConfig()

    # 初始化模型
    model = GNN_Transformer_Fusion(config)
    model.eval()

    # 构造输入
    stft_batch = torch.randn(8, 12, 1, 101, 193)
    adj_matrix = torch.eye(12)  # 2维邻接矩阵

    # 前向传播
    with torch.no_grad():
        logits = model(stft_batch, adj_matrix)

    print(f"✅ 模型测试通过！")
    print(f"输入STFT形状：{stft_batch.shape}")
    print(f"输出logits形状：{logits.shape}（预期：8×6）")
