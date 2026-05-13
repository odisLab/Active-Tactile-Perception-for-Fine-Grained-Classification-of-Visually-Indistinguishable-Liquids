"""
双分支 GNN-Transformer 融合模型
- Phase E 和 Phase H 分别编码
- 支持权重共享和多种融合策略
- 保持下游 GNN → Transformer → Classifier 结构不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchGNNTransformer(nn.Module):
    """双分支 Phase-Aware GNN-Transformer 模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础参数
        self.num_channels = config.NUM_CHANNELS
        self.freq_bins = config.DUAL_FREQ_BINS
        self.e_time_bins = config.E_TARGET_TIME_BINS
        self.h_time_bins = config.H_TARGET_TIME_BINS
        self.gnn_hidden_dim = config.GNN_HIDDEN_DIM
        self.num_classes = config.NUM_CLASSES
        
        # 分支配置
        self.weight_sharing = config.BRANCH_WEIGHT_SHARING
        self.fusion_mode = config.BRANCH_FUSION_MODE
        
        # ========== Phase E 编码器 ==========
        self.stft_compress_e = self._build_stft_encoder(self.e_time_bins)
        
        # ========== Phase H 编码器 ==========
        if self.weight_sharing:
            # 共享权重
            self.stft_compress_h = self.stft_compress_e
            print("⚙️  双分支模型：Phase E/H 编码器权重共享")
        else:
            # 独立权重
            self.stft_compress_h = self._build_stft_encoder(self.h_time_bins)
            print("⚙️  双分支模型：Phase E/H 编码器独立权重")
        
        # ========== 分支融合层 ==========
        if self.fusion_mode == "concat":
            # Concat 后投影回原维度
            self.fusion_projection = nn.Linear(
                self.gnn_hidden_dim * 2,
                self.gnn_hidden_dim
            )
            print(f"⚙️  分支融合模式：Concat + 投影")
            
        elif self.fusion_mode == "gated":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(self.gnn_hidden_dim * 2, self.gnn_hidden_dim),
                nn.Sigmoid()
            )
            print(f"⚙️  分支融合模式：Gated")
            
        elif self.fusion_mode == "attention":
            # 注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=self.gnn_hidden_dim,
                num_heads=config.FUSION_ATTENTION_HEADS,
                batch_first=True
            )
            print(f"⚙️  分支融合模式：Attention ({config.FUSION_ATTENTION_HEADS} heads)")
        else:
            raise ValueError(f"未知的融合模式: {self.fusion_mode}")
        
        # ========== 下游结构（与单分支一致）==========
        # GNN 层
        self.gnn = GraphConvLayer(self.gnn_hidden_dim, self.gnn_hidden_dim)
        
        # Remap 层（将节点特征展平）
        self.remap = nn.Linear(
            self.num_channels * self.gnn_hidden_dim,
            config.TRANSFORMER_LAYERS * self.gnn_hidden_dim
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.gnn_hidden_dim,
            nhead=config.TRANSFORMER_NHEAD,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.TRANSFORMER_LAYERS
        )
        
        # 分类器（与单分支完全一致）
        self.classifier = nn.Sequential(
            nn.Linear(self.gnn_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        print(f"⚙️  模型参数：{self.num_channels} 通道 → {self.num_classes} 类")
    
    def _build_stft_encoder(self, time_bins):
        """
        构建 STFT 编码器（CNN）
        
        输入：(B*N, 1, F, T)
        输出：(B*N, gnn_hidden_dim)
        """
        return nn.Sequential(
            # Conv1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出大小
            
            # Flatten
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.gnn_hidden_dim)
        )
    
    def forward(self, X_e, X_h, adj):
        """
        前向传播
        
        Args:
            X_e: (B, N, 1, F, T_e) - Phase E STFT
            X_h: (B, N, 1, F, T_h) - Phase H STFT
            adj: (B, N, N) 或 (N, N) - 邻接矩阵
        
        Returns:
            logits: (B, num_classes)
            embeddings: dict - 中间特征（用于诊断）
        """
        B = X_e.size(0)
        N = self.num_channels
        
        # ========== Phase E 编码 ==========
        # 展平 batch 和 node 维度：(B, N, 1, F, T_e) → (B*N, 1, F, T_e)
        X_e_flat = X_e.view(B * N, 1, self.freq_bins, self.e_time_bins)
        node_feat_e = self.stft_compress_e(X_e_flat)  # (B*N, gnn_hidden_dim)
        node_feat_e = node_feat_e.view(B, N, self.gnn_hidden_dim)
        
        # ========== Phase H 编码 ==========
        X_h_flat = X_h.view(B * N, 1, self.freq_bins, self.h_time_bins)
        node_feat_h = self.stft_compress_h(X_h_flat)
        node_feat_h = node_feat_h.view(B, N, self.gnn_hidden_dim)
        
        # ========== 分支融合 ==========
        if self.fusion_mode == "concat":
            # Concat + 投影
            node_feat_concat = torch.cat([node_feat_e, node_feat_h], dim=-1)  # (B, N, 2*gnn_hidden_dim)
            node_feat_fused = self.fusion_projection(node_feat_concat)  # (B, N, gnn_hidden_dim)
        
        elif self.fusion_mode == "gated":
            # 门控融合：g = sigmoid(W[e;h])
            # out = g * e + (1-g) * h
            gate_input = torch.cat([node_feat_e, node_feat_h], dim=-1)
            gate = self.gate(gate_input)  # (B, N, gnn_hidden_dim)
            node_feat_fused = gate * node_feat_e + (1 - gate) * node_feat_h
        
        elif self.fusion_mode == "attention":
            # 注意力融合（E 和 H 互相 attend）
            # Stack E and H as sequence: (B, N, 2, gnn_hidden_dim)
            stacked = torch.stack([node_feat_e, node_feat_h], dim=2)
            stacked_flat = stacked.view(B * N, 2, self.gnn_hidden_dim)
            
            # Self-attention
            attended, _ = self.attention(stacked_flat, stacked_flat, stacked_flat)
            # (B*N, 2, gnn_hidden_dim)
            
            # 平均池化
            node_feat_fused = attended.mean(dim=1)  # (B*N, gnn_hidden_dim)
            node_feat_fused = node_feat_fused.view(B, N, self.gnn_hidden_dim)
        
        # ========== GNN 层 ==========
        # 处理邻接矩阵（支持共享邻接矩阵）
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        
        node_feat_gnn = self.gnn(node_feat_fused, adj)  # (B, N, gnn_hidden_dim)
        
        # ========== Remap + Transformer ==========
        # 展平节点特征
        node_feat_flat = node_feat_gnn.view(B, -1)  # (B, N*gnn_hidden_dim)
        remapped = self.remap(node_feat_flat)  # (B, L*gnn_hidden_dim)
        
        # 重塑为序列
        L = self.config.TRANSFORMER_LAYERS
        transformer_input = remapped.view(B, L, self.gnn_hidden_dim)  # (B, L, gnn_hidden_dim)
        
        # Transformer 编码
        transformer_output = self.transformer(transformer_input)  # (B, L, gnn_hidden_dim)
        
        # 池化为单一向量
        pooled = transformer_output.mean(dim=1)  # (B, gnn_hidden_dim)
        
        # ========== 分类器 ==========
        logits = self.classifier(pooled)  # (B, num_classes)
        
        # ========== 返回中间特征（用于诊断）==========
        embeddings = {
            'node_feat_e': node_feat_e.detach(),      # Phase E 节点特征
            'node_feat_h': node_feat_h.detach(),      # Phase H 节点特征
            'node_feat_fused': node_feat_fused.detach(),  # 融合后特征
            'transformer': pooled.detach()            # Transformer 输出
        }
        
        return logits, embeddings


class GraphConvLayer(nn.Module):
    """图卷积层（与单分支一致）"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, X, adj):
        """
        Args:
            X: (B, N, in_dim) - 节点特征
            adj: (B, N, N) - 邻接矩阵
        
        Returns:
            (B, N, out_dim) - 更新后的节点特征
        """
        # 归一化邻接矩阵（度归一化）
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / degree
        
        # 聚合邻居特征：X' = A_norm @ X
        aggregated = torch.bmm(adj_norm, X)  # (B, N, in_dim)
        
        # 线性变换 + 激活
        out = self.linear(aggregated)
        out = F.relu(out)
        
        return out
