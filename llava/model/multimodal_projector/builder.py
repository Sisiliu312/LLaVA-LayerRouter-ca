import os
import numpy as np
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from datetime import datetime

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class AttentionWeightSaver:
    def __init__(self, save_dir='attention_weights', format='pt'):
        """
        初始化保存器
        :param save_dir: 保存目录路径
        :param format: 保存格式 ('pt' for PyTorch, 'npy' for NumPy)
        """
        self.save_dir = save_dir
        self.format = format
        self.counter = 0
        self._create_save_dir()
        
    def _create_save_dir(self):
        """创建保存目录"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def _generate_filename(self):
        """生成按序号递增的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attn_{self.counter:04d}_{timestamp}"
        self.counter += 1
        return os.path.join(self.save_dir, f"{filename}.{self.format}")
    
    def save(self, attn_weights, metadata=None):
        """
        保存attention weights
        :param attn_weights: 要保存的attention weights张量
        :param metadata: 可选的元数据字典
        """
        filename = self._generate_filename()
        
        if self.format == 'pt':
            # 保存为PyTorch文件
            save_dict = {'attn_weights': attn_weights}
            if metadata:
                save_dict['metadata'] = metadata
            torch.save(save_dict, filename)
        elif self.format == 'npy':
            # 保存为NumPy文件
            if isinstance(attn_weights, torch.Tensor):
                attn_weights = attn_weights.cpu().numpy()
            if metadata:
                np.savez(filename, attn_weights=attn_weights, **metadata)
            else:
                np.save(filename, attn_weights)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
            
        print(f"Saved attention weights to: {filename}")
        return filename
    
saver = AttentionWeightSaver(save_dir='/home/data/shika/LLaVA/playground/data/eval/textvqa', format='pt')

class CrossAttention(nn.Module):
    def __init__(self, text_dim, feature_dim):
        super(CrossAttention, self).__init__()
        # 初始化线性变换层
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.constant_(self.W_q.bias, 0.0)
        nn.init.constant_(self.W_k.bias, 0.0)

    def forward(self, text, features):
        """
        text: [batch, text_seq_len, text_dim] 文本特征
        features: [batch, num_patches, feature_dim] 图像特征
        返回: [batch, num_patches] 注意力权重
        """

        # 1. 线性变换
        Q = self.W_q(text)  # [batch, text_seq_len, feature_dim]
        K = self.W_k(features)  # [batch, num_patches, feature_dim]
        # print('Q mean:', Q.mean(), 'std:', Q.std())
        # print('K mean:', K.mean(), 'std:', K.std())
        # print(f"K min: {K.min()}, max: {K.max()}")

        # Q = Q.clamp(min=-50, max=50)
        # K = K.clamp(min=-50, max=50)
        
        # # 2. 应用层归一化
        # Q = self.q_norm(Q)
        # K = self.k_norm(K)
        # print('Q mean:', Q.mean(), 'std:', Q.std())
        # print('K mean:', K.mean(), 'std:', K.std())
        # print(f"K min: {K.min()}, max: {K.max()}")
        
        # 3. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(1, 2))  # [batch, text_seq_len, num_patches]

        # ✅ 注意：这里默认不再进行缩放
        attn_scores = attn_scores / (K.size(-1) ** 0.5)
        # print("attn_scores std:", attn_scores.std().item())
        # print("attn_scores min:", attn_scores.min().item())
        # print("attn_scores max:", attn_scores.max().item())

        # 4. 应用 softmax 获取注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # print("权重为:",attn_weights.mean())
        
        # saver.save(attn_weights)

        attended = torch.matmul(attn_weights, features)  # [B, T, D]

        return attended

class LayerSelectionRouter(nn.Module):
    """
    Router that selects top-5 layers from 24 vision tower layers.
    Initialized to uniformly select layers [1, 6, 11, 16, 21] (0-indexed: [0, 5, 10, 15, 20])
    """
    def __init__(self, dim, num_layers, top_k):
        super().__init__()
        self.num_layers = num_layers
        self.top_k = 5
        self.dim = dim

        print(f"Initializing LayerSelectionRouter with dim={self.dim}, num_layers={self.num_layers}, top_k={self.top_k}")
        
        # Router network with SiLU gating (matching diagram: W1, W2, W3)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, num_layers)
        
        # Initialize to favor uniform selection of [0, 5, 10, 15, 20]
        self._initialize_uniform_selection()
        print("初始化后 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
        
    def _initialize_uniform_selection(self):
        """Initialize router to uniformly select layers [1, 6, 11, 16, 21] (0-indexed)"""
        with torch.no_grad():
            # Target layer indices (0-indexed for [1, 6, 11, 16, 21])
            uniform_indices = [0, 5, 10, 15, 20]
            
            # Initialize W3 bias with large negative values
            self.w3.bias.fill_(-10.0)
            
            # Set positive bias for desired layers
            for idx in uniform_indices:
                self.w3.bias[idx] = 10.0
            
            # Initialize weights to small values for stability
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.xavier_uniform_(self.w2.weight)
            nn.init.xavier_uniform_(self.w3.weight)
            nn.init.constant_(self.w1.bias, 0.0)
            nn.init.constant_(self.w2.bias, 0.0)
    
    def forward(self, text_features):
        """
        Args:
            text_features: Text token features [batch_size, text_len, dim]
        
        Returns:
            layer_weights: Softmax weights for all layers [batch_size, num_layers]
            selected_indices: Indices of top-k selected layers [batch_size, top_k]
        """
        # Average pool over sequence length to get representation
        # [batch_size, text_len, dim] -> [batch_size, dim]
        pooled = text_features.mean(dim=1)
        # self.attention_pool = nn.Sequential(
        #     nn.Linear(dim, 1),
        #     nn.Softmax(dim=1)
        # )
        
        # Apply gating mechanism (matching diagram architecture)
        # W1 with SiLU
        h1 = F.silu(self.w1(pooled))
        
        # W2 with SiLU  
        h2 = F.silu(self.w2(pooled))
        
        # Element-wise multiplication (circle with dot in diagram)
        gated = h1 * h2
        
        print("forward 前 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
        # W3 to get layer logits
        logits = self.w3(gated)  # [batch_size, num_layers]
        
        # Apply softmax to get layer selection probabilities
        layer_probs = F.softmax(logits, dim=-1)

        # Select top-k layers
        top_weights, top_indices = torch.topk(layer_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        return top_indices, top_weights, layer_probs

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_cross_attn(config):
    text_dim = config.text_dim if hasattr(config, 'text_dim') else 4096
    feature_dim = config.feature_dim if hasattr(config, 'feature_dim') else 4096
    
    return CrossAttention(text_dim=text_dim, feature_dim=feature_dim)

def build_layer_router(config):
    dim = config.dim if hasattr(config, 'dim') else 4096
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 24
    top_k = config.top_k if hasattr(config, 'top_k') else 5
    if hasattr(config, 'top_k'):
        print(f"Building LayerSelectionRouter with top_k={config.top_k}")
    
    
    return LayerSelectionRouter(dim=dim, num_layers=num_layers, top_k=top_k)