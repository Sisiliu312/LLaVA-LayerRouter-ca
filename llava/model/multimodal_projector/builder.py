import os
import numpy as np
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from datetime import datetime
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        # self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_norm = LlamaRMSNorm(feature_dim)
        # self.output_norm = LlamaRMSNorm(feature_dim)
        self.q_norm = LlamaRMSNorm(feature_dim)
        self.k_norm = LlamaRMSNorm(feature_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(self, text, features):
        # ============ 步骤1: 检查输入 ============
        # print(f"\n{'='*60}")
        # print("🔍 CrossAttention Debug:")
        # print(f"  text shape: {text.shape}, range: [{text.min():.4f}, {text.max():.4f}]")
        # print(f"  features shape: {features.shape}, range: [{features.min():.4f}, {features.max():.4f}]")
        
        # ============ 步骤2: 线性变换 ============
        features = self.feature_norm(features)
        Q = self.W_q(text)
        K = self.W_k(features)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # print(f"  After linear transform:")
        # print(f"    Q range: [{Q.min():.4f}, {Q.max():.4f}], has_nan={torch.isnan(Q).any()}, has_inf={torch.isinf(Q).any()}")
        # print(f"    K range: [{K.min():.4f}, {K.max():.4f}], has_nan={torch.isnan(K).any()}, has_inf={torch.isinf(K).any()}")
        
        # ✅ 检查权重
        # print(f"  Weight stats:")
        # print(f"    W_q.weight: range=[{self.W_q.weight.min():.4f}, {self.W_q.weight.max():.4f}], norm={self.W_q.weight.norm():.4f}")
        # print(f"    W_k.weight: range=[{self.W_k.weight.min():.4f}, {self.W_k.weight.max():.4f}], norm={self.W_k.weight.norm():.4f}")
        
        # ============ 如果Q或K已经有Inf，直接返回零 ============
        # if torch.isinf(Q).any() or torch.isinf(K).any():
        #     print("⚠️ Q or K contains Inf! Returning zeros to avoid crash.")
        #     return torch.zeros_like(text)
        
        # ============ 步骤3: 计算attention scores ============
        attn_scores = torch.matmul(Q, K.transpose(1, 2))
        
        # if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
        #     print(f"⚠️ attn_scores after matmul: has_nan={torch.isnan(attn_scores).any()}, has_inf={torch.isinf(attn_scores).any()}")
        #     print(f"   attn_scores range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")
        
        attn_scores = attn_scores / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, features)
        # attended = self.output_norm(attended)
        
        # print(f"{'='*60}\n")
        
        return attended

# class LayerSelectionRouter(nn.Module):
#     """
#     Router that selects top-5 layers from 24 vision tower layers.
#     Initialized to uniformly select layers [1, 6, 11, 16, 21] (0-indexed: [0, 5, 10, 15, 20])
#     """
#     def __init__(self, dim, num_layers, top_router):
#         super(LayerSelectionRouter, self).__init__()
#         self.num_layers = num_layers
#         self.top_router = top_router
#         self.dim = dim

#         # print(f"Initializing LayerSelectionRouter with dim={self.dim}, num_layers={self.num_layers}, top_router={self.top_router}")
        
#         # Router network with SiLU gating (matching diagram: W1, W2, W3)
#         self.w1 = nn.Linear(dim, dim)
#         self.w2 = nn.Linear(dim, dim)
#         self.w3 = nn.Linear(dim, num_layers)
        
#         # Initialize to favor uniform selection of [0, 5, 10, 15, 20]
#         self._reset_parameters()
#         # print('初始化时 self.w1 mean:', self.w1.weight.norm().item())
#         print('初始化时 self.w1 shape:', self.w1.weight.shape)
#         # print("初始化时 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
        
#     def _reset_parameters(self):
#         """Initialize router to uniformly select layers [1, 6, 11, 16, 21] (0-indexed)"""
        
#         # checkpoint = torch.load('/hy-tmp/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin')
#         # print("保存的参数:")
#         # for key in checkpoint.keys():
#         #     print(f"  {key}: {checkpoint[key].shape}")

#         nn.init.xavier_uniform_(self.w1.weight)
#         nn.init.xavier_uniform_(self.w2.weight)
#         nn.init.xavier_uniform_(self.w3.weight)
#         nn.init.constant_(self.w1.bias, 0.0)
#         nn.init.constant_(self.w2.bias, 0.0)
#         nn.init.constant_(self.w3.bias, -0.1)
        
#         with torch.no_grad():
#             uniform_indices = [3, 8, 13, 18, 23]
#             for idx in uniform_indices:
#                 self.w3.bias[idx] = 0.1
                    
    
#     def forward(self, text_features):
#         """
#         Args:
#             text_features: Text token features [batch_size, text_len, dim]
        
#         Returns:
#             layer_weights: Softmax weights for all layers [batch_size, num_layers]
#             selected_indices: Indices of top-k selected layers [batch_size, top_router]
#         """
#         # [batch_size, text_len, dim] -> [batch_size, dim]
#         pooled = text_features.mean(dim=1)
#         # self.attention_pool = nn.Sequential(
#         #     nn.Linear(dim, 1),
#         #     nn.Softmax(dim=1)
#         # )
        
#         # print('fordward时 self.w1 mean:', self.w1.weight.norm().item())
#         print('fordward时 self.w1 shape:', self.w1.weight.shape)
#         h1 = F.silu(self.w1(pooled))
        
#         h2 = F.silu(self.w2(pooled))
        
#         gated = h1 * h2
        
#         # print("forward时 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
#         logits = self.w3(gated)
        
#         layer_probs = F.softmax(logits, dim=-1)

#         top_weights, top_indices = torch.topk(layer_probs, self.top_router, dim=-1)
#         top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
#         return top_indices, top_weights, layer_probs

# class LayerSelectionRouter(nn.Module):
#     """
#     Router that selects top-5 layers from 24 vision tower layers.
#     Initialized to uniformly select layers [1, 6, 11, 16, 21] (0-indexed: [0, 5, 10, 15, 20])
#     """
#     def __init__(self, dim, num_layers, top_router):
#         super(LayerSelectionRouter, self).__init__()
#         self.num_layers = num_layers
#         self.top_router = top_router
#         self.dim = dim
        
        
#         self.router = nn.Sequential(
#             nn.Linear(dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_layers)
#         )
        
#         # Initialize to favor uniform selection
#         self._reset_parameters()
        
#     def _reset_parameters(self):
#         """Initialize router to uniformly select layers [1, 6, 11, 16, 21] (0-indexed)"""
#         for m in self.router.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.0)  # ✅ 先全部初始化为 0
        
#         # ✅ 只在最后一层的 bias 上设置偏好
#         uniform_indices = [2, 7, 12, 17, 22]
#         with torch.no_grad():
#             for idx in uniform_indices:
#                 self.router[-1].bias[idx] = 1  # ✅ 给这些层更高的初始权重
    
#     def forward(self, text_features):
#         batch_size, text_len, dim = text_features.shape
        
#         # 每个 token 都预测
#         logits = self.router(text_features)  # [batch_size, text_len, num_layers]
#         layer_probs = F.softmax(logits, dim=-1)
#         layer_probs = layer_probs[0]  # [text_len, num_layers]
        
#         # 投票: 聚合所有 token 的预测
#         aggregated_probs = layer_probs.mean(dim=0)  # [num_layers]
        
#         top_weights, top_indices = torch.topk(aggregated_probs, self.top_router, dim=-1)
#         top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
#         return top_indices, top_weights, aggregated_probs

class AttentionLayerRouter(nn.Module):
    """
    使用 Attention 聚合 token 信息，而不是简单平均
    """
    def __init__(self, dim, num_layers, top_router):
        super(AttentionLayerRouter, self).__init__()
        self.num_layers = num_layers
        self.top_router = top_router
        self.dim = dim
        
        # Attention pooling for text tokens
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, num_layers)
        )
        
        # ✅ 修复：层位置编码应该是 [num_layers, dim] 而不是 [num_layers, 256]
        # self.layer_pos_embed = nn.Parameter(torch.randn(num_layers, dim) * 0.02)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.router[0].weight, gain=0.5)
        nn.init.constant_(self.router[0].bias, 0.0)
        
        with torch.no_grad():
            self.router[2].weight.normal_(0, 0.0001)
            uniform_indices = [22]  # 对应层 [3, 8, 13, 18, 23] 
            for idx in uniform_indices:
                self.router[2].bias[idx] = 1
    
    def compute_diversity_loss(self, layer_probs):
        """
        修正版多样性损失：确保始终为正
        """
        # 1. 组间平衡损失（浅/中/深层均匀分布）
        shallow_prob = layer_probs[:, 0:8].sum(dim=-1)
        middle_prob = layer_probs[:, 8:16].sum(dim=-1)
        deep_prob = layer_probs[:, 16:24].sum(dim=-1)
        
        ideal_prob = 1.0 / 3.0
        group_balance_loss = (
            (shallow_prob - ideal_prob) ** 2 +
            (middle_prob - ideal_prob) ** 2 +
            (deep_prob - ideal_prob) ** 2
        ).mean()
        
        # 2. 熵损失（鼓励在所有层之间均匀分布）
        epsilon = 1e-10
        entropy = -(layer_probs * torch.log(layer_probs + epsilon)).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(float(self.num_layers), 
                        device=layer_probs.device, 
                        dtype=layer_probs.dtype)
        )
        
        # 归一化熵损失：[0, 1]，越接近0越均匀
        normalized_entropy = 1.0 - (entropy / max_entropy)
        entropy_loss = normalized_entropy.mean()
        
        # 3. 组合损失
        total_loss = group_balance_loss + 0.3 * entropy_loss
        
        return total_loss

    def forward(self, text_features, return_loss=False):
        """
        Args:
            text_features: [batch_size, text_len, dim]
            return_loss: 是否返回多样性损失
        
        Returns:
            如果 return_loss=False: (top_indices, top_weights, layer_probs)
            如果 return_loss=True: (top_indices, top_weights, layer_probs, diversity_loss)
        """
        batch_size, text_len, dim = text_features.shape
        
        # ✅ Attention pooling
        attn_weights = self.attention_pool(text_features)
        attn_weights = F.softmax(attn_weights, dim=1)
        # print("Attention weights shape:", attn_weights.shape)
        pooled_features = (text_features * attn_weights).sum(dim=1)
        pooled_features = F.normalize(pooled_features, p=2, dim=-1) * (self.dim ** 0.5)
        # print("Pooled features shape:", pooled_features.shape)
        
        # Router 预测
        logits = self.router(pooled_features)
        # print("Router logits shape:", logits.shape)
        temperature = 2.0
        layer_probs = F.softmax(logits / temperature, dim=-1)
        
        # ✅ 加入层位置信息
        # layer_pos_logits = torch.matmul(pooled_features, self.layer_pos_embed.t())
        # logits = logits + 0.1 * layer_pos_logits
        
        # layer_probs = F.softmax(logits, dim=-1)
        
        top_weights, top_indices = torch.topk(layer_probs, self.top_router, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        full_layer_probs = layer_probs
        
        if batch_size > 1:
            top_indices = top_indices[0]
            top_weights = top_weights[0]
            layer_probs = layer_probs[0]
        else:
            top_indices = top_indices.squeeze(0)
            top_weights = top_weights.squeeze(0)
            layer_probs = layer_probs.squeeze(0)
        
        if return_loss:
            diversity_loss = self.compute_diversity_loss(full_layer_probs)
            return top_indices, top_weights, layer_probs, diversity_loss
        
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
    # print("=" * 60)
    # print("🔍 build_layer_router 调试:")
    # print(f"  hasattr(config, 'dim'): {hasattr(config, 'dim')}")
    # print(f"  hasattr(config, 'num_layers'): {hasattr(config, 'num_layers')}")
    # print(f"  hasattr(config, 'top_router'): {hasattr(config, 'top_router')}")
    
    dim = config.dim if hasattr(config, 'dim') else 4096
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 24
    top_router = config.top_router if hasattr(config, 'top_router') else 5
    
    # print(f"  最终: dim={dim}, num_layers={num_layers}, top_router={top_router}")
    # print(f"  类型: dim={type(dim)}, num_layers={type(num_layers)}, top_router={type(top_router)}")
    # print("=" * 60)
    
    return AttentionLayerRouter(dim=dim, num_layers=num_layers, top_router=top_router)