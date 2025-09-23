import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional   # ✅ 新增
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x
    
# 追加到文件末尾（或放在 MLMHead/MPPHead 之后均可）

class LMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, weight=None,
                 bias=False, use_norm=True, learnable_temp=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5) if use_norm else nn.Identity()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=bias)
        if weight is not None:            # weight tying
            self.decoder.weight = weight
        self.logit_scale = nn.Parameter(torch.tensor(1.0)) if learnable_temp else None

    def forward(self, x):
        x = self.norm(x)
        logits = self.decoder(x)
        if self.logit_scale is not None:
            logits = self.logit_scale * logits
        return logits

# class ProjectionHead(nn.Module):
#     """
#     两层 MLP 投影头： (可选)LayerNorm -> Linear -> GELU -> (Dropout) -> Linear
#     用于将隐空间投到对比空间（dim = out_dim）。
#     """
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         hidden_dim: Optional[int] = None,   # ✅ 兼容老版本 Python
#         use_ln: bool = True,
#         dropout: float = 0.0,
#         activation: str = "gelu",
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or in_dim

#         self.pre_norm = nn.LayerNorm(in_dim, eps=1e-5) if use_ln else nn.Identity()
#         self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
#         self.act = nn.ReLU(inplace=True) if activation.lower() == "relu" else nn.GELU()
#         self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
#         self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

#     def forward(self, x):
#         x = self.pre_norm(x)
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         return x


class ProjectionHead(nn.Module):
    """
    简洁的投影头：LN -> Linear ；将隐空间投到对比空间（dim = proj_dim）
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,   # ✅ 兼容老版本 Python
        use_ln: bool = True,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-5) if use_ln else nn.Identity()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(self.norm(x))


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
