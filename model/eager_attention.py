import math
from typing import Optional, Any
import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from logger import print0 as print

class EagerAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    @torch.compile
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,

        use_cache: bool,
        hidden_states: torch.Tensor,
        position_embeddings,
        past_key_values,
        cache_position,
        **kwargs
    ):
        x = hidden_states
        B, T, C = x.shape
        B, H, T, N = query.shape
        B, KVH, T, N = key.shape
        num_key_value_groups = H // KVH
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output
