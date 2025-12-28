import math
from typing import Optional, Any
import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from logger import print0 as print

class StreamingAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.sliding_window = config.streaming_sliding_window

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
        q, k, v = query, key, value
        del query, key, value
        B, H, L, N = q.shape
        B, KVH, S, N = k.shape
        
        sdpa_kwargs = {}

        if H > KVH:
            if attention_mask is None:
                sdpa_kwargs = {"enable_gqa": True}
            else:
                k = repeat_kv(k, H // KVH)
                v = repeat_kv(v, H // KVH)

        sliding_window = self.sliding_window
        if sliding_window is not None and sliding_window > 0:
            q_idx = torch.arange(S-L, S, device=q.device)[None, None, :, None]
            kv_idx = torch.arange(S, device=q.device)[None, None, None, :]
            window_mask = kv_idx >= q_idx - sliding_window
            if attention_mask is not None:
                assert attention_mask.dtype == torch.bool
                sink_indices = (S - attention_mask.view(B,L,S)[:,-1,:].sum(dim=-1)).view(B) # sink offset per batch idx
                sink_mask = kv_idx == sink_indices.view(B,1,1,1)
                #prefill_mha_mask = q_idx >= S - prefill_n_mha_tokens
                #attention_mask = attention_mask & (sink_mask | window_mask | prefill_mha_mask)
                attention_mask = attention_mask & (sink_mask | window_mask)
            else:
                sink_mask = kv_idx == 0
                attention_mask = kv_idx <= q_idx # causal
                attention_mask = attention_mask & (sink_mask | window_mask)

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, :S]

        is_causal = L > 1 and attention_mask is None
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()

        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal, scale=scaling, **sdpa_kwargs)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output
