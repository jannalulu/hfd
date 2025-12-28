import math
from typing import Optional, Any
import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

from logger import print0 as print

class SDPAAttention(torch.nn.Module):
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

        sdpa_kwargs = {}
        if num_key_value_groups > 1:
            use_gqa_in_sdpa = attention_mask is None
            if not use_gqa_in_sdpa:
                key = repeat_kv(key, num_key_value_groups)
                value = repeat_kv(value, num_key_value_groups)
            else:
                sdpa_kwargs = {"enable_gqa": True}

        # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
        is_causal = kwargs.get('is_causal', False)

        # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
        # - Not in decoding phase (otherwise we want full attention on the single query token)
        # - Attention mask is not to be provided (even if it is a causal pattern)
        # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
        #
        # Quirks on the conditionals:
        # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
        #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
        # - It is important to check first for the shape, otherwise compile will fail with
        #   `argument 'is_causal' must be bool, not SymBool`.
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

        # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
        # We convert it to a bool for the SDPA kernel that only accepts bools.
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()

        # # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
        # # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
        # # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
        # if _is_torch_npu_available:
        #     if attention_mask is not None and attention_mask.dtype != torch.bool:
        #         # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
        #         attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output
