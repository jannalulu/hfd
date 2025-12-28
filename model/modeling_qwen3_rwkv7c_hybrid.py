from .rwkv7c_attention import RWKV7cAttention
from .wrap_hf import create_model_class, StaticStateCacheLayer

Qwen3RWKV7cHybridForCausalLM = create_model_class(
    replacement_attention_class=RWKV7cAttention, 
    base_model_path='transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM', 
    base_attention_path='transformers.models.qwen3.modeling_qwen3.Qwen3Attention',
    replacement_cache_layer_class=StaticStateCacheLayer)
