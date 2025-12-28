from .streaming_attention import StreamingAttention
from .wrap_hf import create_model_class, StaticStateCacheLayer

Qwen3StreamingHybridForCausalLM = create_model_class(
    replacement_attention_class=StreamingAttention, 
    base_model_path='transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM', 
    base_attention_path='transformers.models.qwen3.modeling_qwen3.Qwen3Attention',
    replacement_cache_layer_class=None)
