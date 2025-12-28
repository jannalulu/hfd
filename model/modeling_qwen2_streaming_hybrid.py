from .streaming_attention import StreamingAttention
from .wrap_hf import create_model_class, StaticStateCacheLayer

Qwen2StreamingHybridForCausalLM = create_model_class(
    replacement_attention_class=StreamingAttention, 
    base_model_path='transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM', 
    base_attention_path='transformers.models.qwen2.modeling_qwen2.Qwen2Attention',
    replacement_cache_layer_class=None)
