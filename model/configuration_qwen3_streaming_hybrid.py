from .wrap_hf import create_config_class

Qwen3StreamingHybridConfigParent = create_config_class(base_configuration_path='transformers.models.qwen3.configuration_qwen3.Qwen3Config', )

class Qwen3StreamingHybridConfig(Qwen3StreamingHybridConfigParent):
    def __init__(self, streaming_sliding_window:int|None = None, **kwargs):
        super().__init__(**kwargs)
        self.streaming_sliding_window = streaming_sliding_window
