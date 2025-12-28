from .wrap_hf import create_config_class

Qwen2StreamingHybridConfigParent = create_config_class(base_configuration_path='transformers.models.qwen2.configuration_qwen2.Qwen2Config', )

class Qwen2StreamingHybridConfig(Qwen2StreamingHybridConfigParent):
    def __init__(self, streaming_sliding_window:int|None = None, **kwargs):
        super().__init__(**kwargs)
        self.streaming_sliding_window = streaming_sliding_window
