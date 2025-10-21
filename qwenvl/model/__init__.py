import torch.nn as nn

MODEL_TO_LORA_EXCLUDE_MODULES = {
    "qwen2.5vl": ["audio.layers"],
    "qwen2.5omni": ["audio_tower"],
    "qwen2audio": ["audio_tower", "multi_modal_projector"]
}


class ModelTypeMixin:
    @property
    def model_type(self):
        raise NotImplementedError


class ModelProtocol(ModelTypeMixin, nn.Module):
    """
    A protocol class that inherits both `ModelTypeMixin` and `nn.Module`
    """
    ...
