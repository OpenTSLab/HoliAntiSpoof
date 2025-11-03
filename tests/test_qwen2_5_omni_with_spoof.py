from hydra.utils import instantiate
import torch

if __name__ == "__main__":

    config = {
        '_target_':
            'qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni_with_spoof.Qwen2_5OmniThinkerWithSpoofForConditionalGeneration.from_pretrained',
        'config': {
            '_target_':
                'qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniThinkerWithSpoofConfig.from_pretrained',
            'pretrained_model_name_or_path':
                '/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B',
            'spoof_embed_size':
                2048
        },
        'pretrained_model_name_or_path':
            '/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B',
        'attn_implementation':
            'flash_attention_2',
        'torch_dtype':
            torch.float16
    }
    model = instantiate(config)
    print(model)
