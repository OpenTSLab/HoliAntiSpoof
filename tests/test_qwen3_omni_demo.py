import torch
import soundfile as sf
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from transformers import Qwen3OmniMoeForConditionalGeneration

MODEL_PATH = "/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen3-Omni-30B-A3B-Instruct"

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role":
            "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "archive/cars.jpg"
            # },
            # {
            #     "type": "audio",
            #     "audio": "archive/cough.wav"
            # },
            {
                "type": "video",
                "video": "archive/draw.mp4"
            },
            {
                "type": "text",
                "text": "What can you see and hear? Answer in one short sentence."
            },
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO
)

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)

video_token_id = model.thinker.config.video_token_id
audio_token_id = model.thinker.config.audio_token_id
vision_start_token_id = model.thinker.config.vision_start_token_id
audio_start_token_id = model.thinker.config.audio_start_token_id
vision_end_token_id = model.thinker.config.vision_end_token_id
audio_end_token_id = model.thinker.config.audio_end_token_id
input_ids = torch.Tensor([[
    34,
    45,
    46,
    57,
    vision_start_token_id,
    audio_start_token_id,
] + [video_token_id] * 4 + [audio_token_id] * 25 + [video_token_id] * 4 + [audio_token_id] * 25 +
                          [vision_end_token_id, audio_end_token_id, 77]])
position_ids, rope_deltas = model.thinker.get_rope_index(
    input_ids, None,
    torch.Tensor([[2, 4, 4]]).long(), torch.ones_like(input_ids), True,
    torch.Tensor([382]).long(),
    torch.Tensor([2.0]).float()
)

import ipdb

ipdb.set_trace()

inputs = inputs.to(model.device).to(model.dtype)
text_ids, audio = model.generate(
    **inputs,
    talker_do_sample=False,
    thinker_return_dict_in_generate=True,
    thinker_max_new_tokens=8192,
    thinker_do_sample=False,
    speaker="Ethan",
    use_audio_in_video=USE_AUDIO_IN_VIDEO
)
response = processor.batch_decode(
    text_ids.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print("response: ", response)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
