import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B")

conversation = [
    {
        "role":
            "system",
        "content": [{
            "type":
                "text",
            "text":
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        }],
    },
    {
        "role":
            "user",
        "content": [{
            "type": "video",
            "video": "archive/draw.mp4"
        }, {
            "type": "text",
            "text": "Briefly describe the content in this video."
        }],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
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

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
