import base64
import torch
import torchaudio
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import audioread
import librosa
import numpy as np
from qwen_omni_utils.v2_5.vision_process import process_vision_info
from qwen_omni_utils.v2_5.audio_process import SAMPLE_RATE, _check_if_video_has_audio


def is_petrel_client_available():
    try:
        import petrel_client
        return True
    except ImportError:
        return False


if TYPE_CHECKING:
    from petrel_client.client import Client


def load_audio_from_petrel_oss(audio_path: str, client: "Client") -> tuple[torch.Tensor, int]:
    bytes_data = client.get(audio_path)
    waveform, orig_sr = torchaudio.load(BytesIO(bytes_data))
    return waveform, orig_sr


def process_audio_info(
    conversations: list[dict] | list[list[dict]], use_audio_in_video: bool, petrel_client: Optional["Client"] = None
):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE *
                                         audio_start):None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://"):]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://"):]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue

                if data.startswith("s3://"):
                    waveform, orig_sr = load_audio_from_petrel_oss(data, petrel_client)
                    waveform = torchaudio.functional.resample(waveform, orig_sr, SAMPLE_RATE)
                    if len(waveform.shape) == 2:
                        waveform = waveform.mean(0)
                    waveform = waveform[int(SAMPLE_RATE *
                                            audio_start):None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                    waveform = waveform.numpy()
                else:
                    waveform = librosa.load(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None
                    )[0]

                audios.append(waveform)
    if len(audios) == 0:
        audios = None
    return audios


def process_mm_info(
    conversations, use_audio_in_video, return_video_kwargs=False, petrel_client: Optional["Client"] = None
):
    audios = process_audio_info(conversations, use_audio_in_video, petrel_client)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios, ) + vision
