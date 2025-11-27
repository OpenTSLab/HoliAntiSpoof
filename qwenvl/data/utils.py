import io
import torch
import torchaudio
from typing import TYPE_CHECKING


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
    waveform, orig_sr = torchaudio.load(io.BytesIO(bytes_data))
    return waveform, orig_sr
