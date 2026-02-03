"""
Refactor target:
1. use the processor inner function to calculate audio feature length, instead of hard coding
2. use processor.apply_chat_template, instead of manual applying
"""

from abc import abstractmethod
import copy
import json
import random
from pathlib import Path
import math
import itertools
from dataclasses import dataclass, field
from typing import Dict, Literal, Sequence, List, Any
from collections.abc import Sequence

import h5py
import numpy as np
import librosa
import torch
import torchaudio
import transformers
from glom import glom
from PIL import Image
from decord import VideoReader, cpu
from transformers import PreTrainedTokenizer, WhisperFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeProcessor
# from qwen_omni_utils import process_mm_info

from qwenvl.data.data_qwen import ExcessiveDurationError
from qwenvl.data.processing_qwen2_audio import Qwen2AudioProcessor
from qwenvl.data.utils import is_petrel_client_available, process_mm_info, load_audio_from_petrel_oss
from qwenvl.data.constants import get_dataset_name, get_audio_id

from qwenvl.train.utils import rank0_print

if is_petrel_client_available():
    from petrel_client.client import Client

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
PAD_TOKEN_ID = 151643
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"


def read_jsonl(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class AudioProcessingMixin:

    petrel_client: Client | None

    def get_audio_duration(self, path: str):
        if path.startswith("s3://"):
            waveform, sr = load_audio_from_petrel_oss(path, self.petrel_client)
            return waveform.shape[1] / sr
        else:
            return librosa.get_duration(path=path)


@dataclass(kw_only=True)
class MMQwenDatasetBase(AudioProcessingMixin):
    stage: Literal["training", "inference"] = field(default="training")
    train_type: Literal["sft", "dpo"] = field(default="sft")

    model_type: str
    processor: Qwen2_5OmniProcessor | Qwen3OmniMoeProcessor

    dataset_list: list[str]
    dataset_max_samples: int = field(default=None)
    petrel_oss_config: str | None = None
    max_duration: float | None = None

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        assert self.stage in ("training", "inference")
        if is_petrel_client_available() and self.petrel_oss_config:
            self.petrel_client = Client(self.petrel_oss_config)
        else:
            self.petrel_client = None
        self.load_raw_data()

    def load_raw_data(self, ) -> None:
        rank0_print(f"Loading datasets: {self.dataset_list}")
        list_data_dict = []

        for data in self.dataset_list:
            if isinstance(data, dict):
                data_idxs = data["idxs"]
                data = data["data"]
            else:
                data_idxs = None
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))

            if data_idxs is not None:
                data_idxs = np.load(data_idxs)
            else:
                if self.dataset_max_samples is not None:
                    all_idxs = np.arange(len(annotations))
                    np.random.shuffle(all_idxs)
                    data_idxs = all_idxs[:self.dataset_max_samples]
                else:
                    data_idxs = np.arange(len(annotations))
            annotations = [annotations[idx] for idx in data_idxs]
            list_data_dict += annotations

        self.list_data_dict = list_data_dict
        rank0_print(f"Total samples: {len(self.list_data_dict)}")

    def _get_item(self, i) -> dict[str, Any]:
        source = self.list_data_dict[i]

        audio_duration = 0.0
        for elem in source["conversations"][0]["content"]:
            if elem["type"] == "audio":
                audio_path = elem["audio"]
                audio_duration += self.get_audio_duration(audio_path)

        if self.max_duration is not None and audio_duration > self.max_duration:
            raise ExcessiveDurationError(
                f"Audio {audio_path} duration {audio_duration}s is longer than max duration {self.max_duration}s"
            )

        data_dict = {
            # "index": i,
            "prompt": glom(source, "conversations.0.content.-2.text"),
            "ref": glom(source, "conversations.-1.content.0.text"),
        }

        feat_id_label_dict: dict = self.build_feat_id_label(source)

        data_dict.update(feat_id_label_dict)

        return data_dict

    def __getitem__(self, i) -> dict[str, Any]:

        if self.stage == "inference":
            sample = self._get_item(i)
        else:
            get_item_success = False
            try:
                sample = self._get_item(i)
                get_item_success = True
            except ExcessiveDurationError as e:
                print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")

            while not get_item_success:
                try:
                    sample = self._get_item(random.randint(0, len(self) - 1))
                    get_item_success = True
                except ExcessiveDurationError as e:
                    print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
        return sample

    @abstractmethod
    def build_feat_id_label(self, data_item: dict) -> dict[str, Any]:
        if self.model_type == "qwen2.5omni":
            system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        else:
            system_prompt = ""

        conversations = []
        if system_prompt:
            conversations.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        conversations.extend(data_item["conversations"])

        text = self.processor.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=False, petrel_client=self.petrel_client
        )
        res, expanded_text = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        for k, v in res.items():
            if isinstance(v, torch.Tensor) and k not in ("input_features", "feature_attention_mask"):
                res[k] = v[0]

        expanded_text = expanded_text[0]
        labels = self.build_label(expanded_text)

        if self.stage == "inference":
            len_input = sum(labels == IGNORE_INDEX)
            res["input_ids"] = res["input_ids"][:len_input]
            res["attention_mask"] = torch.ones_like(res["input_ids"])
        else:
            res["labels"] = labels

        return res

    def build_label(
        self,
        text: str,
    ) -> torch.LongTensor:

        all_convs = text.split("<|im_start|>")
        labels = []
        for conv in all_convs:
            if conv:
                if conv.startswith("system\n") or conv.startswith("user\n"):
                    labels += [IGNORE_INDEX] * (
                        self.tokenizer(conv, padding=True, padding_side="left",
                                       return_tensors="pt")["input_ids"].size(1) + 1
                    )
                elif conv.startswith("assistant\n"):
                    labels += [IGNORE_INDEX] * 3  # <|im_start|>assistant\n
                    labels += self.tokenizer(
                        conv[len("assistant\n"):], padding=True, padding_side="left", return_tensors="pt"
                    )["input_ids"].tolist()[0]
                else:
                    raise NotImplementedError

        labels = torch.as_tensor(labels, dtype=torch.long)
        return labels


@dataclass(kw_only=True)
class AudioDataset(MMQwenDatasetBase):
    def __len__(self) -> int:
        return len(self.list_data_dict)

    def _get_item(self, i) -> dict[str, Any]:
        data_dict = super()._get_item(i)
        data_dict["audio"] = self.list_data_dict[i]["conversations"][0]["content"][-1]["audio"]
        return data_dict


@dataclass(kw_only=True)
class AudioSpoofingDataset(AudioDataset):

    data_format: str = "json"

    def _get_item(self, i) -> dict[str, Any]:
        # transform json to text
        source: dict[str, Any] = self.list_data_dict[i]
        data_dict = super()._get_item(i)
        if "keywords" in source:
            keywords = source["keywords"]
        else:
            keywords = None
        data_dict["keywords"] = keywords
        return data_dict


@dataclass(kw_only=True)
class AudioSpoofingWithEmbeddingDataset(AudioSpoofingDataset):
    embedding_dir: str

    def __post_init__(self):
        from qwenvl.data.processing_qwen_omni_spoofing import (
            Qwen2_5OmniWithSpoofingProcessor, Qwen3OmniMoeWithSpoofingProcessor
        )
        assert isinstance(self.processor, (Qwen2_5OmniWithSpoofingProcessor, Qwen3OmniMoeWithSpoofingProcessor))
        self.embedding_dir = Path(self.embedding_dir)
        return super().__post_init__()

    # def load_raw_data(self, ) -> None:
    #     rank0_print(f"Loading datasets: {self.dataset_list}")
    #     list_data_dict = []

    #     for data in self.dataset_list:
    #         if isinstance(data, dict):
    #             data_idxs = data["idxs"]
    #             data = data["data"]
    #         else:
    #             data_idxs = None
    #         file_format = data.split(".")[-1]
    #         if file_format == "jsonl":
    #             annotations = read_jsonl(data)
    #         else:
    #             annotations = json.load(open(data, "r"))

    #         if data_idxs is not None:
    #             data_idxs = np.load(data_idxs)
    #         else:
    #             if self.dataset_max_samples is not None:
    #                 all_idxs = np.arange(len(annotations))
    #                 np.random.shuffle(all_idxs)
    #                 data_idxs = all_idxs[:self.dataset_max_samples]
    #             else:
    #                 data_idxs = np.arange(len(annotations))
    #         annotations = annotations[:self.dataset_max_samples]
    #         list_data_dict += annotations

    #     self.list_data_dict = list_data_dict
    #     rank0_print(f"Total training samples: {len(self.list_data_dict)}")

    def build_feat_id_label(self, data_item):
        res = super().build_feat_id_label(data_item)

        spoof_embeds = []
        for elem in data_item["conversations"][0]["content"]:
            if elem["type"] == "audio":
                audio_path = elem["audio"]
                dataset_name = get_dataset_name(audio_path)
                embed_file = self.embedding_dir / f"{dataset_name}.h5"
                audio_id = get_audio_id(audio_path, dataset_name)
                with h5py.File(embed_file, "r") as hf:
                    embedding = hf[audio_id][()]
                    embedding = np.array(embedding, dtype=np.float32)
                    spoof_embeds.append(embedding)

        res["spoof_embeds"] = torch.as_tensor(np.stack(spoof_embeds))
        return res


@dataclass(kw_only=True)
class AudioVideoDataset(MMQwenDatasetBase):
    """Video dataset for supervised fine-tuning."""

    video_args: dict[str, Any]

    def _get_item(self, i) -> dict[str, torch.Tensor]:
        data_dict = super()._get_item(i)
        source = self.list_data_dict[i]

        data_dict.update({
            "video": source.get("video", None),
            "image": source.get("image", None),
            "audio": source.get("audio", None),
            "use_audio": source.get("use_audio", False),
        })
        return data_dict


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra

    random.seed(2025)

    dataset_config_str = r"""
_target_: qwenvl.data.data_qwen_omni.AudioSpoofingDataset
stage: training
train_type: sft
model_type: qwen2.5omni
dataset_list:
  - data_json/asvspoof2019/train_icl.json
#   - data_json/partial_spoof/train.json
processor:
  _target_: qwenvl.data.processing_qwen2_5_omni.Qwen2_5OmniProcessor.from_pretrained
  pretrained_model_name_or_path: /mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B
data_format: "json"
    """

    collate_config_str = r"""
_target_: qwenvl.data.data_qwen.OmniCollator
torchify_keys:
  - video_second_per_grid
  - spoof_embeds
    """
    config = OmegaConf.create(dataset_config_str)
    dataset = hydra.utils.instantiate(config, _convert_='all')
    item1 = dataset[0]
    item2 = dataset[5000]

    config = OmegaConf.create(collate_config_str)
    collate_fn = hydra.utils.instantiate(config, _convert_='all')

    batch = collate_fn([item1, item2])
    breakpoint()
