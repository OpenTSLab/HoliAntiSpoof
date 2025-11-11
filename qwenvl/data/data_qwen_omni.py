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
import torch
import torchaudio
from PIL import Image

from decord import VideoReader, cpu
import transformers
from transformers import PreTrainedTokenizer, WhisperFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from qwenvl.data.processing_qwen2_audio import Qwen2AudioProcessor
from qwenvl.train.utils import rank0_print

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


@dataclass(kw_only=True)
class MMQwenDatasetBase:
    stage: Literal["training", "inference"] = field(default="training")
    train_type: Literal["sft", "dpo"] = field(default="sft")

    model_type: str
    processor: Qwen2_5OmniProcessor | Qwen3OmniMoeProcessor

    dataset_list: list[str]
    dataset_max_samples: int = field(default=None)

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        assert self.stage in ("training", "inference")
        self.load_raw_data()

    def replace_image_token(self) -> None:
        """
        Replace <image> token with <audio> tokens in the raw data.
        """
        # FIXME "<audio>" should be "<audio>", not "<image>"
        for d in self.list_data_dict:
            if d["conversations"][0]["from"] == "system":
                idx = 1
            else:
                idx = 0
            input_dict = d["conversations"][idx]
            if "<image>" in input_dict["value"] and not "image" in d and "video" in d:
                input_dict["value"] = input_dict["value"].replace("<image>", "<video>")
            if "<image>" in input_dict["value"] and not "image" in d and not "video" in d and "audio" in d:
                input_dict["value"] = input_dict["value"].replace("<image>", "<audio>")

    def load_raw_data(self, ) -> None:
        rank0_print(f"Loading datasets: {self.dataset_list}")
        list_data_dict = []

        for data in self.dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            if self.dataset_max_samples is not None:
                random.shuffle(annotations)
                annotations = annotations[:self.dataset_max_samples]
            list_data_dict += annotations

        self.list_data_dict = list_data_dict
        self.replace_image_token()
        rank0_print(f"Total samples: {len(self.list_data_dict)}")

    def _get_item(self, i) -> dict[str, Any]:
        source = self.list_data_dict[i]

        data_dict = {
            # "index": i,
            "prompt": source["conversations"][:-1],
            "ref": source["conversations"][-1]["value"],
        }

        feat_id_label_dict: dict = self.build_feat_id_label(source)

        data_dict.update(feat_id_label_dict)

        return data_dict

    def __getitem__(self, i) -> dict[str, Any]:
        try:
            sample = self._get_item(i)
        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            if self.stage == "inference":
                rank0_print(f"Error loading {self.list_data_dict[i]}")
                raise e
            else:
                return self._get_item(random.randint(0, len(self) - 1))
        return sample

    @abstractmethod
    def build_feat_id_label(self, data_item: dict) -> dict[str, Any]:
        if self.model_type == "qwen2.5omni":
            system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        else:
            system_prompt = ""
        conversations = self.build_conversation(data_item, system_prompt)
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        res = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v[0]

        labels = self.build_label(text)

        if self.stage == "inference":
            len_input = sum(labels == IGNORE_INDEX)
            res["input_ids"] = res["input_ids"][:len_input]
            res["attention_mask"] = torch.ones_like(res["input_ids"])
        else:
            res["labels"] = labels

        return res

    def build_conversation(self, data_item: dict, system_prompt: str | None = None) -> list[dict]:
        source = copy.deepcopy(data_item["conversations"])
        if system_prompt is not None and source[0]["from"] != "system":
            conversations = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        else:
            conversations = []

        for conv in source:
            if conv["from"] == "system":
                conversations.append({"role": "system", "content": [{"type": "text", "text": conv["value"]}]})
            elif conv["from"] == "human":
                if "<video>\n" in conv["value"]:
                    conversations.append({
                        "role":
                            "user",
                        "content": [{
                            "type": "video",
                            "video": data_item["video"]
                        }, {
                            "type": "text",
                            "text": conv["value"].replace("<video>\n", "")
                        }]
                    })
                elif "<audio>\n" in conv["value"]:
                    conversations.append({
                        "role":
                            "user",
                        "content": [{
                            "type": "audio",
                            "audio": data_item["audio"]
                        }, {
                            "type": "text",
                            "text": conv["value"].replace("<audio>\n", "")
                        }]
                    })
                else:
                    conversations.append({"role": "user", "content": [{"type": "text", "text": conv["value"]}]})
            else:
                conversations.append({"role": "assistant", "content": [{"type": "text", "text": conv["value"]}]})

        return conversations

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
        data_dict["audio"] = self.list_data_dict[i]["audio"]
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
        from qwenvl.model.processing_qwen_omni_spoofing import (
            Qwen2_5OmniWithSpoofingProcessor, Qwen3OmniMoeWithSpoofingProcessor
        )
        assert isinstance(self.processor, (Qwen2_5OmniWithSpoofingProcessor, Qwen3OmniMoeWithSpoofingProcessor))
        return super().__post_init__()

    def load_raw_data(self, ) -> None:
        rank0_print(f"Loading datasets: {self.dataset_list}")
        list_data_dict = []

        for data in self.dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            if self.dataset_max_samples is not None:
                random.shuffle(annotations)
                annotations = annotations[:self.dataset_max_samples]
            list_data_dict += annotations

            dataset_name = Path(data).parent.parts[-1]
            for anno in annotations:
                anno["embedding_file"] = dataset_name

        self.list_data_dict = list_data_dict
        self.replace_image_token()
        rank0_print(f"Total training samples: {len(self.list_data_dict)}")

    def build_feat_id_label(self, data_item):
        res = super().build_feat_id_label(data_item)
        embed_file = Path(self.embedding_dir) / f"{data_item['embedding_file']}.h5"
        with h5py.File(embed_file, "r") as hf:
            embedding = hf[Path(data_item["audio"]).name][()]
            embedding = np.array(embedding, dtype=np.float32)
        res["spoof_embeds"] = embedding
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


@dataclass
class OmniCollator:
    padding_config: dict[str, int] = field(default_factory=lambda: {"input_ids": PAD_TOKEN_ID, "labels": IGNORE_INDEX})
    concat_keys: list[str] = field(
        default_factory=lambda: ["pixel_values_videos", "video_grid_thw", "input_features", "feature_attention_mask"]
    )
    torchify_keys: list[str] = field(default_factory=lambda: ["video_second_per_grid"])

    def __post_init__(self) -> None:
        default = {"input_ids": PAD_TOKEN_ID, "labels": IGNORE_INDEX}
        self.padding_config = {**default, **self.padding_config}

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Sequence[Any] | None]:
        collate_samples: dict[str, Any] = {k: [dic[k] for dic in instances] for k in instances[0]}
        batch_keys = list(collate_samples.keys())

        for key in batch_keys:
            if key in self.padding_config:
                data_batch = torch.nn.utils.rnn.pad_sequence(
                    collate_samples[key], batch_first=True, padding_value=self.padding_config[key]
                )
            elif key in self.concat_keys:
                if collate_samples[key][0] is not None:
                    data_batch = torch.cat(collate_samples[key], dim=0)
                else:
                    data_batch = None
            elif key in self.torchify_keys:
                if collate_samples[key][0] is not None:
                    data_batch = torch.tensor(np.array(collate_samples[key]))
                else:
                    data_batch = None
            else:
                data_batch = collate_samples[key]

            collate_samples[key] = data_batch

        collate_samples["attention_mask"] = collate_samples["input_ids"].ne(PAD_TOKEN_ID).to(torch.int64)

        return collate_samples


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
  - data/asvspoof2019/train.json
  - data/partial_spoof/train.json
processor:
  _target_: transformers.Qwen2_5OmniProcessor.from_pretrained
  pretrained_model_name_or_path: /mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B
data_format: "json"
    """

    collate_config_str = r"""
_target_: qwenvl.data.data_qwen_omni.OmniCollator
torchify_keys:
  - video_second_per_grid
  - spoof_embeds
    """
    config = OmegaConf.create(dataset_config_str)
    dataset = hydra.utils.instantiate(config, _convert_='all')
    item = dataset[0]

    config = OmegaConf.create(collate_config_str)
    collate_fn = hydra.utils.instantiate(config, _convert_='all')

    batch = collate_fn([item])
    breakpoint()
