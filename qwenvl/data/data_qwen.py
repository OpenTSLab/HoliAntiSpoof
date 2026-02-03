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
from transformers import PreTrainedTokenizer, WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

# from qwenvl.data.rope2d import get_rope_index_25, get_rope_index_2
from qwenvl.data.utils import is_petrel_client_available, load_audio_from_petrel_oss
from qwenvl.data.constants import get_audio_id
from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwenvl.data.processing_qwen2_audio import Qwen2AudioProcessor
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


class ExcessiveDurationError(Exception):
    """Specific exception when the audio duration exceeds the limitation"""
    pass


@dataclass(kw_only=True)
class MMQwenDatasetBase:
    stage: Literal["training", "inference"] = field(default="training")
    train_type: Literal["sft", "dpo", "grpo"] = field(default="sft")

    model_type: str
    processor: ProcessorMixin = field(default=None)
    tokenizer: PreTrainedTokenizer = field(default=None)

    dataset_list: list[str]
    dataset_max_samples: int = field(default=None)

    @abstractmethod
    def set_processor(self, ) -> None:
        raise NotImplementedError

    def __post_init__(self, ) -> None:
        assert self.stage in ("training", "inference")
        self.set_processor()
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
        self.replace_image_token()
        rank0_print(f"Total samples: {len(self.list_data_dict)}")

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @abstractmethod
    def update_mm_feature(self, res_dict: dict, mm_data: dict[str, Any]) -> None:
        raise NotImplementedError

    def build_feat_id_label(self, source: dict, mm_data: dict[str, Any]) -> dict[str, Any]:
        convs = copy.deepcopy([source["conversations"]])
        conversations = self.build_conversation(convs)
        res = self.build_id_label_from_conversation(conversations, mm_data)
        self.update_mm_feature(res, mm_data)
        return res

    @abstractmethod
    def build_id_label_from_conversation(
        self,
        conversations: list[dict],
        mm_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Args:
            conversations: list of conversations, in Qwen chat template format
            mm_data: multimodal data
        Returns:
            feat_id_label_dict: dictionary of features and labels
        """
        raise NotImplementedError

    def _get_item(self, i) -> dict[str, Any]:
        source = self.list_data_dict[i]

        data_dict = {
            # "index": i,
            "prompt": source["conversations"][:-1],
        }
        if "value" in source["conversations"][-1] and source["conversations"][-1]["from"] == "gpt":
            data_dict["ref"] = source["conversations"][-1]["value"]

        # load multimodal data
        mm_data = self.load_multimodal_data(source)

        # load text and tokenization
        feat_id_label_dict: dict = self.build_feat_id_label(source, mm_data)

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
    def load_multimodal_data(self, source) -> dict[str, Any]:
        raise NotImplementedError

    def build_conversation(self, sources: list[dict]) -> list[dict]:
        if self.model_type == "qwen2.5omni":
            if sources[0][0]["from"] == "system":
                conversations = []
            else:
                conversations = [{
                    "role":
                        "system",
                    "content": [{
                        "type":
                            "text",
                        "text":
                            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    }]
                }]
            for conv in sources[0]:
                if conv["from"] == "system":
                    conversations.append({"role": "system", "content": [{"type": "text", "text": conv["value"]}]})
                elif conv["from"] == "human":
                    if "<video>\n" in conv["value"]:
                        conversations.append({
                            "role":
                                "user",
                            "content": [{
                                "type": "video"
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
                                "type": "audio"
                            }, {
                                "type": "text",
                                "text": conv["value"].replace("<audio>\n", "")
                            }]
                        })
                    else:
                        conversations.append({"role": "user", "content": [{"type": "text", "text": conv["value"]}]})
                else:
                    conversations.append({"role": "assistant", "content": [{"type": "text", "text": conv["value"]}]})
        elif self.model_type == "qwen2audio":
            conversations = []
            for conv in sources[0]:
                if conv["from"] == "system":
                    conversations.append({"role": "system", "content": [{"type": "text", "text": conv["value"]}]})
                elif conv["from"] == "human":
                    if "<audio>\n" in conv["value"]:
                        conversations.append({
                            "role":
                                "user",
                            "content": [{
                                "type": "audio"
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
        if self.model_type == "qwen2.5omni" or self.model_type == "qwen2audio":
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


@dataclass
class AudioProcessingMixin:

    petrel_oss_config: str | None = None
    max_duration: float | None = None

    def __post_init__(self, ) -> None:
        if is_petrel_client_available() and self.petrel_oss_config:
            self.petrel_client = Client(self.petrel_oss_config)
        else:
            self.petrel_client = None

    def load_audio(
        self,
        audio_source: str,
        sample_rate: int,
    ) -> np.ndarray:
        try:
            if audio_source.startswith("s3://"):
                audio, sr = load_audio_from_petrel_oss(audio_source, self.petrel_client)
            else:
                audio, sr = torchaudio.load(audio_source)
        except Exception as e:
            print(f"Loading {audio_source} failed: {e}")
            raise e
        if len(audio.shape) == 2:
            # audio = audio[0]  # FIXME use the first channel instead of averaging?
            audio = audio.mean(0)
        if self.max_duration is not None:
            duration = audio.shape[0] / sr
            if duration > self.max_duration:
                raise ExcessiveDurationError(
                    f"Audio {audio_source}'s duration {duration}s is longer than max duration {self.max_duration}s"
                )
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = audio.numpy()
        return audio

    def process_audio(
        self,
        model_type: str,
        processor: Qwen2AudioProcessor | WhisperFeatureExtractor,
        audio_source: str | list | None = None,
        audio_wav: np.ndarray | None = None
    ) -> tuple[list | None, list | None]:
        if model_type == "qwen2.5vl":
            audio_kwargs = {"sampling_rate": 16000, "padding": "max_length", "return_attention_mask": False}
        elif model_type == "qwen2.5omni":
            audio_kwargs = {
                'sampling_rate': 16000,
                'padding': 'max_length',
                'return_attention_mask': True,
                'return_tensors': 'pt'
            }
        elif model_type == "qwen2audio":
            audio_kwargs = {
                'sampling_rate': 16000,
                'return_attention_mask': True,
                'padding': 'max_length',
                'return_tensors': 'pt'
            }

        if audio_wav is None:
            if isinstance(audio_source, list):
                audio_data = []
                for file in audio_source:
                    audio = self.load_audio(file, audio_kwargs["sampling_rate"])
                    audio_data.append(audio)
            else:
                audio = self.load_audio(audio_source, audio_kwargs["sampling_rate"])
                audio_data = [audio]
        else:
            audio_data = [audio_wav]

        if model_type == "qwen2.5omni":
            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                feature_attention_mask_idx = []
                input_features_idx = []
                audio_lst = [
                    audio_data[idx][k:k + 300 * audio_kwargs["sampling_rate"]]
                    for k in range(0, len(audio_data[idx]), 300 * audio_kwargs["sampling_rate"])
                ]
                audio_lengths_seg = 0
                for audio_seg in audio_lst:
                    if audio_seg.shape[0] < audio_kwargs["sampling_rate"]:
                        padding = audio_kwargs["sampling_rate"] - audio_seg.shape[0]
                        audio_seg = np.pad(audio_seg, (0, padding), mode="constant", constant_values=0)
                    audio_inputs_seg = processor(audio_seg, **audio_kwargs)
                    attn_seg = audio_inputs_seg.pop("attention_mask")
                    feature_attention_mask_idx.append(attn_seg)
                    input_features_idx.append(audio_inputs_seg.pop("input_features"))
                    input_lengths_seg = (attn_seg.sum(-1) - 1) // 2 + 1
                    audio_lengths_seg += (input_lengths_seg - 2) // 2 + 1

                if audio_lengths_seg <= 0:
                    return None, None

                feature_attention_mask_idx = torch.cat(feature_attention_mask_idx, dim=0)
                input_features_idx = torch.cat(input_features_idx, dim=0)

                audio_inputs.append({
                    "feature_attention_mask": feature_attention_mask_idx,
                    "input_features": input_features_idx
                })
                audio_lengths.append(audio_lengths_seg)

        elif model_type == "qwen2audio":
            audio_inputs = audio_data
            audio_lengths = None

        return audio_inputs, audio_lengths


@dataclass(kw_only=True)
class AudioDataset(MMQwenDatasetBase, AudioProcessingMixin):
    def __post_init__(self) -> None:
        MMQwenDatasetBase.__post_init__(self)
        AudioProcessingMixin.__post_init__(self)

    def set_processor(self) -> None:
        if self.model_type == "qwen2.5omni":
            self.audio_processor = self.processor.feature_extractor
            self.tokenizer = self.processor.tokenizer
        elif self.model_type == "qwen2audio":
            self.audio_processor = self.processor
            self.tokenizer = self.processor.tokenizer
        else:
            raise NotImplementedError

    def load_multimodal_data(self, source: dict) -> dict[str, Any]:
        if isinstance(self.processor, Qwen2_5OmniProcessor):
            processor = self.processor.feature_extractor
        else:
            processor = self.processor
        audio, audio_lengths = self.process_audio(self.model_type, processor, source["audio"])
        mm_data = {"audio": audio, "audio_lengths": audio_lengths}
        return mm_data

    def tokenize_text_after_chat_template(
        self,
        text: str,
        audio: np.ndarray | None,
        audio_lengths,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        if self.model_type == "qwen2.5omni":
            text = self.processor.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths[0]),
                iter([]),
                iter([]),
                iter([]),
                use_audio_in_video=True,
                position_id_per_seconds=25,
                seconds_per_chunk=None,
            )
            assert len(text) == 1
            feat_token_dict = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
            text = text[0]
        elif self.model_type == "qwen2audio":
            feat_token_dict = self.audio_processor(text=text, audio=audio)
            text = self.tokenizer.decode(feat_token_dict["input_ids"][0])
        return text, feat_token_dict

    def update_mm_feature(self, data_dict: dict[str, Any], mm_data: dict[str, Any]) -> None:
        if self.model_type == "qwen2.5omni":
            data_dict.update({
                "pixel_values_videos": None,
                "video_grid_thw": None,
                "video_second_per_grid": None,
                "input_features": mm_data["audio"][0]["input_features"],
                "feature_attention_mask": mm_data["audio"][0]["feature_attention_mask"],
            })
        elif self.model_type == "qwen2audio":
            data_dict.update({
                "input_features": mm_data["input_features"][0],
                "feature_attention_mask": mm_data["feature_attention_mask"][0],
            })

    def build_id_label_from_conversation(
        self,
        conversations: list[dict],
        mm_data: dict[str, Any],
    ) -> dict[str, Any]:
        res = {}
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        text, feat_token_dict = self.tokenize_text_after_chat_template(text, mm_data["audio"], mm_data["audio_lengths"])
        labels = self.build_label(text)

        res.update({
            "input_ids": feat_token_dict["input_ids"][0],
            "attention_mask": feat_token_dict["attention_mask"][0],
        })

        if self.stage == "inference":
            len_input = sum(labels == IGNORE_INDEX)
            res["input_ids"] = res["input_ids"][:len_input]
            res["attention_mask"] = torch.ones_like(res["input_ids"])
        else:
            res["labels"] = labels

        return res

    def _get_item(self, i) -> dict[str, Any]:
        data_dict = super()._get_item(i)
        data_dict["audio"] = self.list_data_dict[i]["audio"]
        return data_dict


@dataclass
class SpoofingMixin:

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
class AudioSpoofingDataset(SpoofingMixin, AudioDataset):
    pass


@dataclass(kw_only=True)
class AudioSpoofingRealFakeDataset(AudioSpoofingDataset):
    def load_raw_data(self) -> None:
        super().load_raw_data()

        for idx in range(len(self.list_data_dict)):
            item = self.list_data_dict[idx]
            item["conversations"][0]["value"] = "<audio>\nDetermine whether this audio clip is a spoof or not."
            item["conversations"][-1]["value"] = json.loads(item["conversations"][-1]["value"])["real_or_fake"]


@dataclass(kw_only=True)
class AudioSpoofingRawTextDataset(AudioSpoofingDataset):
    def json_to_raw_text(self, json_text: str) -> str:
        item = json.loads(json_text)
        if item["real_or_fake"] == "real":
            output = "The utterance is real."
        else:
            output = "The utterance is fake."
            if "spoof_method" in item:
                output += f" The spoof method is {item['spoof_method']}."
            if "fake_region" in item:
                if item["fake_region"] == "all":
                    region_text = "the entire utterance"
                else:
                    region_text = ", ".join([f"{r[0]:.2f}-{r[1]:.2f} seconds" for r in item["fake_region"]])
                output += f" The fake region is {region_text}."
            if "semantic_influence" in item:
                output += f" The spoofing may result in the following influence: {item['semantic_influence']}."
        return output

    def load_raw_data(self) -> None:
        super().load_raw_data()

        for idx in range(len(self.list_data_dict)):
            item = self.list_data_dict[idx]
            analysis_text = self.json_to_raw_text(item["conversations"][-1]["value"])
            item["conversations"][-1]["value"] = analysis_text


@dataclass(kw_only=True)
class AudioSpoofingWithEmbeddingDataset(AudioSpoofingDataset):
    embedding_dir: str

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

    def tokenize_text_after_chat_template(
        self,
        text: str,
        audio: np.ndarray | None,
        audio_lengths,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        if self.model_type == "qwen2.5omni":
            text = self.processor.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths[0] + 1),  # insert spoof embedding
                iter([]),
                iter([]),
                iter([]),
                use_audio_in_video=True,
                position_id_per_seconds=25,
                seconds_per_chunk=None,
            )
            assert len(text) == 1
            feat_token_dict = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
            text = text[0]
        else:
            raise NotImplementedError
        return text, feat_token_dict

    def load_multimodal_data(self, source) -> dict[str, Any]:
        mm_data = super().load_multimodal_data(source)
        embed_file = Path(self.embedding_dir) / f"{source['embedding_file']}.h5"
        dataset_name = Path(embed_file).stem
        with h5py.File(embed_file, "r") as hf:
            audio_id = get_audio_id(source["audio"], dataset_name)
            embedding = hf[audio_id][()]
            embedding = np.array(embedding, dtype=np.float32)
        mm_data["spoof_embeds"] = embedding
        return mm_data

    def update_mm_feature(self, data_dict, mm_data) -> None:
        super().update_mm_feature(data_dict, mm_data)
        data_dict["spoof_embeds"] = mm_data["spoof_embeds"]


@dataclass(kw_only=True)
class AudioVideoDataset(MMQwenDatasetBase, AudioProcessingMixin):
    """Video dataset for supervised fine-tuning."""

    image_processor: BaseImageProcessor = field(default=None)
    audio_processor: SequenceFeatureExtractor | Qwen2AudioProcessor | WhisperFeatureExtractor = field(default=None)
    omni_processor: Qwen2_5OmniProcessor = field(default=None)

    video_args: dict[str, Any]

    def set_processor(self) -> None:
        if self.image_processor is not None:
            self.image_processor.max_pixels = self.video_args["max_pixels"]
            self.image_processor.min_pixels = self.video_args["min_pixels"]
            self.image_processor.size["longest_edge"] = self.video_args["max_pixels"]
            self.image_processor.size["shortest_edge"] = self.video_args["min_pixels"]
        elif self.model_type == "qwen2.5omni":
            self.image_processor = self.omni_processor.image_processor
            self.audio_processor = self.omni_processor.feature_extractor
            self.tokenizer = self.omni_processor.tokenizer
        elif self.model_type == "qwen2audio":
            # `audio_processor` is passed
            self.tokenizer = self.audio_processor.tokenizer
        else:
            raise NotImplementedError

    def __post_init__(self, ) -> None:
        self.video_max_total_pixels = self.video_args.get("max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = self.video_args.get("min_total_pixels", 256 * 28 * 28)
        super().__post_init__()

    @property
    def lengths(self) -> list[int]:
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self) -> list[int]:
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = (cur_len if ("image" in sample) or ("video" in sample) else -cur_len)
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self) -> np.ndarray:
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        torchcodec_video = self.video_torchcodec(video_file)
        return torchcodec_video

    def video_torchcodec(self, video_file):
        vr = VideoReader(video_file, num_threads=1)
        total_frame_num = len(vr)

        video_length = total_frame_num / vr.get_avg_fps()

        interval = self.video_args.get("base_interval", 0.5)
        num_frames_to_sample = round(video_length / interval)
        min_frames = self.video_args.get("min_frames", 1)
        max_frames = self.video_args.get("video_max_frames", 600)
        target_frames = min(max(num_frames_to_sample, min_frames), max_frames)
        frame_idx = np.linspace(0, total_frame_num - 1, target_frames, dtype=int)

        video = vr.get_batch(frame_idx).asnumpy()  # video: (F, H, W, C)
        video = np.array(video)

        if self.model_type == "qwen2.5omni":
            video = torch.from_numpy(video)
            video_proc = self.image_processor(images=None, videos=video, return_tensors="pt")
            fps = len(frame_idx) / video_length  # 1 / interval
            fps = [fps] * 1
            video_proc["video_second_per_grid"] = [
                self.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            return video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]
        else:
            raise NotImplementedError

    def load_image(self, image: list | str):
        if isinstance(image, list):
            if len(image) > 1:
                image = [file for file in image]
                results = [self.process_image_unified(file) for file in image]
                image, grid_thw = zip(*results)
            else:
                image = image[0]
                image, grid_thw = self.process_image_unified(image)
                image = [image]
        else:
            image, grid_thw = self.process_image_unified(image)
            image = [image]
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [merged_thw.prod() // self.image_processor.merge_size**2 for merged_thw in grid_thw_merged]
        return image, grid_thw, grid_thw_merged

    def load_video(self, video: list | str, use_audio: bool):
        if isinstance(video, list):
            if len(video) > 1:
                video = [file for file in video]
                results = [self.process_video(file) for file in video]
                video, grid_thw, second_per_grid_ts = zip(*results)
            else:
                video = video[0]
                video, grid_thw, second_per_grid_ts = self.process_video(video)
                video = [video]
        else:
            video, grid_thw, second_per_grid_ts = self.process_video(video)
            video = [video]
        if use_audio:
            audio, audio_lengths = self.process_audio(self.model_type, self.audio_processor, video)
        else:
            audio, audio_lengths = None, None

        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]

        return video, grid_thw, grid_thw_merged, second_per_grid_ts, audio, audio_lengths

    def tokenize_text_after_chat_template(
        self,
        text: str,
        video_grid_thw,
        second_per_grid_ts,
        audio: np.ndarray | None,
        audio_lengths,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        if self.model_type == "qwen2.5omni":
            text = self.omni_processor.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths[0]) if audio is not None else iter([]),
                iter([]),
                iter(video_grid_thw[0]) if video_grid_thw is not None else iter([]),
                video_second_per_grid=iter(second_per_grid_ts) if video_grid_thw is not None else iter([]),
                use_audio_in_video=audio is not None,
                position_id_per_seconds=25,
                seconds_per_chunk=2.0 * second_per_grid_ts[0] if second_per_grid_ts is not None else None,
            )
            assert len(text) == 1
            feat_token_dict = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
            text = text[0]
        elif self.model_type == "qwen2audio":
            feat_token_dict = self.audio_processor(text=text, audio=audio)
            text = self.tokenizer.decode(feat_token_dict["input_ids"][0])
        return text, feat_token_dict

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

    def load_multimodal_data(self, source: dict) -> dict[str, Any]:
        audio, audio_lengths, image, image_grid_thw, image_grid_thw_merged,  video, video_grid_thw, \
            video_grid_thw_merged, second_per_grid_ts = [None] * 9

        if "image" in source:
            image, image_grid_thw, image_grid_thw_merged = self.load_image(self.list_data_dict[i]["image"])

        if "video" in source:
            use_audio = "use_audio" in source and source["use_audio"]
            video, video_grid_thw, video_grid_thw_merged, second_per_grid_ts, audio, audio_lengths = \
                self.load_video(source["video"], use_audio)

        if "audio" in source:
            audio, audio_lengths = self.process_audio(self.model_type, self.audio_processor, source["audio"])

        if "audio" in source and self.video_args["force_audio"]:
            assert audio is not None and audio[0]["input_features"] is not None and audio[0][
                "feature_attention_mask"] is not None and audio_lengths is not None

        return {
            "audio": audio,
            "audio_lengths": audio_lengths,
            "image": image,
            "image_grid_thw": image_grid_thw,
            "image_grid_thw_merged": image_grid_thw_merged,
            "video": video,
            "video_grid_thw": video_grid_thw,
            "video_grid_thw_merged": video_grid_thw_merged,
            "second_per_grid_ts": second_per_grid_ts,
        }

    def update_mm_feature(self, data_dict: dict[str, Any], mm_data: dict[str, Any]) -> None:
        if self.model_type == "qwen2.5omni":
            data_dict.update({
                "pixel_values_videos":
                    mm_data["video"][0] if mm_data["video"][0] is not None else None,
                "video_grid_thw":
                    mm_data["video_grid_thw"][0] if mm_data["video_grid_thw"][0] is not None else None,
                "video_second_per_grid":
                    mm_data["video_second_per_grid"][0] if mm_data["video_second_per_grid"][0] is not None else None,
                "input_features":
                    mm_data["audio"][0]["input_features"] if mm_data["audio"][0] is not None else None,
                "feature_attention_mask":
                    mm_data["audio"][0]["feature_attention_mask"] if mm_data["audio"][0] is not None else None,
            })
        elif self.model_type == "qwen2.5vl":
            if mm_data["image"]:
                data_dict.update({
                    "pixel_values": torch.cat(mm_data["image"], dim=0),  # FIXME `pixel_values_image`
                    "image_grid_thw": torch.cat([thw.unsqueeze(0) for thw in mm_data["image_grid_thw"]], dim=0)
                })
            # video exist in the data
            if mm_data["video"]:
                data_dict.update({
                    "pixel_values_videos": torch.cat(mm_data["video"], dim=0),  # FIXME `pixel_values_image`
                    "video_grid_thw": torch.cat([thw.unsqueeze(0) for thw in mm_data["video_grid_thw"]], dim=0)
                })
            if mm_data["audio"]:
                data_dict.update({
                    "pixel_values_videos": torch.cat(mm_data["video"], dim=0),  # FIXME `pixel_values_image`
                    "video_grid_thw": torch.cat([thw.unsqueeze(0) for thw in mm_data["video_grid_thw"]], dim=0)
                })
        elif self.model_type == "qwen2audio":
            data_dict.update({
                "audio_feature": torch.cat(mm_data["audio"], dim=0),
                "audio_lengths": mm_data["audio_lengths"],
            })

    def build_id_label_from_conversation(
        self,
        conversations: list[dict],
        mm_data: dict[str, Any],
    ) -> dict[str, Any]:
        res = {}
        text = self.omni_processor.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        text, feat_token_dict = self.tokenize_text_after_chat_template(
            text, mm_data["video_grid_thw"], mm_data["second_per_grid_ts"], mm_data["audio"], mm_data["audio_lengths"]
        )
        labels = self.build_label(text)

        res.update({
            "input_ids": feat_token_dict["input_ids"][0],
            "attention_mask": feat_token_dict["attention_mask"][0],
        })

        if self.stage == "inference":
            len_input = sum(labels == IGNORE_INDEX)
            res["input_ids"] = res["input_ids"][:, :len_input]
            res["attention_mask"] = torch.ones_like(res["input_ids"])
        else:
            res["labels"] = labels

        return res


class MMQwenPreferenceDataBase(MMQwenDatasetBase):
    def build_feat_id_label(
        self,
        source: dict,
        mm_data: dict[str, Any],
    ) -> dict[str, Any]:
        chosen_convs = copy.deepcopy(source["conversations"])
        chosen_convs[-1]["value"] = chosen_convs[-1].pop("chosen")
        chosen_convs = self.build_conversation([chosen_convs])

        rejected_convs = copy.deepcopy(source["conversations"])
        rejected_convs[-1]["value"] = rejected_convs[-1].pop("rejected")
        rejected_convs = self.build_conversation([rejected_convs])

        chosen_dict = self.build_id_label_from_conversation(chosen_convs, mm_data)
        rejected_dict = self.build_id_label_from_conversation(rejected_convs, mm_data)

        res = {
            "chosen_input_ids": chosen_dict["input_ids"],
            "chosen_attention_mask": chosen_dict["attention_mask"],
            "chosen_labels": chosen_dict["labels"],
            "rejected_input_ids": rejected_dict["input_ids"],
            "rejected_attention_mask": rejected_dict["attention_mask"],
            "rejected_labels": rejected_dict["labels"],
        }
        self.update_mm_feature(res, mm_data)
        return res

    def _get_item(self, i) -> dict[str, Any]:
        source = self.list_data_dict[i]
        res = super()._get_item(i)
        res.update({
            "chosen": source["conversations"][-1]["chosen"],
            "rejected": source["conversations"][-1]["rejected"],
        })
        return res


@dataclass(kw_only=True)
class AudioPreferenceDataset(MMQwenPreferenceDataBase, AudioDataset):

    train_type: Literal["dpo"] = field(default="dpo")


@dataclass(kw_only=True)
class AudioSpoofingPreferenceDataset(SpoofingMixin, AudioPreferenceDataset):
    pass


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

        if "input_ids" in collate_samples:
            collate_samples["attention_mask"] = collate_samples["input_ids"].ne(PAD_TOKEN_ID).to(torch.int64)

        return collate_samples


@dataclass
class OmniPreferenceCollator(OmniCollator):

    padding_config: dict[str, int] = field(
        default_factory=lambda: {
            "chosen_input_ids": PAD_TOKEN_ID,
            "chosen_labels": IGNORE_INDEX,
            "rejected_input_ids": PAD_TOKEN_ID,
            "rejected_labels": IGNORE_INDEX
        }
    )

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Sequence[Any] | None]:
        collate_samples = super().__call__(instances)
        collate_samples["chosen_attention_mask"] = collate_samples["chosen_input_ids"].ne(PAD_TOKEN_ID).to(torch.int64)
        collate_samples["rejected_attention_mask"] = collate_samples["rejected_input_ids"].ne(PAD_TOKEN_ID).to(
            torch.int64
        )
        return collate_samples


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra

    random.seed(2025)

    dataset_config_str = r"""
_target_: qwenvl.data.data_qwen.AudioSpoofingDataset
stage: training
train_type: sft
model_type: qwen2.5omni
dataset_list:
  - data_json/asvspoof2019/train.json
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
