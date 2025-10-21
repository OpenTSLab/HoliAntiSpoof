import argparse
import os
from pathlib import Path
import logging
import json
from copy import deepcopy

import numpy as np
import torch
import transformers
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    GRPOArguments,
)
from qwenvl.train.trainer import GRPOQwenVLTrainer
from qwenvl.train.train_qwen import (
    set_seed,
    apply_liger_kernel_to_qwen2_5_vl,
    collate_fn_single_sample,
    initialize_model,
    initialize_processor,
    load_maybe_lora_ckpt,
    print_trainable_blocks,
    set_model_gradient_checkpointing,
    set_lora,
    set_model_frozen_trainable,
)
from qwenvl.train.utils import KeywordsStoppingCriteria
from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from qwenvl.model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

local_rank = None


def calculate_overlap_ratio(segments_a, segments_b):
    """
    Compute overlap ratio (IoU) between two lists of time segments in a fully vectorized way.
    Each segment list is a list of [start, end] pairs.

    Args:
        segments_a (list[list[float]]): First list of segments.
        segments_b (list[list[float]]): Second list of segments.

    Returns:
        float: Intersection over Union (IoU) of the two segment sets.
    """
    # Convert to numpy arrays for vectorized operations
    a = np.array(segments_a, dtype=np.float32)  # Shape: [N, 2]
    b = np.array(segments_b, dtype=np.float32)  # Shape: [M, 2]

    # Calculate total length of each set
    total_len_a = np.sum(np.maximum(0, a[:, 1] - a[:, 0]))
    total_len_b = np.sum(np.maximum(0, b[:, 1] - b[:, 0]))

    # Expand dimensions to create pairwise comparisons between all segments
    # a[:, None, 0] -> shape [N, 1], b[None, :, 0] -> shape [1, M]
    start_max = np.maximum(a[:, None, 0], b[None, :, 0])  # Shape: [N, M]
    end_min = np.minimum(a[:, None, 1], b[None, :, 1])  # Shape: [N, M]

    # Compute intersection lengths for all pairs
    overlaps = np.maximum(0, end_min - start_max)  # Shape: [N, M]

    # Total intersection is the sum of all overlaps
    total_overlap = np.sum(overlaps)

    # Compute IoU: intersection / union
    union_len = total_len_a + total_len_b - total_overlap
    return (total_overlap / union_len).item() if union_len > 0 else 0.0


def real_fake_reward_function(
    completion: dict,
    label: dict,
):
    if "real_or_fake" not in completion:
        return 0.0
    if completion["real_or_fake"] == label["real_or_fake"]:
        return 1.0
    else:
        return 0.0


def spoof_method_reward_function(
    completion: dict,
    label: dict,
):
    if "spoof_method" not in label:  # real utterance, no spoofing
        return 1.0
    if "spoof_method" not in completion:  # should output spoof_method but not found
        return 0.0
    if completion["spoof_method"] == label["spoof_method"]:
        return 1.0
    else:
        return 0.0


def fake_region_reward_function(
    completion: dict,
    label: dict,
):
    if "fake_region" not in label:  # real utterance, no spoofing
        return 1.0
    if "fake_region" not in completion:  # should output fake_region but not found
        return 0.0
    # if len(label["fake_region"]) == 0:
    #     if len(completion["fake_region"]) == 0:
    #         return 1.0
    #     else:
    #         return 0.0
    if label["fake_region"] == "all":
        if completion["fake_region"] == "all":
            return 1.0
        else:
            return 0.0
    try:
        reward = calculate_overlap_ratio(label["fake_region"], completion["fake_region"])
    except Exception:
        reward = 0.0
    return reward


def keywords_reward_function(
    completion: dict,
    keywords: str | None,
):
    if keywords is None:
        return 1.0

    if "semantic_influence" not in completion:
        return 0.0

    reward = 0.0
    keywords = keywords.split()
    num_keywords = len(keywords)
    for keyword in keywords:
        if keyword in completion["semantic_influence"]:
            reward += 1.0 / num_keywords
    return reward


def reward_function(
    completions: list[str],
    labels_text: list[str],
    keywords: list[str | None],
    **kwargs,
):
    rewards = []
    for completion, label_text, keywords_ in zip(completions, labels_text, keywords):
        reward = 0.0
        label = json.loads(label_text.strip())
        try:
            completion = json.loads(completion.strip())
            json_format_reward = 1.0
            real_fake_reward = real_fake_reward_function(completion, label)
            spoof_method_reward = spoof_method_reward_function(completion, label)
            fake_region_reward = fake_region_reward_function(completion, label)
            keywords_reward = keywords_reward_function(completion, keywords_)
            reward = json_format_reward + real_fake_reward + spoof_method_reward + \
                fake_region_reward + keywords_reward
        except json.JSONDecodeError:
            reward = 0.0
        rewards.append(reward)
    return rewards


def train():
    global local_rank

    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", required=True, type=str, help="Path to config YAML file.")
    entry_parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()

    seed = 2025
    set_seed(seed)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))

    config_dir = Path(args.config_file).parent.absolute().__str__()
    config_fname = Path(args.config_file).name
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_fname, overrides=args.options)
    OmegaConf.resolve(config)

    model_args, data_args, training_args = parser.parse_dict(config, allow_extra_keys=True)
    # `dist.init_process_group` has been called in the parsing of `training_args`
    rank = dist.get_rank()

    if rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(training_args.output_dir, "config.yaml"))
    dist.barrier()

    data_args.model_type = model_args.model_type

    assert data_args.train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {data_args.train_type} is not supported"
    assert model_args.model_type in ["qwen2.5vl", "qwen2.5omni"], f"model_type {model_args.model_type} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank

    tokenizer = initialize_processor(model_args, data_args, training_args)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model = initialize_model(model_args, training_args)
    model.requires_grad_(False)

    set_model_gradient_checkpointing(model, training_args)

    if model_args.lora_ckpt is not None:
        # model = load_maybe_lora_ckpt(
        # model_args,
        # model,
        # model_args.lora_ckpt,
        # is_lora=True,
        # )
        if rank == 0:
            model.save_pretrained(os.path.join(training_args.output_dir, "base"))
        dist.barrier()

    if training_args.beta == 0.0:
        ref_model = None
    else:
        ref_model = deepcopy(model)
        ref_model.eval()
        ref_model.requires_grad_(False)

    set_model_frozen_trainable(model_args, model)
    model = set_lora(model_args, model)
    print_trainable_blocks(model, is_main_process=(rank == 0))

    trainer = GRPOQwenVLTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=reward_function,
        ref_model=ref_model,
        **data_module
    )

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
