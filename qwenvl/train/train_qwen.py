# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from transformers import PreTrainedModel, TrainingArguments
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft import LoraConfig

from qwenvl.train.utils import (
    apply_liger_kernel_to_qwen2_5_vl,
    initialize_model,
    load_config,
    print_trainable_blocks,
    set_lora,
    set_model_frozen_trainable,
    set_model_gradient_checkpointing,
    set_seed,
)
from qwenvl.train.trainer import QwenVLTrainer


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():

    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", "-c", required=True, type=str, help="Path to config YAML file.")
    entry_parser.add_argument("--overrides", "-o", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()

    seed = 2025
    set_seed(seed)

    parser = transformers.HfArgumentParser((TrainingArguments))

    config = load_config(args.config_file, args.overrides)

    training_args, = parser.parse_dict(config["training_args"], allow_extra_keys=True)
    # `dist.init_process_group` is called in the parsing of `training_args`
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    exp_dir = Path(config["training_args"]["output_dir"])

    if rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, exp_dir / "config.yaml")
    dist.barrier()

    train_type = config["global"]["train_type"]
    assert train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {train_type} is not supported"
    assert config["global"]["model_type"] in [
        "qwen2.5vl", "qwen2.5omni", "qwen2audio"
    ], f"model_type {config['model_type']} is not supported"

    apply_liger_kernel_to_qwen2_5_vl()

    model: PreTrainedModel = initialize_model(config["model"], training_args)
    if "lora_ckpt" in config["model"] and config["model"]["lora_ckpt"]:
        if rank == 0:
            model.save_pretrained(exp_dir / "base")
        dist.barrier()

    model.requires_grad_(False)
    set_model_gradient_checkpointing(model, training_args)

    use_lora = "lora_config" in config["model"]
    set_model_frozen_trainable(config["model"]["trainable_config"], model, use_lora)
    model = set_lora(model, config["model"])

    print_trainable_blocks(model)

    train_dataset = instantiate(config["train_dataset"], _convert_="all")
    val_dataset = instantiate(config["val_dataset"], _convert_="all")
    data_collator = instantiate(config["data_collator"], _convert_="all")
    metric = instantiate(config["metric"], _convert_="all")
    callbacks = instantiate(config["callbacks"], _convert_="all")

    trainer = QwenVLTrainer(
        model=model,
        processing_class=train_dataset.tokenizer,
        args=training_args,
        train_type=train_type,
        compute_metrics=metric,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if list(exp_dir.glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
