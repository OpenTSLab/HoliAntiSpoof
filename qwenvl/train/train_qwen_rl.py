import argparse
from pathlib import Path
import logging
from copy import deepcopy

import transformers
import torch.distributed as dist
from hydra.utils import instantiate
from transformers import PreTrainedModel
from omegaconf import OmegaConf

from qwenvl.train.rl_argument import GRPOArguments
from qwenvl.train.grpo_trainer import GRPOQwenVLTrainer
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


def train():

    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", "-c", required=True, type=str, help="Path to config YAML file.")
    entry_parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()

    seed = 2025
    set_seed(seed)

    parser = transformers.HfArgumentParser((GRPOArguments))

    config = load_config(args.config_file, args.options)

    training_args, = parser.parse_dict(config["trainer"], allow_extra_keys=True)
    # `dist.init_process_group` has been called in the parsing of `training_args`
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    exp_dir = Path(config["trainer"]["output_dir"])

    if rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, exp_dir / "config.yaml")
    dist.barrier()

    train_type = config["global"]["train_type"]
    assert train_type in ["dpo", "grpo"], f"train_type {train_type} is not supported in RL training"

    apply_liger_kernel_to_qwen2_5_vl()

    train_dataset = instantiate(config["train_dataset"], _convert_="all")
    val_dataset = instantiate(config["val_dataset"], _convert_="all")
    data_collator = instantiate(config["data_collator"], _convert_="all")

    model: PreTrainedModel = initialize_model(config["model"], training_args)

    model.requires_grad_(False)
    set_model_gradient_checkpointing(model, training_args)

    if training_args.beta == 0.0:
        ref_model = None
    else:
        ref_model = deepcopy(model)
        ref_model.eval()
        ref_model.requires_grad_(False)

    use_lora = "lora_config" in config["model"]
    set_model_frozen_trainable(config["model"]["trainable_config"], model, use_lora)
    model = set_lora(model, config["model"])

    print_trainable_blocks(model)

    reward_func = instantiate(config["reward_fn"], _convert_="all")
    metric = instantiate(config["metric"], _convert_="all")
    trainer = GRPOQwenVLTrainer(
        model=model,
        processing_class=train_dataset.tokenizer,
        args=training_args,
        reward_funcs=reward_func,
        compute_metrics=metric,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    if list(exp_dir.glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
