# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
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
from safetensors.torch import load_file
import transformers
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from qwenvl.train.utils import get_torch_dtype, set_seed
from qwenvl.train.argument import TrainingArguments
from qwenvl.train.trainer import QwenVLTrainer
from qwenvl.model import ModelProtocol, MODEL_TO_LORA_EXCLUDE_MODULES


def collate_fn_single_sample(batch):
    return batch[0]


def collate_fn_single_sample_qwen_audio(batch):
    batch[0]["input_features"] = batch[0]["input_features"].unsqueeze(0)
    batch[0]["feature_attention_mask"] = batch[0]["feature_attention_mask"].unsqueeze(0)
    return batch[0]


def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen2.5 model...")

    assert not (cross_entropy and
                fused_linear_cross_entropy), ("cross_entropy and fused_linear_cross_entropy cannot both be True.")

    from qwenvl.model.qwen2_5_vl import modeling_qwen2_5_vl
    from qwenvl.model.qwen2_5_omni import modeling_qwen2_5_omni

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
        modeling_qwen2_5_omni.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
        modeling_qwen2_5_omni.Qwen2RMSNorm = LigerRMSNorm
    # if cross_entropy:
    #     modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss
    # if fused_linear_cross_entropy:
    #     modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_vl_lce_forward
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
        modeling_qwen2_5_omni.Qwen2MLP = LigerSwiGLUMLP


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


def set_model_frozen_trainable(trainable_cfg: dict, model: ModelProtocol, use_lora: bool):
    if model.model_type == "qwen2.5vl":
        if trainable_cfg["vision_encoder"]:
            model.visual.requires_grad_(True)
        else:
            model.visual.requires_grad_(False)

        if trainable_cfg["vision_adapter"]:
            model.visual.merger.requires_grad_(True)
        else:
            model.visual.merger.requires_grad_(False)

        if trainable_cfg["audio_encoder"]:
            model.audio.requires_grad_(True)
        else:
            model.audio.requires_grad_(False)

        if trainable_cfg["audio_adapter"]:
            model.audio.qformer.requires_grad_(True)
            model.audio.q_tokens.requires_grad_(True)
            model.audio.audio_proj.requires_grad_(True)
        else:
            model.audio.qformer.requires_grad_(False)
            model.audio.q_tokens.requires_grad_(False)
            model.audio.audio_proj.requires_grad_(False)

        if trainable_cfg["llm"]:
            assert not use_lora, "when LoRA is used, LLM must be frozen"
            model.model.requires_grad_(True)
            model.lm_head.requires_grad_(True)
        else:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)

    elif model.model_type == "qwen2.5omni":
        if trainable_cfg["vision_encoder"]:
            model.visual.requires_grad_(True)
        else:
            model.visual.requires_grad_(False)

        if trainable_cfg["vision_adapter"]:
            model.visual.merger.requires_grad_(True)
        else:
            model.visual.merger.requires_grad_(False)

        if trainable_cfg["audio_encoder"]:
            model.audio_tower.requires_grad_(True)
        else:
            model.audio_tower.requires_grad_(False)

        if trainable_cfg["audio_adapter"]:
            model.audio_tower.ln_post.requires_grad_(True)
            model.audio_tower.proj.requires_grad_(True)
        else:
            model.audio_tower.ln_post.requires_grad_(False)
            model.audio_tower.proj.requires_grad_(False)

        if trainable_cfg["llm"]:
            assert not use_lora, "when LoRA is used, LLM must be frozen"
            model.model.requires_grad_(True)
            model.lm_head.requires_grad_(True)
        else:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)

    elif model.model_type == "qwen2audio":
        if trainable_cfg["audio_encoder"]:
            model.audio_tower.requires_grad_(True)
        else:
            model.audio_tower.requires_grad_(False)

        if trainable_cfg["audio_adapter"]:
            model.multi_modal_projector.requires_grad_(True)
        else:
            model.multi_modal_projector.requires_grad_(False)

        if trainable_cfg["llm"]:
            assert not use_lora, "when LoRA is used, LLM must be frozen"
            model.language_model.requires_grad_(True)
        else:
            model.language_model.requires_grad_(False)


def load_non_lora_params_from_ckpt(
    model: PeftModel,
    ckpt_path: str,
    load_modules: list[str] | None = None,
):
    if load_modules is None:
        load_modules = []
    load_module_prefixs = []
    for module_name in load_modules:
        load_module_prefixs.append(f"base_model.model.{module_name}")
    state_dict: dict = torch.load(ckpt_path, map_location="cpu")
    audio_state_dict = {}
    for k, v in state_dict["module"].items():
        found_in_load_modules = False
        for prefix in load_module_prefixs:
            if k.startswith(prefix):
                found_in_load_modules = True
                break
        if found_in_load_modules:
            audio_state_dict[k] = v
    model.load_state_dict(audio_state_dict, strict=False)


def set_model_gradient_checkpointing(model: nn.Module, training_args: TrainingArguments):
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs["use_reentrant"] = False


def initialize_model(config: dict, training_args: TrainingArguments):
    # 1. instantiate model based on config
    # 2. load lora ckpt if provided
    #
    # this is used for initializing model before training and inference, during inference, it
    # is called before loading the "real" checkpoint in experiment directory
    torch_dtype = get_torch_dtype(training_args.bf16, training_args.fp16)
    config["model"]["architecture"]["torch_dtype"] = torch_dtype
    model: nn.Module = instantiate(config["model"]["architecture"])

    if "lora_ckpt" in config["model"] and config["model"]["lora_ckpt"]:
        model = load_maybe_lora_ckpt(
            model,
            config["model"]["lora_ckpt"],
            is_lora=True,
        )

    return model


def load_maybe_lora_ckpt(model: ModelProtocol, pretrained_ckpt: str, is_lora: bool):
    if is_lora:
        model = apply_or_load_lora_safely(model, lora_config=None, lora_ckpt=pretrained_ckpt)

        if model.model_type in ["qwen2.5omni", "qwen2audio"]:
            audio_ckpt = list(Path(pretrained_ckpt).glob("global_step*/mp_rank_00_model_states.pt"))[0]
            load_non_lora_params_from_ckpt(model, audio_ckpt, MODEL_TO_LORA_EXCLUDE_MODULES[model.model_type])

        model = model.merge_and_unload()
    else:
        ckpt = load_file(Path(pretrained_ckpt) / "model.safetensors")
        model.load_state_dict(ckpt, strict=False)
    return model


def apply_or_load_lora_safely(
    model: ModelProtocol, lora_config: LoraConfig | None = None, lora_ckpt: str | None = None
):
    """
    A generic helper that temporarily removes certain submodules before LoRA injection,
    then restores them afterward.

    This is useful when you want to exclude specific parts of a model
    (e.g., projection layers, towers, etc.) from being wrapped by LoRA.

    Args:
        model: 
            The base model before LoRA injection.
        lora_config:
            Configuration used by get_peft_model().
        lora_ckpt:
            Path to the LoRA checkpoint.
            

    Returns:
        model: ModelProtocol
            The LoRA-injected model with the excluded modules reattached.
    """
    exclude_attrs = MODEL_TO_LORA_EXCLUDE_MODULES[model.model_type]

    assert lora_config is not None or lora_ckpt is not None

    # Step 1. Save excluded modules and temporarily delete them from the model
    excluded = {}
    for attr_path in exclude_attrs:
        parts = attr_path.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]

        excluded[attr_path] = getattr(parent, attr_name)
        delattr(parent, attr_name)

    # Step 2. Inject LoRA (PEFT) or load LoRA from checkpoint
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    elif lora_ckpt is not None:
        model = PeftModel.from_pretrained(model, lora_ckpt)

    # Step 3. Restore previously excluded modules
    for attr_path, module in excluded.items():
        parts = attr_path.split('.')
        # After LoRA injection, the actual model is usually under model.model
        parent = model.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], module)

    return model


def set_lora(model: ModelProtocol, model_cfg: dict):
    if "lora_config" not in model_cfg:
        return model
    trainable_cfg = model_cfg["trainable_config"]
    modules_to_save = []
    if model.model_type == "qwen2.5vl":
        if trainable_cfg["vision_encoder"]:
            modules_to_save.append("visual")
        if trainable_cfg["vision_adapter"]:
            modules_to_save.append("visual.merger")
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio")
        if trainable_cfg["audio_adapter"]:
            modules_to_save.append("audio.qformer")
            modules_to_save.append("audio.q_tokens")
            modules_to_save.append("audio.audio_proj")
    elif model.model_type == "qwen2.5omni":
        if trainable_cfg["vision_encoder"]:
            modules_to_save.append("visual")
        if trainable_cfg["vision_adapter"]:
            modules_to_save.append("visual.merger")
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio_tower")
        if trainable_cfg["audio_adapter"]:
            modules_to_save.append("audio_tower.ln_post")
            modules_to_save.append("audio_tower.proj")
    elif model.model_type == "qwen2audio":
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio_tower")
        if trainable_cfg["audio_adapter"]:
            modules_to_save.append("multi_modal_projector")

    lora_config = instantiate(model_cfg["lora_config"], modules_to_save=modules_to_save, _convert_="all")
    model = apply_or_load_lora_safely(model, lora_config=lora_config, lora_ckpt=None)

    for k, v in model.named_parameters():
        if "lora" in k:
            v.requires_grad_(True)

    return model


def print_trainable_blocks(model: nn.Module, is_main_process: bool = True):
    if not is_main_process:
        return
    cnt, total = 0, 0
    print("Trainable blocks: ")
    for k, v in model.named_parameters():
        if v.requires_grad:
            print(k, v.shape)
            cnt += 1
        total += 1
    print(f"Total blocks: {total}, trainable blocks: {cnt}")


def train():

    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", required=True, type=str, help="Path to config YAML file.")
    entry_parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()

    seed = 2025
    set_seed(seed)

    parser = transformers.HfArgumentParser((TrainingArguments))

    config_dir = Path(args.config_file).parent.absolute().__str__()
    config_fname = Path(args.config_file).name
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_fname, overrides=args.options)
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)

    training_args, = parser.parse_dict(config["trainer"], allow_extra_keys=True)
    # `dist.init_process_group` is called in the parsing of `training_args`
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
    assert train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {train_type} is not supported"
    assert config["global"]["model_type"] in [
        "qwen2.5vl", "qwen2.5omni", "qwen2audio"
    ], f"model_type {config['model_type']} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    train_dataset = instantiate(config["train_dataset"], _convert_="all")
    val_dataset = instantiate(config["val_dataset"], _convert_="all")
    data_collator = instantiate(config["data_collator"], _convert_="all")

    model: PreTrainedModel = initialize_model(config, training_args)
    if "lora_ckpt" in config["model"] and config["model"]["lora_ckpt"]:
        if rank == 0:
            model.save_pretrained(exp_dir / "base")
        dist.barrier()

    model.requires_grad_(False)
    set_model_gradient_checkpointing(model, training_args)

    use_lora = "lora_config" in config["model"]
    set_model_frozen_trainable(config["model"]["trainable_config"], model, use_lora)
    model = set_lora(model, config["model"])

    print_trainable_blocks(model, is_main_process=(rank == 0))
    metric = instantiate(config["metric"], _convert_="all")
    trainer = QwenVLTrainer(
        model=model,
        processing_class=train_dataset.tokenizer,
        args=training_args,
        train_type=train_type,
        compute_metrics=metric,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
