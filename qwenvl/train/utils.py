import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import StoppingCriteria, TrainingArguments
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from peft import LoraConfig, get_peft_model, PeftModel

from ..model import ModelProtocol, MODEL_TO_LORA_EXCLUDE_MODULES


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = output_ids.shape[0]
        all_matched = True
        for batch_idx in range(batch_size):
            sample_output_ids = output_ids[batch_idx:batch_idx + 1]
            offset = min(sample_output_ids.shape[1] - self.start_len, 3)
            self.keyword_ids = [keyword_id.to(sample_output_ids.device) for keyword_id in self.keyword_ids]
            keyword_matched = False
            for keyword_id in self.keyword_ids:
                if torch.all(sample_output_ids[0, -keyword_id.shape[0]:] == keyword_id):
                    keyword_matched = True
                    break
            if not keyword_matched:
                outputs = self.tokenizer.batch_decode(sample_output_ids[:, -offset:], skip_special_tokens=True)[0]
                for keyword in self.keywords:
                    if keyword in outputs:
                        keyword_matched = True
                        break
            if not keyword_matched:
                all_matched = False
                break
        return all_matched


def get_torch_dtype(bf16: bool = False, fp16: bool = False) -> torch.dtype | None:
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolver for hydra configs, which can be used in YAML
    files for dynamically setting values
    """
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("get_torch_dtype", get_torch_dtype, replace=True)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_file: str, overrides: list[str] = []) -> dict:
    config_dir = Path(config_file).parent.absolute().__str__()
    config_fname = Path(config_file).name
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_fname, overrides=overrides)
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)
    return config


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

    rank0_print("Applying Liger kernels to Qwen2.5 model...")

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


# def apply_or_load_lora_safely(
#     model: ModelProtocol, lora_config: LoraConfig | None = None, lora_ckpt: str | None = None
# ) -> ModelProtocol:
#     # TODO use `modules_to_save` instead of del and assign to save modules into "adapter_model.safetensors"
#     """
#     A generic helper that temporarily removes certain submodules before LoRA injection,
#     then restores them afterward.

#     This is useful when you want to exclude specific parts of a model
#     (e.g., projection layers, towers, etc.) from being wrapped by LoRA.

#     Args:
#         model:
#             The base model before LoRA injection.
#         lora_config:
#             Configuration used by get_peft_model().
#         lora_ckpt:
#             Path to the LoRA checkpoint.

#     Returns:
#         model: ModelProtocol
#             The LoRA-injected model with the excluded modules reattached.
#     """
#     exclude_attrs = MODEL_TO_LORA_EXCLUDE_MODULES[model.model_type]

#     assert lora_config is not None or lora_ckpt is not None

#     # Step 1. Save excluded modules and temporarily delete them from the model
#     excluded = {}
#     for attr_path in exclude_attrs:
#         try:
#             parts = attr_path.split('.')
#             parent = model
#             for p in parts[:-1]:
#                 parent = getattr(parent, p)
#             attr_name = parts[-1]

#             excluded[attr_path] = getattr(parent, attr_name)
#             delattr(parent, attr_name)
#         except AttributeError as e:  # if exclude modules not found in model, just skip
#             rank0_print(
#                 f"Attribute {attr_path} not found when caching non-LoRA parts with error: {e}, please check whether there are typos"
#             )
#             pass

#     # Step 2. Inject LoRA (PEFT) or load LoRA from checkpoint
#     if lora_config is not None:
#         model = get_peft_model(model, lora_config)
#     elif lora_ckpt is not None:
#         model = PeftModel.from_pretrained(model, lora_ckpt)

#     # Step 3. Restore previously excluded modules
#     for attr_path, module in excluded.items():
#         parts = attr_path.split('.')
#         # After LoRA injection, the actual model is usually under model.model
#         parent = model.model
#         for p in parts[:-1]:
#             parent = getattr(parent, p)
#         setattr(parent, parts[-1], module)

#     return model


def load_non_lora_params_from_ckpt(
    model: PeftModel,
    ckpt_path: str,
    load_modules: list[str] | None = None,
) -> None:
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


def load_maybe_lora_ckpt(model: ModelProtocol, pretrained_ckpt: str, is_lora: bool) -> ModelProtocol:
    if is_lora:
        # model = apply_or_load_lora_safely(model, lora_config=None, lora_ckpt=pretrained_ckpt)
        model = PeftModel.from_pretrained(model, pretrained_ckpt)

        # if model.model_type in ["qwen2.5omni", "qwen2audio"]:
        #     audio_ckpt = list(Path(pretrained_ckpt).glob("global_step*/mp_rank_00_model_states.pt"))[0]
        #     load_non_lora_params_from_ckpt(model, audio_ckpt, MODEL_TO_LORA_EXCLUDE_MODULES[model.model_type])

        model = model.merge_and_unload()
    else:
        ckpt = load_file(Path(pretrained_ckpt) / "model.safetensors")
        model.load_state_dict(ckpt, strict=False)
    return model


def initialize_model(config: dict, training_args: TrainingArguments) -> ModelProtocol:
    # 1. instantiate model based on config
    # 2. load lora ckpt if provided
    #
    # this is used for initializing model before training and inference, during inference, it
    # is called before loading the "real" checkpoint in experiment directory
    torch_dtype = get_torch_dtype(training_args.bf16, training_args.fp16)
    config["architecture"]["torch_dtype"] = torch_dtype
    model: nn.Module = instantiate(config["architecture"], _convert_="all")

    if "lora_ckpt" in config and config["lora_ckpt"]:
        model = load_maybe_lora_ckpt(
            model,
            config["lora_ckpt"],
            is_lora=True,
        )

    return model


def set_model_gradient_checkpointing(model: nn.Module, training_args: TrainingArguments) -> None:
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


def print_trainable_blocks(model: nn.Module) -> None:
    cnt, total = 0, 0
    rank0_print("Trainable blocks: ")
    for k, v in model.named_parameters():
        if v.requires_grad:
            rank0_print(k, v.shape)
            cnt += 1
        total += 1
    rank0_print(f"Total blocks: {total}, trainable blocks: {cnt}")


def set_model_frozen_trainable(trainable_cfg: dict, model: ModelProtocol, use_lora: bool) -> None:
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

        if hasattr(model, "spoof_proj"):
            model.spoof_proj.requires_grad_(True)

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


def set_lora(model: ModelProtocol, model_cfg: dict) -> ModelProtocol:
    if "lora_config" not in model_cfg:
        return model
    trainable_cfg = model_cfg["trainable_config"]
    modules_to_save = []
    if model.model_type == "qwen2.5vl":
        if trainable_cfg["vision_encoder"]:
            modules_to_save.append("visual")
        # if trainable_cfg["vision_adapter"]:
        #     modules_to_save.append("visual.merger")
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio")
        # if trainable_cfg["audio_adapter"]:
        #     modules_to_save.append("audio.qformer")
        #     modules_to_save.append("audio.q_tokens")
        #     modules_to_save.append("audio.audio_proj")
    elif model.model_type == "qwen2.5omni":
        if trainable_cfg["vision_encoder"]:
            modules_to_save.append("visual")
        if trainable_cfg["vision_adapter"]:
            modules_to_save.append("visual.merger")
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio_tower")
        # if trainable_cfg["audio_adapter"]:
        #     modules_to_save.append("audio_tower.ln_post")
        #     modules_to_save.append("audio_tower.proj")
        if hasattr(model, "spoof_proj"):
            modules_to_save.append("spoof_proj")
    elif model.model_type == "qwen2audio":
        if trainable_cfg["audio_encoder"]:
            modules_to_save.append("audio_tower")
        if trainable_cfg["audio_adapter"]:
            modules_to_save.append("multi_modal_projector")

    lora_config: LoraConfig = instantiate(model_cfg["lora_config"], modules_to_save=modules_to_save, _convert_="all")
    # model = apply_or_load_lora_safely(model, lora_config=lora_config, lora_ckpt=None)
    model = get_peft_model(model, lora_config)

    for k, v in model.named_parameters():
        if "lora" in k:
            v.requires_grad_(True)

    return model
