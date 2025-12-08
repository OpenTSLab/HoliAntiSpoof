# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch import autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_comet_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_liger_kernel_available, is_peft_available

from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.models import create_reference_model, prepare_deepspeed
from trl.models.utils import prepare_fsdp
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.dpo_config import FDivergenceConstants, FDivergenceType
from trl.trainer.utils import (
    RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    flush_right,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss

from qwenvl.train.trainer import QwenVLTrainer
from qwenvl.train.rl_argument import DPOArguments
from qwenvl.data.data_qwen import IGNORE_INDEX


def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right, and pad with pad_token_id"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id


class DPOQwenVLTrainer(QwenVLTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args ([`DPOArguments`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. DPO supports [preference](#preference) type and. The format of the samples can
            be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
    """
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: DPOArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin,
                                         ProcessorMixin]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):

        if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
            self.padding_value = processing_class.pad_token_id
        elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
            self.padding_value = processing_class.tokenizer.pad_token_id
        else:
            raise ValueError(
                "`pad_token_id` is missing in the `processing_class`. Please set `tokenizer.pad_token` (e.g.,"
                " `tokenizer.pad_token = tokenizer.eos_token`) before instantiating the trainer."
            )

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
                # TODO xxn ref_model has done ".eval()", why should dropout be disabled again?

        # Liger kernel
        if args.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_loss=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if args.loss_type != "sigmoid":
                raise ValueError(
                    "You set `use_liger_loss=True` but the loss type is not `sigmoid`. "
                    "Please set `loss_type='sigmoid'` to use the liger kernel."
                )
            self.dpo_loss_fn = LigerFusedLinearDPOLoss(
                ignore_index=args.label_pad_token_id,
                beta=args.beta,
                use_ref_model=not args.reference_free,
                average_log_prob=False,
            )
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        # self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_logits_to_keep = args.use_logits_to_keep

        if args.padding_free:
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
        self.padding_free = args.padding_free

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        # self._precomputed_train_ref_log_probs = False
        # self._precomputed_eval_ref_log_probs = False

        if (
            args.loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"] and
            args.label_smoothing > 0
        ):
            warnings.warn(
                f"You are using the {args.loss_type} loss type that does not support label smoothing. The "
                "`label_smoothing` parameter will be ignored. Set `label_smoothing` to `0.0` to remove this warning.",
                UserWarning,
            )

        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            warnings.warn(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
                UserWarning,
            )

        self.use_weighting = args.use_weighting
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        # self.dataset_num_proc = args.dataset_num_proc

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        # self.model_accepts_loss_kwargs = False

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        # if self.is_deepspeed_enabled:
        #     if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
        #         raise ValueError(
        #             "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
        #         )

        if self.ref_model is None:
            # if not (self.is_peft_model or self.precompute_ref_log_probs):
            #     raise ValueError(
            #         "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
            #     )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            # if self.precompute_ref_log_probs:
            #     raise ValueError(
            #         "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
            #     )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

        self._peft_has_been_casted_to_bf16 = False
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            peft_module_casting_to_bf16(model)
            # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
            self._peft_has_been_casted_to_bf16 = True

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        with torch.no_grad():
            ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`. The losses tensor contains the DPO
            loss for each example in the batch. The `chosen_rewards` and `rejected_rewards` tensors contain the rewards
            for the chosen and rejected responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) -
                F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) *
                (1 - self.label_smoothing) + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)

        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta))**2

        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) -
                                   delta) - F.logsigmoid(-(self.beta * rejected_logratios - delta))

        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta)**2 + (b + 0.5 / self.beta)**2

        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) -
                0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing) -
                F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing) -
                F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def _compute_loss_liger(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        args: DPOArguments = self.args
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        # Repeat non chosen/rejected tensors to perform generation for chosen and rejected in a single forward pass
        for k, v in batch.items():
            if k not in args.unused_items_for_generation:
                if isinstance(v, torch.Tensor):
                    if k.startswith("chosen_") or k.startswith("rejected_"):
                        continue
                    else:
                        model_kwargs[k] = torch.cat((v, v), dim=0)
                else:
                    model_kwargs[k] = v

        # For decoder-only models
        max_seq_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        num_examples = batch["chosen_input_ids"].shape[0]
        input_ids = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_seq_length, pad_value=self.padding_value),
                pad_to_length(batch["rejected_input_ids"], max_seq_length, pad_value=self.padding_value)
            ),
            dim=0,
        )
        attention_mask = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_seq_length, pad_value=0.0),
                pad_to_length(batch["rejected_attention_mask"], max_seq_length, pad_value=0.0),
            ),
            dim=0,
        )
        # Mask the prompt but not the completion for the loss
        loss_mask = torch.zeros_like(attention_mask)
        label_ids = torch.cat(
            (
                pad_to_length(batch["chosen_labels"], max_seq_length, pad_value=IGNORE_INDEX),
                pad_to_length(batch["rejected_labels"], max_seq_length, pad_value=IGNORE_INDEX)
            ),
            dim=0,
        )
        loss_mask[label_ids != IGNORE_INDEX] = 1

        # Flush and truncate
        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                attention_mask = attention_mask[:, :self.max_length]
                input_ids = input_ids[:, :self.max_length]
                loss_mask = loss_mask[:, :self.max_length]
            elif self.truncation_mode == "keep_end":
                # Flush right before truncating left, then flush left
                # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                loss_mask = loss_mask[:, -self.max_length:]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                    "'keep_start']."
                )
        else:
            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        # Add logits_to_keep optimization
        if self.use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
            model_kwargs["logits_to_keep"] = logits_to_keep

        model_kwargs["output_hidden_states"] = True

        # Add padding-free training support
        if self.padding_free:
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        outputs = unwrapped_model(
            input_ids,
            use_cache=False,
            **model_kwargs,
        )
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state[:, :-1]
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1][:, :-1]

        # Get reference hidden states if needed
        ref_hidden_states = None
        if not self.reference_free and self.ref_model is not None:
            unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)

            ref_outputs = unwrapped_ref_model(
                input_ids,
                use_cache=False,
                **model_kwargs,
            )
            if hasattr(ref_outputs, "last_hidden_state"):
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]
            elif hasattr(ref_outputs, "hidden_states"):
                ref_hidden_states = ref_outputs.hidden_states[-1][:, :-1]
        elif not self.reference_free:
            unwrapped_ref_model = unwrapped_model

            with self.null_ref_context():
                ref_outputs = unwrapped_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **model_kwargs,
                )
                if hasattr(ref_outputs, "last_hidden_state"):
                    ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]
                elif hasattr(ref_outputs, "hidden_states"):
                    ref_hidden_states = ref_outputs.hidden_states[-1][:, :-1]

        masked_input_ids = torch.where(loss_mask != 0, input_ids, self.label_pad_token_id)
        labels = masked_input_ids[:, 1:]  # Shift right for casual LM
        # Get the LM head
        lm_head = unwrapped_model.get_output_embeddings()

        # Get reference model weights if needed
        ref_weight = None
        ref_bias = None
        if not self.reference_free:
            if self.ref_model is not None:
                ref_lm_head = unwrapped_ref_model.get_output_embeddings()
            else:
                with self.null_ref_context():
                    ref_lm_head = unwrapped_model.get_output_embeddings()
            ref_weight = ref_lm_head.weight
            ref_bias = ref_lm_head.bias if hasattr(ref_lm_head, "bias") else None

        # hidden_states = hidden_states.to(lm_head.weight.dtype)
        # ref_hidden_states = ref_hidden_states.to(ref_weight.dtype)
        # Compute loss using Liger kernel
        loss_output = self.dpo_loss_fn(
            lm_head.weight,
            hidden_states,
            labels,
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            ref_input=ref_hidden_states if not self.reference_free else None,
            ref_weight=ref_weight if not self.reference_free else None,
            ref_bias=ref_bias if not self.reference_free else None,
        )
        (
            loss,
            (chosen_logps, rejected_logps, chosen_logits_mean, rejected_logits_mean, nll_loss, *aux_outputs),
        ) = loss_output

        output = {
            "loss": loss,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logits_mean,
            "mean_rejected_logits": rejected_logits_mean,
            "nll_loss": nll_loss,
            "chosen_rewards": aux_outputs[0],
            "rejected_rewards": aux_outputs[1],
        }
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ):
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """

        args: DPOArguments = self.args
        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Repeat non chosen/rejected tensors to perform generation for chosen and rejected in a single forward pass
        for k, v in batch.items():
            if k not in args.unused_items_for_generation:
                if isinstance(v, torch.Tensor):
                    if k.startswith("chosen_") or k.startswith("rejected_"):
                        continue
                    else:
                        model_kwargs[k] = torch.cat((v, v), dim=0)
                else:
                    model_kwargs[k] = v

        # Concatenate the prompt and completion inputs
        # NOTE here [chosen/rejected]_input_ids contain both prompt and completion tokens!
        # We can use `label_ids` to get completion mask (label_ids != IGNORE_INDEX)
        max_seq_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        num_examples = batch["chosen_input_ids"].shape[0]
        input_ids = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_seq_length, pad_value=self.padding_value),
                pad_to_length(batch["rejected_input_ids"], max_seq_length, pad_value=self.padding_value)
            ),
            dim=0,
        )
        attention_mask = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_seq_length, pad_value=0.0),
                pad_to_length(batch["rejected_attention_mask"], max_seq_length, pad_value=0.0),
            ),
            dim=0,
        )
        # Mask the prompt but not the completion for the loss
        loss_mask = torch.zeros_like(attention_mask)
        label_ids = torch.cat(
            (
                pad_to_length(batch["chosen_labels"], max_seq_length, pad_value=IGNORE_INDEX),
                pad_to_length(batch["rejected_labels"], max_seq_length, pad_value=IGNORE_INDEX)
            ),
            dim=0,
        )
        loss_mask[label_ids != IGNORE_INDEX] = 1

        # Flush and truncate
        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                attention_mask = attention_mask[:, :self.max_length]
                input_ids = input_ids[:, :self.max_length]
                loss_mask = loss_mask[:, :self.max_length]
            elif self.truncation_mode == "keep_end":
                # Flush right before truncating left, then flush left
                # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                loss_mask = loss_mask[:, -self.max_length:]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                    "'keep_start']."
                )
        else:
            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        if self.use_logits_to_keep:
            # Compute logits_to_keep based on loss_mask pattern:
            # [[0, 0, 0, x, x, x, x],
            #  [0, 0, 0, x, x, x, 0]]
            #         ^ start computing logits from here ([:, -(7-3+1):])
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
            model_kwargs["logits_to_keep"] = logits_to_keep

        model_kwargs["output_hidden_states"] = True

        if self.padding_free:
            # Flatten the input_ids, position_ids, and loss_mask
            # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
            #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            # Align labels with logits
            # logits:    -,  -, [x2, x3, x4, x5, x6]
            #                     ^ --------- ^       after logits[:, :-1, :]
            # labels:   [y0, y1, y2, y3, y4, y5, y6]
            #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
            # loss_mask: [0,  0,  0,  1,  1,  1,  1]
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        if args.ld_alpha is not None and not is_ref_model:
            # Compute response lengths based on loss_mask
            completion_lengths = loss_mask.sum(dim=1)

            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        args: DPOArguments = self.args

        if args.use_liger_loss:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            chosen_rewards = model_output["chosen_rewards"]
            rejected_rewards = model_output["rejected_rewards"]
        else:
            model_output = self.concatenated_forward(model, batch)

            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
            )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if args.rpo_alpha is not None:
            losses = losses + args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )

        # if args.rpo_alpha is not None:
        #     metrics["nll_loss"] = (
        #         self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
        #     )
        # if self.aux_loss_enabled:
        #     metrics["aux_loss"] = (
        #         self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
        #     )

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:

        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)

        if return_outputs:
            return loss, metrics

        return loss

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.padding_value,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.padding_value,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.padding_value,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.padding_value)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        ref_output = pad_to_length(ref_output, self.max_length, self.padding_value)
        ref_output_decoded = self.processing_class.batch_decode(ref_output, skip_special_tokens=True)

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[list[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
    #     `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """

    #     # Sample and save to game log if requested (for one batch to save time)
    #     if self.generate_during_eval:
    #         # Generate random indices within the range of the total number of samples
    #         num_samples = len(dataloader.dataset)
    #         random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

    #         # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
    #         random_batch_dataset = dataloader.dataset.select(random_indices)
    #         random_batch = self.data_collator(random_batch_dataset)
    #         random_batch = self._prepare_inputs(random_batch)

    #         policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, random_batch)

    #     # Base evaluation
    #     initial_output = super().evaluation_loop(
    #         dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
    #     )

    #     return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)
