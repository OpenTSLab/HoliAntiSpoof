from pathlib import Path
from typing import (
    Dict,
    Optional,
    Optional,
)
import time
import json

import numpy as np
from packaging import version
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.utils import (
    is_accelerate_available,
    is_torch_xla_available,
)
from transformers.trainer import (
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    denumpify_detensorize,
)
from transformers.integrations.deepspeed import deepspeed_init, is_deepspeed_available
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size,
)
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]

    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

TRAINER_STATE_NAME = "trainer_state.json"


class QwenVLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.train_type = kwargs.pop("train_type", "sft")
        super().__init__(*args, **kwargs)

        # Initialize log file path
        self.train_log_path = Path(self.args.output_dir) / "train.log"
        # Create parent directory if it doesn't exist
        self.train_log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model) if self.is_deepspeed_enabled or
                (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8") else
                self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                preds = logits[0].argmax(dim=-1)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(preds)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                        processing_class=self.processing_class,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        # if (
        #     self.compute_metrics is not None and all_preds is not None and all_labels is not None and
        #     not self.args.batch_eval_metrics
        # ):
        if self.compute_metrics is not None and not self.args.batch_eval_metrics:
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs),
                processing_class=self.processing_class,
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # def calc_dpo_loss(self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1):
    #     lm_head = self.model.lm_head.weight
    #     dpo_loss, (
    #         chosen_logp, reject_logp, chosen_logit, reject_logit, chosen_nll_loss, chosen_rewards, reject_rewards
    #     ) = self.dpo_loss_fct(lm_head, policy_input, policy_target, ref_input=ref_input, ref_weight=lm_head)
    #     if ce_loss is not None:
    #         loss = dpo_loss + beta * ce_loss
    #     else:
    #         loss = dpo_loss
    #     print(f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}")
    #     return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, and write to train.log file.
        
        Override to add file logging functionality. Logs are written to train.log when
        an epoch ends (indicated by presence of 'epoch' key in logs).
        """
        # Call parent's log method first
        super().log(logs, start_time)

        if "eval_loss" in logs and self.args.should_save:
            metric_name = self.compute_metrics.name
            log_entry = {
                "epoch": f"{logs['epoch']:.3g}",
                "eval_loss": f"{logs['eval_loss']:.3g}",
                f"eval_{metric_name}": f"{logs[f'eval_{metric_name}']:.3g}"
            }

            # Write to file (append mode)
            with open(self.train_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        train_output = super()._inner_training_loop(
            batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval
        )
        self._save_checkpoint(self.model, trial)  # save the last checkpoint at the end of training
        return train_output


class QwenVLTrainerWithPdb(QwenVLTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import ipdb
        ipdb.set_trace()
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
