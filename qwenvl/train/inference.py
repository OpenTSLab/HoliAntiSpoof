import argparse
from pathlib import Path
import re
import json

import torch
import torch.nn as nn
import transformers
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    InferenceArguments,
)
from qwenvl.train.train_qwen import (
    apply_liger_kernel_to_qwen2_5_vl,
    initialize_model,
    initialize_processor,
    load_maybe_lora_ckpt,
    collate_fn_single_sample,
    collate_fn_single_sample_qwen_audio,
)
from qwenvl.train.utils import get_torch_dtype
from qwenvl.data.data_qwen import LazySupervisedDataset
from qwenvl.train.utils import KeywordsStoppingCriteria


def inference():
    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    entry_parser.add_argument("--ckpt_dir", "-ckpt", type=str, required=True, help="path to checkpoint directory")
    entry_parser.add_argument("--exp_dir", type=str, required=False, help="path to experiment directory")
    entry_parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()
    ckpt_dir = args.ckpt_dir
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = Path(ckpt_dir).parent

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    config_fpath = args.config_file
    config_dir = Path(config_fpath).parent.absolute().__str__()
    config_fname = Path(config_fpath).name
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_fname, overrides=args.options)
    config = OmegaConf.to_container(config, resolve=True)

    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    exp_config = OmegaConf.to_container(exp_config, resolve=True)
    config.update(exp_config)

    model_args, data_args, training_args, infer_args = parser.parse_dict(config, allow_extra_keys=True)
    data_args.run_test = True
    data_args.model_type = model_args.model_type
    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    tokenizer = initialize_processor(model_args, data_args, training_args)
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        dataset_list=infer_args.infer_datasets,
        data_args=data_args,
        dataset_max_samples=100000,
    )

    rank = dist.get_rank()
    local_rank = training_args.local_rank
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    model = initialize_model(exp_config, training_args)

    # load experiment checkpoint
    if "lora_config" in exp_config["model"]:
        is_lora = True
    else:
        is_lora = False
    model = load_maybe_lora_ckpt(model, ckpt_dir, is_lora=is_lora)

    model.eval()
    model.to(torch.device(f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    unwrapped_model = model.module

    result = []

    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    collate_fn = collate_fn_single_sample if model_args.model_type != "qwen2audio" \
        else collate_fn_single_sample_qwen_audio
    loader = DataLoader(
        dataset,
        batch_size=infer_args.eval_batch_size,
        sampler=dist_sampler,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    for batch_idx, inputs in enumerate(tqdm(loader, desc=f"RANK {rank}", disable=rank != 0)):
        if inputs:
            res_i = {
                "video": inputs.pop("video", None),
                "image": inputs.pop("image", None),
                "prompt": inputs.pop("prompt", None),
                "ref": inputs.pop("ref", None),
                "audio": inputs.pop("audio", None),
                "use_audio": inputs.pop("use_audio", False),
            }

            new_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    new_inputs[k] = v.to(unwrapped_model.device)
                elif k == 'video_second_per_grid' and v is not None:
                    new_inputs[k] = torch.tensor([v], device=unwrapped_model.device)
                elif k == "keywords":
                    continue
                elif v is None:
                    continue
                else:
                    new_inputs[k] = v
            inputs = new_inputs

            keywords = ["<|im_end|>", "<|endoftext|>"]
            pattern = re.compile("|".join(re.escape(k) for k in keywords))
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs["input_ids"])
            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    **inputs,
                    max_new_tokens=model_args.max_new_tokens,
                    do_sample=infer_args.do_sample,
                    top_p=0.9,
                    stopping_criteria=[stopping_criteria],
                )
            output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
            output_text = tokenizer.decode(
                output_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            matched = pattern.search(output_text)
            if matched:
                output_text = output_text[:matched.start()]
            res_i["pred"] = output_text
            result.append(res_i)

    output_fpath = exp_dir / infer_args.output_fname
    tmp_dir = output_fpath.with_suffix("")
    if rank == 0:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    with open(tmp_dir / f"infer_results_rank{rank}.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    dist.barrier()

    if rank == 0:
        merged_output = []
        for file in tmp_dir.glob("infer_results_rank*.json"):
            merged_output.extend(json.load(open(file, "r")))
        output_fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_fpath, "w") as f:
            json.dump(merged_output, f, indent=2, ensure_ascii=False)
        print(f"Inference results saved to {output_fpath.__str__()}")
        for file in tmp_dir.glob("infer_results_rank*.json"):
            file.unlink()
        tmp_dir.rmdir()


if __name__ == "__main__":
    inference()
