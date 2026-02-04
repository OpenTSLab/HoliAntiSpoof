import argparse
from pathlib import Path
import re
import json

import torch
import transformers
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import TrainingArguments

from qwenvl.train.utils import (
    KeywordsStoppingCriteria,
    load_maybe_lora_ckpt,
    apply_liger_kernel_to_qwen2_5_vl,
    initialize_model,
    rank0_print,
)
from evaluation.parsing_utils import SpoofingParser


def search_sequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    if len(haystack) < len(needle):
        return -1
    L = len(needle)
    matches = (haystack.unfold(0, L, 1) == needle).all(dim=1)
    idxs = matches.nonzero(as_tuple=True)[0]
    return idxs[0] if len(idxs) > 0 else -1


def inference():
    entry_parser = argparse.ArgumentParser()
    entry_parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    entry_parser.add_argument("--overrides", "-o", nargs="+", default=[], help="Override options in the config file.")

    args = entry_parser.parse_args()

    config_fpath = args.config_file
    config_dir = Path(config_fpath).parent.absolute().__str__()
    config_fname = Path(config_fpath).name
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_fname, overrides=args.overrides)
    OmegaConf.set_struct(config, False)  # allow new key, to merge with `exp_config`

    # === Load config from experiment directory ===
    # Step:
    #   - Get `exp_dir` from config file or overrides
    #   - Read training config from `exp_dir / "config.yaml"`
    #   - Merge it with inference config from `args.config_file`
    #   - Apply command line overrides finally

    ckpt_dir = config["ckpt_dir"]
    exp_dir = config["exp_dir"]
    assert ckpt_dir is not None
    if exp_dir is None:
        exp_dir = Path(ckpt_dir).parent

    # === Read training config from `exp_dir / "config.yaml"` ===
    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    del exp_config["data_dict"]  # TODO maybe `data_dict` is not the only possible data keys
    del exp_config["global"]

    # === Merge it with inference config from `args.config_file` ===
    config = OmegaConf.merge(config, exp_config)
    config = OmegaConf.to_container(config, resolve=True)

    # === Parse HF training args (for distributed env) ===
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args, = parser.parse_dict(config["training_args"], allow_extra_keys=True)

    apply_liger_kernel_to_qwen2_5_vl()

    rank = dist.get_rank()
    local_rank = training_args.local_rank
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    model = initialize_model(config["model"], training_args)

    # === Load experiment checkpoint ===
    if "lora_config" in config["model"] and config["model"]["lora_config"]:
        is_lora = True
    else:
        is_lora = False

    model = load_maybe_lora_ckpt(model, ckpt_dir, is_lora=is_lora)

    # === Prepare model for distributed evaluation ===
    model.eval()
    model.to(torch.device(f"cuda:{local_rank}"))
    unwrapped_model = model

    dataset = instantiate(config["test_dataloader"]["dataset"], _convert_="all")
    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    dataloader = instantiate(config["test_dataloader"], sampler=dist_sampler, _convert_="all")

    result = []

    # text_parser = SpoofingParser(config["global"]["data_format"])
    if config["global"]["data_format"] == "json":
        real_token_vocab_idx = dataset.tokenizer.convert_tokens_to_ids("real")
        fake_token_vocab_idx = dataset.tokenizer.convert_tokens_to_ids("fake")
        real_fake_seq_idx = 6
    elif config["global"]["data_format"] == "cot":
        real_token_vocab_idx = dataset.tokenizer.encode(" real", add_special_tokens=False)[0]
        fake_token_vocab_idx = dataset.tokenizer.encode(" a", add_special_tokens=False)[0]  # "a spoof"
        search_needle = dataset.tokenizer.encode(
            "</think>\n\nThe utterance is", add_special_tokens=False, return_tensors="pt"
        )[0].to(unwrapped_model.device)
    elif config["global"]["data_format"] == "real_fake_raw":
        real_token_vocab_idx = dataset.tokenizer.convert_tokens_to_ids("real")
        fake_token_vocab_idx = dataset.tokenizer.convert_tokens_to_ids("fake")
        real_fake_seq_idx = 0
    elif config["global"]["data_format"] == "raw_text":
        real_token_vocab_idx = dataset.tokenizer.encode(" real", add_special_tokens=False)[0]
        fake_token_vocab_idx = dataset.tokenizer.encode(" fake", add_special_tokens=False)[0]
        real_fake_seq_idx = 4

    set_breakpoint = config.get("set_breakpoint", False)

    for batch_idx, inputs in enumerate(tqdm(dataloader, desc=f"RANK {rank}", disable=rank != 0)):
        if inputs:
            res_i = {
                "video": inputs.pop("video")[0] if "video" in inputs else None,
                "image": inputs.pop("image")[0] if "image" in inputs else None,
                "prompt": inputs.pop("prompt")[0] if "prompt" in inputs else None,
                "ref": inputs.pop("ref")[0] if "ref" in inputs else None,
                "audio": inputs.pop("audio")[0] if "audio" in inputs else None,
                "use_audio": inputs.pop("use_audio")[0] if "use_audio" in inputs else None,
            }

            inputs_on_device = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_on_device[k] = v.to(unwrapped_model.device)
                else:
                    inputs_on_device[k] = v

            for unused_item in config["unused_items"]:
                inputs_on_device.pop(unused_item)

            keywords = ["<|im_end|>", "<|endoftext|>"]
            pattern = re.compile("|".join(re.escape(k) for k in keywords))
            stopping_criteria = KeywordsStoppingCriteria(keywords, dataset.tokenizer, inputs_on_device["input_ids"])
            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    **inputs_on_device,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=config["do_sample"],
                    top_p=0.9,
                    stopping_criteria=[stopping_criteria],
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            output_trimmed = outputs.sequences[0, len(inputs_on_device["input_ids"][0]):]
            output_text = dataset.tokenizer.decode(
                output_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            if config["global"]["data_format"] == "cot":
                if "</think>\n\nThe utterance is" not in output_text:
                    real_fake_seq_idx = -3
                else:
                    real_fake_seq_idx = search_sequence(output_trimmed, search_needle) + len(search_needle)

            real_logit = outputs.scores[real_fake_seq_idx][0, real_token_vocab_idx].item()
            fake_logit = outputs.scores[real_fake_seq_idx][0, fake_token_vocab_idx].item()
            prob = torch.softmax(outputs.scores[real_fake_seq_idx], dim=-1)
            real_prob = prob[0, real_token_vocab_idx].item()
            fake_prob = prob[0, fake_token_vocab_idx].item()

            matched = pattern.search(output_text)
            if matched:
                output_text = output_text[:matched.start()]

            if torch.cuda.device_count() == 1 and set_breakpoint:
                print(f"fake logit: {fake_logit}, real logit: {real_logit}, output: {output_text}, ref: {res_i['ref']}")
                import ipdb
                ipdb.set_trace()

            res_i["pred"] = output_text
            res_i["real_logit"] = real_logit
            res_i["fake_logit"] = fake_logit
            res_i["real_prob"] = real_prob
            res_i["fake_prob"] = fake_prob
            result.append(res_i)

    output_fpath = exp_dir / config["output_fname"]
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
