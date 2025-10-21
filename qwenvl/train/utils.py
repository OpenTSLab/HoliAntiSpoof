import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import StoppingCriteria
from omegaconf import OmegaConf


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
