import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    attn_implementation: str = field(default="flash_attention_2")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    infer_fname: str = field(default="inference.json")


@dataclass
class InferenceArguments:
    infer_datasets: list[str] = field(default_factory=list)
    output_fname: str = field(default="inference.json")
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    eval_batch_size: int = field(default=1)
