from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from peft import PeftModel
from omegaconf import OmegaConf
from hydra.utils import instantiate

from qwenvl.model import MODEL_TO_LORA_EXCLUDE_MODULES
from qwenvl.train.utils import load_non_lora_params_from_ckpt

torch_dtype = torch.bfloat16

ckpt_dir = "experiments/all_data/r_64/checkpoint-200000"
ckpt_dir = Path(ckpt_dir)
exp_dir = ckpt_dir.parent
config = OmegaConf.load(exp_dir / "config.yaml")
config = OmegaConf.to_container(config)

config["model"]["architecture"]["torch_dtype"] = torch_dtype
model = instantiate(config["model"]["architecture"], _convert_="all")

model = PeftModel.from_pretrained(model, ckpt_dir)
