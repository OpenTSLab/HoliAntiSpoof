import argparse
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
from hydra import compose, initialize, initialize_config_dir
from hydra.core.config_store import ConfigStore

entry_parser = argparse.ArgumentParser()
entry_parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
entry_parser.add_argument("--exp_dir", type=str, required=False, default=None, help="path to experiment directory")
entry_parser.add_argument("--ckpt_dir", type=str, required=False, default=None, help="path to checkpoint directory")
entry_parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

args = entry_parser.parse_args()

ckpt_dir = args.ckpt_dir
exp_dir = args.exp_dir
assert ckpt_dir is not None
if exp_dir is None:
    exp_dir = Path(ckpt_dir).parent

config_fpath = args.config_file
config_dir = Path(config_fpath).parent.absolute().__str__()
config_fname = Path(config_fpath).name
with initialize_config_dir(config_dir=config_dir, version_base=None):
    config = compose(config_name=config_fname, overrides=args.options)

OmegaConf.set_struct(config, False)  # allow new key, to merge with `exp_config`

exp_config = OmegaConf.load(exp_dir / "config.yaml")
del exp_config["data_dict"]
# === Merge it with inference config from `args.config_file` ===
config = OmegaConf.merge(config, exp_config)
config = OmegaConf.to_container(config, resolve=True)

pprint(config["test_dataloader"])
