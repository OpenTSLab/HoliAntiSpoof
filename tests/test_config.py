import argparse
from pathlib import Path

from omegaconf import OmegaConf
import hydra

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", "-c", required=True, type=str, help="Path to config YAML file.")
parser.add_argument("--options", nargs="+", default=[], help="Override options in the config file.")

args = parser.parse_args()

config_dir = Path(args.config_file).parent.absolute().__str__()
config_fname = Path(args.config_file).name
with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
    config = hydra.compose(config_name=config_fname, overrides=args.options)
OmegaConf.resolve(config)
config = OmegaConf.to_container(config)
