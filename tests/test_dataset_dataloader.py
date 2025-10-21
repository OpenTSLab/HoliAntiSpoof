import argparse

from omegaconf import OmegaConf
import hydra

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", required=True)

args = parser.parse_args()
config = OmegaConf.load(args.config)
dataset = hydra.utils.instantiate(config, _convert_="all")

for idx, item in enumerate(tqdm(dataset)):
    if idx >= 1000:
        break
    # pass
    # import ipdb
    # ipdb.set_trace()
