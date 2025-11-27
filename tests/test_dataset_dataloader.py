import argparse

from omegaconf import OmegaConf
import hydra

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", required=False, default=None)

args = parser.parse_args()

if args.config:
    config = OmegaConf.load(args.config)
    if "data_dict" in config:
        dataset = hydra.utils.instantiate(config["data_dict"]["train"], _convert_="all")
        collate_fn = hydra.utils.instantiate(config["train_dataloader"]["collate_fn"], _convert_='all')
    else:
        dataset = hydra.utils.instantiate(config["dataset"], _convert_="all")
        collate_fn = hydra.utils.instantiate(config["collate_fn"], _convert_='all')

else:
    dataset_config_str = r"""
_target_: qwenvl.data.data_qwen.AudioSpoofingPreferenceDataset
stage: training
train_type: dpo
model_type: qwen2.5omni
dataset_list:
  - data_json/asvspoof2019/train_preference.json
#   - data/partial_spoof/train.json
processor:
  _target_: qwenvl.data.processing_qwen2_5_omni.Qwen2_5OmniProcessor.from_pretrained
  pretrained_model_name_or_path: /mnt/shared-storage-user/brainllm-share/checkpoints/Qwen2.5-Omni-7B
data_format: "json"
# embedding_dir: /mnt/shared-storage-user/xuxuenan/workspace/nii_anti_deepfake/embeddings/mms_300m
    """

    collate_config_str = r"""
_target_: qwenvl.data.data_qwen.OmniPreferenceCollator
padding_config:
  chosen_input_ids: 151643
  chosen_labels: -100
  rejected_input_ids: 151643
  rejected_labels: -100
torchify_keys:
  - video_second_per_grid
  - spoof_embeds
    """
    config = OmegaConf.create(dataset_config_str)
    dataset = hydra.utils.instantiate(config, _convert_='all')

    config = OmegaConf.create(collate_config_str)
    collate_fn = hydra.utils.instantiate(config, _convert_='all')

for idx, item in enumerate(tqdm(dataset)):
    if idx >= 1000:
        break

item1 = dataset[0]
item2 = dataset[1]
batch = collate_fn([item1, item2])
print(batch)
