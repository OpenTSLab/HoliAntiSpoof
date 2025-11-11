# README

## Data Format

```json
[
  {
    "video": "xxx",
    "audio": "xxx",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nYour Prompt Here."
      },
      {
        "from": "gpt",
        "value": "xxx"
      }
    ]
  },
  // ...
]
```

For audio-only data, simply skip "video" item and only leave "audio" and "conversations" items in the JSON object.

## Training

```bash
torchrun xxx \
  qwenvl/train/train_qwen.py \
  --config_file configs/train.yaml
```

Training on ASVSpoof2019 and using SpoofingWithEmbeddingDataset:
```bash
torchrun xxx \
  qwenvl/train/train_qwen.py \
  --config_file configs/train.yaml \
  --options \
  data/datasets@data_dict.data_lists=asvspoof2019 \
  data@data_dict=spoofing_with_embed
```

Arguments after `--options` follow hydra override rules.

## Inference

```bash
torchrun xxx \
    qwenvl/train/inference.py \
    -c configs/infer.yaml \
    -ckpt $ckpt_path \
    --options \
    data_dict.test.dataset_list.0=xxxx.json \
    ++output_fname=xxxx.json \
    ++test_dataloader.batch_size=1
```

Results will be saved to `$ckpt_path/../xxxx.json`.

See scripts in `scripts` for more examples.
