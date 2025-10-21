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

## Training

```bash
torchrun xxx \
  qwenvl/train/train_qwen.py \
  --config_file configs/train.yaml \
  --options \

```

## Inference
(Not checked yet)

```bash
torchrun xxx \
    qwenvl/train/inference.py \
    -c configs/infer.yaml \
    -ckpt $ckpt_path \
    --options \
    infer_datasets.0=data/partial_edit/test_2000.json \
    ++output_fname=$output_dir/partial_edit.json \
    ++eval_batch_size=1
```
