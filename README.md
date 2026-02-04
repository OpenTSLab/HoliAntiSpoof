# HoliAntiSpoof: Audio LLM for Holistic Speech Anti-Spoofing

This repository is the official code release for the paper "HoliAntiSpoof: Audio LLM for Holistic Speech Anti-Spoofing".

## Data and PreTrained Model
DailyTalkEdit and the semantic influence annotation of PartialEdit are provided [here](https://huggingface.co/datasets/wsntxxn/DailyTalkEdit).

The pretrained HoliAntiSpoof (DoRA rank = 64) checkpoint is available [here](https://huggingface.co/wsntxxn/HoliAntiSpoof).

## Quick Start

### Data Preparation

Prepare the anti-spoofing data based on formats in `example_data/partial_edit.json`, where `"audio"` is the absolute audio path.

### Training

```bash
export PYTHONPATH=.
torchrun <args> \
    qwenvl/train/train_qwen.py \
    -c example_configs/train_sft.yaml \
    -o \
    training_args.output_dir=experiments/holi_anti_spoof
```
Arguments after `-o`/`--overrides` follow hydra override rules, i.e., `arg1=val1 arg2=val2 ++arg3=val3`.

### Inference

```bash
torchrun <args> \
    qwenvl/train/inference.py \
    -c example_configs/infer.yaml \
    -ckpt <ckpt_path> \
    -o \
    data_dict.test.dataset_list.0=/path/to/infer/data.json \
    ++output_fname=path/to/output.json \
    ++test_dataloader.batch_size=1
```
Inference results will be saved to `<ckpt_path>/../path/to/output.json`.

### Evaluation

```bash
python evaluation/eval_real_fake.py \
    -c example_configs/eval/asvspoof2019.yaml \
    infer_dir=<infer_dir>
```
Corresponding settings are specified in `example_configs/eval/asvspoof2019.yaml`.

## Citation
