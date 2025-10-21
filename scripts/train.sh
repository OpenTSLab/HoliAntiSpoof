export PYTHONPATH=.

TORCHRUN="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/py310/bin/torchrun"

$TORCHRUN \
    qwenvl/train/train_qwen.py \
    --config_file configs/train.yaml \
    --options \
    trainer.output_dir=experiments/all_data/train_audio_encoder_dora_r_64 \
    trainer.run_name=all_data_train_encoder_dora_r_64