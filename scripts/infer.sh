cd /mnt/shared-storage-user/xuxuenan/workspace/qwen_training

export PYTHONPATH=.

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

ckpt_path="experiments/json_format/all_data_wavefake_ljspeech/qwen2_5omni/lora_r_64_alpha_128_audio_encoder_trainable_steps_200k_metric_loss_lr_1e-5/checkpoint-200000"
output_dir="infer_step200000"

TORCHRUN="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/py310/bin/torchrun"

# torchrun --nproc_per_node=${N_GPUS} \
$TORCHRUN --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
    qwenvl/train/inference.py \
    -c configs/infer.yaml \
    -ckpt $ckpt_path \
    --options \
    infer_datasets.0=data/partial_edit/test_2000.json \
    ++output_fname=$output_dir/partial_edit.json \
    ++eval_batch_size=1
