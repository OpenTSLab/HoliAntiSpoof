export PYTHONPATH=.
export TRITON_CACHE_DIR=/mnt/shared-storage-user/xuxuenan/triton_cache

TORCHRUN="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/qwen/bin/torchrun"

cd /mnt/shared-storage-user/xuxuenan/workspace/qwen_training

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

$TORCHRUN --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
    qwenvl/train/train_qwen_rl.py \
    --config_file configs/train_grpo.yaml \
    --options \
    trainer.output_dir=experiments/all_data/r_64_grpo \
    trainer.run_name=r_64_grpo