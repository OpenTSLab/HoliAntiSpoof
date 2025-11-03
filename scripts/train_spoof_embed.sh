export PYTHONPATH=.

TORCHRUN="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/py310/bin/torchrun"


$TORCHRUN --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
    qwenvl/train/train_qwen.py \
    --config_file configs/train.yaml \
    --options \
    data@data_dict=spoofing_with_embed \
    trainer.output_dir=experiments/all_data/r_64_mms_300m \
    trainer.run_name=r_64_mms_300m \
    model=qwen2_5_omni_with_spoof