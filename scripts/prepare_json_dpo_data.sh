cd /mnt/shared-storage-user/xuxuenan/workspace/qwen_training

export PYTHONPATH=.

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

ckpt_path="experiments/all_data/r_64/infer_step20000"
output_path="experiments/all_data/r_64/infer_step20000_preference_pairs"

# Parse command line arguments
__snapshot_before=$(mktemp)
declare -p > "$__snapshot_before"

# Check for --help option
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --ckpt_path PATH     Path to checkpoint directory
                       Default: experiments/all_data/r_64/infer_step20000
  --help, -h           Show this help message and exit

Examples:
  $0 --ckpt_path experiments/my_model/checkpoint-100000

EOF
    rm -f "$__snapshot_before"
    exit 0
fi

while [[ $# -gt 0 ]]; do
    key="$1"
    val="$2"

    if [[ "$key" =~ ^--(.+) ]]; then
        var_name="${BASH_REMATCH[1]}"

        if [[ -n "$val" && ! "$val" =~ ^-- ]]; then
            if grep -q -E "^declare .* $var_name=" "$__snapshot_before"; then
                eval "$var_name=\"\$val\""
            fi
            shift 2
        else
            shift 1
        fi
    else
        shift 1
    fi
done

rm -f "$__snapshot_before"

torchrun_bin="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/qwen/bin/torchrun"

train_datasets=(
  "asvspoof2019 data_json/asvspoof2019/train.json"
  "codecfake_ntu data_json/codecfake_ntu/train.json"
  "ljspeech data_json/ljspeech/train.json"
  "partial_edit data_json/partial_edit/train.json"
  "partial_spoof data_json/partial_spoof/train.json"
  "recent_tts data_json/recent_tts/train.json"
#   "sine data_json/sine/train.json"
  "sine_no_vocoder data_json/sine/train_no_vocoder.json"
  "vctk data_json/vctk/train.json"
  "wavefake data_json/wavefake/train.json"
)

val_datasets=(
  "asvspoof2019 data_json/asvspoof2019/dev.json"
  "codecfake_ntu data_json/codecfake_ntu/val.json"
  "ljspeech data_json/ljspeech/val.json"
  "partial_edit data_json/partial_edit/val.json"
  "partial_spoof data_json/partial_spoof/dev.json"
  "recent_tts data_json/recent_tts/val.json"
#   "sine data_json/sine/val.json"
  "sine_no_vocoder data_json/sine/val_no_vocoder.json"
  "vctk data_json/vctk/val.json"
  "wavefake data_json/wavefake/val.json"
)


# for pair in "${train_datasets[@]}"; do
#     read -r dataset_name test_file <<< "$pair"

#     echo "Running inference for dataset: ${dataset_name}"

#     ${torchrun_bin} --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
#         qwenvl/train/prepare_preference_pairs.py \
#         -c configs/infer.yaml \
#         --options \
#         ckpt_dir=$ckpt_path \
#         data_dict.test.dataset_list.0=${test_file} \
#         ++output_fname=${output_path}/train/${dataset_name}.json \
#         ++test_dataloader.batch_size=1 \
#         ++num_generations=8 \
#         ++temperature=1.0 \
#         ++data_dict.test.dataset_max_samples=1000000
# done

for pair in "${val_datasets[@]}"; do
    read -r dataset_name test_file <<< "$pair"

    echo "Running inference for dataset: ${dataset_name}"

    ${torchrun_bin} --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        qwenvl/train/prepare_preference_pairs.py \
        -c configs/infer.yaml \
        --options \
        ckpt_dir=$ckpt_path \
        data_dict.test.dataset_list.0=${test_file} \
        ++output_fname=${output_path}/val/${dataset_name}.json \
        ++test_dataloader.batch_size=1 \
        ++num_generations=8 \
        ++temperature=1.0 \
        ++data_dict.test.dataset_max_samples=1000000
done