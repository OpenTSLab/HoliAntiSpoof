cd /mnt/shared-storage-user/xuxuenan/workspace/qwen_training

export PYTHONPATH=.

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

ckpt_path="experiments/all_data/r_64/infer_step20000"
output_dir="infer_step20000"

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
  --output_dir DIR     Output directory name for inference results
                       Default: infer_step20000
  --help, -h           Show this help message and exit

Examples:
  $0 --ckpt_path my_checkpoint --output_dir my_output
  $0 --ckpt_path experiments/my_model/checkpoint-100000 --output_dir infer_step100000

Description:
  This script runs inference on multiple datasets:
  - asvspoof2019
  - codecfake_ntu
  - ljspeech
  - partial_edit
  - partial_spoof
  - recent_tts
  - sine
  - vctk
  - wavefake

  The script automatically detects the number of available GPUs and runs
  distributed inference using torchrun.
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

TORCHRUN="/mnt/shared-storage-user/xuxuenan/miniconda3/envs/qwen/bin/torchrun"

datasets=(
  "asvspoof2019 data/asvspoof2019/eval_2000.json"
  "codecfake_ntu data/codecfake_ntu/test_2000.json"
  "ljspeech data/ljspeech/test.json"
  "partial_edit data/partial_edit/test_2000.json"
  "partial_spoof data/partial_spoof/eval_2000.json"
  "recent_tts data/recent_tts/test_2000.json"
  "sine data/sine/test_1500_no_vocoder.json"
  "vctk data/vctk/test_2000.json"
  "wavefake data/wavefake/test.json"
)


for pair in "${datasets[@]}"; do
    read -r dataset_name test_file <<< "$pair"

    echo "Running inference for dataset: ${dataset_name}"

    $TORCHRUN --nproc_per_node=${N_GPUS} --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        qwenvl/train/inference.py \
        -c configs/infer.yaml \
        --options \
        ckpt_dir=$ckpt_path \
        data_dict.test.dataset_list.0=${test_file} \
        ++output_fname=$output_dir/${dataset_name}.json \
        ++test_dataloader.batch_size=1
done