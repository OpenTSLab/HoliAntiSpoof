infer_dir=experiments/all_data_wavefake_ljspeech/qwen2_5omni/lora_r_64_alpha_128_audio_encoder_trainable_steps_20k_metric_loss_lr_5e-6/infer_step12000
eval_keyword=1

# Parse command line arguments
__snapshot_before=$(mktemp)
declare -p > "$__snapshot_before"

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

echo "eval_all.sh: Running with the following parameters:"
echo "infer_dir: $infer_dir"
echo "eval_keyword: $eval_keyword"

export PYTHONPATH=.

# python evaluation/eval_real_fake.py \
#     --config-name eval_asvspoof2019 \
#     infer_dir=$infer_dir

python evaluation/eval_real_fake.py \
    --config-name eval_composite \
    infer_dir=$infer_dir

python evaluation/eval_spoof_method.py \
    --config-name eval_composite \
    infer_dir=$infer_dir \
    ++is_coarse=True

python evaluation/eval_spoof_method.py \
    --config-name eval_composite \
    infer_dir=$infer_dir \
    ++is_coarse=False

python evaluation/eval_fake_region.py \
    --config-name eval_fake_region \
    infer_dir=$infer_dir

if [ "${eval_keyword}" -eq 1 ]; then
    python evaluation/eval_edit_keyword.py \
        --config-name eval_composite \
        infer_dir=$infer_dir
fi