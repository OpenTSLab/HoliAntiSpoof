infer_dir="experiments/all_data/r_64/infer_step20000"

# Parse command line arguments
__snapshot_before=$(mktemp)
declare -p > "$__snapshot_before"

# Check for --help option
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --infer_dir DIR      Path to inference results directory
                       Default: experiments/all_data/r_64/infer_step20000
  --help, -h           Show this help message and exit

Examples:
  $0 --infer_dir experiments/my_model/infer_step10000

Description:
  This script runs a series of evaluation tasks, including:
  - real / fake evaluation
  - fake method evaluation
  - fake region evaluation
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

echo "eval_all.sh: Running with the following parameters:"
echo "infer_dir: $infer_dir"

export PYTHONPATH=.

python evaluation/eval_real_fake.py \
    --config-name eval_had \
    infer_dir=$infer_dir

python evaluation/eval_spoof_method.py \
    --config-name eval_had \
    infer_dir=$infer_dir \
    ++is_coarse=True

python evaluation/eval_fake_region.py \
    --config-name eval_had \
    infer_dir=$infer_dir