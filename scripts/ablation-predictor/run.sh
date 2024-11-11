#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/ablation-predictor

output_dir=$(pwd)/output_figures

# 检查 src 目录是否在 PYTHONPATH 中
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi

python "$script_dir/loss_random_attn.py"  \
        --model_name facebook/opt-1.3b \
        > "$script_dir/loss_opt-1.3b_random.log"

python "$script_dir/loss_random_mlp.py"  \
        --model_name facebook/opt-1.3b \
        > "$script_dir/loss_opt-1.3b_mlp.log"

python "$script_dir/loss_exposer.py"  \
        --model_name facebook/opt-1.3b \
        > "$script_dir/loss_opt-1.3b_exposer.log"

python "$script_dir/plot_loss_curve.py" \
        --cur_path "$script_dir" \
        --output "$output_dir/exp-ablation-predictor-loss.png"
