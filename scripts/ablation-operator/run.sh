#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/ablation-operator
save_json_attn="$script_dir/record_attn.json"
save_json_mlp="$script_dir/record_mlp.json"

output_dir=$(pwd)/output_figures

# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json_attn" ]; then
    echo "[]" > "$save_json_attn"
else
    > "$save_json_attn"
    echo "[]" > "$save_json_attn"
fi

if [ ! -f "$save_json_mlp" ]; then
    echo "[]" > "$save_json_mlp"
else
    > "$save_json_mlp"
    echo "[]" > "$save_json_mlp"
fi

# 检查 src 目录是否在 PYTHONPATH 中
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi
mkdir -p "$script_dir/output"
python "$script_dir/benchmark_attn.py"  \
        --save_json "$save_json_attn" \
        > "$script_dir/output/attn.log"

python "$script_dir/benchmark_mlp.py"  \
        --save_json "$save_json_mlp" \
        > "$script_dir/output/mlp.log"

python "$script_dir/plot_attn.py" \
        --save_json "$save_json_attn" \
        --output "$output_dir/exp-ablation-operator-attn.png"

python "$script_dir/plot_mlp.py" \
        --save_json "$save_json_mlp" \
        --output "$output_dir/exp-ablation-operator-mlp.png"
