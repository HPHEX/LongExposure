#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/ablation-breakdown
save_json="$script_dir/record.json"

output_dir=$(pwd)/output_figures

# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json" ]; then
    echo "[]" > "$save_json"
else
    > "$save_json"
    echo "[]" > "$save_json"
fi



# 设置 PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi

mkdir -p "$script_dir/output"
python "$script_dir/exposer_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl  \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/exposer_adapter.log"

python "$script_dir/exposer_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/exposer_bitfit.log"

python "$script_dir/exposer_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/exposer_lora.log"

python "$script_dir/torch_full.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/torch_full.log"

python "$script_dir/torch_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/torch_lora.log"

python "$script_dir/torch_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/torch_adapter.log"

python "$script_dir/torch_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --seq_len 512 \
        --save_json "$save_json" \
        --batch_size 4 > "$script_dir/output/torch_bitfit.log"


python "$script_dir/plot_breakdown.py" \
        --save_json "$save_json" \
        --output "$output_dir/exp-ablation-breakdown.png" 