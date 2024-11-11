#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/overall-memory
save_json_350m="$script_dir/record_350m.json"
save_json_1b="$script_dir/record_1.3b.json"

output_dir=$(pwd)/output_figures

# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json_350m" ]; then
    echo "[]" > "$save_json_350m"
else
    > "$save_json_350m"
    echo "[]" > "$save_json_350m"
fi

if [ ! -f "$save_json_1b" ]; then
    echo "[]" > "$save_json_1b"
else
    > "$save_json_1b"
    echo "[]" > "$save_json_1b"
fi

# 设置 PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi

# 打印 PYTHONPATH 以确认
echo "PYTHONPATH: $PYTHONPATH"

# ------------------------------350m------------------------------
mkdir -p "$script_dir/output/opt350m"
python "$script_dir/torch_full.py" \
    --model_name facebook/opt-350m \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_350m" \
    > "$script_dir/output/opt350m/l128_torch_full.log"

python "$script_dir/torch_lora.py" \
    --model_name facebook/opt-350m \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_350m" \
    > "$script_dir/output/opt350m/l128_torch_lora.log"

python "$script_dir/exposer_lora.py" \
    --model_name facebook/opt-350m \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_350m" \
    > "$script_dir/output/opt350m/l128_exposer_lora.log"

python "$script_dir/exposer_lora_opt.py" \
    --model_name facebook/opt-350m \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_350m" \
    > "$script_dir/output/opt350m/l128_exposer_lora_opt.log"

# ------------------------------1.3b------------------------------
mkdir -p "$script_dir/output/opt1.3b"
python "$script_dir/torch_full.py" \
    --model_name facebook/opt-1.3b \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_1b" \
    > "$script_dir/output/opt1.3b/l128_torch_full.log"

python "$script_dir/torch_lora.py" \
    --model_name facebook/opt-1.3b \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_1b" \
    > "$script_dir/output/opt1.3b/l128_torch_lora.log"

python "$script_dir/exposer_lora.py" \
    --model_name facebook/opt-1.3b \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_1b" \
    > "$script_dir/output/opt1.3b/l128_exposer_lora.log"

python "$script_dir/exposer_lora_opt.py" \
    --model_name facebook/opt-1.3b \
    --data ./dataset/valid_opt.jsonl \
    --seq_len 128 \
    --batch_size 4 \
    --output_json "$save_json_1b" \
    > "$script_dir/output/opt1.3b/l128_exposer_lora_opt.log"

python "$script_dir/plot_memory_opt-350m.py" \
    --save_json "$save_json_350m" \
    --save_png "$output_dir/exp-overall-memory-350m.png"

python "$script_dir/plot_memory_opt-1.3b.py" \
    --save_json "$save_json_1b" \
    --save_png "$output_dir/exp-overall-memory-1.3b.png"