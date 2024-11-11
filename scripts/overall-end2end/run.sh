#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/overall-end2end
save_json="$script_dir/record.json"

output_dir=$(pwd)/output_figures

# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json" ]; then
    echo "[]" > "$save_json"
else
    > "$save_json"
    echo "[]" > "$save_json"
fi

# 检查 src 目录是否在 PYTHONPATH 中
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$src_dir"
else
    if [[ ":$PYTHONPATH:" != *":$src_dir:"* ]]; then
        export PYTHONPATH="$PYTHONPATH:$src_dir"
    fi
fi

# ------------------------------1.3b_512------------------------------
mkdir -p "$script_dir/output/opt1.3b_512"
python "$script_dir/torch_full.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/torch_full.log"
python "$script_dir/torch_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/torch_lora.log"
python "$script_dir/torch_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/torch_adapter.log"
python "$script_dir/torch_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/torch_bitfit.log"

python "$script_dir/exposer_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/exposer_lora.log"
python "$script_dir/exposer_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/exposer_adapter.log"
python "$script_dir/exposer_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt1.3b_512/exposer_bitfit.log"

#-----------------------------1.3b_1024------------------------------
mkdir -p "$script_dir/output/opt1.3b_1024"
python "$script_dir/torch_full.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/torch_full.log"
python "$script_dir/torch_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/torch_lora.log"
python "$script_dir/torch_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/torch_adapter.log"
python "$script_dir/torch_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/torch_bitfit.log"

python "$script_dir/exposer_lora.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/exposer_lora.log"
python "$script_dir/exposer_adapter.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/exposer_adapter.log"
python "$script_dir/exposer_bitfit.py" \
        --model_name facebook/opt-1.3b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt1.3b_1024/exposer_bitfit.log"

# ------------------------------2.7b_512------------------------------
mkdir -p "$script_dir/output/opt2.7b_512"
python "$script_dir/torch_full.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/torch_full.log"
python "$script_dir/torch_lora.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/torch_lora.log"
python "$script_dir/torch_adapter.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/torch_adapter.log"
python "$script_dir/torch_bitfit.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/torch_bitfit.log"

python "$script_dir/exposer_lora.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/exposer_lora.log"
python "$script_dir/exposer_adapter.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/exposer_adapter.log"
python "$script_dir/exposer_bitfit.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 512 --batch_size 4 > "$script_dir/output/opt2.7b_512/exposer_bitfit.log"

#-----------------------------2.7b_1024------------------------------
mkdir -p "$script_dir/output/opt2.7b_1024"
python "$script_dir/torch_full.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/torch_full.log"
python "$script_dir/torch_lora.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/torch_lora.log"
python "$script_dir/torch_adapter.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/torch_adapter.log"
python "$script_dir/torch_bitfit.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/torch_bitfit.log"

python "$script_dir/exposer_lora.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/exposer_lora.log"
python "$script_dir/exposer_adapter.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/exposer_adapter.log"
python "$script_dir/exposer_bitfit.py" \
        --model_name facebook/opt-2.7b \
        --data ./dataset/valid_opt.jsonl \
        --output_json "$save_json" \
        --seq_len 1024 --batch_size 4 > "$script_dir/output/opt2.7b_1024/exposer_bitfit.log"



python "$script_dir/plot_end2end_a100.py" --save_json "$save_json" --output "$output_dir/exp-overall-end2end-a100.png"
