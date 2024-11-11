#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/scale-card
save_json_125m="$script_dir/record_125m.json"
save_json_350m="$script_dir/record_350m.json"
save_json_1b="$script_dir/record_1.3b.json"

output_dir=$(pwd)/output_figures

# 检查并创建或清空 JSON 文件
if [ ! -f "$save_json_125m" ]; then
    echo "[]" > "$save_json_125m"
else
    > "$save_json_125m"
    echo "[]" > "$save_json_125m"
fi

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

#125M
CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-125m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_125m" \
    --num_cards 4

#350m
CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-350m \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_350m" \
    --num_cards 4

#1.3B
CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_lora_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_adapter_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 4

CUDA_VISIBLE_DEVICES=0  deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 1

CUDA_VISIBLE_DEVICES=0,1 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 2

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed "$script_dir/exposer_bitfit_scale.py" \
    --data ./dataset/valid_opt.jsonl \
    --model_name facebook/opt-1.3b \
    --seq_len 1024 \
    --batch_size 2 \
    --save_json "$save_json_1b" \
    --num_cards 4

python "$script_dir/plot_scale_card.py" \
    --save_json "$save_json_125m" \
    --output "$output_dir/exp-scale-card-125m.png"

python "$script_dir/plot_scale_card.py" \
    --save_json "$save_json_350m" \
    --output "$output_dir/exp-scale-card-350m.png"

python "$script_dir/plot_scale_card.py" \
    --save_json "$save_json_1b" \
    --output "$output_dir/exp-scale-card-1.3b.png"