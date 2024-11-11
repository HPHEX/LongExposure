#!/bin/bash

set -x

current_dir=$(pwd)
src_dir=$(pwd)/src
script_dir=$(pwd)/src/experiments/scale-model
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

#large + 512
python "$script_dir/torch_full.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/torch_lora.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_lora.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/torch_adapter.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_adapter.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/torch_bitfit.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_bitfit.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

#large + 1024
python "$script_dir/torch_full.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/torch_lora.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_lora.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/torch_adapter.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_adapter.py" \
        --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/torch_bitfit.py" --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_bitfit.py"  --model_name gpt2-large \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

#xl+512
python "$script_dir/torch_full.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512   

python "$script_dir/torch_lora.py"  --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_lora.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/torch_adapter.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_adapter.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/torch_bitfit.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

python "$script_dir/exposer_bitfit.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 512

#xl+1024
python "$script_dir/torch_full.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024
    
python "$script_dir/torch_lora.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_lora.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/torch_adapter.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_adapter.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/torch_bitfit.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/exposer_bitfit.py" \
        --model_name gpt2-xl \
        --save_json "$save_json" \
        --data ./dataset/valid_gpt.jsonl \
        --batch_size 4   \
        --seq_len 1024

python "$script_dir/plot_scale_model.py" \
        --save_json "$save_json" \
        --output "$output_dir/exp-scale_model.png"
