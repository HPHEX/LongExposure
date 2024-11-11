#!/bin/bash

set -x

mkdir -p ./output_figures

python ./plotting/ablation-attention/plot_benchmark_sparsity.py \
    --save_json ./logs/ablation-attention/record.json \
    --output ./output_figures/exp-ablation-attn-sparsity.png

python ./plotting/ablation-attention/plot_benchmark_time.py \
    --save_json ./logs/ablation-attention/record.json \
    --output ./output_figures/exp-ablation-attn-time.png

python ./plotting/ablation-breakdown/plot_breakdown.py \
    --save_json ./logs/ablation-breakdown/record.json \
    --output ./output_figures/exp-ablation-breakdown.png

python ./plotting/ablation-mlp/plot_benchmark_sparsity.py \
    --data_path ./logs/ablation-mlp/ \
    --output ./output_figures/exp-ablation-mlp-sparsity.png

python ./plotting/ablation-mlp/plot_benchmark_time.py \
    --save_json ./logs/ablation-mlp/record.json \
    --output ./output_figures/exp-ablation-mlp-time.png

python ./plotting/ablation-operator/plot_attn.py \
    --save_json ./logs/ablation-operator/record_attn.json \
    --output ./output_figures/exp-ablation-operator-attn.png

python ./plotting/ablation-operator/plot_mlp.py \
    --save_json ./logs/ablation-operator/record_mlp.json \
    --output ./output_figures/exp-ablation-operator-mlp.png

python ./plotting/ablation-predictor/plot_loss_curve.py \
    --cur_path ./logs/ablation-predictor/ \
    --output ./output_figures/exp-ablation-predictor.png

python ./plotting/overall-end2end/plot_end2end_a100.py \
    --save_json ./logs/overall-end2end/record.json \
    --output ./output_figures/exp-overall-end2end-a100.png

python ./plotting/overall-memory/plot_memory_opt-1.3b.py \
    --save_json ./logs/overall-memory/record_1.3b.json \
    --save_pdf ./output_figures/exp-overall-memory-opt-1.3b.png

python ./plotting/overall-memory/plot_memory_opt-350m.py \
    --save_json ./logs/overall-memory/record_350m.json \
    --save_pdf ./output_figures/exp-overall-memory-opt-350m.png

python ./plotting/scale-model/plot_scale_model.py \
    --save_json ./logs/scale-model/record.json \
    --output ./output_figures/exp-scale-model.png