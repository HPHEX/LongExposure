import json

import torch
import torch.nn as nn
import numpy as np

from exposer.layers.sparse_mlp.fc1_matmul import fc1_matmul
from exposer.layers.sparse_mlp.fc2_matmul import fc2_matmul


device = torch.device("cuda:0")


class DenseMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenseMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc2_weight = nn.Parameter(torch.randn(output_size, hidden_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.matmul(x, self.fc1_weight.t())
        x = self.relu(x)
        x = torch.matmul(x, self.fc2_weight.t())
        return x


class ColumnSparseMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ColumnSparseMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc2_weight = nn.Parameter(torch.randn(output_size, hidden_size))
        self.relu = nn.ReLU()

    def forward(self, x, NZ_INDICES):
        fc1_weight = self.fc1_weight[NZ_INDICES, :]
        fc2_weight = self.fc2_weight[:, NZ_INDICES]
        x = x @ fc1_weight.t()
        x = x @ fc2_weight.t()
        return x


class BlockSparseFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc2_weight = torch.nn.Parameter(torch.randn(hidden_size, output_size))

    def forward(self, x, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS):
        x = fc1_matmul(x, self.fc1_weight, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS)
        x = fc2_matmul(x, self.fc2_weight, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS)
        return x


# Define Constants
warmup = 10
repetitions = 50

batch_size = 8 * 512
sparsity_block = 128
sparsity_value_threshold = 0.05
sparsity_ratio_threshold = 0.9

test_cases = ['dense', 'column_sparse', 'block_sparse']

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
    argparser.add_argument('--save_json', type=str, default=None )
    args = argparser.parse_args()
    model_name = args.model_name
    model_name = model_name.split('/')[-1]

    data_path = "./src/experiments/ablation-mlp/data/" + model_name + "/mlp_activations.npy"
    figures_path = "./src/experiments/ablation-mlp/figures/" + model_name

    mlp_activations = np.load(data_path)
    mlp_activations = torch.tensor(mlp_activations, device=device)
    _layers, _hidden_size = mlp_activations.shape
    input_size = _hidden_size // 4
    
    dummy_inputs = torch.randn(batch_size, input_size, device=device, dtype=torch.float16, requires_grad=True)
    dummy_output_grads = torch.randn(batch_size, input_size, device=device, dtype=torch.float16)

    dense_mlp = DenseMLP(input_size, _hidden_size, input_size).half().to(device)
    column_sparse_mlp = ColumnSparseMLP(input_size, _hidden_size, input_size).half().to(device)
    block_sparse_ffn = BlockSparseFFN(input_size, _hidden_size, input_size).half().to(device)

    data_re = []
    for i in range(_layers):
        _activations = mlp_activations[i]

        NZ_INDICES = torch.nonzero(_activations).view(-1)
        num_blocks = _hidden_size // sparsity_block
        nz_block_indices = []
        for j in range(0, _hidden_size, sparsity_block):
            blk_activations = _activations[j:j+sparsity_block]
            max_val = blk_activations.max()
            num_zeros = (blk_activations < sparsity_value_threshold * max_val).sum()
            if num_zeros / sparsity_block < sparsity_ratio_threshold:
                nz_block_indices.append(j // sparsity_block)
        NZ_BLOCK_INDICES = torch.tensor(nz_block_indices, device=device, dtype=torch.int64)
        NUM_NZ_BLOCKS = len(nz_block_indices)

        dense_time = 0
        column_sparse_time = 0
        block_sparse_time = 0
        for test_case in test_cases:
            if test_case == 'dense':
                for id in range(warmup + repetitions):
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output = dense_mlp(dummy_inputs)
                    output.backward(dummy_output_grads)
                    end_time.record()
                    torch.cuda.synchronize()
                    if id >= warmup:
                        dense_time += start_time.elapsed_time(end_time) / 1000.0
                dense_time /= repetitions
            elif test_case == 'column_sparse':
                for id in range(warmup + repetitions):
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output = column_sparse_mlp(dummy_inputs, NZ_INDICES)
                    output.backward(dummy_output_grads)
                    end_time.record()
                    torch.cuda.synchronize() 
                    if id >= warmup:
                        column_sparse_time += start_time.elapsed_time(end_time) / 1000.0
                column_sparse_time /= repetitions
            else:
                for id in range(warmup + repetitions):
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    output = block_sparse_ffn(dummy_inputs, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS)
                    output.backward(dummy_output_grads)
                    end_time.record()
                    torch.cuda.synchronize()
                    if id >= warmup:
                        block_sparse_time += start_time.elapsed_time(end_time) / 1000.0
                block_sparse_time /= repetitions
        print(f"{{'layer': {i}, 'dense_time': {dense_time}, 'column_sparse_time': {column_sparse_time}, 'block_sparse_time': {block_sparse_time}}}")
        data_re.append({'layer': i, 'dense_time': dense_time, 'column_sparse_time': column_sparse_time, 'block_sparse_time': block_sparse_time})
    with open(args.save_json, 'w') as f:
        json.dump(data_re, f)
