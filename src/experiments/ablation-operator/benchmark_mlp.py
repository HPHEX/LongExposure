import argparse
import json
import torch  
from exposer.layers.sparse_mlp.fc1_matmul import _fc1_matmul
from exposer.layers.sparse_mlp.fc2_matmul import _fc2_matmul

parser = argparse.ArgumentParser(description='Benchmarking the performance of different sparse attention implementations')
parser.add_argument('--save_json', type=str, default='None', help='save')

args = parser.parse_args()

# Define device  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Define constants
warmup = 10
repetitions = 50

batch_size = 8
num_heads = 32
seq_len = 2048
head_dim = 64

M = batch_size * seq_len
K = num_heads * head_dim
N = 4 * K

A = torch.randn((M, K), device=device, dtype=torch.float16)
B = torch.randn((K, N), device=device, dtype=torch.float16)

test_cases = ['dense', 'dss', 'dsd']


with open(args.save_json, 'r') as f:
    data_re = json.load(f)
    
for sparsity in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    BLOCK_N = 128
    NUM_BLOCKS = N // BLOCK_N
    NUM_NZ_BLOCKS = int(NUM_BLOCKS * (1 - sparsity))
    NZ_BLOCK_INDICES = torch.randperm(NUM_BLOCKS)[:NUM_NZ_BLOCKS].cuda()

    time_dense = 0
    time_dss = 0
    time_dsd = 0
    for test_case in test_cases:
        if test_case == 'dense':
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                _ = torch.matmul(A, B)
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_dense += start_time.elapsed_time(end_time) / 1000.0
        elif test_case == 'dss':
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                _ = _fc2_matmul._fwd(A, B, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype=None, allow_tf32=True, fp8_fast_accum=True, output_dtype=None)
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_dss += start_time.elapsed_time(end_time) / 1000.0
        elif test_case == 'dsd':
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                _ = _fc1_matmul._fwd(A, B, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype=None, allow_tf32=True, fp8_fast_accum=True, output_dtype=None)
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_dsd += start_time.elapsed_time(end_time) / 1000.0
    time_dense /= repetitions
    time_dss /= repetitions
    time_dsd /= repetitions
    print(f"{{'sparsity': {sparsity}, 'dense': {time_dense}, 'dss': {time_dss}, 'dsd': {time_dsd}}}")
    data_re.append({'sparsity': sparsity, 'dense': time_dense, 'dss': time_dss, 'dsd': time_dsd})

with open(args.save_json, 'w') as f:
    json.dump(data_re,f)