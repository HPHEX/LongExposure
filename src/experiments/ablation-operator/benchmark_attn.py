import json
import argparse
import torch  
from exposer.ops.triton.blocksparse_matmul import matmul


parser = argparse.ArgumentParser(description='Benchmarking the performance of different sparse attention implementations')
parser.add_argument('--save_json', type=str, default='None', help='save')

args = parser.parse_args()

  
# Define device  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Define constants
warmup = 10
repetitions = 50

block_size = 64
batch_size = 8
num_heads = 32
seq_len = 2048
head_dim = 64

test_cases = ['dense_sdd', 'dense_dsd', 'sdd', 'dsd', 'torch_sdd', 'torch_dsd']

with open(args.save_json, 'r') as f:
    data_re = json.load(f)
for sparsity_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # Create a random sparsity layout with controlled sparsity
    num_sparsity_blocks = seq_len // block_size
    sparsity_layout = torch.zeros((num_heads, num_sparsity_blocks, num_sparsity_blocks), device=device, dtype=torch.int64)
    sparsity_layout.bernoulli_(1 - sparsity_ratio)
    
    # Initialize matmul operations  
    sparse_dot_sdd_nt = matmul(sparsity_layout, block_size, 'sdd', device, trans_a=False, trans_b=True)
    sparse_dot_dsd_nn = matmul(sparsity_layout, block_size, 'dsd', device, trans_a=False, trans_b=False)
    
    # Create random dense matrices  
    A = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
    B = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
    C = torch.randn((batch_size, num_heads * block_size * block_size, num_sparsity_blocks, num_sparsity_blocks), device=device, dtype=torch.float16)
    D = torch.randn((batch_size, num_heads, seq_len, seq_len), device=device, dtype=torch.float16)
    
    time_dense_sdd = 0
    time_sdd = 0
    time_dense_dsd = 0
    time_dsd = 0
    time_torch_sdd = 0
    time_torch_dsd = 0

    for test_case in test_cases:
        if test_case == 'dense_sdd':
            # print(f"Running {test_case} with sparsity ratio {sparsity_ratio}")
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                torch.matmul(A, B.permute(0, 1, 3, 2))
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_dense_sdd += start_time.elapsed_time(end_time) / 1000.0
        elif test_case == 'sdd':
            # print(f"Running {test_case} with sparsity ratio {sparsity_ratio}")
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                sparse_dot_sdd_nt(A, B)
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_sdd += start_time.elapsed_time(end_time) / 1000.0
        elif test_case == 'dense_dsd':
            # print(f"Running {test_case} with sparsity ratio {sparsity_ratio}")
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                torch.matmul(D, A)
                end_time.record()
                torch.cuda.synchronize() 
                if id >= warmup:
                    time_dense_dsd += start_time.elapsed_time(end_time) / 1000.0
        elif test_case == 'dsd':
            # print(f"Running {test_case} with sparsity ratio {sparsity_ratio}")
            for id in range(warmup + repetitions):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                sparse_dot_dsd_nn(C, B)
                end_time.record()
                torch.cuda.synchronize()
                if id >= warmup:
                    time_dsd += start_time.elapsed_time(end_time) / 1000.0
    time_dense_sdd /= repetitions
    time_sdd /= repetitions
    time_dense_dsd /= repetitions
    time_dsd /= repetitions
    time_torch_sdd /= repetitions
    time_torch_dsd /= repetitions
    
    # Print out the timing results
    print(f"{{'sparsity': {sparsity_ratio}, 'dense_sdd': {time_dense_sdd}, 'sdd': {time_sdd}, 'dense_dsd': {time_dense_dsd}, 'dsd': {time_dsd}, 'torch_sdd': {time_torch_sdd}, 'torch_dsd': {time_torch_dsd}}}")
    data_re.append({'sparsity': sparsity_ratio, 'dense_sdd': time_dense_sdd, 'sdd': time_sdd, 'dense_dsd': time_dense_dsd, 'dsd': time_dsd, 'torch_sdd': time_torch_sdd, 'torch_dsd': time_torch_dsd})

with open(args.save_json, 'w') as f:
    json.dump(data_re, f)
