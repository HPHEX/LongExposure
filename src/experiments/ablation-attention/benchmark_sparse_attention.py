from typing import Optional, Tuple

import json
import torch
import torch.nn as nn
import numpy as np
from transformers import OPTConfig

from exposer.utils.config_utils import get_opt_lora_config
from exposer.layers.sparse_attention.sparsity_config import (
    SparsityConfig,
    BigBirdSparsityConfig,
    LongformerSparsityConfig
)
from exposer.layers.peft.lora import LoRAMergedLinear
from exposer.ops.triton.blocksparse_matmul import matmul
from exposer.ops.triton.blocksparse_softmax import softmax


device = torch.device("cuda:0")
  
# Function to measure execution time  
def measure_time(func, *args):  
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    inputs = args[0]
    output_grads = args[1]
    output = func(inputs)
    output.backward(output_grads)
    end_time.record()
    torch.cuda.synchronize() 
    return start_time.elapsed_time(end_time) / 1000.0 


# Define Constants
warmup = 10
repetitions = 50

batch_size = 8
seq_len = 512
sparsity_block = 16
sparsity_threshold = 10

class SparseSelfAttention(nn.Module):
    def __init__(self, sparsity_layout: torch.Tensor):
        super().__init__()
        # initialize sparse layout and register as buffer
        master_layout = sparsity_layout
        self.register_buffer("master_layout", master_layout)
        self._need_layout_synchronization = True
        # ops
        self.sparse_dot_sdd_nt = matmul(sparsity_layout, sparsity_block, 'sdd', device, trans_a=False, trans_b=True)
        self.sparse_dot_dsd_nn = matmul(sparsity_layout, sparsity_block, 'dsd', device, trans_a=False, trans_b=False)
        self.sparse_softmax = softmax(sparsity_layout, sparsity_block, device)

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        assert query.dtype == torch.half, "sparse attention only supports training in fp16 currently, please file a github issue if you need fp32 support"
        bsz, num_heads, tgt_len, head_dim = query.size()
        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)
        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)
        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)
        scaling = float(head_dim)**-0.5
        # attention scores
        attn_output_weights = self.sparse_dot_sdd_nt(query, key)
        attn_output_weights = self.sparse_softmax(attn_output_weights, scale=scaling, rel_logits=rpe, is_causal=True)
        # outputs
        attn_output = self.sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output



class OPTSparseAttention(nn.Module):
    """Implements Sparse Self Attention layer of Bert model"""
    def __init__(self, config: OPTConfig, sparse_config: SparsityConfig):
        super().__init__()
        self.config = config
        self.sparse_self_attention = SparseSelfAttention(sparse_config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"f" and `num_heads`: {self.num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = LoRAMergedLinear(
            in_features=self.embed_dim, 
            out_features=self.embed_dim,                 
            r=config.lora_attn_dim,
            lora_alpha=config.lora_attn_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=[True],
            merge_weights=False,
        )
        self.q_proj = LoRAMergedLinear(
            in_features=self.embed_dim, 
            out_features=self.embed_dim, 
            r=config.lora_attn_dim,
            lora_alpha=config.lora_attn_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=[True],
            merge_weights=False,
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        attn_output = self.sparse_self_attention(query_states, key_states, value_states, attn_mask=attention_mask)
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()
    model_name = args.model_name

    config = get_opt_lora_config(model_name)
    num_heads = config.num_attention_heads
    embed_dim = config.hidden_size
    model_name = model_name.split("/")[-1]

    attn_scores_path = f"./src/experiments/ablation-attention/data/{model_name}/attn_scores.npy"
    attn_scores = np.load(attn_scores_path)
    attn_scores = torch.from_numpy(attn_scores).to(device)
    print(attn_scores.shape)  # (num_layers, num_heads, seq_len, seq_len)
    assert attn_scores.shape[1] == num_heads
    assert attn_scores.shape[2] == seq_len
    assert attn_scores.shape[3] == seq_len
    data_re =[]
    for _layer in range(attn_scores.shape[0]):
        record = {'layer': 0, 'dense_time': 0.00791, 'bigbird_time': 0.00491, 'longformer_time': 0.00504, 'shadowy_time': 0.00586, 'exposer_time': 0.00424, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25781, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.08950}
        # print(f"Layer {_layer}")
        num_blocks = seq_len // sparsity_block
        # Dense Layout
        dense_layout = torch.ones((num_heads, num_blocks, num_blocks), dtype=torch.int16, device=device)
        record['layer'] = _layer
        record['dense_sparsity'] = dense_layout.sum().item() / dense_layout.numel()
        # print('dense_layout sparsity ratio:', dense_layout.sum().item() / dense_layout.numel())
        # BigBird
        bigbird_config = BigBirdSparsityConfig(  
            num_heads=num_heads,  
            block=sparsity_block,  
            num_random_blocks=4,  
            num_sliding_window_blocks=8,  
            num_global_blocks=4,  
            attention='unidirectional'
        )
        bigbird_layout = bigbird_config.make_layout(seq_len)
        record['bigbird_sparsity'] = bigbird_layout.sum().item() / bigbird_layout.numel()
        # print('bigbird_layout sparsity ratio:', bigbird_layout.sum().item() / bigbird_layout.numel())
        # Longformer
        longformer_config = LongformerSparsityConfig(  
            num_heads=num_heads,  
            block=sparsity_block,  
            num_sliding_window_blocks=8,
            global_block_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            attention='unidirectional'
        )
        longformer_layout = longformer_config.make_layout(seq_len)
        record['longformer_sparsity'] = longformer_layout.sum().item() / longformer_layout.numel()
        # print('longformer_layout sparsity ratio:', longformer_layout.sum().item() / longformer_layout.numel())
        # Shadowy Layout
        shadowy_layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int16, device=device)
        _attn_scores = attn_scores[_layer].max(dim=0).values
        for i in range(0, seq_len, sparsity_block):
            for j in range(0, seq_len, sparsity_block):
                blk_i = i // sparsity_block
                blk_j = j // sparsity_block
                block = _attn_scores[i:i+sparsity_block, j:j+sparsity_block]
                if block.sum() != 0:
                    shadowy_layout[:, blk_i, blk_j] = 1
        record['shadowy_sparsity'] = shadowy_layout.sum().item() / shadowy_layout.numel()   
        # print('shadowy_layout sparsity ratio:', shadowy_layout.sum().item() / shadowy_layout.numel())
        # Exposer Layout
        exposer_layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int16, device=device)
        for _head in range(num_heads):
            for i in range(0, seq_len, sparsity_block):
                for j in range(0, seq_len, sparsity_block):
                    blk_i = i // sparsity_block
                    blk_j = j // sparsity_block
                    block = attn_scores[_layer, _head, i:i+sparsity_block, j:j+sparsity_block]
                    if block.sum() > sparsity_threshold:
                        exposer_layout[_head, blk_i, blk_j] = 1
        record['exposer_sparsity'] = exposer_layout.sum().item() / exposer_layout.numel()   
        # print('exposer_layout sparsity ratio:', exposer_layout.sum().item() / exposer_layout.numel())
        # print(exposer_layout.sum().item() / exposer_layout.numel())
        sparsity_layout = None
        for test_case in ["dense", "bigbird", "longformer", "shadowy", "exposer"]:
            if test_case == "dense":
                sparsity_layout = dense_layout
            elif test_case == "bigbird":
                sparsity_layout = bigbird_layout
            elif test_case == "longformer":
                sparsity_layout = longformer_layout
            elif test_case == "shadowy":
                sparsity_layout = shadowy_layout
            elif test_case == "exposer":
                sparsity_layout = exposer_layout

            # Initialize model
            model = OPTSparseAttention(config, sparsity_layout).half().to(device)

            # Create dummy input
            hidden_states = torch.randn((batch_size, seq_len, embed_dim), dtype=torch.float16, device=device)
            output_grads = torch.randn((batch_size, seq_len, embed_dim), dtype=torch.float16, device=device)

            # Perform a warm-up for GPU to avoid startup overhead
            for _ in range(warmup):
                _ = model(hidden_states)

            time_sparse = 0
            for _ in range(repetitions):
                time_sparse += measure_time(model, hidden_states, output_grads)

            # Print out the timing results
            record[f"{test_case}_time"] = time_sparse / repetitions
            # print(f"{test_case} Time: {time_sparse / repetitions:.5f} s")
        print(record)
        data_re.append(record)
    with open(args.save_json, 'w') as f:
        json.dump(data_re, f)
    