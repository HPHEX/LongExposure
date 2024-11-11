from transformers import OPTConfig, GPT2Config

from exposer.layers.sparse_attention import (
    LocalSlidingWindowSparsityConfig,
    FixedSparsityConfig,
    VariableSparsityConfig,
    BigBirdSparsityConfig,
    LongformerSparsityConfig,
    DenseSparsityConfig,
    RandomSparsityConfig
)


def get_custom_opt_config(model_name='facebook/opt-125m',
                          lora_attn_dim=4, 
                          lora_attn_alpha=32,
                          lora_dropout=0.0,
                          lora_r_dropout=0.0,
                          attn_implementation='eager',
                          trace_attn_scores=False,
                          trace_mlp_activations=False,
                          enable_attn_head_similarity=False,
                          attn_head_similarity_method='cosine',
                          attn_head_similarity_threshold=0.8,
                          enable_mlp_block_approx=False,
                          mlp_block_approx_method='threshold',
                          mlp_block_approx_threshold=0.8,
                          mlp_seq_blk_size=32,
                          mlp_ffn_blk_size=32,
                          enable_sparse_attn=False,
                          sparse_config_num_heads=12,
                          sparse_config_block=32,
                          sparse_config_different_layout_per_head=False):
    config = OPTConfig.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    config._attn_implementation = attn_implementation
    config.trace_attn_scores = trace_attn_scores
    config.trace_mlp_activations = trace_mlp_activations
    config.enable_attn_head_similarity = enable_attn_head_similarity
    config.attn_head_similarity_method = attn_head_similarity_method
    config.attn_head_similarity_threshold = attn_head_similarity_threshold
    config.enable_mlp_block_approx = enable_mlp_block_approx
    config.mlp_block_approx_method = mlp_block_approx_method
    config.mlp_block_approx_threshold = mlp_block_approx_threshold
    config.mlp_seq_blk_size = mlp_seq_blk_size
    config.mlp_ffn_blk_size = mlp_ffn_blk_size
    config.enable_sparse_attn = enable_sparse_attn
    config.sparse_config_num_heads = sparse_config_num_heads
    config.sparse_config_block = sparse_config_block
    config.sparse_config_different_layout_per_head = sparse_config_different_layout_per_head
    return config


def get_opt_config(model_name='facebook/opt-125m'):
    config = OPTConfig.from_pretrained(model_name)
    return config


def get_custom_opt_profile_config(model_name='facebook/opt-125m',
                                  trace_attn_scores=False,
                                  trace_mlp_activations=False,
                                  trace_attn_inputs=False,
                                  trace_mlp_inputs=False):
    config = OPTConfig.from_pretrained(model_name)
    config.trace_attn_scores = trace_attn_scores
    config.trace_mlp_activations = trace_mlp_activations
    config.trace_attn_inputs = trace_attn_inputs
    config.trace_mlp_inputs = trace_mlp_inputs
    return config


def get_opt_profile_attn_config(model_name='facebook/opt-125m', 
                                trace_attn_inputs=True,
                                trace_attn_scores=True):
    config = OPTConfig.from_pretrained(model_name)
    config.trace_attn_inputs = trace_attn_inputs
    config.trace_attn_scores = trace_attn_scores
    return config


def get_opt_profile_mlp_config(model_name='facebook/opt-125m',
                               trace_mlp_inputs=True,
                               trace_mlp_activations=True):
    config = OPTConfig.from_pretrained(model_name)
    config.trace_mlp_inputs = trace_mlp_inputs
    config.trace_mlp_activations = trace_mlp_activations
    return config


def get_opt_lora_config(model_name='facebook/opt-125m',
                        lora_attn_dim=4, 
                        lora_attn_alpha=32,
                        lora_dropout=0.0,
                        lora_r_dropout=0.0):
    config = OPTConfig.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    return config


def get_opt_adapter_config(model_name='facebook/opt-125m',
                           adapter_dim=64):
    config = OPTConfig.from_pretrained(model_name)
    config.adapter_dim = adapter_dim
    return config


def get_opt_prefix_config(model_name='facebook/opt-125m',
                          prefix_len=128):
    config = OPTConfig.from_pretrained(model_name)
    config.prefix_len = prefix_len
    return config


def get_opt_exposer_lora_attn_config(model_name='facebook/opt-125m',
                                     lora_attn_dim=4, 
                                     lora_attn_alpha=32,
                                     lora_dropout=0.0,
                                     lora_r_dropout=0.0,
                                     sparse_config='bigbird'):
    config = OPTConfig.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'random':
        config.sparsity_config = RandomSparsityConfig(config.num_attention_heads)
    else:
        config.sparsity_config = DenseSparsityConfig(config.num_attention_heads)
    config.sparsity_config_type = sparse_config
    return config


def get_opt_exposer_lora_mlp_config(model_name='facebook/opt-125m',
                                    lora_attn_dim=4, 
                                    lora_attn_alpha=32,
                                    lora_dropout=0.0,
                                    lora_r_dropout=0.0,
                                    mlp_block_size=128,
                                    mlp_threshold=0.80):
    config = OPTConfig.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    config.mlp_block_size = mlp_block_size
    config.mlp_threshold = mlp_threshold
    return config


def get_opt_exposer_lora_config(model_name='facebook/opt-125m',
                                lora_attn_dim=4, 
                                lora_attn_alpha=32,
                                lora_dropout=0.0,
                                lora_r_dropout=0.0,
                                sparse_config='bigbird',
                                mlp_block_size=128,
                                mlp_threshold=0.80):
    config = OPTConfig.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    config.mlp_block_size = mlp_block_size
    config.mlp_threshold = mlp_threshold
    return config


def get_opt_exposer_adapter_config(model_name='facebook/opt-125m',
                                   sparse_config='bigbird',
                                   mlp_block_size=128,
                                   mlp_threshold=0.80,
                                   adapter_dim=64):
    config = OPTConfig.from_pretrained(model_name)
    config.adapter_dim = adapter_dim
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    config.mlp_block_size = mlp_block_size
    config.mlp_threshold = mlp_threshold
    return config


def get_opt_exposer_prefix_config(model_name='facebook/opt-125m',
                                  prefix_len=128,
                                  sparse_config='bigbird',
                                  mlp_block_size=128,
                                  mlp_threshold=0.80):
    config = OPTConfig.from_pretrained(model_name)
    config.prefix_len = prefix_len
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    config.mlp_block_size = mlp_block_size
    config.mlp_threshold = mlp_threshold
    return config


def get_opt_exposer_bitfit_config(model_name='facebook/opt-125m',
                                  sparse_config='bigbird',
                                  mlp_block_size=128,
                                  mlp_threshold=0.80):
    config = OPTConfig.from_pretrained(model_name)
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    config.mlp_block_size = mlp_block_size
    config.mlp_threshold = mlp_threshold
    return config


def get_custom_gpt2_config(model_name='gpt2',
                      lora_attn_dim=4, 
                      lora_attn_alpha=32,
                      lora_dropout=0.0,
                      lora_r_dropout=0.0):
    config = GPT2Config.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    return config


def get_gpt2_config(model_name='gpt2'):
    config = GPT2Config.from_pretrained(model_name)
    return config


def get_gpt2_lora_config(model_name='gpt2',
                         lora_attn_dim=4, 
                         lora_attn_alpha=32,
                         lora_dropout=0.0,
                         lora_r_dropout=0.0):
    config = GPT2Config.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    return config


def get_gpt2_adapter_config(model_name='gpt2',
                            adapter_dim=64):
    config = GPT2Config.from_pretrained(model_name)
    config.adapter_dim = adapter_dim
    return config


def get_gpt2_exposer_lora_config(model_name='gpt2',
                                 lora_attn_dim=4, 
                                 lora_attn_alpha=32,
                                 lora_dropout=0.0,
                                 lora_r_dropout=0.0,
                                 sparse_config='bigbird'):
    config = GPT2Config.from_pretrained(model_name)
    config.lora_attn_dim = lora_attn_dim
    config.lora_attn_alpha = lora_attn_alpha
    config.lora_dropout = lora_dropout
    config.lora_r_dropout = lora_r_dropout
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    return config


def get_gpt2_exposer_adapter_config(model_name='gpt2',
                                    sparse_config='bigbird',
                                    adapter_dim=64):
    config = GPT2Config.from_pretrained(model_name)
    config.adapter_dim = adapter_dim
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    return config


def get_gpt2_exposer_bitfit_config(model_name='facebook/opt-125m',
                                   sparse_config='bigbird'):
    config = GPT2Config.from_pretrained(model_name)
    config.sparse_config_num_heads = config.num_attention_heads
    if sparse_config == 'bigbird':
        config.sparsity_config = BigBirdSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'longformer':
        config.sparsity_config = LongformerSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'fixed':
        config.sparsity_config = FixedSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'variable':
        config.sparsity_config = VariableSparsityConfig(config.num_attention_heads)
    elif sparse_config == 'local':
        config.sparsity_config = LocalSlidingWindowSparsityConfig(config.num_attention_heads)
    return config
