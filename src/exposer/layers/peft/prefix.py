# Adapted from https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(config.prefix_len, config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )

    def forward(self, prefix: torch.Tensor):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values
