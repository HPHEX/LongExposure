import torch

from exposer.layers.sparse_mlp.fc1_matmul import fc1_matmul
from exposer.layers.sparse_mlp.fc2_matmul import fc2_matmul


class SparseMLP(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(embed_dim, hidden_dim))
        self.fc2_weight = torch.nn.Parameter(torch.randn(hidden_dim, output_dim))

    def forward(self, x, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS):
        x = fc1_matmul(x, self.fc1_weight, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS)
        x = fc2_matmul(x, self.fc2_weight, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS)
        return x
