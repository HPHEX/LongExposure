import torch

import exposer.ops.stk as stk


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random_ops.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.matrix_ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (dense.to(cuda_device).requires_grad_(True),
            sparse.to(cuda_device).requires_grad_(True))
