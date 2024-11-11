import torch
import triton
import triton.language as tl

from exposer.ops.triton.matmul_perf_model import early_config_prune, estimate_matmul_time

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


def upcast_if_fp8(a):
    if "fp8" in str(a):
        return torch.float16
    return a


def get_higher_dtype(a, b):
    a = upcast_if_fp8(a)
    b = upcast_if_fp8(b)
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        # BLOCK_N must be 128
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _fc1_fwd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            NZ_BLOCK_INDICES,
            NUM_NZ_BLOCKS: tl.constexpr,
            acc_dtype: tl.constexpr,
            allow_tf32: tl.constexpr,
            fp8_fast_accum: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr
        ):
    # matrix multiplication
    pid_z = tl.program_id(1)
    # re-order program ID for better L2 performance
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = NUM_NZ_BLOCKS
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    block_offset = tl.load(NZ_BLOCK_INDICES + pid_n)
    rn = block_offset * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = block_offset * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.autotune(
    configs=[
        # BLOCK_N must be 128
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _fc1_bwd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            NZ_BLOCK_INDICES,
            NUM_NZ_BLOCKS: tl.constexpr,
            acc_dtype: tl.constexpr,
            allow_tf32: tl.constexpr,
            fp8_fast_accum: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr
        ):
    # matrix multiplication
    pid_z = tl.program_id(1)
    # re-order program ID for better L2 performance
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    STRIDE_A = BLOCK_K * SPLIT_K * stride_ak
    STRIDE_B = BLOCK_K * SPLIT_K * stride_bk
    # for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
    for k in range(0, NUM_NZ_BLOCKS):
        block_offset = tl.load(NZ_BLOCK_INDICES + k)
        _A = A + STRIDE_A * block_offset
        _B = B + STRIDE_B * block_offset
        if EVEN_K:
            a = tl.load(_A)
            b = tl.load(_B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(_A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(_B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _fc1_matmul(torch.autograd.Function):
    @staticmethod
    def _fwd(a, b, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype, allow_tf32, fp8_fast_accum, output_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        # common type between a and b
        ab_dtype = get_higher_dtype(a.dtype, b.dtype)

        # allocates output
        if (output_dtype is None):
            output_dtype = ab_dtype

        c = torch.empty((M, N), device=device, dtype=output_dtype)

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
        }

        if acc_dtype is None:
            acc_dtype = supported_acc_dtypes[ab_dtype][0]
        else:
            assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
            assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
            assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

        def to_tl_type(ty):
            return getattr(tl, str(ty).split(".")[-1])

        acc_dtype = to_tl_type(acc_dtype)
        ab_dtype = to_tl_type(ab_dtype)
        output_dtype = to_tl_type(output_dtype)

        # Tensor cores support input with mixed float8 types.
        if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
            ab_dtype = None
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * NUM_NZ_BLOCKS, META['SPLIT_K'])
        _fc1_fwd_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            NZ_BLOCK_INDICES,
            NUM_NZ_BLOCKS=NUM_NZ_BLOCKS,
            acc_dtype=acc_dtype,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8, AB_DTYPE=ab_dtype
        )
        return c

    @staticmethod
    def _bwd(a, b, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype, allow_tf32, fp8_fast_accum, output_dtype):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        # common type between a and b
        ab_dtype = get_higher_dtype(a.dtype, b.dtype)

        # allocates output
        if (output_dtype is None):
            output_dtype = ab_dtype

        c = torch.empty((M, N), device=device, dtype=output_dtype)

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.int32, )
        }

        if acc_dtype is None:
            acc_dtype = supported_acc_dtypes[ab_dtype][0]
        else:
            assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
            assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
            assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

        def to_tl_type(ty):
            return getattr(tl, str(ty).split(".")[-1])

        acc_dtype = to_tl_type(acc_dtype)
        ab_dtype = to_tl_type(ab_dtype)
        output_dtype = to_tl_type(output_dtype)

        # Tensor cores support input with mixed float8 types.
        if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
            ab_dtype = None
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _fc1_bwd_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            NZ_BLOCK_INDICES,
            NUM_NZ_BLOCKS=NUM_NZ_BLOCKS,
            acc_dtype=acc_dtype,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8, AB_DTYPE=ab_dtype
        )
        return c

    @staticmethod
    def forward(ctx, a, b, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype=None, allow_tf32=True, fp8_fast_accum=True, output_dtype=None):
        ctx.save_for_backward(a, b)  # save_for_backward can only save variables
        ctx.NZ_BLOCK_INDICES = NZ_BLOCK_INDICES
        ctx.NUM_NZ_BLOCKS = NUM_NZ_BLOCKS
        return _fc1_matmul._fwd(a, b, NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, 
                                 acc_dtype=acc_dtype, allow_tf32=allow_tf32, 
                                 fp8_fast_accum=fp8_fast_accum, output_dtype=output_dtype)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        NZ_BLOCK_INDICES = ctx.NZ_BLOCK_INDICES
        NUM_NZ_BLOCKS = ctx.NUM_NZ_BLOCKS
        grad_a = None
        if ctx.needs_input_grad[0]:
            grad_a = _fc1_matmul._bwd(dc, b.t(), NZ_BLOCK_INDICES, NUM_NZ_BLOCKS, acc_dtype=dc.dtype, allow_tf32=True, fp8_fast_accum=True, output_dtype=a.dtype)
        if ctx.needs_input_grad[1]:
            NotImplementedError("backward for b is not implemented, as it is not needed in PEFT")
        return grad_a, None, None, None


fc1_matmul = _fc1_matmul.apply
