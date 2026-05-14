# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import triton
import triton.language as tl

from rrattn.ops.rrattention import flat_group_gemm_fuse_reshape_rr_kernel
from rrattn.ops.xattention import (
    flat_group_gemm_fuse_reshape_kernel,
    softmax_fuse_block_sum_kernel_causal,
    softmax_fuse_block_sum_kernel_non_causal,
)


@triton.jit
def flat_group_gemm_kernel(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_on,
    chunk_start,
    chunk_end,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
        return

    Q_ptrs = (
        Q
        + batch_id * stride_qz
        + head_id * stride_qh
        + block_m * BLOCK_M * stride_qn
    )
    K_ptrs = (
        K
        + batch_id * stride_kz
        + head_id * stride_kh
        + block_n * BLOCK_N * stride_kn
    )

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_qn
        + tl.arange(0, BLOCK_K)[None, :]
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * stride_kn
        + tl.arange(0, BLOCK_K)[:, None]
    )

    num_iters = HEAD_DIM // BLOCK_K
    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for iter in range(num_iters):
        q = tl.load(Q_ptrs + iter * BLOCK_K)
        k = tl.load(K_ptrs + iter * BLOCK_K)
        o += tl.dot(q, k)

    O_ptrs = (
        Out
        + batch_id * stride_oz
        + head_id * stride_oh
        + block_m * BLOCK_M * stride_on
        + block_n * BLOCK_N
    )
    O_ptrs = (
        O_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * stride_on
        + tl.arange(0, BLOCK_N)[None, :]
    )

    tl.store(O_ptrs, o.to(Out.type.element_ty))


def softmax_fuse_block_sum(
    attn_weights_slice,
    reshaped_block_size,
    segment_size,
    chunk_start,
    chunk_end,
    real_q_len,
    scale,
    is_causal=True,
):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    assert k_len % segment_size == 0
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = torch.empty(
        (
            batch_size,
            num_heads,
            q_len // reshaped_block_size,
            k_len // reshaped_block_size,
        ),
        dtype=attn_weights_slice.dtype,
        device=attn_weights_slice.device,
    )

    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output


def flat_group_gemm(query_states, key_states, chunk_start, chunk_end):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    output = torch.empty(
        (batch_size, num_heads, q_len, kv_len),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (q_len // BLOCK_M, kv_len // BLOCK_N, batch_size * num_heads)
    flat_group_gemm_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    return output


def flat_group_gemm_fuse_reshape(
    query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    assert q_len % (stride * BLOCK_M) == 0
    assert kv_len % (stride * BLOCK_N) == 0

    grid = (
        q_len // stride // BLOCK_M,
        kv_len // stride // BLOCK_N,
        batch_size * num_heads,
    )
    flat_group_gemm_fuse_reshape_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output


def flat_group_gemm_fuse_reshape_rr(
    query_states, key_states, stride, chunk_start, chunk_end, is_causal=True
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert key_states.shape[1] == num_heads
    assert key_states.shape[3] == head_dim

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    BLOCK_M = 128
    BLOCK_N = 128
    assert q_len % (stride * BLOCK_M) == 0
    assert kv_len % (stride * BLOCK_N) == 0

    grid = (
        q_len // stride // BLOCK_M,
        kv_len // stride // BLOCK_N,
        batch_size * num_heads,
    )
    flat_group_gemm_fuse_reshape_rr_kernel[grid](
        query_states,
        key_states,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        chunk_start,
        chunk_end,
        num_heads,
        stride,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        is_causal,
    )

    return output
