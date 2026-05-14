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

import triton
import triton.language as tl


@triton.jit
def scan_maxmin_chunked(
    input_ptr,
    output_max_ptr,
    output_min_ptr,
    seqlen,
    num_chunks,
    chunk_size: tl.constexpr,
    BN: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    i_tile = tl.program_id(0)
    i_bh = tl.program_id(1)

    p_tile = i_tile * BN + tl.arange(0, BN)
    mask_tile = p_tile < seqlen
    b_tile = tl.load(input_ptr + i_bh * seqlen + p_tile, mask=mask_tile)

    b_omax = tl.where(mask_tile, b_tile, INT_MIN).reshape(
        (BN // chunk_size, chunk_size)
    )
    b_omax = tl.max(b_omax, axis=1)

    b_omin = tl.where(mask_tile, b_tile, INT_MAX).reshape(
        (BN // chunk_size, chunk_size)
    )
    b_omin = tl.min(b_omin, axis=1)

    offs_out = tl.arange(0, BN // chunk_size) + i_tile * (BN // chunk_size)
    mask_out = offs_out < num_chunks
    tl.store(
        output_max_ptr + i_bh * num_chunks + offs_out, b_omax, mask=mask_out
    )
    tl.store(
        output_min_ptr + i_bh * num_chunks + offs_out, b_omin, mask=mask_out
    )


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)

    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(
        tl.sum(tl.where(mask == 0, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    right = tl.broadcast_to(
        tl.sum(tl.where(mask == 1, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    idtype = tl.core.get_int_dtype(
        bitwidth=x.dtype.primitive_bitwidth, signed=True
    )
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)

    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order

    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def bitonic_argsort_device(
    x, ids, n_dims: tl.constexpr, descending: tl.constexpr = tl.core.CONSTEXPR_0
):
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(
            x, ids, i, 2 if i < n_dims else descending, n_dims
        )
    return x, ids


@triton.jit
def top_p_kernel(
    X_ptr,
    Out_ptr,
    stride_row,
    threshold_p,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_ptr = X_ptr + pid * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_load = offsets < N_COLS

    x_raw = tl.load(row_start_ptr + offsets, mask=mask_load, other=0.0).to(
        tl.float32
    )
    row_sum = tl.sum(x_raw, axis=0)

    out_row_ptr = Out_ptr + pid * stride_row
    if row_sum == 0.0:
        tl.store(
            out_row_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=tl.int8),
            mask=mask_load,
        )
        return

    actual_cutoff = row_sum * threshold_p
    padding_val = float("-inf")
    x_for_sort = tl.where(mask_load, x_raw, padding_val)
    ids = tl.arange(0, BLOCK_SIZE)

    x_sorted, ids_sorted = bitonic_argsort_device(
        x_for_sort, ids, NUM_DIMS, descending=1
    )

    cum_probs = tl.cumsum(x_sorted, axis=0)
    mask_keep = (cum_probs - x_sorted) < actual_cutoff
    mask_keep = mask_keep & (x_sorted > padding_val)

    mask_store = ids_sorted < N_COLS
    tl.store(out_row_ptr + ids_sorted, mask_keep.to(tl.int8), mask=mask_store)


@triton.jit
def flat_group_gemm_fuse_reshape_rr_kernel(
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
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    is_caual: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_caual:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    Q_ptrs = (
        Q
        + batch_id * stride_qz
        + head_id * stride_qh
        + block_m * BLOCK_M * STRIDE * stride_qn
    )
    K_ptrs = (
        K
        + batch_id * stride_kz
        + head_id * stride_kh
        + block_n * BLOCK_N * STRIDE * stride_kn
    )

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE)
        + tl.arange(0, HEAD_DIM)[None, :]
        + stride_qn * (head_id % STRIDE)
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE)
        + tl.arange(0, HEAD_DIM)[:, None]
    )

    q = tl.load(Q_ptrs)
    k = tl.load(K_ptrs)
    for iter in range(1, STRIDE):
        k += tl.load(K_ptrs + iter * stride_kn)

    o = tl.dot(q, k)

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


@triton.jit
def rrattn_gemm_qchunk_gqa_kernel(
    Q,
    K,
    Out,
    seqlen_q,
    seqlen_k,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    shift_tokens,
    out_stride_b,
    out_stride_h,
    out_stride_q,
    HQ: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_HEADS_PER_CTA: tl.constexpr,
    is_causal: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    i_bhg = tl.program_id(2).to(tl.int64)

    G: tl.constexpr = HQ // H
    GROUPS_PER_KV: tl.constexpr = (
        G + GQA_HEADS_PER_CTA - 1
    ) // GQA_HEADS_PER_CTA

    i_b = i_bhg // (H * GROUPS_PER_KV)
    rem = i_bhg % (H * GROUPS_PER_KV)
    i_hkv = rem // GROUPS_PER_KV
    i_group = rem % GROUPS_PER_KV
    q_head_base = i_hkv * G + i_group * GQA_HEADS_PER_CTA

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    global_q_stride_start = chunk_q_start + block_m * BLOCK_M
    q_strides_global = global_q_stride_start + offs_m
    q_rows_local = block_m * BLOCK_M + offs_m
    k_strides_global = block_n * BLOCK_N + offs_n

    if is_causal:
        q_stride_end = chunk_q_start + (block_m + 1) * BLOCK_M
        max_q_token = (q_stride_end - 1) * STRIDE + (STRIDE - 1) + shift_tokens
        if max_q_token < block_n * BLOCK_N * STRIDE:
            return

    k_token_base = k_strides_global * STRIDE
    b_k = tl.zeros([HEAD_DIM, BLOCK_N], dtype=tl.float32)
    for lane in tl.static_range(STRIDE):
        k_token_ids = k_token_base + lane
        k_valid = k_token_ids < seqlen_k
        p_k = (
            K
            + i_b * (seqlen_k * H * HEAD_DIM).to(tl.int64)
            + k_token_ids[None, :] * (H * HEAD_DIM)
            + i_hkv * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[:, None]
        )
        b_k += tl.load(p_k, mask=k_valid[None, :], other=0.0).to(tl.float32)

    k_oob = k_strides_global >= n_k_strides
    q_oob = q_rows_local >= chunk_q_strides

    for g_local in tl.static_range(GQA_HEADS_PER_CTA):
        q_head = q_head_base + g_local
        q_head_valid = q_head < ((i_hkv + 1) * G)
        q_head_safe = tl.minimum(q_head, HQ - 1)
        head_offset = q_head % STRIDE
        q_token_ids = q_strides_global * STRIDE + head_offset
        q_valid = (q_token_ids < seqlen_q) & q_head_valid

        p_q = (
            Q
            + i_b * (seqlen_q * HQ * HEAD_DIM).to(tl.int64)
            + q_token_ids[:, None] * (HQ * HEAD_DIM)
            + q_head_safe * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[None, :]
        )
        b_q = tl.load(p_q, mask=q_valid[:, None], other=0.0)
        o = tl.dot(b_q, b_k.to(b_q.dtype))
        o = tl.where(k_oob[None, :], -1.0e6, o)
        o = tl.where(q_oob[:, None], -1.0e6, o)

        p_out = (
            Out
            + i_b * out_stride_b
            + q_head_safe * out_stride_h
            + q_rows_local[:, None] * out_stride_q
            + (block_n * BLOCK_N + offs_n)[None, :]
        )
        store_mask = (
            q_head_valid
            & (q_rows_local[:, None] < chunk_q_strides)
            & (k_strides_global[None, :] < n_k_strides)
        )
        tl.store(p_out, o.to(Out.type.element_ty), mask=store_mask)


@triton.jit
def rrattn_nomask_softmax_reduce_kernel(
    In,
    Out,
    OutBoundaryMask,
    scale,
    in_stride_b,
    in_stride_h,
    in_stride_q,
    out_stride_b,
    out_stride_h,
    out_stride_qb,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    num_q_blocks,
    num_k_blocks,
    seqlen_q,
    shift_tokens,
    STRIDE: tl.constexpr,
    ratio: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_qblock_local = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    q_stride_base_local = i_qblock_local * ratio
    q_stride_base_global = chunk_q_start + q_stride_base_local
    q_block_id = q_stride_base_global // ratio
    q_block_valid = (q_stride_base_local < chunk_q_strides) & (
        q_block_id < num_q_blocks
    )

    offs_q = tl.arange(0, ratio)
    q_rows_local = q_stride_base_local + offs_q
    q_valid = q_rows_local < chunk_q_strides

    head_offset = i_h % STRIDE
    q_token_ids = q_stride_base_global * STRIDE + offs_q * STRIDE + head_offset
    q_token_valid = q_token_ids < seqlen_q

    p_in_base = (
        In
        + i_b * in_stride_b
        + i_h * in_stride_h
        + q_stride_base_local * in_stride_q
    )
    p_out = (
        Out
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )
    p_out_mask = (
        OutBoundaryMask
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )

    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.full([ratio], 1.0, dtype=tl.float32)

    num_segments = (n_k_strides + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    num_active_segments = num_segments
    if is_causal:
        last_q_token = (
            (q_stride_base_global + ratio - 1) * STRIDE
            + head_offset
            + shift_tokens
        )
        last_q_stride = last_q_token // STRIDE
        active_k_strides = tl.minimum(
            n_k_strides,
            tl.where(last_q_token >= 0, last_q_stride + 1, 0),
        )
        num_active_segments = (
            active_k_strides + SEGMENT_SIZE - 1
        ) // SEGMENT_SIZE
        diag_segment_idx = tl.maximum(last_q_stride, 0) // SEGMENT_SIZE

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal and seg_idx == diag_segment_idx:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)

        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        l_local = tl.sum(tl.math.exp2(X - m_new[:, None]), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i

    BLOCKS_PER_SEG: tl.constexpr = SEGMENT_SIZE // ratio
    offs_kb = tl.arange(0, BLOCKS_PER_SEG)
    zero_mask = tl.zeros([BLOCKS_PER_SEG], dtype=tl.int8)

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal and seg_idx == diag_segment_idx:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(m_i[:, None] < -1.0e5, 0.0, X)
        X = tl.where(q_token_valid[:, None], X, 0.0)

        X_reshaped = X.reshape(ratio, BLOCKS_PER_SEG, ratio)
        block_sums = tl.sum(tl.sum(X_reshaped, 2), 0)

        k_block_base = seg_start // ratio
        k_block_ids = k_block_base + offs_kb
        valid_store = q_block_valid & (k_block_ids < num_k_blocks)
        tl.store(
            p_out + k_block_ids,
            block_sums.to(Out.type.element_ty),
            mask=valid_store,
        )
        tl.store(
            p_out_mask + k_block_ids,
            zero_mask,
            mask=valid_store,
        )

    if is_causal:
        zero_vals = tl.zeros([BLOCKS_PER_SEG], dtype=tl.float32)
        for seg_idx in range(num_active_segments, num_segments):
            seg_start = seg_idx * SEGMENT_SIZE
            k_block_base = seg_start // ratio
            k_block_ids = k_block_base + offs_kb
            valid_store = q_block_valid & (k_block_ids < num_k_blocks)
            tl.store(
                p_out + k_block_ids,
                zero_vals.to(Out.type.element_ty),
                mask=valid_store,
            )
            tl.store(
                p_out_mask + k_block_ids,
                zero_mask,
                mask=valid_store,
            )


@triton.jit
def _load_bounds(
    base_offset,
    k_offsets,
    load_mask,
    ptr_start_lt,
    ptr_end_lt,
    ptr_start_ut,
    ptr_end_ut,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    pad_lt = INT_MAX
    pad_ut = INT_MIN

    b_lts = tl.load(
        ptr_start_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
    )

    need_lte: tl.constexpr = (causal and mode == 2) or (
        not causal and mode == 4
    )
    if need_lte:
        b_lte = tl.load(
            ptr_end_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
        )
    else:
        b_lte = tl.full(b_lts.shape, pad_lt, dtype=tl.int32)

    if causal:
        b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)
    else:
        if mode == 4:
            b_uts = tl.load(
                ptr_start_ut + base_offset + k_offsets,
                mask=load_mask,
                other=pad_ut,
            )
        else:
            b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    need_ute: tl.constexpr = (not causal) and (mode == 2 or mode == 4)
    if need_ute:
        b_ute = tl.load(
            ptr_end_ut + base_offset + k_offsets, mask=load_mask, other=pad_ut
        )
    else:
        b_ute = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    return b_lts, b_lte, b_uts, b_ute


@triton.jit
def _is_block_fully_masked(block_rows, lts_max, lte_min, uts_max, ute_min):
    in_lt = (block_rows[:, None] >= lts_max[None, :]) & (
        block_rows[:, None] < lte_min[None, :]
    )
    in_ut = (block_rows[:, None] >= uts_max[None, :]) & (
        block_rows[:, None] < ute_min[None, :]
    )
    return in_lt | in_ut


@triton.jit
def _check_fully_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_strict_lt_start,
    ptrs_strict_lt_end,
    ptrs_strict_ut_start,
    ptrs_strict_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    fm_lts, fm_lte, fm_uts, fm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_strict_lt_start,
        ptrs_strict_lt_end,
        ptrs_strict_ut_start,
        ptrs_strict_ut_end,
        causal=causal,
        mode=mode,
    )
    fm_geo = _is_block_fully_masked(q_rows, fm_lts, fm_lte, fm_uts, fm_ute)
    return fm_geo | (~k_load_mask[None, :])


@triton.jit
def _is_block_partially_masked(block_rows, lts_min, lte_max, uts_min, ute_max):
    overlap_lt = (block_rows[:, None] < lte_max[None, :]) & (
        block_rows[:, None] >= lts_min[None, :]
    )
    overlap_ut = (block_rows[:, None] < ute_max[None, :]) & (
        block_rows[:, None] >= uts_min[None, :]
    )
    return overlap_lt | overlap_ut


@triton.jit
def _check_partially_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_perm_lt_start,
    ptrs_perm_lt_end,
    ptrs_perm_ut_start,
    ptrs_perm_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    pm_lts, pm_lte, pm_uts, pm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_perm_lt_start,
        ptrs_perm_lt_end,
        ptrs_perm_ut_start,
        ptrs_perm_ut_end,
        causal=causal,
        mode=mode,
    )
    return _is_block_partially_masked(q_rows, pm_lts, pm_lte, pm_uts, pm_ute)


@triton.jit
def rrattn_flashmask_softmax_reduce_qchunk_kernel(
    In,
    Out,
    OutBoundaryMask,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    scale,
    in_stride_b,
    in_stride_h,
    in_stride_q,
    out_stride_b,
    out_stride_h,
    out_stride_qb,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    num_q_blocks,
    num_k_blocks,
    seqlen_q,
    shift_tokens,
    HQ: tl.constexpr,
    HIDS: tl.constexpr,
    STRIDE: tl.constexpr,
    ratio: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    mode: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_qblock_local = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    GIDS: tl.constexpr = HQ // HIDS
    i_hid = i_h // GIDS

    q_stride_base_local = i_qblock_local * ratio
    q_stride_base_global = chunk_q_start + q_stride_base_local
    q_block_id = q_stride_base_global // ratio
    q_block_valid = (q_stride_base_local < chunk_q_strides) & (
        q_block_id < num_q_blocks
    )

    offs_q = tl.arange(0, ratio)
    q_rows_local = q_stride_base_local + offs_q
    q_valid = q_rows_local < chunk_q_strides
    q_strides_global = q_stride_base_global + offs_q

    head_offset = i_h % STRIDE
    q_token_ids = q_strides_global * STRIDE + head_offset
    q_token_valid = q_token_ids < seqlen_q

    p_in_base = (
        In
        + i_b * in_stride_b
        + i_h * in_stride_h
        + q_stride_base_local * in_stride_q
    )
    p_out = (
        Out
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )
    p_out_mask = (
        OutBoundaryMask
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )

    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.full([ratio], 1.0, dtype=tl.float32)

    num_segments = (n_k_strides + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    num_active_segments = num_segments
    if is_causal:
        last_q_token = (
            (q_stride_base_global + ratio - 1) * STRIDE
            + head_offset
            + shift_tokens
        )
        last_q_stride = last_q_token // STRIDE
        active_k_strides = tl.minimum(
            n_k_strides,
            tl.where(last_q_token >= 0, last_q_stride + 1, 0),
        )
        num_active_segments = (
            active_k_strides + SEGMENT_SIZE - 1
        ) // SEGMENT_SIZE

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)

        curr_stride_offset = (
            i_b * n_k_strides * HIDS + i_hid * n_k_strides + seg_start
        )
        curr_load_mask = (seg_start + offs_k) < n_k_strides
        fully_masked_stride_mask = _check_fully_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=is_causal,
            mode=mode,
        )
        X = tl.where(fully_masked_stride_mask, -1.0e6, X)

        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        l_local = tl.sum(tl.math.exp2(X - m_new[:, None]), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i

    BLOCKS_PER_SEG: tl.constexpr = SEGMENT_SIZE // ratio
    offs_kb = tl.arange(0, BLOCKS_PER_SEG)
    zero_mask = tl.zeros([BLOCKS_PER_SEG], dtype=tl.int8)

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale

        causal_visible = tl.full([ratio, SEGMENT_SIZE], True, dtype=tl.int1)
        if is_causal:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_visible = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_visible, X, -1.0e6)

        curr_stride_offset = (
            i_b * n_k_strides * HIDS + i_hid * n_k_strides + seg_start
        )
        curr_load_mask = (seg_start + offs_k) < n_k_strides
        fully_masked_stride_mask = _check_fully_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=is_causal,
            mode=mode,
        )
        X = tl.where(fully_masked_stride_mask, -1.0e6, X)

        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(m_i[:, None] < -1.0e5, 0.0, X)
        X = tl.where(q_token_valid[:, None], X, 0.0)

        X_reshaped = X.reshape(ratio, BLOCKS_PER_SEG, ratio)
        block_sums = tl.sum(tl.sum(X_reshaped, 2), 0)

        partially_masked_stride_mask = _check_partially_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemin,
            lt_end_nstridemax,
            ut_start_nstridemin,
            ut_end_nstridemax,
            causal=is_causal,
            mode=mode,
        )
        real_partial = (
            ~fully_masked_stride_mask
        ) & partially_masked_stride_mask
        if is_causal:
            real_partial = real_partial & causal_visible
        partial_blocks = real_partial.to(tl.int32).reshape(
            ratio, BLOCKS_PER_SEG, ratio
        )
        partial_block_mask = tl.sum(tl.sum(partial_blocks, 2), 0) > 0

        k_block_base = seg_start // ratio
        k_block_ids = k_block_base + offs_kb
        valid_store = q_block_valid & (k_block_ids < num_k_blocks)
        tl.store(
            p_out + k_block_ids,
            block_sums.to(Out.type.element_ty),
            mask=valid_store,
        )
        tl.store(
            p_out_mask + k_block_ids,
            partial_block_mask.to(tl.int8),
            mask=valid_store,
        )

    if is_causal:
        zero_vals = tl.zeros([BLOCKS_PER_SEG], dtype=tl.float32)
        for seg_idx in range(num_active_segments, num_segments):
            seg_start = seg_idx * SEGMENT_SIZE
            k_block_base = seg_start // ratio
            k_block_ids = k_block_base + offs_kb
            valid_store = q_block_valid & (k_block_ids < num_k_blocks)
            tl.store(
                p_out + k_block_ids,
                zero_vals.to(Out.type.element_ty),
                mask=valid_store,
            )
            tl.store(
                p_out_mask + k_block_ids,
                zero_mask,
                mask=valid_store,
            )


__all__ = [
    "flat_group_gemm_fuse_reshape_rr_kernel",
    "rrattn_flashmask_softmax_reduce_qchunk_kernel",
    "rrattn_gemm_qchunk_gqa_kernel",
    "rrattn_nomask_softmax_reduce_kernel",
    "scan_maxmin_chunked",
    "top_p_kernel",
]
