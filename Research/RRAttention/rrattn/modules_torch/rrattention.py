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

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rrattn.ops.rrattention import flat_group_gemm_fuse_reshape_rr_kernel
from rrattn.ops.xattention import (
    softmax_fuse_block_sum_kernel_causal,
    softmax_fuse_block_sum_kernel_non_causal,
)

from .utils import find_blocks_chunked


@dataclass(frozen=True)
class RRAttnConfig:
    # Keep this in sync with rrattn.modules.rrattention.RRAttnConfig.
    block_m: int = 128
    block_n: int = 32
    num_warps: int = 4
    num_stages: int = 1
    segment_size: int = 128


TUNABLE_FIELDS = (
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "segment_size",
)


def gpu_info() -> tuple[str, int | None]:
    if not torch.cuda.is_available():
        return "", None
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.name.lower(), props.major
    except Exception:
        return "", None


GPU_NAME, GPU_MAJOR = gpu_info()


def get_rrattn_config(
    head_dim: int, gpu_name: str | None = None
) -> RRAttnConfig:
    gpu_name = (gpu_name or GPU_NAME or "").lower()

    if "h100" in gpu_name or "h800" in gpu_name:
        if head_dim <= 64:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=4,
                num_stages=1,
                segment_size=256,
            )
        elif head_dim <= 128:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=4,
                num_stages=3,
                segment_size=256,
            )
        else:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=8,
                num_stages=1,
                segment_size=256,
            )

    if head_dim <= 64:
        return RRAttnConfig(
            block_m=128, block_n=16, num_warps=4, num_stages=1, segment_size=256
        )
    elif head_dim <= 128:
        return RRAttnConfig(
            block_m=128, block_n=32, num_warps=4, num_stages=2, segment_size=256
        )
    else:
        return RRAttnConfig(num_warps=4, num_stages=1)


def _normalize_config(
    config: RRAttnConfig | None, *, head_dim: int
) -> RRAttnConfig:
    if config is not None:
        return config
    return get_rrattn_config(head_dim)


def _resolve_segment_size(
    config_segment_size: int, reshaped_block_size: int, k_reshaped_seq_len: int
) -> int:
    segment_size = int(config_segment_size)
    while (
        segment_size > k_reshaped_seq_len and segment_size > reshaped_block_size
    ):
        segment_size //= 2

    if (
        segment_size >= reshaped_block_size
        and segment_size % reshaped_block_size == 0
        and k_reshaped_seq_len % segment_size == 0
    ):
        return segment_size

    max_segment_size = (
        min(segment_size, k_reshaped_seq_len) // reshaped_block_size
    ) * reshaped_block_size
    for candidate in range(
        max_segment_size, reshaped_block_size - 1, -reshaped_block_size
    ):
        if k_reshaped_seq_len % candidate == 0:
            return candidate
    return reshaped_block_size


def _block_sparse_attn_func(*args, **kwargs):
    from block_sparse_attn import block_sparse_attn_func

    return block_sparse_attn_func(*args, **kwargs)


enable_profile = False
attn_time_ms = 0.0
estimate_func_time_ms = 0.0


def set_profile(enable=True):
    global enable_profile
    enable_profile = enable


def is_enable_profile():
    global enable_profile
    return enable_profile


def set_attn_time(attn_time=0.0):
    global attn_time_ms
    attn_time_ms = attn_time


def get_attn_time():
    global attn_time_ms
    return attn_time_ms


def add_attn_time(attn_time):
    global attn_time_ms
    attn_time_ms += attn_time


def set_estimate_func_time(xattn_estimate_func_time=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = xattn_estimate_func_time


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(xattn_estimate_func_time):
    global estimate_func_time_ms
    estimate_func_time_ms += xattn_estimate_func_time


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


def flat_group_gemm_fuse_reshape_rr(
    query_states,
    key_states,
    stride,
    chunk_start,
    chunk_end,
    is_causal=True,
    block_m=128,
    block_n=128,
    num_warps=None,
    num_stages=None,
):
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    assert key_states.shape[0] == batch_size
    assert num_heads % key_states.shape[1] == 0
    assert key_states.shape[3] == head_dim
    gqa_groups = num_heads // key_states.shape[1]
    if gqa_groups != 1:
        key_states = key_states.repeat_interleave(gqa_groups, dim=1)

    output = torch.empty(
        (batch_size, num_heads, q_len // stride, kv_len // stride),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    block_m = int(block_m)
    block_n = int(block_n)
    assert q_len % (stride * block_m) == 0
    assert kv_len % (stride * block_n) == 0

    grid = (
        q_len // stride // block_m,
        kv_len // stride // block_n,
        batch_size * num_heads,
    )
    launch_kwargs = {}
    if num_warps is not None:
        launch_kwargs["num_warps"] = int(num_warps)
    if num_stages is not None:
        launch_kwargs["num_stages"] = int(num_stages)

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
        block_m,
        block_n,
        is_causal,
        **launch_kwargs,
    )

    return output


def rrattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    layer_idx=None,
    config: RRAttnConfig | None = None,
) -> torch.Tensor:
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    config = _normalize_config(config, head_dim=head_dim)

    # 1. 计算相关参数
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    offset_token_chunk_num = k_chunk_num - q_chunk_num
    segment_size = _resolve_segment_size(
        config.segment_size, reshaped_block_size, k_reshaped_seq_len
    )

    attn_sum_list = []

    # 2. Padding
    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(
            query_states, (0, 0, 0, q_num_to_pad), value=0
        ).to("cuda")
    else:
        pad_query_states = query_states

    assert use_triton is True, "rrattn requires use_triton=True"

    # 5. 分chunk处理
    for chunk_idx in range(q_chunk_num):
        if kdb != 1:
            raise ValueError("use_triton and kdb cannot be used together")
        attn_weights_slice = flat_group_gemm_fuse_reshape_rr(
            pad_query_states[
                :,
                :,
                (chunk_idx * reshaped_chunk_size) * stride : (
                    chunk_idx * reshaped_chunk_size + reshaped_chunk_size
                )
                * stride,
                :,
            ],
            pad_key_states,
            stride,
            (k_block_num - q_block_num) * reshaped_block_size
            + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size
            + chunk_idx * reshaped_chunk_size
            + reshaped_chunk_size,
            is_causal=causal,
            block_m=config.block_m,
            block_n=config.block_n,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        attn_sum = softmax_fuse_block_sum(
            attn_weights_slice,
            reshaped_block_size,
            segment_size,
            (k_block_num - q_block_num) * reshaped_block_size
            + chunk_idx * reshaped_chunk_size,
            (k_block_num - q_block_num) * reshaped_block_size
            + chunk_idx * reshaped_chunk_size
            + reshaped_chunk_size,
            k_reshaped_seq_len - k_reshaped_num_to_pad,
            1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
            is_causal=causal,
        )

        attn_sum_list.append(attn_sum)

    # 12. 合并
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = find_blocks_chunked(
        attn_sums,
        0,  # current_index for simple mode
        threshold,
        None,
        decoding=False,
        mode="prefill",
        causal=causal,
    )

    # 应用额外的约束
    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num,
                    q_block_num,
                    dtype=bool,
                    device=key_states.device,
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )

    simple_masks[
        :,
        :,
        (q_len + block_size - 1) // block_size - 1,
        : (k_len + block_size - 1) // block_size,
    ] = True

    return attn_sums, simple_masks


def rrattn_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    layer_idx=None,
    config: RRAttnConfig | None = None,
):
    batch_size, num_kv_heads, k_len, head_dim = key_states.shape
    _, num_q_heads, q_len, _ = query_states.shape
    assert num_q_heads % num_kv_heads == 0, (
        "MHA/GQA requires num_q_heads % num_kv_heads == 0"
    )

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )

    chunk_size = min(
        (q_len + (block_size * stride) - 1)
        // (block_size * stride)
        * (block_size * stride),
        chunk_size,
    )

    if is_enable_profile():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    attn_sums, approx_simple_mask = rrattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        layer_idx=layer_idx,
        config=config,
    )
    if is_enable_profile():
        # torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        add_estimate_func_time(elapsed_time_ms)

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    approx_simple_mask = approx_simple_mask[
        :, :, :q_block_num, :k_block_num
    ].contiguous()

    ####################
    assert block_size == 128
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(
        q_len, num_q_heads, head_dim
    )
    key_states = key_states.transpose(1, 2).view(k_len, num_kv_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(
        k_len, num_kv_heads, head_dim
    )
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_q_heads)],
        device=query_states.device,
        dtype=torch.int32,
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    if is_enable_profile():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    attn_output = _block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask,
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(
        batch_size, q_len, num_q_heads, head_dim
    ).transpose(1, 2)
    if is_enable_profile():
        # torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        add_attn_time(elapsed_time_ms)
    ################################

    del query_states
    num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_q_heads

    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    sparse_ratio = 1.0 - (approx_simple_mask.sum() / num_to_compute)
    del approx_simple_mask, attn_sums
    return attn_output, sparse_ratio


__all__ = [
    "RRAttnConfig",
    "get_rrattn_config",
    "rrattn_estimate",
    "rrattn_prefill",
    "set_profile",
    "is_enable_profile",
    "set_attn_time",
    "get_attn_time",
    "add_attn_time",
    "set_estimate_func_time",
    "get_estimate_func_time",
    "add_estimate_func_time",
]
