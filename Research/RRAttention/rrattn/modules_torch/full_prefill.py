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

import torch
import torch.nn.functional as F

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


def set_estimate_func_time(estimate_func_time=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = estimate_func_time


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(estimate_func_time):
    global estimate_func_time_ms
    estimate_func_time_ms += estimate_func_time


def full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    attention_mask=None,
):
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(query_states.shape[-1])

    if causal:
        q_len = query_states.shape[-2]
        k_len = key_states.shape[-2]
        q_pos = torch.arange(q_len, device=query_states.device) + k_len - q_len
        k_pos = torch.arange(k_len, device=query_states.device)
        causal_mask = q_pos[:, None] < k_pos[None, :]
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            attention_mask = torch.where(attention_mask == 0, True, False)
        attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    return torch.matmul(attn_weights, value_states)


def flash_full_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
):
    from flash_attn import flash_attn_func

    if is_enable_profile():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    if is_enable_profile():
        end_event.record()
        torch.cuda.synchronize()
        add_attn_time(start_event.elapsed_time(end_event))

    return attn_output


__all__ = [
    "full_prefill",
    "flash_full_prefill",
    "set_profile",
    "is_enable_profile",
    "set_attn_time",
    "get_attn_time",
    "add_attn_time",
    "set_estimate_func_time",
    "get_estimate_func_time",
    "add_estimate_func_time",
]
