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

import types
from collections.abc import Iterable

import torch

from .flexprefill import flex_prefill
from .full_prefill import flash_full_prefill
from .rrattention import rrattn_prefill
from .xattention import xattn_prefill

SUPPORTED_METHODS = ("xattn", "rrattn", "flex", "full")


def validate_method(method: str):
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method={method!r}; supported methods are: xattn, rrattn, flex, full"
        )


def select_layer_value(value, layer_idx: int):
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def get_decoder_layers(model) -> Iterable:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return []


def configure_attention_layer(
    attn,
    layer_idx: int,
    method: str,
    threshold: float,
    stride: int,
    **kwargs,
):
    attn.method = method
    attn.threshold = select_layer_value(threshold, layer_idx)
    attn.stride = select_layer_value(stride, layer_idx)
    attn.sparse_ratio = None
    for key, value in kwargs.items():
        setattr(attn, key, select_layer_value(value, layer_idx))


def bind_attention_forward(attn, new_forward):
    if not hasattr(attn, "_rrattn_original_forward"):
        attn._rrattn_original_forward = attn.forward
    attn.forward = types.MethodType(new_forward, attn)


def patch_attention_layers(
    model,
    attention_cls,
    new_forward,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    **kwargs,
):
    validate_method(method)
    patched = 0
    seen = set()

    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if attention_cls is not None and not isinstance(attn, attention_cls):
            continue
        configure_attention_layer(
            attn,
            layer_idx=layer_idx,
            method=method,
            threshold=threshold,
            stride=stride,
            **kwargs,
        )
        bind_attention_forward(attn, new_forward)
        seen.add(id(attn))
        patched += 1

    if hasattr(model, "named_modules"):
        for _, attn in model.named_modules():
            if id(attn) in seen:
                continue
            if attention_cls is not None and not isinstance(
                attn, attention_cls
            ):
                continue
            layer_idx = getattr(attn, "layer_idx", patched)
            configure_attention_layer(
                attn,
                layer_idx=layer_idx,
                method=method,
                threshold=threshold,
                stride=stride,
                **kwargs,
            )
            bind_attention_forward(attn, new_forward)
            seen.add(id(attn))
            patched += 1

    if patched == 0:
        if attention_cls is None:
            attention_name = "attention"
        elif isinstance(attention_cls, tuple):
            attention_name = "/".join(cls.__name__ for cls in attention_cls)
        else:
            attention_name = attention_cls.__name__
        raise ValueError(f"No {attention_name} layers found")

    for module in (model, getattr(model, "model", None)):
        if module is not None and hasattr(module, "config"):
            module.config._attn_implementation = "flash_attention_2"
    return model


def prefill_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = True,
    scaling: float | None = None,
) -> torch.Tensor:
    method = getattr(module, "method", "rrattn")
    validate_method(method)
    threshold = getattr(module, "threshold", 0.9)
    stride = getattr(module, "stride", 8)

    if method == "xattn":
        attn_output, sparse_ratio = xattn_prefill(
            query_states,
            key_states,
            value_states,
            norm=1,
            stride=stride,
            threshold=threshold,
            use_triton=getattr(module, "use_triton", True),
            keep_sink=getattr(module, "keep_sink", True),
            keep_recent=getattr(module, "keep_recent", True),
            chunk_size=getattr(module, "chunk_size", 16384),
            layer_idx=getattr(module, "layer_idx", None),
        )
        module.sparse_ratio = sparse_ratio
        return attn_output.transpose(1, 2)

    if method == "rrattn":
        attn_output, sparse_ratio = rrattn_prefill(
            query_states,
            key_states,
            value_states,
            norm=1,
            stride=stride,
            threshold=threshold,
            use_triton=getattr(module, "use_triton", True),
            keep_sink=getattr(module, "keep_sink", True),
            keep_recent=getattr(module, "keep_recent", True),
            chunk_size=getattr(module, "chunk_size", 16384),
            layer_idx=getattr(module, "layer_idx", None),
        )
        module.sparse_ratio = sparse_ratio
        return attn_output.transpose(1, 2)

    if method == "flex":
        result = flex_prefill(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            gamma=threshold,
            tau=getattr(module, "tau", 0.1),
            min_budget=getattr(module, "min_budget", None),
            max_budget=getattr(module, "max_budget", None),
            gqa_interleave=getattr(module, "gqa_interleave", False),
            softmax_scale=scaling,
            block_size=getattr(module, "block_size", 128),
        )
        if isinstance(result, tuple):
            attn_output, sparse_ratio = result
        else:
            attn_output = result
            sparse_ratio = 0.0
        module.sparse_ratio = sparse_ratio
        return attn_output

    attn_output = flash_full_prefill(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=causal,
        softmax_scale=scaling,
    )
    module.sparse_ratio = 0.0
    return attn_output


def decode_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    from flash_attn import flash_attn_func

    batch_size = query_states.shape[0]
    if batch_size != 1:
        raise ValueError(
            "flash attention decode currently supports batch_size=1"
        )

    q_len = query_states.shape[-2]
    dropout_p = 0.0
    if getattr(module, "training", False):
        dropout_p = getattr(module, "attention_dropout", 0.0)

    del attention_mask

    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=dropout_p,
        softmax_scale=getattr(module, "scaling", None),
        causal=False if q_len == 1 else getattr(module, "is_causal", True),
    )
    return attn_output


def attention_forward(
    module,
    hidden_states: torch.Tensor,
    apply_rotary_pos_emb,
    repeat_kv,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, module.num_heads, module.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, module.num_key_value_heads, module.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, module.num_key_value_heads, module.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = module.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, module.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = module.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    scaling = getattr(module, "scaling", None)
    if key_states.shape[2] == query_states.shape[2]:
        attn_output = prefill_attention(
            module,
            query_states,
            key_states,
            value_states,
            causal=getattr(module, "is_causal", True),
            scaling=scaling,
        )
    else:
        attn_output = decode_attention(
            module, query_states, key_states, value_states, attention_mask
        )

    hidden_size = getattr(
        module, "hidden_size", module.num_heads * module.head_dim
    )
    attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()
    attn_output = module.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value
