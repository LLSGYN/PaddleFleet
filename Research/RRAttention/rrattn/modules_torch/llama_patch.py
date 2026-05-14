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
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from .patch_utils import attention_forward, patch_attention_layers


def get_llama_attention_classes():
    from transformers.models.llama import modeling_llama

    classes = []
    for name in (
        "LlamaAttention",
        "LlamaFlashAttention2",
        "LlamaSdpaAttention",
    ):
        cls = getattr(modeling_llama, name, None)
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


@torch.no_grad()
def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    return attention_forward(
        self,
        hidden_states,
        apply_rotary_pos_emb,
        repeat_kv,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )


def patch_llama_attention(
    model,
    method: str = "rrattn",
    threshold: float = 0.9,
    stride: int = 8,
    **kwargs,
):
    return patch_attention_layers(
        model,
        get_llama_attention_classes(),
        new_attention_forward,
        method=method,
        threshold=threshold,
        stride=stride,
        **kwargs,
    )


__all__ = ["patch_llama_attention", "new_attention_forward"]
