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

_EXPORTS = {
    "full_prefill": ("rrattn.modules_torch.full_prefill", "full_prefill"),
    "flash_full_prefill": (
        "rrattn.modules_torch.full_prefill",
        "flash_full_prefill",
    ),
    "flex_prefill": ("rrattn.modules_torch.flexprefill", "flex_prefill"),
    "xattn_prefill": ("rrattn.modules_torch.xattention", "xattn_prefill"),
    "rrattn_prefill": ("rrattn.modules_torch.rrattention", "rrattn_prefill"),
    "rrattn_estimate": (
        "rrattn.modules_torch.rrattention",
        "rrattn_estimate",
    ),
    "RRAttnConfig": ("rrattn.modules_torch.rrattention", "RRAttnConfig"),
    "get_rrattn_config": (
        "rrattn.modules_torch.rrattention",
        "get_rrattn_config",
    ),
    "patch_llama_attention": (
        "rrattn.modules_torch.llama_patch",
        "patch_llama_attention",
    ),
    "patch_qwen_attention": (
        "rrattn.modules_torch.qwen_patch",
        "patch_qwen_attention",
    ),
    "patch_ernie_attention": (
        "rrattn.modules_torch.ernie_patch",
        "patch_ernie_attention",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
