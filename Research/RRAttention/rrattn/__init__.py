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
    "rrattn_estimate": ("rrattn.modules.rrattention", "rrattn_estimate"),
    "rrattn_prefill": ("rrattn.modules.rrattention", "rrattn_prefill"),
    "patch_llama_attention": (
        "rrattn.modules.llama_patch",
        "patch_llama_attention",
    ),
    "patch_qwen_attention": (
        "rrattn.modules.qwen_patch",
        "patch_qwen_attention",
    ),
    "patch_ernie_attention": (
        "rrattn.modules.ernie_patch",
        "patch_ernie_attention",
    ),
    "RRAttnConfig": ("rrattn.modules.rrattention", "RRAttnConfig"),
    "get_rrattn_config": ("rrattn.modules.rrattention", "get_rrattn_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name == "paddle":
            raise ModuleNotFoundError(
                "The top-level rrattn API is Paddle-only and requires "
                "PaddlePaddle. Torch users should import from "
                "rrattn.modules_torch."
            ) from exc
        raise
    globals()[name] = value
    return value
