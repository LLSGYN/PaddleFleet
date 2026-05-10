#!/usr/bin/env python

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

from __future__ import annotations

import argparse
import contextlib
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import paddle

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.speed_test import (  # noqa: E402
    clear_cache,
    infer_model_type,
    load_model,
    load_patch,
    load_tokenizer,
    quick_get_random_kv_samples,
    synchronize,
)


DEFAULT_SEQ_LENS = "8192,16384,32768,65536,131072"
DEFAULT_THRESHOLDS = "0.9,0.95"
DEFAULT_WINDOW_SIZES = "128,256,512,1024,2048,4096,8192"
def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, str):
        values = [int(item) for item in value.split(",") if item.strip()]
    else:
        values = [int(item) for item in value]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, str):
        values = [float(item) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def build_swa_startend_row_indices(
    batch_size: int,
    seq_len: int,
    window_size: int,
) -> paddle.Tensor:
    startend_row_indices = paddle.arange(
        window_size,
        seq_len + window_size,
        dtype="int32",
    ).reshape((1, 1, seq_len, 1))
    return paddle.clip(startend_row_indices, max=seq_len).repeat_interleave(
        batch_size,
        axis=0,
    )


@contextlib.contextmanager
def patched_swa_attention_branch():
    import rrattn.patch_utils as patch_utils
    from paddle.nn.functional.flash_attention import flashmask_attention
    from paddleformers.nn.attention.eager_attention import repeat_kv

    original_attention_branch = patch_utils.attention_branch

    def swa_attention_branch(
        module,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        dropout: float = 0.0,
        causal: bool = True,
        scaling: float | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        method = getattr(module, "method", "rrattn")
        if method != "swa":
            return original_attention_branch(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                dropout=dropout,
                causal=causal,
                scaling=scaling,
            )

        del attention_mask
        if attn_mask_startend_row_indices is None:
            raise ValueError("SWA benchmark requires startend_row_indices")
        if key_states.shape[2] != query_states.shape[2]:
            raise NotImplementedError("SWA benchmark only supports prefill")

        key_states = repeat_kv(key_states, module.num_key_value_groups)
        value_states = repeat_kv(value_states, module.num_key_value_groups)

        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = flashmask_attention(
            query_states,
            key_states,
            value_states,
            startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not module.training else dropout,
            causal=causal,
            softmax_scale=scaling,
        )
        if isinstance(attn_output, (list, tuple)):
            attn_output = attn_output[0]

        clear_sparse_ratio(module)
        attn_output = attn_output.reshape(
            [attn_output.shape[0], attn_output.shape[1], -1]
        ).contiguous()
        return attn_output, None

    patch_utils.attention_branch = swa_attention_branch
    try:
        yield
    finally:
        patch_utils.attention_branch = original_attention_branch


def clear_sparse_ratio(module):
    buffers = getattr(module, "_buffers", None)
    if buffers is not None and "sparse_ratio" in buffers:
        module.sparse_ratio = None
    else:
        module.sparse_ratio = 0.0


def set_patched_attention_config(
    model,
    method: str,
    threshold: float | None = None,
    stride: int | None = None,
) -> int:
    from rrattn.patch_utils import get_decoder_layers

    count = 0
    seen = set()
    for layer in get_decoder_layers(model):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if not hasattr(attn, "_rrattn_original_forward"):
            continue
        attn.method = method
        if threshold is not None:
            attn.threshold = threshold
        if stride is not None:
            attn.stride = stride
        seen.add(id(attn))
        count += 1

    if hasattr(model, "named_sublayers"):
        for _, attn in model.named_sublayers():
            if id(attn) in seen:
                continue
            if not hasattr(attn, "_rrattn_original_forward"):
                continue
            attn.method = method
            if threshold is not None:
                attn.threshold = threshold
            if stride is not None:
                attn.stride = stride
            seen.add(id(attn))
            count += 1

    if count == 0:
        raise ValueError("No patched attention layers found")
    return count


def measure_ttft(
    model,
    input_ids: paddle.Tensor,
    attention_mask: paddle.Tensor | None,
    device: str,
    attn_mask_startend_row_indices: paddle.Tensor | None = None,
) -> float:
    synchronize(device)
    start_event = None
    end_event = None
    if device.startswith("gpu"):
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    start_time = time.perf_counter()
    model_kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    if attn_mask_startend_row_indices is not None:
        model_kwargs[
            "attn_mask_startend_row_indices"
        ] = attn_mask_startend_row_indices

    with paddle.no_grad():
        model(**model_kwargs)
    synchronize(device)

    elapsed_wall_ms = (time.perf_counter() - start_time) * 1000.0
    if device.startswith("gpu"):
        end_event.record()
        synchronize(device)
        return float(start_event.elapsed_time(end_event))
    return elapsed_wall_ms


def slice_sample(
    sample: dict,
    seq_len: int,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    input_ids = sample["input_ids"][..., :seq_len]
    attention_mask = sample["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask[..., :seq_len]
    return input_ids, attention_mask


def summarize_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array))


@dataclass(frozen=True)
class Case:
    method: str
    seq_len: int
    threshold: float | None = None
    window_size: int | None = None


def iter_cases(
    seq_lens: list[int],
    thresholds: list[float],
    window_sizes: list[int],
) -> list[Case]:
    cases = []
    for seq_len in seq_lens:
        cases.append(Case("full", seq_len, threshold=1.0))
    for threshold in thresholds:
        for seq_len in seq_lens:
            cases.append(Case("rrattn", seq_len, threshold=threshold))
    for window_size in window_sizes:
        for seq_len in seq_lens:
            cases.append(Case("swa", seq_len, window_size=window_size))
    return cases


def benchmark_case(
    model,
    samples: list[dict],
    case: Case,
    device: str,
    stride: int,
    warmup_iters: int,
) -> dict:
    startend_row_indices = None
    if case.method == "swa":
        batch_size = int(samples[0]["input_ids"].shape[0])
        startend_row_indices = build_swa_startend_row_indices(
            batch_size=batch_size,
            seq_len=case.seq_len,
            window_size=case.window_size,
        )

    for _ in range(warmup_iters):
        input_ids, attention_mask = slice_sample(samples[0], case.seq_len)
        measure_ttft(
            model,
            input_ids,
            attention_mask,
            device,
            attn_mask_startend_row_indices=startend_row_indices,
        )
        clear_cache(device)

    ttft_times = []
    for sample in samples:
        input_ids, attention_mask = slice_sample(sample, case.seq_len)
        elapsed_ms = measure_ttft(
            model,
            input_ids,
            attention_mask,
            device,
            attn_mask_startend_row_indices=startend_row_indices,
        )
        ttft_times.append(elapsed_ms)
        clear_cache(device)

    ttft_mean = summarize_mean(ttft_times)
    saved_stride = ""
    if case.method == "rrattn":
        saved_stride = stride
    return {
        "method": case.method,
        "seq_len": case.seq_len,
        "threshold": "" if case.threshold is None else case.threshold,
        "window_size": "" if case.window_size is None else case.window_size,
        "stride": saved_stride,
        "ttft_mean_ms": ttft_mean,
    }


def validate_prompt_lengths(
    samples: list[dict],
    seq_lens: list[int],
    n_kv_num: int,
):
    prompt_lens = [int(sample["input_ids"].shape[-1]) for sample in samples]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    max_target_len = max(seq_lens)
    if min_prompt_len < max_target_len:
        raise ValueError(
            "Generated prompts are too short for the requested seq_lens: "
            f"shortest={min_prompt_len}, longest={max_prompt_len}, "
            f"required={max_target_len}. Increase --n-kv-num above "
            f"{n_kv_num} or lower --seq-lens."
        )


def output_path(model_name: str, output_dir: str) -> Path:
    return (
        Path(output_dir)
        / f"benchmark_rrattention_{Path(model_name).name}_result.csv"
    )


def format_threshold(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def case_column_name(row: dict) -> str:
    if row["method"] == "full":
        return "full"
    if row["method"] == "rrattn":
        return f"rrattn_t{format_threshold(row['threshold'])}"
    return f"swa_w{row['window_size']}"


def csv_columns(
    thresholds: list[float],
    window_sizes: list[int],
) -> list[str]:
    columns = ["full"]
    columns.extend(
        f"rrattn_t{format_threshold(threshold)}" for threshold in thresholds
    )
    columns.extend(f"swa_w{window_size}" for window_size in window_sizes)
    return columns


def write_wide_csv(
    path: Path,
    rows: list[dict],
    seq_lens: list[int],
    thresholds: list[float],
    window_sizes: list[int],
):
    columns = csv_columns(thresholds, window_sizes)
    table = {
        seq_len: {"seq_len": seq_len, **{column: "" for column in columns}}
        for seq_len in seq_lens
    }
    for row in rows:
        seq_len = int(row["seq_len"])
        table[seq_len][case_column_name(row)] = row["ttft_mean_ms"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_len", *columns])
        writer.writeheader()
        writer.writerows(table[seq_len] for seq_len in seq_lens)


def print_case_result(row: dict):
    if row["method"] in {"rrattn", "full"}:
        config = f"threshold={row['threshold']}"
    else:
        config = f"window_size={row['window_size']}"
    print(
        "{method:<6} seq_len={seq_len:<6} {config:<16} "
        "ttft_mean={ttft:.2f}ms".format(
            method=row["method"],
            seq_len=row["seq_len"],
            config=config,
            ttft=row["ttft_mean_ms"],
        )
    )


def warn_if_non_hopper(device: str):
    if not device.startswith("gpu"):
        return
    try:
        major, minor = paddle.device.cuda.get_device_capability()
    except Exception:
        return
    if major < 9:
        print(
            "warning: FA v3 benchmark is intended for Hopper GPUs; "
            f"current capability is {major}.{minor}."
        )


def patch_paddle_compat_for_paddlefleet_import():
    compat = getattr(paddle, "compat", None)
    if compat is None:
        return
    if callable(getattr(compat, "enable_torch_proxy", None)):
        return

    enable_compat = getattr(paddle, "enable_compat", None)
    if callable(enable_compat):
        compat.enable_torch_proxy = enable_compat
    else:
        compat.enable_torch_proxy = lambda *args, **kwargs: None


def dry_run(
    seq_lens: list[int],
    thresholds: list[float],
    window_sizes: list[int],
):
    paddle.set_device("cpu")
    cases = iter_cases(seq_lens, thresholds, window_sizes)
    print(f"case_count={len(cases)}")
    for case in cases:
        if case.method in {"rrattn", "full"}:
            print(
                f"{case.method} seq_len={case.seq_len} "
                f"threshold={case.threshold}"
            )
        else:
            print(f"swa seq_len={case.seq_len} window_size={case.window_size}")

    mask = build_swa_startend_row_indices(
        batch_size=1,
        seq_len=8,
        window_size=4,
    )
    expected = [4, 5, 6, 7, 8, 8, 8, 8]
    actual = mask.reshape([-1]).numpy().tolist()
    if actual != expected:
        raise AssertionError(
            f"SWA mask dry-run mismatch: expected={expected}, actual={actual}"
        )
    print("dry-run SWA mask check passed")


def main(
    model_name: str,
    model_type: str = "auto",
    seq_lens: str = DEFAULT_SEQ_LENS,
    thresholds: str = DEFAULT_THRESHOLDS,
    window_sizes: str = DEFAULT_WINDOW_SIZES,
    stride: int = 8,
    n_times: int = 3,
    warmup_iters: int = 1,
    n_kv_num: int = 6000,
    gold_index: int = 3000,
    dtype: str = "bfloat16",
    device: str = "gpu:0",
    output_dir: str = ".",
    seed: int = 20260420,
    dry_run_only: bool = False,
):
    if n_times <= 0:
        raise ValueError("--n-times must be positive")
    if warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative")

    seq_len_list = parse_int_list(seq_lens)
    threshold_list = parse_float_list(thresholds)
    window_size_list = parse_int_list(window_sizes)
    cases = iter_cases(seq_len_list, threshold_list, window_size_list)

    if dry_run_only:
        dry_run(seq_len_list, threshold_list, window_size_list)
        return

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.set_flags({"FLAGS_flash_attn_version": 3})
    paddle.set_device(device)

    if model_type == "auto":
        model_type = infer_model_type(model_name)

    warn_if_non_hopper(device)
    patch_paddle_compat_for_paddlefleet_import()

    model = load_model(model_name, model_type, dtype)
    model.eval()
    patch_fn = load_patch(model_type)
    patch_fn(
        model,
        method="rrattn",
        threshold=threshold_list[0],
        stride=stride,
    )
    tokenizer = load_tokenizer(model_name)
    samples = quick_get_random_kv_samples(
        model_name,
        tokenizer,
        gold_index=gold_index,
        n_kv_num=n_kv_num,
        n_sample=n_times,
        device=device,
    )
    validate_prompt_lengths(samples, seq_len_list, n_kv_num)

    fa_version = paddle.base.framework.get_flags(
        ["FLAGS_flash_attn_version"]
    )["FLAGS_flash_attn_version"]
    print(
        f"model={model_name} model_type={model_type} device={device} "
        f"dtype={dtype} fa_version={fa_version}"
    )
    print(f"cases={len(cases)} n_times={n_times} warmup_iters={warmup_iters}")

    rows = []
    with patched_swa_attention_branch():
        for case in cases:
            if case.method == "rrattn":
                set_patched_attention_config(
                    model,
                    method="rrattn",
                    threshold=case.threshold,
                    stride=stride,
                )
            elif case.method == "full":
                set_patched_attention_config(
                    model,
                    method="full",
                    threshold=case.threshold,
                    stride=1,
                )
            else:
                set_patched_attention_config(model, method="swa")

            row = benchmark_case(
                model,
                samples,
                case,
                device,
                stride,
                warmup_iters,
            )
            row.update(
                {
                    "model": Path(model_name).name,
                    "model_type": model_type,
                    "dtype": dtype,
                    "device": device,
                    "fa_version": fa_version,
                }
            )
            rows.append(row)
            print_case_result(row)

    saved_path = output_path(model_name, output_dir)
    write_wide_csv(
        saved_path,
        rows,
        seq_len_list,
        threshold_list,
        window_size_list,
    )
    print(f"saved: {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "llama", "qwen", "ernie", "ernie_moe"],
    )
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--window-sizes", default=DEFAULT_WINDOW_SIZES)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n-times", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--n-kv-num", type=int, default=6000)
    parser.add_argument("--gold-index", type=int, default=3000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run_only",
        help="Print the benchmark matrix and validate SWA mask construction.",
    )
    main(**vars(parser.parse_args()))
