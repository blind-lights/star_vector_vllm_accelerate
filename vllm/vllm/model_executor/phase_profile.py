"""CUDA-event phase timer.

torch.profiler 在这块 sm_120 上 CUPTI 初始化失败，trace 里没有 kernel。
这里用 CUDA Event 记录阶段和叶子模块的 GPU 时间，只在 start_profile 到
stop_profile 之间启用。
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

_enabled = False
_stack: List[str] = []
_phase_events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
_op_events: List[Tuple[str, str, str, torch.cuda.Event, torch.cuda.Event]] = []
_handles: List[torch.utils.hooks.RemovableHandle] = []

_LINEAR_TYPES = {
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
}
_LEAF_CLASSES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.Embedding,
    nn.GELU,
    nn.MultiheadAttention,
)


def _timed_module(module: nn.Module) -> bool:
    type_name = type(module).__name__
    if type_name in _LINEAR_TYPES or type_name in {
            "Attention", "ParallelLMHead", "VocabParallelEmbedding"
    }:
        return True
    if "GELU" in type_name or type_name in {"QuickGELU", "NewGELU", "GELUActivation"}:
        return True
    return isinstance(module, _LEAF_CLASSES)


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled
    if enabled:
        _stack.clear()
        _phase_events.clear()
        _op_events.clear()


def attach(model: nn.Module) -> None:
    """给叶子模块挂 forward hook。重复调用先拆掉旧 hook。"""
    detach()

    def pre_hook(module: nn.Module, _args, qualified: str = "",
                 type_name: str = ""):
        if not _enabled or not _stack:
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.__dict__["_sv_op"] = (qualified, type_name, _stack[-1], start, end)

    def post_hook(module: nn.Module, _args, _output):
        recorded = module.__dict__.pop("_sv_op", None)
        if recorded is None:
            return
        recorded[4].record()
        _op_events.append(recorded)

    for qualified, module in model.named_modules():
        if not qualified or not _timed_module(module):
            continue
        type_name = type(module).__name__
        _handles.append(
            module.register_forward_pre_hook(
                lambda mod, args, q=qualified, t=type_name: pre_hook(
                    mod, args, q, t)))
        _handles.append(module.register_forward_hook(post_hook))


def detach() -> None:
    while _handles:
        _handles.pop().remove()


@contextmanager
def phase(name: str) -> Iterator[None]:
    use_nvtx = os.environ.get("STARVECTOR_NVTX") == "1"
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    timing = _enabled and torch.cuda.is_available()
    start = None
    end = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _stack.append(name)
        start.record()
    try:
        yield
    finally:
        if timing:
            end.record()
            _phase_events.append((name, start, end))
            _stack.pop()
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


def _region(qualified: str) -> str:
    if "image_encoder" in qualified or "visual" in qualified:
        return "vision"
    if "image_projection" in qualified:
        return "adapter"
    if "lm_head" in qualified:
        return "lm_head"
    return "llm"


def _role(qualified: str, type_name: str) -> str:
    lowered = qualified.lower()
    if type_name in {"Attention", "MultiheadAttention"} or ".attn" in lowered:
        return "attention"
    if "lm_head" in lowered:
        return "lm_head"
    if "conv" in type_name.lower() or ".conv" in lowered:
        return "conv"
    if "Norm" in type_name or "norm" in lowered or ".ln_" in lowered:
        return "norm"
    if "GELU" in type_name or "gelu" in lowered or "act" in lowered:
        return "activation"
    if type_name in _LINEAR_TYPES or type_name == "Linear":
        if ".mlp." in lowered:
            return "mlp_linear"
        if ".attn." in lowered:
            return "attn_linear"
        return "linear"
    if type_name == "Embedding":
        return "embedding"
    return type_name


def dump(path: str) -> Optional[str]:
    if not _phase_events and not _op_events:
        return None
    torch.cuda.synchronize()
    phases: Dict[str, Dict[str, float]] = {}
    for name, start, end in _phase_events:
        bucket = phases.setdefault(name, {"count": 0, "total_ms": 0.0})
        bucket["count"] += 1
        bucket["total_ms"] += start.elapsed_time(end)
    ops: Dict[str, Dict[str, float]] = {}
    for qualified, type_name, phase_name, start, end in _op_events:
        key = f"{phase_name}/{_region(qualified)}/{_role(qualified, type_name)}"
        bucket = ops.setdefault(key, {"count": 0, "total_ms": 0.0})
        bucket["count"] += 1
        bucket["total_ms"] += start.elapsed_time(end)
    payload = {"phases": phases, "ops": ops}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
