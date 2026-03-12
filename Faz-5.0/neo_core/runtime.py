from __future__ import annotations

from dataclasses import dataclass
import threading

import taichi as ti
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    torch_device: str
    taichi_arch: str
    bridge_mode: str
    zero_copy_mode: str
    zero_copy_ready: bool
    reason: str


_runtime_lock = threading.Lock()
_runtime_info: RuntimeInfo | None = None


def _probe_torch_cuda() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned false"

    try:
        probe = torch.zeros((1,), device="cuda", dtype=torch.float32)
        probe += 1.0
        torch.cuda.synchronize()
        return True, "shared-vram path available"
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return False, str(exc)


def ensure_runtime(prefer_cuda: bool = False) -> RuntimeInfo:
    global _runtime_info
    if _runtime_info is not None:
        return _runtime_info

    with _runtime_lock:
        if _runtime_info is not None:
            return _runtime_info

        use_cuda = False
        reason = "sm_120 collision: stable async CPU bridge selected"
        if prefer_cuda:
            use_cuda, cuda_reason = _probe_torch_cuda()
            if use_cuda:
                reason = "shared-vram path available"
            else:
                reason = f"sm_120 collision: {cuda_reason}"

        torch_device = "cuda" if use_cuda else "cpu"
        taichi_arch = ti.cuda if use_cuda else ti.cpu
        taichi_arch_name = "cuda" if use_cuda else "cpu"
        zero_copy_mode = "shared-vram" if use_cuda else "disabled_async_cpu_bridge"
        bridge_mode = "torch-direct" if use_cuda else "async_cpu_bridge"

        if not ti.lang.impl.get_runtime().prog:
            ti.init(arch=taichi_arch, random_seed=42)

        _runtime_info = RuntimeInfo(
            torch_device=torch_device,
            taichi_arch=taichi_arch_name,
            bridge_mode=bridge_mode,
            zero_copy_mode=zero_copy_mode,
            zero_copy_ready=use_cuda,
            reason=reason,
        )
        return _runtime_info
