"""phosphorus.native —— C99 原生内核（T4）：zig cc 编译 / ctypes 绑定 / 并行切分。

对外只暴露三个入口：
  * `kernels()`      —— 获取内核封装（首次调用触发编译/加载；不可用抛 NativeUnavailable）
  * `kernels_or_none()` / `is_available()` —— 不抛异常的能力探测
  * `unavailable_reason()` —— 上一次不可用的原因（供 WARN 日志）

设计红线（全部来自 2026-10-17 实测）：
  * 显式 `-target`，否则产物含 AVX-512，旧机器 SIGILL
  * 内容寻址唯一文件名，Windows 已加载 DLL 不可覆盖
  * 绝对路径 `CDLL`（cdecl）；禁用 `PyDLL`（持 GIL）
  * 不可用时静默/告警降级 numpy，绝不 crash
"""

from __future__ import annotations

from .bindings import (
    PHOS_FLAG_AVX2,
    PHOS_FLAG_SCALAR,
    NativeKernels,
    kernels,
    kernels_or_none,
    unavailable_reason,
)
from .build import (
    NativeUnavailable,
    cache_dir,
    csrc_dir,
    ensure_library,
    library_or_none,
)
from .parallel import default_threads, map_rows, row_blocks

__all__ = [
    "PHOS_FLAG_AVX2",
    "PHOS_FLAG_SCALAR",
    "NativeKernels",
    "NativeUnavailable",
    "cache_dir",
    "csrc_dir",
    "default_threads",
    "ensure_library",
    "is_available",
    "kernels",
    "kernels_or_none",
    "library_or_none",
    "map_rows",
    "row_blocks",
    "unavailable_reason",
]


def is_available() -> bool:
    """原生后端是否可用（不抛异常）。"""
    return kernels_or_none() is not None
