"""Phosphorus（晨星）—— 三值原生（1.58-bit）LLM 的训练 + 投机解码推理引擎。

设计要点（详见 docs/ARCHITECTURE.md）：
    * **运行时零 torch 依赖**：`import phosphorus` 与 `import phosphorus.engine.*`
      只依赖 numpy + tqdm（core 分组）。训练相关子包（`phosphorus.model` /
      `phosphorus.train`）才惰性导入 torch。
    * **三级后端回退**：native(zig cc) -> numpy -> pure-python，见 `kernels.registry`。

本模块只做版本与能力探测，不导入任何重依赖。
"""

from __future__ import annotations

import importlib.util
import os
import platform
from typing import Final

from .scheduler import Scheduler

__version__: Final[str] = "0.1.0"
__author__: Final[str] = "晨星（CJX0712）"
__license__: Final[str] = "Apache-2.0"

# .phos 格式版本（ARCHITECTURE.md §3）
PHOS_VERSION_MAJOR: Final[int] = 1
PHOS_VERSION_MINOR: Final[int] = 0


def has_torch() -> bool:
    """返回当前环境是否可用 torch（不真正导入，只探测元数据）。"""
    return importlib.util.find_spec("torch") is not None


def has_ziglang() -> bool:
    """返回当前环境是否可用 ziglang（O4：零系统编译器依赖的构建链）。"""
    return importlib.util.find_spec("ziglang") is not None


def default_backend() -> str:
    """默认计算后端：`PHOS_BACKEND` 环境变量优先，否则 auto。"""
    return os.environ.get("PHOS_BACKEND", "auto")


def default_threads() -> int:
    """默认线程数：`PHOS_THREADS` 环境变量优先，否则 CPU 逻辑核数。"""
    env = os.environ.get("PHOS_THREADS")
    if env and env.isdigit():
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def env_summary() -> str:
    """返回一行环境摘要，便于报告与 issue 复现。"""
    import sys

    return (
        f"phosphorus {__version__} | python {sys.version.split()[0]} | "
        f"{platform.system()} {platform.machine()} | "
        f"torch={has_torch()} ziglang={has_ziglang()} threads={default_threads()}"
    )


__all__ = [
    "PHOS_VERSION_MAJOR",
    "PHOS_VERSION_MINOR",
    "__author__",
    "__license__",
    "__version__",
    "Scheduler",
    "default_backend",
    "default_threads",
    "env_summary",
    "has_torch",
    "has_ziglang",
]
