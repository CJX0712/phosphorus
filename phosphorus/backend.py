"""实现后端注册表：`PHOS_BACKEND` 解析 + native/numpy 回退（T4 C1，补 T3 C5 欠账）。

⚠️ 两个**正交**的轴，不要混淆：
  * **数学后端**（`engine.BACKENDS`）：`fp32` / `int8` / `tl` / `i2s`
    —— 选哪条数值路径（量化方案）。
  * **实现后端**（本模块）：`native`（C99 内核）/ `numpy`
    —— 同一条数学路径由谁来算。

`resolve()` 只回答第二个问题；数学后端仍由 `PhosModel(backend=...)` 选择。

回退语义（docs/TASKS.md T3 C5）：
  * `auto`   ：native 可用就用，不可用静默（DEBUG 日志）回落 numpy
  * `native` ：显式指定却不可用 → **WARNING** + 回落 numpy（用户必须知情）
  * `numpy` / `fp32` / `int8`：纯 numpy 实现，不加载原生库
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("phosphorus.backend")

_NATIVE_IMPL = "native"
_NUMPY_IMPL = "numpy"
_KNOWN = ("auto", "native", "numpy", "fp32", "int8", "tl", "i2s")

_status_cache: tuple[bool, str] | None = None


def _probe_native() -> tuple[bool, str]:
    """探测原生后端（进程内缓存一次）。"""
    global _status_cache
    if _status_cache is None:
        try:
            from .native import kernels as _kernels

            k = _kernels()
            _status_cache = (True, f"abi={k.abi_version()} {k.build_info()}")
        except Exception as exc:  # noqa: BLE001 - 任何失败都只是"不可用"
            _status_cache = (False, str(exc))
    return _status_cache


def native_status() -> tuple[bool, str]:
    """(是否可用, 原因/描述)。"""
    return _probe_native()


def resolve(requested: str | None = None) -> str:
    """把请求的后端名解析为实际实现：返回 `"native"` 或 `"numpy"`。

    `requested` 为空时读 `PHOS_BACKEND`（默认 `auto`）。
    """
    req = (requested or os.environ.get("PHOS_BACKEND") or "auto").strip().lower()
    if req not in _KNOWN:
        raise ValueError(f"未知后端 {req!r}，可选 {_KNOWN}")
    if req in (_NUMPY_IMPL, "fp32", "int8"):
        return _NUMPY_IMPL
    available, reason = _probe_native()
    if available:
        return _NATIVE_IMPL
    # native 不可用：auto 静默、显式 native 告警
    logger.log(
        logging.WARNING if req == _NATIVE_IMPL else logging.DEBUG,
        "phosphorus: native 实现后端不可用（%s），回退 numpy",
        reason,
    )
    return _NUMPY_IMPL


def available_backends() -> tuple[str, ...]:
    """当前实际可用的实现后端（与实测一致，T3 C5 要求）。"""
    available, _ = _probe_native()
    return (_NATIVE_IMPL, _NUMPY_IMPL) if available else (_NUMPY_IMPL,)
