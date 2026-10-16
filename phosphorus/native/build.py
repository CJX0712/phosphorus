"""zig cc 编译 C99 原生内核 + 内容寻址缓存（T4 C1）。

设计要点（全部来自 2026-10-17 实测，见 docs/BENCHMARKS.md）：
  * 必须显式 `-target <triple>`：`zig cc` 不带 `-target` 时按 host CPU 产出，
    本机会定义 `__AVX512F__`，产物在 pre-AVX-512 机器上直接 SIGILL。
  * 缓存产物用**内容寻址唯一文件名**：Windows 上已 `CDLL` 加载的 DLL 不可
    覆盖 / 删除 / `os.replace`（WinError 5），源码一变必须产出新文件名。
  * 编译到 `<cache>/<hash>.tmp` 再 `os.replace` 原子落位；并发编译用
    `O_CREAT|O_EXCL` 文件锁，120 s 拿不到锁就放弃并降级（不阻塞启动）。
  * ctypes 只吃**绝对路径**；用 `CDLL`（cdecl）而非 `PyDLL`（持 GIL）。
  * 冷启动路径绝不调用 zig：hash 命中即 `CDLL`，实测 ≈ 2–4 ms。
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("phosphorus.native")

ABI_VERSION = 1

# C 源文件（位于包内：wheel 只含 `phosphorus*`，顶层 csrc/ 装不进包）
CSRC = ("phos.h", "phos_gemm.c", "phos_pack.c", "phos_ops.c")
_MAIN_SOURCES = ("phos_gemm.c", "phos_pack.c", "phos_ops.c")

_LOCK_TIMEOUT_S = 120.0
_COMPILE_TIMEOUT_S = 300.0


class NativeUnavailable(RuntimeError):
    """原生后端不可用：未装 ziglang / 编译失败 / ABI 不匹配 / 加载失败。"""


def csrc_dir() -> Path:
    """C 源码目录（包内 `phosphorus/csrc/`）。"""
    return Path(__file__).resolve().parent.parent / "csrc"


def cache_dir() -> Path:
    """编译产物缓存目录（可用 `PHOS_NATIVE_DIR` 覆盖，便于 CI / 只读 HOME）。"""
    env = os.environ.get("PHOS_NATIVE_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "phosphorus" / "native"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _triple() -> str:
    """zig `-target` 三元组：显式指定，避免 host CPU 特化（AVX-512 SIGILL）。"""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        arch = "x86_64"
    sysname = platform.system()
    if sysname == "Windows":
        return f"{arch}-windows-gnu"
    if sysname == "Darwin":
        return f"{arch}-macos-none"
    return f"{arch}-linux-gnu"


def _ext() -> str:
    return {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")


def _zig_version() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "ziglang", "version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return (out.stdout or out.stderr).strip() or "unknown"
    except Exception:  # noqa: BLE001 - 版本探测失败不应阻塞后续判断
        return "unknown"


def _compile_flags() -> list[str]:
    """编译开关：进入 hash，也进入 `phos_build_info()` 便于核对。"""
    return [
        "-O2",
        "-std=c99",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",  # 防 FMA 改变浮点累加顺序（位精确红线）
        "-fno-fast-math",
    ]


def _source_digest() -> str:
    """内容寻址 hash：ABI + zig 版本 + triple + 编译开关 + 全部源文件 sha256。"""
    h = hashlib.sha256()
    h.update(f"abi={ABI_VERSION}\0".encode())
    h.update(f"zig={_zig_version()}\0".encode())
    h.update(f"triple={_triple()}\0".encode())
    for f in sorted(_compile_flags()):
        h.update(f"flag={f}\0".encode())
    for name in CSRC:
        p = csrc_dir() / name
        h.update(f"src={name}\0".encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _artifact_name(digest: str) -> str:
    return f"phos_native-{digest}{_ext()}"


def _load(path: Path) -> ctypes.CDLL:
    """绝对路径 CDLL + ABI 校验；任何异常都转为 NativeUnavailable。"""
    try:
        lib = ctypes.CDLL(str(path.resolve()))
    except OSError as exc:
        raise NativeUnavailable(f"加载 {path.name} 失败: {exc}") from exc
    try:
        got = int(lib.phos_abi_version())
    except AttributeError as exc:
        raise NativeUnavailable(f"{path.name} 缺少 phos_abi_version 符号") from exc
    if got != ABI_VERSION:
        raise NativeUnavailable(f"ABI 不匹配：产物 {got} != 期望 {ABI_VERSION}")
    return lib


def _compile(digest: str, out: Path) -> None:
    if importlib.util.find_spec("ziglang") is None:
        raise NativeUnavailable(
            "ziglang 未安装：请 `pip install 'phosphorus[native]'`（或 pip install ziglang）"
        )
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    cmd = [
        sys.executable,
        "-m",
        "ziglang",
        "cc",
        *_compile_flags(),
        "-target",
        _triple(),
        f"-I{csrc_dir()}",
        "-o",
        str(tmp),
        *[str(csrc_dir() / n) for n in _MAIN_SOURCES],
    ]
    logger.info("phosphorus.native: 编译原生内核 -> %s", out.name)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise NativeUnavailable(f"zig cc 超时（{_COMPILE_TIMEOUT_S}s）") from exc
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise NativeUnavailable(f"zig cc 失败（rc={proc.returncode}）: {proc.stderr[-800:]}")
    os.replace(tmp, out)  # 原子落位；目标文件名唯一，绝不覆盖已加载 DLL
    meta = {
        "abi": ABI_VERSION,
        "zig": _zig_version(),
        "triple": _triple(),
        "flags": _compile_flags(),
        "digest": digest,
        "ts": time.time(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class _FileLock:
    """`O_CREAT|O_EXCL` 自旋锁：拿不到就等，超时返回 False（降级而非阻塞）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> bool:
        deadline = time.time() + _LOCK_TIMEOUT_S
        while time.time() < deadline:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(self.fd, str(os.getpid()).encode())
                return True
            except FileExistsError:
                time.sleep(0.05)
        return False

    def __exit__(self, *exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            with contextlib.suppress(OSError):
                self.path.unlink()


def ensure_library() -> ctypes.CDLL:
    """返回可用的原生库 CDLL；失败抛 `NativeUnavailable`（由上层决定降级）。"""
    digest = _source_digest()
    out = cache_dir() / _artifact_name(digest)

    if out.exists():  # 热路径：不调用 zig，实测 ≈ 2–4 ms
        try:
            return _load(out)
        except NativeUnavailable:
            logger.warning("phosphorus.native: 缓存产物 %s 不可用，重新编译", out.name)

    with _FileLock(cache_dir() / "build.lock") as locked:
        if not locked:
            raise NativeUnavailable("等待 native 编译锁超时（另一进程正在编译）")
        if out.exists():
            try:
                return _load(out)
            except NativeUnavailable:
                out.unlink(missing_ok=True)
        _compile(digest, out)
        return _load(out)


def library_or_none() -> ctypes.CDLL | None:
    """不抛异常版本：拿不到就返回 None（调用方自行降级）。"""
    try:
        return ensure_library()
    except NativeUnavailable as exc:
        logger.debug("phosphorus.native: 不可用（%s）", exc)
        return None
