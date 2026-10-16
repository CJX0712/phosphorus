"""ctypes 绑定 + numpy 友好的 wrapper（T4 C1/C2）。

约定：
  * 只用 `ctypes.CDLL`（cdecl）—— 实测 zig cc 的 `__declspec(dllexport)`
    导出的是默认 cdecl；`PyDLL` 会持有 GIL，破坏 row-block 并行。
  * 所有数组必须 C-contiguous 且 dtype 精确匹配；标量走 `c_float`/`c_int`。
  * `flags` 由本模块统一决定：`PHOS_FLAG_AVX2` 仅在运行时探测通过时置位，
    `PHOS_FLAG_SCALAR` 供测试强制走标量权威路径。
"""

from __future__ import annotations

import ctypes

import numpy as np

from .build import NativeUnavailable, ensure_library

PHOS_OK = 0
PHOS_FLAG_SCALAR = 1
PHOS_FLAG_AVX2 = 2

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_u8_p = ctypes.POINTER(ctypes.c_uint8)
_c_i8_p = ctypes.POINTER(ctypes.c_int8)
_c_i32_p = ctypes.POINTER(ctypes.c_int32)
_c_f32_p = ctypes.POINTER(ctypes.c_float)


class NativeKernels:
    """原生内核的 ctypes 封装（一个实例对应一个已加载的 CDLL）。"""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        self._prototype()

    def _prototype(self) -> None:
        L = self.lib
        L.phos_abi_version.restype = ctypes.c_int
        L.phos_build_info.restype = ctypes.c_char_p
        L.phos_cpu_has_avx2.restype = ctypes.c_int

        L.phos_tl_gemm_i32.restype = ctypes.c_int
        L.phos_tl_gemm_i32.argtypes = [
            _c_u8_p,
            _c_i8_p,
            _c_i32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        L.phos_i2s_gemm_i32.restype = ctypes.c_int
        L.phos_i2s_gemm_i32.argtypes = [
            _c_i8_p,
            _c_u8_p,
            _c_i32_p,
            _c_i32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        L.phos_tl_gemm_f32.restype = ctypes.c_int
        L.phos_tl_gemm_f32.argtypes = [
            _c_u8_p,
            _c_i8_p,
            _c_f32_p,
            ctypes.c_double,
            _c_f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        L.phos_i2s_gemm_f32.restype = ctypes.c_int
        L.phos_i2s_gemm_f32.argtypes = [
            _c_i8_p,
            _c_u8_p,
            _c_i32_p,
            _c_f32_p,
            ctypes.c_double,
            _c_f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        L.phos_pack5.restype = ctypes.c_int
        L.phos_pack5.argtypes = [_c_i8_p, _c_u8_p, ctypes.c_int]
        L.phos_unpack5.restype = ctypes.c_int
        L.phos_unpack5.argtypes = [_c_u8_p, _c_i8_p, ctypes.c_int]

        L.phos_rmsnorm_f32.restype = ctypes.c_int
        L.phos_rmsnorm_f32.argtypes = [
            _c_f32_p,
            _c_f32_p,
            _c_f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_uint,
        ]
        L.phos_softmax_f32.restype = ctypes.c_int
        L.phos_softmax_f32.argtypes = [
            _c_f32_p,
            _c_f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        L.phos_rope_f32.restype = ctypes.c_int
        L.phos_rope_f32.argtypes = [
            _c_f32_p,
            _c_f32_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_uint,
        ]

    # ---- 能力 / 信息 ----------------------------------------------------- #
    def abi_version(self) -> int:
        return int(self.lib.phos_abi_version())

    def build_info(self) -> str:
        return self.lib.phos_build_info().decode("ascii", "replace")

    def cpu_has_avx2(self) -> bool:
        return bool(self.lib.phos_cpu_has_avx2())

    def default_flags(self) -> int:
        return PHOS_FLAG_AVX2 if self.cpu_has_avx2() else 0

    # ---- GEMM ------------------------------------------------------------ #
    def i2s_gemm_i32(
        self,
        W_i8: np.ndarray,
        xu8: np.ndarray,
        row_wsum: np.ndarray,
        n0: int = 0,
        n1: int | None = None,
        flags: int | None = None,
    ) -> np.ndarray:
        """P0 主路径：解包 int8 权重 × uint8 激活 − 128·row_wsum，int32 累加。"""
        W = np.ascontiguousarray(W_i8, dtype=np.int8)
        X = np.ascontiguousarray(xu8, dtype=np.uint8)
        R = np.ascontiguousarray(row_wsum, dtype=np.int32)
        N, K = W.shape
        n1 = N if n1 is None else n1
        out = np.zeros(N, dtype=np.int32)
        rc = self.lib.phos_i2s_gemm_i32(
            W.ctypes.data_as(_c_i8_p),
            X.ctypes.data_as(_c_u8_p),
            R.ctypes.data_as(_c_i32_p),
            out.ctypes.data_as(_c_i32_p),
            N,
            K,
            n0,
            n1,
            self._flags(flags),
        )
        _check(rc, "phos_i2s_gemm_i32")
        return out

    def i2s_gemm_f32(
        self,
        W_i8: np.ndarray,
        xu8: np.ndarray,
        row_wsum: np.ndarray,
        w_scale: np.ndarray,
        sx: float,
        n0: int = 0,
        n1: int | None = None,
        flags: int | None = None,
    ) -> np.ndarray:
        """I2_S 生产路径：fp32 输出 = (acc · w_scale) · sx（double 域左结合）。"""
        W = np.ascontiguousarray(W_i8, dtype=np.int8)
        X = np.ascontiguousarray(xu8, dtype=np.uint8)
        R = np.ascontiguousarray(row_wsum, dtype=np.int32)
        S = np.ascontiguousarray(w_scale, dtype=np.float32)
        N, K = W.shape
        n1 = N if n1 is None else n1
        out = np.zeros(N, dtype=np.float32)
        rc = self.lib.phos_i2s_gemm_f32(
            W.ctypes.data_as(_c_i8_p),
            X.ctypes.data_as(_c_u8_p),
            R.ctypes.data_as(_c_i32_p),
            S.ctypes.data_as(_c_f32_p),
            ctypes.c_double(sx),
            out.ctypes.data_as(_c_f32_p),
            N,
            K,
            n0,
            n1,
            self._flags(flags),
        )
        _check(rc, "phos_i2s_gemm_f32")
        return out

    def tl_gemm_i32(
        self,
        packed: np.ndarray,
        xq_pad: np.ndarray,
        N: int,
        K_pad: int,
        n0: int = 0,
        n1: int | None = None,
        flags: int | None = None,
    ) -> np.ndarray:
        """P1 路径：直接吃 5-bit 打包流的 TL 内核（内存受限场景）。"""
        P = np.ascontiguousarray(packed, dtype=np.uint8)
        X = np.ascontiguousarray(xq_pad, dtype=np.int8)
        n1 = N if n1 is None else n1
        out = np.zeros(N, dtype=np.int32)
        rc = self.lib.phos_tl_gemm_i32(
            P.ctypes.data_as(_c_u8_p),
            X.ctypes.data_as(_c_i8_p),
            out.ctypes.data_as(_c_i32_p),
            N,
            K_pad,
            n0,
            n1,
            self._flags(flags),
        )
        _check(rc, "phos_tl_gemm_i32")
        return out

    def tl_gemm_f32(
        self,
        packed: np.ndarray,
        xq_pad: np.ndarray,
        w_scale: np.ndarray,
        sx: float,
        N: int,
        K_pad: int,
        n0: int = 0,
        n1: int | None = None,
        flags: int | None = None,
    ) -> np.ndarray:
        P = np.ascontiguousarray(packed, dtype=np.uint8)
        X = np.ascontiguousarray(xq_pad, dtype=np.int8)
        S = np.ascontiguousarray(w_scale, dtype=np.float32)
        n1 = N if n1 is None else n1
        out = np.zeros(N, dtype=np.float32)
        rc = self.lib.phos_tl_gemm_f32(
            P.ctypes.data_as(_c_u8_p),
            X.ctypes.data_as(_c_i8_p),
            S.ctypes.data_as(_c_f32_p),
            ctypes.c_double(sx),
            out.ctypes.data_as(_c_f32_p),
            N,
            K_pad,
            n0,
            n1,
            self._flags(flags),
        )
        _check(rc, "phos_tl_gemm_f32")
        return out

    # ---- pack / unpack ---------------------------------------------------- #
    def pack5(self, w: np.ndarray) -> np.ndarray:
        """打包 n_groups 组三值权重（w 长度必须是 3 的倍数）。"""
        W = np.ascontiguousarray(w, dtype=np.int8)
        if W.size % 3 != 0:
            raise ValueError(f"pack5 需要长度是 3 的倍数，得到 {W.size}")
        n_groups = W.size // 3
        nbytes = (5 * n_groups + 7) // 8
        out = np.zeros(nbytes, dtype=np.uint8)
        rc = self.lib.phos_pack5(W.ctypes.data_as(_c_i8_p), out.ctypes.data_as(_c_u8_p), n_groups)
        _check(rc, "phos_pack5")
        return out

    def unpack5(self, packed: np.ndarray, n_groups: int) -> np.ndarray:
        P = np.ascontiguousarray(packed, dtype=np.uint8)
        out = np.zeros(3 * n_groups, dtype=np.int8)
        rc = self.lib.phos_unpack5(P.ctypes.data_as(_c_u8_p), out.ctypes.data_as(_c_i8_p), n_groups)
        _check(rc, "phos_unpack5")
        return out

    # ---- ops -------------------------------------------------------------- #
    def rmsnorm(
        self, x: np.ndarray, w: np.ndarray, eps: float, flags: int | None = None
    ) -> np.ndarray:
        X = np.ascontiguousarray(x, dtype=np.float32)
        W = np.ascontiguousarray(w, dtype=np.float32)
        out = np.empty_like(X)
        rc = self.lib.phos_rmsnorm_f32(
            X.ctypes.data_as(_c_f32_p),
            W.ctypes.data_as(_c_f32_p),
            out.ctypes.data_as(_c_f32_p),
            X.shape[0],
            X.shape[1],
            ctypes.c_float(eps),
            self._flags(flags),
        )
        _check(rc, "phos_rmsnorm_f32")
        return out

    def softmax(self, x: np.ndarray, flags: int | None = None) -> np.ndarray:
        X = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(X)
        rc = self.lib.phos_softmax_f32(
            X.ctypes.data_as(_c_f32_p),
            out.ctypes.data_as(_c_f32_p),
            X.shape[0],
            X.shape[1],
            self._flags(flags),
        )
        _check(rc, "phos_softmax_f32")
        return out

    def rope(self, x: np.ndarray, pos0: int, theta: float, flags: int | None = None) -> np.ndarray:
        """x/y: [T, H, Dh] float32 C-contiguous。"""
        X = np.ascontiguousarray(x, dtype=np.float32)
        T, H, Dh = X.shape
        out = np.empty_like(X)
        rc = self.lib.phos_rope_f32(
            X.ctypes.data_as(_c_f32_p),
            out.ctypes.data_as(_c_f32_p),
            T,
            H,
            Dh,
            pos0,
            ctypes.c_float(theta),
            self._flags(flags),
        )
        _check(rc, "phos_rope_f32")
        return out

    def _flags(self, flags: int | None) -> int:
        return self.default_flags() if flags is None else int(flags)


def _check(rc: int, what: str) -> None:
    if rc != PHOS_OK:
        raise RuntimeError(f"{what} 返回错误码 {rc}")


_LIB: ctypes.CDLL | None = None
_KERNELS: NativeKernels | None = None
_REASON = ""


def kernels() -> NativeKernels:
    """进程内单例：首次调用时编译/加载；不可用抛 `NativeUnavailable`。"""
    global _LIB, _KERNELS, _REASON
    if _KERNELS is not None:
        return _KERNELS
    try:
        _LIB = ensure_library()
        _KERNELS = NativeKernels(_LIB)
    except NativeUnavailable as exc:
        _REASON = str(exc)
        raise
    return _KERNELS


def kernels_or_none() -> NativeKernels | None:
    try:
        return kernels()
    except NativeUnavailable:
        return None


def unavailable_reason() -> str:
    return _REASON
