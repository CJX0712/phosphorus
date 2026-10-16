"""row-block 并行切分（T4 C1）。

C 内核是纯函数、可重入（`phos.h` 约束 #1）；`ctypes.CDLL` 的外部调用会**释放 GIL**，
因此 `ThreadPoolExecutor` 在 **N 维行块**粒度切分可获得真实并行。
各线程只写 `y` 的互斥行区间 —— 天然无数据竞争，无需任何锁。

⚠️ 切 K 维会破坏累加完整性（ARCHITECTURE.md §5.1），本模块只切 N 维。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


def default_threads() -> int:
    """并行度：`PHOS_THREADS` 优先，否则 min(8, cpu_count)。"""
    env = os.environ.get("PHOS_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1))


def row_blocks(N: int, n_threads: int) -> list[tuple[int, int]]:
    """把 [0, N) 切成至多 n_threads 个互斥行区间。"""
    if N <= 0:
        return []
    t = max(1, min(int(n_threads), N))
    per = -(-N // t)  # ceil
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < N:
        end = min(N, start + per)
        blocks.append((start, end))
        start = end
    return blocks


def map_rows(fn: Callable[[int, int], None], N: int, n_threads: int | None = None) -> None:
    """对行区间 [0, N) 并行调用 `fn(n0, n1)`。"""
    blocks = row_blocks(N, default_threads() if n_threads is None else n_threads)
    if len(blocks) <= 1:
        for b in blocks:
            fn(*b)
        return
    with ThreadPoolExecutor(max_workers=len(blocks)) as ex:
        list(ex.map(lambda b: fn(*b), blocks))
