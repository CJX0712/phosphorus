"""数据加载与批处理采样。

借鉴来源：
    * GPT/LLaMA 风格的 next-token 自回归切块（Karpathy, nanoGPT）— A-16
    * 不引入额外框架；class 只依赖 numpy + 本地分词器。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer


class TextDataset:
    """把语料 token 化并缓存为 .bin，提供自回归批采样。

    复用约定：token 以 int16 存储（vocab ≤ 32767 足够）。缓存文件名由文本
    路径 + vocab_size 派生，避免重复 token 化。
    """

    def __init__(self, tokenizer: Tokenizer, tokens: np.ndarray, block_size: int) -> None:
        self.tok = tokenizer
        self.tokens = tokens.astype(np.int16)
        self.block_size = block_size

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer: Tokenizer,
        block_size: int = 256,
        cache_dir: str | Path | None = None,
        text_limit: int = 1_200_000,
    ) -> TextDataset:
        corpus_path = Path(corpus_path)
        cache_dir = Path(cache_dir or Path(__file__).resolve().parents[1] / "data" / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{corpus_path.stem}.vocab{tokenizer.vocab_size}.bin"
        if cache_path.exists():
            tokens = np.frombuffer(cache_path.read_bytes(), dtype=np.int16)
        else:
            text = corpus_path.read_text(encoding="utf-8", errors="ignore")[:text_limit]
            tokens = np.array(tokenizer.encode(text), dtype=np.int16)
            cache_path.write_bytes(tokens.tobytes())
        return cls(tokenizer, tokens, block_size)

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """随机采 batch_size 个 (x, y) 块，形状 [B, T]。"""
        idx = rng.integers(0, len(self), size=batch_size)
        x = np.stack([self.tokens[i : i + self.block_size] for i in idx])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in idx])
        return x.astype(np.int64), y.astype(np.int64)

    def val_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """取尾部 n 个块作验证集（确定性）。"""
        start = max(0, len(self) - n * self.block_size)
        idx = np.arange(start, start + n * self.block_size, self.block_size)
        x = np.stack([self.tokens[i : i + self.block_size] for i in idx])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in idx])
        return x.astype(np.int64), y.astype(np.int64)
