"""数据加载：自回归批采样形状正确、确定性、无越界。"""

import numpy as np
import pytest

from phosphorus.data import TextDataset  # noqa: E402
from phosphorus.tokenizer import Tokenizer  # noqa: E402


@pytest.fixture
def dataset():
    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = Tokenizer.train(text, vocab_size=400)
    ids = np.array(tok.encode(text), dtype=np.int16)
    return TextDataset(tok, ids, block_size=32)


def test_batch_shape(dataset):
    rng = np.random.default_rng(0)
    x, y = dataset.batch(8, rng)
    assert x.shape == (8, 32) and y.shape == (8, 32)
    # 目标必须为 int64：PyTorch F.cross_entropy 要求 Long 目标（不变量：训练可跑通）
    assert x.dtype == np.int64 and y.dtype == np.int64
    # y 是 x 的右移一位
    assert np.array_equal(x[:, 1:], y[:, :-1])


def test_val_batch_deterministic(dataset):
    x1, y1 = dataset.val_batch(4)
    x2, y2 = dataset.val_batch(4)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_len_nonnegative(dataset):
    assert len(dataset) >= 0
    assert len(dataset) == max(0, len(dataset.tokens) - dataset.block_size)
