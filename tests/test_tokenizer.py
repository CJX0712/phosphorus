"""不变量 ⑤：分词器 encode → decode 严格恒等（任意 UTF-8，含 CJK / emoji / 控制字符）。"""

import pytest

from phosphorus.tokenizer import Tokenizer

_CORPUS = (
    "To be, or not to be, that is the question.\n"
    "Romeo and Juliet met under the balcony at midnight.\n"
    "The quick brown fox jumps over the lazy dog 1234567890 times.\n"
)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(_CORPUS, vocab_size=400)


def test_roundtrip_ascii(tok):
    s = "To be, or not to be, that is the question."
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_whitespace(tok):
    s = "  multiple   spaces   and\nnewlines\tand tabs  "
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_cjk(tok):
    s = "晨星打造世界顶级开源 AI 项目 phosphorus 三值原生推理引擎"
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_emoji(tok):
    s = "hello 🌟 world 🚀 and a fox 🦊 jumps!"
    assert tok.decode(tok.encode(s)) == s


def test_encode_is_deterministic(tok):
    s = "Romeo and Juliet met under the balcony at midnight."
    assert tok.encode(s) == tok.encode(s)


def test_special_tokens(tok):
    assert "<|bos|>" in tok.specials
    assert "<|eos|>" in tok.specials
