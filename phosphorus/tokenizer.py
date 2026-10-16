"""BPE 分词器（复刻 GPT-2 byte-level BPE，零额外依赖）。

借鉴来源：
    * Sennrich et al., 2016, "Neural Machine Translation of Many Languages
      with a Single Encoder-Decoder" (BPE 原始算法) — A-12
    * Radford et al., 2019, GPT-2 `encoder.py` (byte-level 映射，保证任意
      UTF-8 字节无损) — A-13
    * Karpathy/minbpe (开源 BPE 教学实现，本实现取其 merge 优先级策略) — A-13

选型理由：自写而非引入 `regex` / `tokenizers` 库。
    * GPT-2 预分词正则用 `\\p{L}`/`\\p{N}`，Python 内置 `re` 不支持，引入
      `regex` 会污染 core 依赖；本实现用 `unicodedata` 手写等价字符类。
    * byte-level 映射 + 纯字符级 merge，保证 `decode(encode(s)) == s`
      对任意 UTF-8 文本（含 emoji/CJK/控制字符）逐字节严格成立（不变量 ⑤）。
"""

from __future__ import annotations

import heapq
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# byte-level 映射（GPT-2 bytes_to_unicode）                                   #
# --------------------------------------------------------------------------- #


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_ENC = bytes_to_unicode()
_DEC = {v: k for k, v in _ENC.items()}


# --------------------------------------------------------------------------- #
# 预分词（手写等价 GPT-2 正则，避免引入 regex 库）                           #
# --------------------------------------------------------------------------- #


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _pretokenize(text: str) -> list[str]:
    """把文本切成 GPT-2 风格的预分词单元（每个单元是原始字符串）。

    等价正则：'s 't 're 've 'm 'll 'd | ?\\p{L}+ | ?\\p{N}+ |
    ?[^\\s\\p{L}\\p{N}]+ | \\s+(?!\\S) | \\s+
    此处用逐字符状态机手写，不依赖 `regex`。
    """
    words: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == " ":
            # 吃掉前导空格（可能多个），附着到下一个 token 或自成空白 token
            j = i
            while j < n and text[j] == " ":
                j += 1
            # GPT-2: 连续空白最后一个空格独立，其余附着下一个词；简化为整块空白自成一词
            if j == n:
                words.append(text[i:j])
            else:
                words.append(text[i:j])  # 整块空白
            i = j
            continue
        if ch in ("'",) and i + 1 < n and text[i + 1] in "stmvd":
            # 收缩词 's 't 'm 'd
            words.append(text[i : i + 2])
            i += 2
            continue
        if ch == "'" and text[i : i + 3] in ("'re", "'ve", "'ll"):
            words.append(text[i : i + 3])
            i += 3
            continue
        if _is_letter(ch):
            j = i
            while j < n and (_is_letter(text[j]) or text[j] == "'"):
                # 词内允许撇号（如 don't）
                if text[j] == "'" and not (j + 1 < n and text[j + 1] in "stmvd"):
                    break
                j += 1
            words.append(text[i:j])
            i = j
            continue
        if _is_number(ch):
            j = i
            while j < n and _is_number(text[j]):
                j += 1
            words.append(text[i:j])
            i = j
            continue
        # 其他字符：至多一个（标点/符号），GPT-2 用 ` ?[^\s\p{L}\p{N}]+`
        j = i
        while j < n and not _is_letter(text[j]) and not _is_number(text[j]) and text[j] != " ":
            if text[j] == "'":
                break
            j += 1
        if j == i:
            j = i + 1
        words.append(text[i:j])
        i = j
    return words


# --------------------------------------------------------------------------- #
# Tokenizer                                                                   #
# --------------------------------------------------------------------------- #


class Tokenizer:
    """byte-level BPE 分词器。

    vocab: 符号字符串列表，下标即 id。
    merges: 有序 (a, b) 列表，下标即合并 rank。
    specials: 特殊 token 名 -> id（编码时先按特殊 token 切分）。
    """

    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        specials: dict[str, int] | None = None,
    ) -> None:
        self.vocab = list(vocab)
        self.merges = [(a, b) for a, b in merges]
        self.specials = dict(specials or {})
        self.ranks: dict[tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self.sym2id = {s: i for i, s in enumerate(self.vocab)}
        # 特殊 token 在符号表中必须存在
        for name, tid in self.specials.items():
            # 允许特殊 token 追加到末尾（若其 id 越界或符号不匹配）
            if (tid >= len(self.vocab) or self.vocab[tid] != name) and name not in self.sym2id:
                self.sym2id[name] = len(self.vocab)
                self.vocab.append(name)
        self._cache: dict[str, list[int]] = {}

    # ----------------------------- 训练 ----------------------------- #
    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: list[str] | None = None
    ) -> Tokenizer:
        """在 `text` 上训练 BPE，目标词表 `vocab_size`（实际可能更小）。"""
        specials = list(special_tokens or ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"])
        # 初始词表 = 全部 256 个 byte-level 符号（GPT-2 标准）。
        # 必须为「全体字节」而非「训练文本出现过的字节」，否则任意未见 UTF-8
        # 字节（CJK / emoji）在 encode 时会 KeyError，破坏 decode(encode(s))==s 的
        # 无损不变量（⑤）。base vocab 齐备后，任何文本至少能逐字节编解码。
        vocab: list[str] = [_ENC[b] for b in range(256)]
        sym2id = {s: i for i, s in enumerate(vocab)}
        # 预分词并把每个 word 表示为符号列表（按频次去重，只保留唯一词 + 词频）
        word_freq: Counter = Counter()
        for w in _pretokenize(text):
            syms = tuple(_ENC[b] for b in w.encode("utf-8"))
            if syms:
                word_freq[syms] += 1
        words: list[tuple[str, ...]] = list(word_freq.keys())
        # 词频按 word_id 索引（不能用「符号元组 -> 频次」字典：合并后元组会变，
        # 查表会得到 0，导致计数永远无法扣减——这是增量式 BPE 最容易踩的坑）。
        freqs: list[int] = [word_freq[w] for w in words]

        # ---- Sennrich 2016 增量式合并：pair 计数 + 反向索引 + 懒失效最大堆 ----
        # 朴素实现每轮都要重扫全语料（O(merges × 语料长度)），8192 词表下不可行；
        # 这里维护 pair -> count 与 pair -> 受影响词集合，每轮只更新命中该 pair 的词，
        # 并用最大堆取最高频 pair（堆中过期条目在弹出时按当前计数校验后丢弃）。
        pair_counts: Counter = Counter()
        where: dict[tuple[str, str], set[int]] = {}
        for wid, syms in enumerate(words):
            f = freqs[wid]
            for p in zip(syms[:-1], syms[1:]):
                pair_counts[p] += f
                where.setdefault(p, set()).add(wid)
        heap: list[tuple[int, str, str]] = [(-c, a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        target_merges = max(0, vocab_size - len(vocab))
        while len(merges) < target_merges and heap:
            negc, a, b = heapq.heappop(heap)
            pair = (a, b)
            if pair_counts.get(pair, 0) != -negc or pair_counts[pair] <= 0:
                continue  # 堆中过期条目（计数已被后续合并改动）
            merges.append(pair)
            merged_sym = a + b
            if merged_sym not in sym2id:
                sym2id[merged_sym] = len(vocab)
                vocab.append(merged_sym)
            for wid in list(where.pop(pair, ())):
                syms = words[wid]
                f = freqs[wid]
                for q in zip(syms[:-1], syms[1:]):
                    pair_counts[q] -= f
                    s = where.get(q)
                    if s is not None:
                        s.discard(wid)
                new_syms: list[str] = []
                i, n = 0, len(syms)
                while i < n:
                    if i < n - 1 and syms[i] == a and syms[i + 1] == b:
                        new_syms.append(merged_sym)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                new_tuple = tuple(new_syms)
                words[wid] = new_tuple
                for q in zip(new_tuple[:-1], new_tuple[1:]):
                    pair_counts[q] += f
                    where.setdefault(q, set()).add(wid)
                    heapq.heappush(heap, (-pair_counts[q], q[0], q[1]))

        # 特殊 token 追加到词表尾部
        for name in specials:
            if name not in sym2id:
                sym2id[name] = len(vocab)
                vocab.append(name)
        special_map = {name: sym2id[name] for name in specials}
        tok = cls(vocab, merges, special_map)
        return tok

    # ----------------------------- 编码 ----------------------------- #
    def _merge_word(self, word: list[str]) -> list[str]:
        if not word:
            return []
        while True:
            best_idx, best_rank = -1, math.inf
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            word = word[:best_idx] + [word[best_idx] + word[best_idx + 1]] + word[best_idx + 2 :]
        return word

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        if text in self._cache and not add_special:
            return self._cache[text]
        ids: list[int] = []
        # 先按特殊 token 切分
        if self.specials:
            parts = self._split_specials(text)
        else:
            parts = [(False, text)]
        for is_special, seg in parts:
            if is_special:
                ids.append(self.specials[seg])
                continue
            for w in _pretokenize(seg):
                syms = [_ENC[b] for b in w.encode("utf-8")]
                merged = self._merge_word(syms)
                for s in merged:
                    ids.append(self.sym2id[s])
        if add_special and "<|bos|>" in self.specials:
            ids = [self.specials["<|bos|>"]] + ids
        if not add_special:
            self._cache[text] = ids
        return ids

    def _split_specials(self, text: str):
        result = []
        i = 0
        names = sorted(self.specials.keys(), key=len, reverse=True)
        while i < len(text):
            matched = None
            for name in names:
                if text.startswith(name, i):
                    matched = name
                    break
            if matched:
                result.append((True, matched))
                i += len(matched)
            else:
                # 收集到下一个特殊 token 之前
                j = i
                while j < len(text):
                    hit = False
                    for name in names:
                        if text.startswith(name, j):
                            hit = True
                            break
                    if hit:
                        break
                    j += 1
                result.append((False, text[i:j]))
                i = j
        return result

    # ----------------------------- 解码 ----------------------------- #
    def decode(self, ids: list[int]) -> str:
        out_chars = []
        for tid in ids:
            out_chars.append(self.vocab[tid])
        s = "".join(out_chars)
        # byte-level 逆映射 -> 原始字节 -> utf-8
        raw = bytearray()
        for ch in s:
            raw.append(_DEC[ch])
        return raw.decode("utf-8", errors="replace")

    # ----------------------------- 序列化 ----------------------------- #
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        data = {
            "type": "bpe",
            "version": 1,
            "pretokenizer": "gpt2",
            "byte_level": True,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "specials": self.specials,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data["vocab"],
            [tuple(m) for m in data["merges"]],
            data.get("specials"),
        )

    def save_to_bytes(self) -> bytes:
        """序列化分词器为 UTF-8 JSON 字节（用于 .phos tokenizer blob）。"""
        data = {
            "type": "bpe",
            "version": 1,
            "pretokenizer": "gpt2",
            "byte_level": True,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "specials": self.specials,
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @classmethod
    def load_from_bytes(cls, b: bytes) -> Tokenizer:
        data = json.loads(b.decode("utf-8"))
        return cls(data["vocab"], [tuple(m) for m in data["merges"]], data.get("specials"))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def train_from_corpus(
    path: str | Path, vocab_size: int = 8192, limit_chars: int = 1_200_000
) -> Tokenizer:
    """从语料文件训练 BPE。"""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")[:limit_chars]
    return Tokenizer.train(text, vocab_size)
