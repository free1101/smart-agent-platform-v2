"""文本分块与中文分词。

中文场景注意:
- 分块按标点/换行切,避免切断语义
- 分词用「单字 + 双字 bigram + 英文词」,过滤高频停用字,bigram 携带更多语义
"""

from __future__ import annotations

import re

# 中文/通用分句标点
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")

# 单词 tokenizer(英文/数字)
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# 高频功能字/词:几乎不携带检索语义,过滤掉降低噪音
STOP_CHARS = set("的了吗呢吧啊呀哦嘛唉哟你我他她它们这那啥怎怎么什么样哪些哪里那儿这")
# 注意:不含「不/能/会/要/没」等有语义的否定/情态字,避免误伤


def split_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """将文本切成块,块之间带少量重叠以保持上下文。"""
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        while len(sent) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if not current:
            current = sent
        elif len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            chunks.append(current)
            current = (current[-overlap:] if overlap else "") + sent

    if current:
        chunks.append(current)
    return chunks


def tokenize(text: str) -> list[str]:
    """统一 tokenizer:中文单字 + 双字 bigram(过滤停用字)+ 英文词。

    供 BM25 打分与关键词精排共用,保证词表一致(可解释)。
    bigram 重复出现一次以加权,增强关键词路的语义区分度。
    """
    tokens: list[str] = []
    for ch in text:
        if ch.strip() and ch not in STOP_CHARS:
            tokens.append(ch)
    for i in range(len(text) - 1):
        bigram = text[i : i + 2]
        if bigram.strip() and not all(c in STOP_CHARS for c in bigram):
            tokens.append(bigram)
            tokens.append(bigram)  # bigram 加权
    tokens += _WORD_RE.findall(text.lower())
    return tokens
