"""混合检索器单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.text_splitter import split_text, tokenize


def _sample_docs() -> list[dict]:
    return [
        {"id": 1, "content": "我们推拿房的营业时间是每天上午9点到晚上10点。", "category": "营业时间", "keywords": ["营业时间", "开门", "关门"], "source": "t"},
        {"id": 2, "content": "全身推拿120元/60分钟,肩颈推拿80元/30分钟。", "category": "服务项目", "keywords": ["价格", "全身推拿", "肩颈推拿", "多少钱"], "source": "t"},
        {"id": 3, "content": "高峰期建议至少提前一天预约,迟到15分钟视为取消。", "category": "预约政策", "keywords": ["预约", "迟到", "取消", "提前"], "source": "t"},
        {"id": 4, "content": "我们提供免费WiFi、茶水,以及全套一次性用品。", "category": "常见问题", "keywords": ["wifi", "茶水", "用品", "自带"], "source": "t"},
    ]


def test_split_text():
    # 长文本(超过 chunk_size)应切出多块,且块间保留重叠
    text = "今天天气真好。适合出去走走。" * 20  # 约 300 字符
    chunks = split_text(text, chunk_size=60, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c) > 0 for c in chunks)
    assert all(len(c) <= 70 for c in chunks)  # chunk_size + overlap


def test_tokenize():
    tokens = tokenize("全身推拿多少钱")
    # 包含单字与 bigram
    assert "全身" in tokens
    assert "推拿" in tokens
    assert "钱" in tokens


def test_hybrid_search_top1():
    r = HybridRetriever()
    r.build(_sample_docs())
    hits = r.search("全身推拿多少钱", top_k=3)
    assert hits, "应有检索结果"
    assert hits[0].doc_id == 2, "价格问题应命中服务项目"


def test_hybrid_search_keyword_hit():
    r = HybridRetriever()
    r.build(_sample_docs())
    hits = r.search("迟到了会取消吗", top_k=3)
    assert hits and hits[0].doc_id == 3, "迟到应命中预约政策"


def test_hybrid_search_empty():
    r = HybridRetriever()
    hits = r.search("随便问问", top_k=3)
    assert hits == []


def test_reranker():
    r = HybridRetriever()
    r.build(_sample_docs())
    hits = r.search("营业时间", top_k=4)
    rr = Reranker()
    ranked = rr.rerank("营业时间", hits)
    assert ranked[0].doc_id == 1
