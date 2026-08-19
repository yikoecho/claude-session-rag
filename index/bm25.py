"""
index/bm25.py — BM25Okapi 索引构建与查询
"""

import json
import re
import threading
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from utils.config import ARCHIVE_FILE, JSONL_INDEX_FILE

# 加载 jieba 时静默
jieba.setLogLevel("CRITICAL")

_bm25_chunks: list[str] = []
_bm25_index: BM25Okapi | None = None
_bm25_lock = threading.Lock()


_CHINESE_RE = re.compile(r"[一-鿿㐀-䶿]+")


def _tokenize(text: str) -> list[str]:
    """混合分词：中文用 jieba，英文/数字用正则。"""
    tokens: list[str] = []
    # 先用 jieba 切中文词
    for word in jieba.cut(text):
        word = word.strip()
        if word and _CHINESE_RE.fullmatch(word):
            tokens.append(word)
    # 再用正则提取非中文 token（英文单词、数字等）
    non_chinese = _CHINESE_RE.sub(" ", text.lower())
    tokens.extend(re.findall(r"[a-z0-9_]+", non_chinese))
    return tokens


def build_bm25_index() -> None:
    global _bm25_chunks, _bm25_index
    chunks: list[str] = []

    # session_archive.md — 按 ### 开头切块
    if ARCHIVE_FILE.exists():
        text = ARCHIVE_FILE.read_text(encoding="utf-8")
        blocks = re.split(r"(?=^### )", text, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if block:
                chunks.append(block)

    # session_index.jsonl — key + text 字段拼一行
    if JSONL_INDEX_FILE.exists():
        for line in JSONL_INDEX_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                date = obj.get("date", "")
                key = obj.get("key", "")
                txt = obj.get("text", "")
                chunks.append(f"[jsonl] {date} {key}：{txt}")
            except json.JSONDecodeError:
                continue

    if not chunks:
        print("[bm25] 没有可索引的内容，跳过")
        return

    tokenized = [_tokenize(c) for c in chunks]
    with _bm25_lock:
        _bm25_chunks = chunks
        _bm25_index = BM25Okapi(tokenized)

    print(f"[bm25] 索引已建立，共 {len(chunks)} 个 chunk")


def bm25_search(query: str, top_k: int = 5) -> str:
    with _bm25_lock:
        if _bm25_index is None or not _bm25_chunks:
            return ""
        scores = _bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        lines = []
        for i in ranked:
            if scores[i] <= 0:
                break
            snippet = _bm25_chunks[i][:200].replace("\n", " ")
            lines.append(snippet)
        return "\n".join(lines)


def bm25_search_raw(query: str, top_k: int = 20) -> list[str]:
    """返回 BM25 排序后的 chunk 文本列表（已过滤 score<=0）。"""
    with _bm25_lock:
        if _bm25_index is None or not _bm25_chunks:
            return []
        scores = _bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [_bm25_chunks[i] for i in ranked if scores[i] > 0]


def reload_bm25_async() -> None:
    threading.Thread(target=build_bm25_index, daemon=True).start()
