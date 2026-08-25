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

_bm25_chunks: list[tuple[str, str]] = []  # (text, source) where source = "archive" | "entity"
_bm25_index: BM25Okapi | None = None
_bm25_lock = threading.Lock()


_CHINESE_RE = re.compile(r"[一-鿿㐀-䶿]+")

# 中文停用词（高频虚词，IDF接近零，不应参与BM25计分）
_STOPWORDS = frozenset({
    "的", "了", "在", "是", "有", "和", "就", "不", "都", "也", "很", "着",
    "一", "这", "那", "与", "但", "而", "或", "从", "以", "及", "对", "为",
    "我", "你", "他", "她", "它", "们", "我们", "你们", "他们", "她们",
    "这个", "那个", "这些", "那些", "什么", "怎么", "哪", "哪个", "哪些",
    "一个", "一些", "一下", "一起", "一直", "一样", "一边", "一点",
    "然后", "所以", "因为", "但是", "不过", "如果", "虽然", "还是", "已经",
    "可以", "没有", "这样", "那样", "现在", "之后", "之前", "上面", "下面",
    "过来", "过去", "出来", "进来", "回来", "去", "来", "过", "到",
    "说", "做", "看", "用", "把", "让", "被", "比", "给", "向",
    "时候", "东西", "问题", "事情", "地方", "方面", "情况", "时间",
    "自己", "大家", "别人", "其他", "其实", "可能", "应该", "需要",
})


def _tokenize(text: str) -> list[str]:
    """混合分词：中文用 jieba，英文/数字用正则。过滤停用词。"""
    tokens: list[str] = []
    # 先用 jieba 切中文词，过滤停用词
    for word in jieba.cut(text):
        word = word.strip()
        if word and _CHINESE_RE.fullmatch(word) and word not in _STOPWORDS:
            tokens.append(word)
    # 再用正则提取非中文 token（英文单词、数字等）
    non_chinese = _CHINESE_RE.sub(" ", text.lower())
    tokens.extend(re.findall(r"[a-z0-9_]+", non_chinese))
    return tokens


def build_bm25_index() -> None:
    global _bm25_chunks, _bm25_index
    chunks: list[tuple[str, str]] = []  # (text, source)

    # session_archive.md — 按 ### 开头切块
    if ARCHIVE_FILE.exists():
        text = ARCHIVE_FILE.read_text(encoding="utf-8")
        blocks = re.split(r"(?=^### )", text, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if block:
                chunks.append((block, "archive"))

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
                entities = obj.get("entities", {})
                entity_terms = []
                if isinstance(entities, dict):
                    for v in entities.values():
                        if isinstance(v, list):
                            entity_terms.extend([str(x) for x in v])
                elif isinstance(entities, list):
                    entity_terms.extend([str(x) for x in entities])
                # aliases 也进 BM25（"你以后会怎么称呼这件事"的自然说法）
                aliases = obj.get("aliases", [])
                alias_str = " ".join(str(a) for a in aliases) if aliases else ""
                entity_str = " ".join(entity_terms)
                chunks.append((f"[jsonl] {date} {key}：{txt} {entity_str} {alias_str}".strip(), "entity"))
            except json.JSONDecodeError:
                continue

    if not chunks:
        print("[bm25] 没有可索引的内容，跳过")
        return

    tokenized = [_tokenize(text) for text, _ in chunks]
    with _bm25_lock:
        _bm25_chunks = chunks
        _bm25_index = BM25Okapi(tokenized)

    archive_count = sum(1 for _, s in chunks if s == "archive")
    entity_count = sum(1 for _, s in chunks if s == "entity")
    print(f"[bm25] 索引已建立，共 {len(chunks)} 个 chunk (archive={archive_count}, entity={entity_count})")


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
            text, _ = _bm25_chunks[i]
            snippet = text[:200].replace("\n", " ")
            lines.append(snippet)
        return "\n".join(lines)


def bm25_search_raw(query: str, top_k: int = 20) -> list[dict]:
    """返回 BM25 排序后的 chunk 列表（已过滤 score<=0），每条包含 text、source、score。"""
    with _bm25_lock:
        if _bm25_index is None or not _bm25_chunks:
            return []
        scores = _bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in ranked:
            if scores[i] <= 0:
                break
            text, source = _bm25_chunks[i]
            results.append({"text": text, "source": source, "score": float(scores[i])})
        return results


def reload_bm25_async() -> None:
    threading.Thread(target=build_bm25_index, daemon=True).start()
