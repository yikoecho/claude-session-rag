"""
index/vector.py — LanceDB 连接 + 向量搜索
"""

import threading

import lancedb
from lancedb.index import FTS
import openai

from utils.config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    LANCE_DB_PATH, TABLE_NAME,
)

print(f"[search_server] 初始化 client & db...")
_client = openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
_db = lancedb.connect(LANCE_DB_PATH)

try:
    _table = _db.open_table(TABLE_NAME)
    print(f"[search_server] LanceDB 已连接，表行数: {_table.count_rows()}")

    def _build_fts():
        try:
            _table.create_index("text", config=FTS(), replace=True)
            print("[search_server] FTS 索引已建立")
        except Exception as fe:
            print(f"[search_server] ⚠ FTS 索引建立失败: {fe}")

    threading.Thread(target=_build_fts, daemon=True).start()
except Exception as e:
    _table = None
    print(f"[search_server] ⚠ 表加载失败: {e}")


def get_table():
    return _table


def get_client():
    return _client


def vector_search_raw(query: str, top_k: int = 20, threshold: float = 0.3) -> list[dict]:
    """返回向量搜索结果，每项含 text 和 vector（date: snippet 格式）。"""
    if _table is None:
        return []
    try:
        resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vec = resp.data[0].embedding
        rows = _table.search(vec).metric("cosine").limit(top_k).select(["text", "vector", "timestamp_start"]).to_pandas()
    except Exception as e:
        print(f"[hybrid] 向量搜索失败: {e}")
        return []
    results = []
    for _, row in rows.iterrows():
        score = 1 - row.get("_distance", 1)
        if score < threshold:
            continue
        date = str(row["timestamp_start"])[:10]
        text_str = f"{date}: {str(row['text'])[:200].replace(chr(10), ' ')}"
        vec_val = list(row["vector"]) if row["vector"] is not None else None
        results.append({"text": text_str, "vector": vec_val, "score": round(score, 6)})
    return results
