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
            _table.create_index(config=FTS(), replace=True)
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


def vector_search_raw(query: str, top_k: int = 20, threshold: float = 0.3) -> list[str]:
    """返回向量搜索排序后的文本列表（date: snippet 格式）。"""
    if _table is None:
        return []
    try:
        resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vec = resp.data[0].embedding
        rows = _table.search(vec).metric("cosine").limit(top_k).to_pandas()
    except Exception as e:
        print(f"[hybrid] 向量搜索失败: {e}")
        return []
    results = []
    for _, row in rows.iterrows():
        score = 1 - row.get("_distance", 1)
        if score < threshold:
            continue
        date = str(row["timestamp_start"])[:10]
        text = row["text"][:200].replace("\n", " ")
        results.append(f"{date}: {text}")
    return results
