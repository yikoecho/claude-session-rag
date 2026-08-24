#!/usr/bin/env python3
"""
search.py — JSONL 语义记忆检索
可独立运行测试，也可作为模块导入到 hook 中。

独立测试:
  python search.py "你的查询内容"

作为模块:
  from search import semantic_search, should_trigger_search
  
  if should_trigger_search(user_message):
      results = semantic_search(user_message, top_k=3)
      # results = [{"text": "...", "score": 0.82, "time": "...", ...}, ...]
"""

import os
import re
import sys

import lancedb
import openai

# 读 /root/.env
from pathlib import Path as _Path
_env_path = _Path("/root/.env")
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ─── 配置 ───
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "ollama")
LANCE_DB_PATH = os.environ.get("LANCE_DB_PATH", "/root/semantic/memory_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://172.18.0.2:11434/v1")
TABLE_NAME = "conversations"

# ─── 回溯词触发 ───
TRIGGER_WORDS = [
    # 时间回溯
    "记得", "以前", "当时", "那时候", "那时", "之前", "过去", "曾经", "从前",
    "上次", "前几天", "有一次", "小时候",
    # 人物/关系
    "前任", "ex", "妈妈", "爸爸", "父母", "家人", "家庭", "家里",
    "童年", "原生家庭", "老师", "同学", "朋友",
    # 情感回溯
    "第一次", "最开始", "一开始", "刚开始",
    "你还记得", "我跟你说过", "我提过", "我之前说",
    "我们聊过", "你知道的",
]

# 编译正则（任意一个词命中即触发）
_trigger_pattern = re.compile("|".join(re.escape(w) for w in TRIGGER_WORDS))


def should_trigger_search(message: str) -> bool:
    """判断用户消息是否包含回溯词，应触发语义检索"""
    return bool(_trigger_pattern.search(message))


def get_embedding(text: str, client: openai.OpenAI) -> list[float]:
    """单条文本 embedding"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def semantic_search(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> list[dict]:
    """
    语义检索，返回最相关的 chunk 列表。

    Returns:
        [{"text": str, "score": float, "timestamp_start": str,
          "timestamp_end": str, "session_id": str, "msg_count": int}, ...]
    """
    client = openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    db = lancedb.connect(LANCE_DB_PATH)

    try:
        table = db.open_table(TABLE_NAME)
    except Exception:
        return []

    query_vec = get_embedding(query, client)

    results = (
        table.search(query_vec)
        .metric("cosine")
        .limit(top_k)
        .to_pandas()
    )

    output = []
    for _, row in results.iterrows():
        score = 1 - row.get("_distance", 1)  # cosine metric：distance = 1 - cosine_sim
        if score < score_threshold:
            continue
        output.append({
            "text": row["text"],
            "score": round(score, 4),
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"],
            "session_id": row["session_id"],
            "msg_count": int(row["msg_count"]),
        })

    return output


def format_context(results: list[dict]) -> str:
    """
    把检索结果格式化为可注入上下文的文本。
    直接塞进 additionalContext 用。
    """
    if not results:
        return ""

    parts = ["<recalled_memories>"]
    for i, r in enumerate(results, 1):
        time_range = r["timestamp_start"][:10]  # 只取日期
        parts.append(
            f"<memory index=\"{i}\" date=\"{time_range}\" relevance=\"{r['score']}\">"
        )
        parts.append(r["text"])
        parts.append("</memory>")
    parts.append("</recalled_memories>")

    return "\n".join(parts)


# ─── 独立运行 / hook 调用 ───
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--raw", action="store_true", help="只输出文本内容，不输出调试信息")
    args = parser.parse_args()

    query = " ".join(args.query)

    if not args.raw:
        print(f"🔍 查询: {query}")
        print(f"   触发词检测: {'✅ 命中' if should_trigger_search(query) else '❌ 未命中'}")
        print()

    results = semantic_search(query, top_k=args.top_k, score_threshold=args.threshold)

    if not results:
        if not args.raw:
            print("没有找到相关记忆")
        sys.exit(0)

    if args.raw:
        # hook 模式：每条结果输出"日期: 内容前120字"
        for r in results:
            date = r["timestamp_start"][:10]
            snippet = r["text"][:120].replace("\n", " ")
            print(f"{date}: {snippet}")
    else:
        for i, r in enumerate(results, 1):
            print(f"━━━ 结果 {i} (相似度: {r['score']}) ━━━")
            print(f"时间: {r['timestamp_start'][:10]}")
            print(f"内容:\n{r['text'][:500]}")
            print()
        print("─── 注入格式 ───")
        print(format_context(results))
