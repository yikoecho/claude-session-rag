#!/usr/bin/env python3
"""
build_index.py — session_archive.md + JSONL → LanceDB 语义索引
用法:
  python build_index.py                   # 使用默认路径
  python build_index.py /path/to/archive.md

环境变量（从 /root/.env 读）:
  EMBEDDING_API_KEY=sk-...
  LANCE_DB_PATH=./memory_db        (可选，默认 /root/semantic/memory_db)
  EMBEDDING_MODEL=text-embedding-v3  (可选)
  JSONL_DIR=/root/.claude/projects/-root  (可选，JSONL数据源目录)
"""

import os
import re
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime

# 读 /root/.env
_env_path = Path("/root/.env")
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import lancedb
import openai
import pyarrow as pa
import tiktoken

# ─── 配置 ───
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "ollama")
LANCE_DB_PATH = os.environ.get("LANCE_DB_PATH", "/root/semantic/memory_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://172.18.0.2:11434/v1")
EMBEDDING_DIM = 1024
BATCH_SIZE = 20

ARCHIVE_PATH = "/root/.claude/session_archive.md"
JSONL_DIR = os.environ.get("JSONL_DIR", "/root/.claude/projects/-root")
TABLE_NAME = "conversations"

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_chunks(text: str, session_id: str, timestamp: str) -> list[dict]:
    """把一段文本按 tiktoken 切成 400-token 块，50-token overlap。"""
    tokens = _tokenizer.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_text = _tokenizer.decode(tokens[start:end])
        if len(chunk_text.strip()) >= 20:
            chunk_id = hashlib.md5(f"{session_id}:{start}:{chunk_text[:80]}".encode()).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "session_id": session_id,
                "text": chunk_text,
                "timestamp_start": timestamp,
                "timestamp_end": timestamp,
                "msg_count": 1,
                "uuids": "[]",
            })
        if end == len(tokens):
            break
        start += step
    return chunks


# ─── 解析 session_archive.md ───
def parse_archive(filepath: str) -> list[dict]:
    """
    按 '---' 分段，每段再用 tiktoken 切 400-token 块（50 overlap）。
    """
    text = Path(filepath).read_text(encoding="utf-8")
    raw_sections = re.split(r'\n---\n', text)

    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue
        if section.startswith("# Session Archive"):
            continue

        ts_match = re.search(r'###\s+(\d{4}-\d{2}-\d{2}[^\n]*)', section)
        timestamp = ts_match.group(1).strip() if ts_match else ""

        session_match = re.search(r'##\s+Session[：:]\s*([^\n]+)', section)
        session_id = session_match.group(1).strip() if session_match else "unknown"

        chunks.extend(_token_chunks(section, session_id, timestamp))

    return chunks


# ─── 解析 JSONL 数据源 ───
def _extract_text_from_content(content) -> str:
    """从 message.content 提取纯文本（跳过 tool_use / tool_result）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype in ("tool_use", "tool_result"):
                    continue
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "thinking":
                    continue
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return ""


def parse_jsonl_dir(jsonl_dir: str) -> list[dict]:
    """
    读取目录下所有 .jsonl，提取 role=assistant 的文本内容，
    按 tiktoken 切块后返回 chunk 列表。
    """
    dirpath = Path(jsonl_dir)
    if not dirpath.exists():
        print(f"  ⚠ JSONL目录不存在，跳过: {jsonl_dir}")
        return []

    jsonl_files = sorted(dirpath.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ⚠ 未找到 .jsonl 文件: {jsonl_dir}")
        return []

    print(f"  📄 找到 {len(jsonl_files)} 个 JSONL 文件")
    chunks = []

    for jfile in jsonl_files:
        session_id = f"jsonl:{jfile.stem}"
        assistant_texts = []
        try:
            for line in jfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "assistant":
                    continue
                msg = obj.get("message", {})
                if msg.get("role") != "assistant":
                    continue
                text = _extract_text_from_content(msg.get("content", ""))
                if text and len(text) >= 20:
                    ts = obj.get("timestamp", "")[:10]
                    assistant_texts.append((ts, text))
        except Exception as e:
            print(f"  ⚠ 读取 {jfile.name} 失败: {e}")
            continue

        for ts, text in assistant_texts:
            chunks.extend(_token_chunks(text, session_id, ts))

    print(f"  📦 JSONL解析完成，共 {len(chunks)} 个 chunk")
    return chunks


# ─── Embedding ───
def embed_batch(texts: list[str], client: openai.OpenAI) -> list[list[float]]:
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"  ⚠ Embedding 失败，{wait}s 后重试: {e}")
                time.sleep(wait)
            else:
                raise


def get_processed_chunks(db) -> set:
    try:
        table = db.open_table(TABLE_NAME)
        return set(table.to_arrow().to_pydict()["chunk_id"])
    except Exception:
        return set()


def count_chunks(db) -> int:
    try:
        return db.open_table(TABLE_NAME).count_rows()
    except Exception:
        return 0


def build_index(archive_path: str = ARCHIVE_PATH, jsonl_dir: str = JSONL_DIR):
    if not EMBEDDING_API_KEY:
        print("❌ 请设置 EMBEDDING_API_KEY")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )
    db = lancedb.connect(LANCE_DB_PATH)
    processed_ids = get_processed_chunks(db)

    print(f"📂 数据库: {LANCE_DB_PATH}")
    print(f"📊 已有 {len(processed_ids)} 个 chunk")
    print(f"📖 解析: {archive_path}")

    chunks = parse_archive(archive_path)
    print(f"   session_archive 解析出 {len(chunks)} 个 chunk")

    print(f"📖 JSONL 数据源: {jsonl_dir}")
    jsonl_chunks = parse_jsonl_dir(jsonl_dir)
    chunks.extend(jsonl_chunks)

    print(f"   合计 {len(chunks)} 个 chunk")
    new_chunks = [c for c in chunks if c["chunk_id"] not in processed_ids]
    print(f"   其中 {len(new_chunks)} 个是新的")

    if not new_chunks:
        print("✅ 没有新内容需要处理")
        return

    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("timestamp_start", pa.string()),
        pa.field("timestamp_end", pa.string()),
        pa.field("msg_count", pa.int32()),
        pa.field("uuids", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])

    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🚀 开始 embedding ({total_batches} 批，model={EMBEDDING_MODEL})...")

    for batch_idx in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        vectors = embed_batch([c["text"] for c in batch], client)

        records = [
            {**c, "vector": vec}
            for c, vec in zip(batch, vectors)
        ]

        try:
            table = db.open_table(TABLE_NAME)
            table.add(records)
        except Exception:
            db.create_table(TABLE_NAME, records, schema=schema)

        print(f"   ✅ 批次 {batch_num}/{total_batches} ({len(batch)} chunks)")

        if batch_idx + BATCH_SIZE < len(new_chunks):
            time.sleep(0.3)

    print(f"\n🎉 完成！数据库共 {count_chunks(db)} 个 chunk")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ARCHIVE_PATH
    jdir = sys.argv[2] if len(sys.argv) > 2 else JSONL_DIR
    build_index(path, jdir)
