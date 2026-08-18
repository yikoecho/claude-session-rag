#!/usr/bin/env python3
"""
build_index.py — session_archive.md → LanceDB semantic index

Usage:
  python build_index.py                   # use defaults from env
  python build_index.py /path/to/archive.md

Environment variables (can be set in .env file):
  EMBEDDING_API_KEY     API key for embedding service
  EMBEDDING_BASE_URL    Base URL for embedding API (default: SiliconFlow)
  EMBEDDING_MODEL       Embedding model name (default: BAAI/bge-m3)
  LANCEDB_PATH          Path to LanceDB storage (default: ./data/lancedb)
  SESSION_ARCHIVE_PATH  Path to session archive file
"""

import os
import re
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime

# Load .env if present
_env_path = Path(".env")
if not _env_path.exists():
    _env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import lancedb
import openai
import pyarrow as pa

# ─── Configuration ───
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "./data/lancedb")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_DIM = 1024
BATCH_SIZE = 20

SESSION_ARCHIVE_PATH = os.environ.get("SESSION_ARCHIVE_PATH", "./data/session_archive.md")
TABLE_NAME = "conversations"


# ─── Parse session_archive.md ───
def parse_archive(filepath: str) -> list[dict]:
    """
    Split by '---' separators, extract each section as a chunk.
    Extract ### timestamp lines as metadata.
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

        # Extract timestamp (### 2026-xx-xx HH:MM format)
        ts_match = re.search(r'###\s+(\d{4}-\d{2}-\d{2}[^\n]*)', section)
        timestamp = ts_match.group(1).strip() if ts_match else ""

        # Extract session range (## Session: ... format)
        session_match = re.search(r'##\s+Session[：:]\s*([^\n]+)', section)
        session_id = session_match.group(1).strip() if session_match else "unknown"

        chunk_id = hashlib.md5(section.encode()).hexdigest()

        # Truncate overly long sections
        content = section if len(section) <= 2000 else section[:2000]

        chunks.append({
            "chunk_id": chunk_id,
            "session_id": session_id,
            "text": content,
            "timestamp_start": timestamp,
            "timestamp_end": timestamp,
            "msg_count": 1,
            "uuids": "[]",
        })

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
                print(f"  Warning: embedding failed, retrying in {wait}s: {e}")
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


def build_index(archive_path: str = SESSION_ARCHIVE_PATH):
    if not EMBEDDING_API_KEY:
        print("Error: EMBEDDING_API_KEY is not set")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )
    db = lancedb.connect(LANCEDB_PATH)
    processed_ids = get_processed_chunks(db)

    print(f"Database: {LANCEDB_PATH}")
    print(f"Existing chunks: {len(processed_ids)}")
    print(f"Archive: {archive_path}")

    chunks = parse_archive(archive_path)
    print(f"Parsed {len(chunks)} chunks")

    new_chunks = [c for c in chunks if c["chunk_id"] not in processed_ids]
    print(f"New chunks to index: {len(new_chunks)}")

    if not new_chunks:
        print("Nothing to do — index is up to date.")
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
    print(f"\nStarting embedding ({total_batches} batches, model={EMBEDDING_MODEL})...")

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

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks) done")

        if batch_idx + BATCH_SIZE < len(new_chunks):
            time.sleep(0.3)

    print(f"\nDone. Database now has {count_chunks(db)} chunks.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else SESSION_ARCHIVE_PATH
    build_index(path)
