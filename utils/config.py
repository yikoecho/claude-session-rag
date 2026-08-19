"""
utils/config.py — 环境变量加载 + 常量定义
"""

import os
from pathlib import Path

# 读 /root/.env
_env = Path("/root/.env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "ollama")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://172.18.0.2:11434/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
LANCE_DB_PATH = os.environ.get(
    "LANCE_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_db"),
)
TABLE_NAME = "conversations"
JSONL_DIR = Path(
    os.environ.get("JSONL_DIR", os.path.expanduser("~/.claude/projects/-root"))
)
PORT = int(os.environ.get("SEARCH_PORT", "15200"))

ARCHIVE_FILE = Path(
    os.environ.get("ARCHIVE_FILE", os.path.expanduser("~/.claude/session_archive.md"))
)
JSONL_INDEX_FILE = Path(
    os.environ.get(
        "JSONL_INDEX_FILE", os.path.expanduser("~/.claude/session_index.jsonl")
    )
)

# Recall Agent
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RECALL_MODEL = os.environ.get("RECALL_MODEL", "anthropic/claude-haiku-4.5")
RECALL_ENABLED = bool(OPENROUTER_API_KEY)
