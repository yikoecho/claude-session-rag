"""
utils/config.py — 环境变量加载 + 常量定义
"""

import os
from pathlib import Path

# Load .env (repo dir first, then legacy /root/.env for backward compat)
_env = Path(__file__).resolve().parent.parent / ".env"
if not _env.exists():
    _env = Path("/root/.env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "ollama")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
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

# ---- LLM backend (recall filter + entity enrichment) ----
# LLM_BACKEND: "none" | "ollama" | "siliconflow" | "api" | "openrouter"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# 向后兼容：老变量 RECALL_API_KEY / RECALL_BASE_URL 仍可用
_legacy_api_key = os.environ.get("RECALL_API_KEY", "")
_legacy_base_url = os.environ.get("RECALL_BASE_URL", "")

# 自动推断默认后端：有 OpenRouter key 就用它，否则关闭
_default_backend = "openrouter" if (OPENROUTER_API_KEY or _legacy_api_key) else "none"
LLM_BACKEND = os.environ.get("LLM_BACKEND", _default_backend).lower()

# 各后端的 endpoint / key / 默认模型
if LLM_BACKEND == "ollama":
    LLM_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    LLM_API_KEY = "ollama"
    RECALL_MODEL = os.environ.get("RECALL_MODEL", "qwen2.5:3b")
    ENRICH_MODEL = os.environ.get("ENRICH_MODEL", "qwen2.5:3b")
elif LLM_BACKEND == "siliconflow":
    LLM_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("EMBEDDING_API_KEY", "")
    RECALL_MODEL = os.environ.get("RECALL_MODEL", "Qwen/Qwen3-8B")
    ENRICH_MODEL = os.environ.get("ENRICH_MODEL", "Qwen/Qwen3-8B")
else:  # api / openrouter（none 时这些值用不到）
    LLM_BASE_URL = _legacy_base_url or "https://openrouter.ai/api/v1"
    LLM_API_KEY = OPENROUTER_API_KEY or _legacy_api_key
    RECALL_MODEL = os.environ.get("RECALL_MODEL", "anthropic/claude-haiku-4-5")
    ENRICH_MODEL = os.environ.get("ENRICH_MODEL", "anthropic/claude-haiku-4-5")

RECALL_ENABLED = LLM_BACKEND != "none"
