#!/usr/bin/env python3
"""
recall_keywords.py — Extract search keywords from a user prompt.

Reads a prompt from stdin (up to 300 chars after stripping XML tags).
Outputs two lines:
  Line 1: pipe-separated keywords suitable for a keyword/BM25 search
  Line 2: a retrospective query string (may be empty) for temporal lookups

If QUERY_REWRITE_ENABLED=true, sends the last 3 conversation turns
(read from CONTEXT_FILE if set) to an LLM to produce a condensed search
query, which replaces the jieba-extracted keywords on line 1.

Usage:
    echo "你还记得上次我们聊的那件事吗" | python3 recall_keywords.py

Requires: jieba  (pip install jieba)
Optional: openai  (pip install openai)  — for query rewrite
"""

import os
import re
import sys
import json
from pathlib import Path

# Load .env (repo dir first, then /root/.env for backward compat)
_env_path = Path(__file__).parent.parent / ".env"
if not _env_path.exists():
    _env_path = Path("/root/.env")
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import jieba
jieba.setLogLevel(20)

QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").lower() == "true"
RECALL_API_KEY  = os.environ.get("RECALL_API_KEY", "")
RECALL_BASE_URL = os.environ.get("RECALL_BASE_URL", "https://openrouter.ai/api/v1")
RECALL_MODEL    = os.environ.get("RECALL_MODEL", "anthropic/claude-haiku-4-5")
CONTEXT_FILE    = os.environ.get("CONTEXT_FILE", "")  # optional jsonl of recent turns

STOPWORDS = {
    "好了", "我", "在", "的", "是", "了", "你", "他", "她", "它",
    "这", "那", "就", "都", "也", "还", "啊", "吧", "呢", "吗",
    "嗯", "哦", "然后", "但是", "因为", "所以", "可以", "一个",
    "什么", "怎么", "这个", "那个",
}

# Trigger words that indicate the user is asking about the past,
# mapped to a human-readable retrospective query sent to the memory store.
RETRO_MAP = {
    "记得":   "过去的事",
    "以前":   "以前",
    "当时":   "当时",
    "前任":   "前任",
    "妈妈":   "妈妈 家庭",
    "家庭":   "家庭",
    "童年":   "童年 小时候",
    "小时候": "小时候 童年",
    "爸爸":   "爸爸 家庭",
    "父亲":   "父亲 家庭",
    "母亲":   "母亲 妈妈",
    "上次":   "上次",
    "那时":   "那时候",
}


def _load_recent_context(n: int = 3) -> list[dict]:
    """Load the last n conversation turns from CONTEXT_FILE (jsonl)."""
    if not CONTEXT_FILE:
        return []
    try:
        lines = Path(CONTEXT_FILE).read_text(errors="ignore").strip().splitlines()
        turns = []
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role") or (obj.get("message") or {}).get("role", "")
            content = obj.get("content") or (obj.get("message") or {}).get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if role and content and content.strip():
                turns.append({"role": role, "content": content.strip()[:300]})
                if len(turns) >= n:
                    break
        return list(reversed(turns))
    except Exception:
        return []


def _rewrite_query(prompt: str, context: list[dict]) -> str:
    """Ask Haiku to compress the conversation into a concise search query."""
    if not RECALL_API_KEY:
        return ""
    try:
        import openai
        client = openai.OpenAI(api_key=RECALL_API_KEY, base_url=RECALL_BASE_URL)
        ctx_text = "\n".join(f"{t['role']}: {t['content']}" for t in context)
        system = (
            "You are a search query optimizer. "
            "Given recent conversation context and the latest user message, "
            "output a single concise search query (5-15 words) that captures "
            "what past memories would be most relevant. "
            "Output ONLY the query, no explanation."
        )
        user_msg = f"Recent context:\n{ctx_text}\n\nLatest message: {prompt}"
        resp = client.chat.completions.create(
            model=RECALL_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_msg}],
            max_tokens=60,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────
raw = sys.stdin.read().strip()
text = re.sub(r'<[^>]+>', '', raw).strip()[:300]

words = [w for w in jieba.lcut(text) if w not in STOPWORDS and len(w) >= 2]
keywords = words[:5]

retro_query = ""
for trigger, ob_query in RETRO_MAP.items():
    if trigger in text:
        retro_query = ob_query
        break

if QUERY_REWRITE_ENABLED and RECALL_API_KEY:
    context = _load_recent_context(n=3)
    rewritten = _rewrite_query(text, context)
    if rewritten:
        keywords = [rewritten]

print("|".join(keywords))
print(retro_query)
