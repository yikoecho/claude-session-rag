#!/usr/bin/env python3
"""
search_server.py — LanceDB semantic search + optional Recall Agent (Haiku filter)

Listens on 127.0.0.1:15200 (configurable via SEARCH_SERVER_PORT)

GET  /search?q=<query>&top_k=3&threshold=0.45  — semantic search
POST /recall                                     — Haiku recall filter (optional)

The /recall endpoint accepts:
  {"prompt": "current user message", "candidates": ["memory line 1", ...]}
Returns:
  {"results": ["filtered line 1", ...]}

If OPENROUTER_API_KEY is not set, /recall passes candidates through unchanged.
"""

import os
import json
import subprocess
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Load .env if present
_env = Path(".env")
if not _env.exists():
    _env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import httpx
import lancedb
from lancedb.index import FTS
import openai

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
LANCEDB_PATH = os.environ.get("LANCEDB_PATH", "./data/lancedb")
TABLE_NAME = "conversations"
PORT = int(os.environ.get("SEARCH_SERVER_PORT", "15200"))

# Optional: path to Claude Code project .jsonl files for grep fallback
# Set JSONL_DIR to enable grep fallback on raw conversation history
JSONL_DIR = os.environ.get("JSONL_DIR", "")

# ── Recall Agent (optional Haiku filter) ──
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RECALL_MODEL = os.environ.get("RECALL_MODEL", "anthropic/claude-haiku-4-5")
RECALL_ENABLED = bool(OPENROUTER_API_KEY)

# ── Initialize on startup ──
print(f"[search_server] Initializing...")
_client = openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
_db = lancedb.connect(LANCEDB_PATH)
_http_client = httpx.Client(timeout=10.0)

try:
    _table = _db.open_table(TABLE_NAME)
    print(f"[search_server] LanceDB connected, rows: {_table.count_rows()}")
    def _build_fts():
        try:
            _table.create_index(config=FTS(), replace=True)
            print("[search_server] FTS index built")
        except Exception as fe:
            print(f"[search_server] Warning: FTS index failed: {fe}")
    threading.Thread(target=_build_fts, daemon=True).start()
except Exception as e:
    _table = None
    print(f"[search_server] Warning: table load failed: {e}")

if RECALL_ENABLED:
    print(f"[search_server] Recall Agent enabled, model: {RECALL_MODEL}")
else:
    print(f"[search_server] Recall Agent disabled (set OPENROUTER_API_KEY to enable)")


# ══════════════════════════════════════════
#  Search
# ══════════════════════════════════════════

def search(query: str, top_k: int = 3, threshold: float = 0.45) -> str:
    if _table is None:
        return ""

    # Vector semantic search
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = resp.data[0].embedding
    vec_results = _table.search(vec).metric("cosine").limit(top_k).to_pandas()

    # FTS keyword search
    try:
        fts_results = _table.search(query, query_type="fts").limit(top_k).to_pandas()
    except Exception:
        fts_results = None

    # Merge results, deduplicate by chunk_id
    merged = {}
    for _, row in vec_results.iterrows():
        score = 1 - row.get("_distance", 1)
        if score < threshold:
            continue
        cid = row["chunk_id"]
        merged[cid] = {"row": row, "score": score}

    if fts_results is not None and not fts_results.empty:
        for _, row in fts_results.iterrows():
            cid = row["chunk_id"]
            if cid not in merged:
                if threshold > 0.5:
                    continue
                merged[cid] = {"row": row, "score": 0.5}

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    lines = []
    for item in ranked:
        row = item["row"]
        date = str(row["timestamp_start"])[:10]
        snippet = row["text"][:120].replace("\n", " ")
        lines.append(f"{date}: {snippet}")

    # Optional: grep fallback on raw .jsonl conversation history
    if JSONL_DIR:
        fallback = _jsonl_fallback(query)
        if fallback:
            lines.append(fallback)

    return "\n".join(lines)


def _extract_content(obj: dict) -> str:
    """Extract readable text from a jsonl conversation entry."""
    msg = obj.get("message", {})
    content = msg.get("content") if msg else obj.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return ""


def _jsonl_fallback(query: str, context_window: int = 3) -> str:
    """
    Grep raw .jsonl conversation history for query.
    Returns earliest hit (with context) + latest hit.
    Only active when JSONL_DIR is set.
    """
    if not JSONL_DIR:
        return ""

    jsonl_dir = Path(JSONL_DIR)
    if not jsonl_dir.exists():
        return ""

    try:
        grep_result = subprocess.run(
            ["grep", "-F", "-r", "-n", "--include=*.jsonl", query, str(jsonl_dir)],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        return ""

    if not grep_result.stdout.strip():
        return ""

    seen_texts = set()
    hits = []
    for line in grep_result.stdout.strip().splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        filepath, lineno_str, raw_json = parts[0], parts[1], parts[2].strip()
        try:
            obj = json.loads(raw_json)
            lineno = int(lineno_str)
        except (json.JSONDecodeError, ValueError):
            continue
        # Skip tool_result entries (injected context, timestamps unreliable)
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list) and any(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in content
        ):
            continue
        text = _extract_content(obj)
        if not text or query not in text:
            continue
        dedup_key = text[:100]
        if dedup_key in seen_texts:
            continue
        seen_texts.add(dedup_key)
        ts = str(obj.get("timestamp", ""))
        hits.append((ts, filepath, lineno, text))

    if not hits:
        return ""

    hits.sort(key=lambda x: x[0])
    oldest = hits[0]
    newest = hits[-1]

    def fmt_line(text: str, ts: str, role: str = "") -> str:
        prefix = f"[{ts[:10]} {role}]" if role else f"[{ts[:10]}]"
        return f"{prefix} {text[:200]}"

    def load_context(filepath: str, lineno: int) -> list[str]:
        try:
            all_lines = Path(filepath).read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        start = max(0, lineno - 1 - context_window)
        end = min(len(all_lines), lineno + context_window)
        msgs = []
        for raw in all_lines[start:end]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = _extract_content(o)
            if not t:
                continue
            role = o.get("message", {}).get("role", "")
            ts = str(o.get("timestamp", ""))
            msgs.append(fmt_line(t, ts, role))
        return msgs

    parts_out = []

    ctx = load_context(oldest[1], oldest[2])
    if ctx:
        parts_out.append("# Earliest occurrence")
        parts_out.extend(ctx)

    if newest[0] != oldest[0] or newest[1] != oldest[1] or newest[2] != oldest[2]:
        ts, _, _, text = newest
        parts_out.append("# Latest occurrence")
        parts_out.append(fmt_line(text, ts))

    if not parts_out:
        return ""

    return "[jsonl_fallback]\n" + "\n".join(parts_out)


# ══════════════════════════════════════════
#  Recall Agent (optional Haiku filter)
# ══════════════════════════════════════════

RECALL_PROMPT_TEMPLATE = """You are a memory relevance filter. Given a list of candidate memory snippets and the current conversation message, select only the ones that are genuinely relevant to the current context.

Rules:
1. Keep only snippets that are truly relevant to the current topic, emotion, or context — max 3
2. "Looks similar" does not mean "should appear" — discard topic-adjacent but unnecessary items
3. If nothing is relevant, return an empty list — do not force relevance
4. Do not fabricate reasoning to make a snippet seem relevant

Current message:
{prompt}

Candidate memories (0-indexed):
{candidates}

Return JSON only, no other text:
{{"keep": [0, 2], "reason": "brief explanation"}}
or
{{"keep": [], "reason": "no relevant memories"}}"""


def recall_agent(prompt: str, candidates: list[str]) -> list[str]:
    """Call Haiku via OpenRouter to filter candidate memories."""
    if not RECALL_ENABLED:
        return candidates  # Pass through if not configured

    if not candidates:
        return []

    formatted = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))

    full_prompt = RECALL_PROMPT_TEMPLATE.format(
        prompt=prompt,
        candidates=formatted,
    )

    try:
        resp = _http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RECALL_MODEL,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 150,
                "temperature": 0,
            },
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        # Parse JSON (tolerant: strip markdown fences)
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        keep_indices = result.get("keep", [])

        if not keep_indices:
            print(f"[recall_agent] No relevant memories ({result.get('reason', '')})")
            return []

        filtered = [candidates[i] for i in keep_indices if i < len(candidates)]
        print(f"[recall_agent] {len(candidates)} → {len(filtered)} ({result.get('reason', '')})")
        return filtered

    except Exception as e:
        print(f"[recall_agent] Failed, passing through: {e}")
        return candidates  # Don't block on failure


# ══════════════════════════════════════════
#  HTTP Handler
# ══════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        top_k = int(params.get("top_k", [3])[0])
        threshold = float(params.get("threshold", [0.45])[0])

        if not query:
            self.send_response(400)
            self.end_headers()
            return

        try:
            result = search(query, top_k, threshold)
        except Exception as e:
            result = ""
            print(f"[search_server] Search error: {e}")

        body = result.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/recall":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(raw)
            prompt = data.get("prompt", "")
            candidates = data.get("candidates", [])
        except (json.JSONDecodeError, AttributeError):
            self.send_response(400)
            self.end_headers()
            return

        filtered = recall_agent(prompt, candidates)

        body = json.dumps({"results": filtered}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[search_server] Listening on 127.0.0.1:{PORT}")
    server.serve_forever()
