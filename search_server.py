#!/usr/bin/env python3
"""
search_server.py — LanceDB 语义搜索 + Recall Agent 常驻服务
监听 127.0.0.1:15200

GET  /search?q=<query>&top_k=3&threshold=0.45  → 原有语义搜索
POST /recall                                     → Recall Agent（Haiku 筛选）

Recall Agent 接收候选记忆 + 当前对话，调用 OpenRouter Haiku
判断哪些真正相关，返回筛选后的结果或空。
"""

import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# 读 /root/.env
_env = Path("/root/.env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import json
import re
import subprocess
import threading
import httpx
import lancedb
from lancedb.index import FTS
import openai
from rank_bm25 import BM25Okapi

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "ollama")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://172.18.0.2:11434/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
LANCE_DB_PATH = os.environ.get("LANCE_DB_PATH", "/root/semantic/memory_db")
TABLE_NAME = "conversations"
JSONL_DIR = Path("/root/.claude/projects/-root")
PORT = 15200

# ── Recall Agent 配置 ──
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
RECALL_MODEL = os.environ.get("RECALL_MODEL", "anthropic/claude-haiku-4.5")
RECALL_ENABLED = bool(OPENROUTER_API_KEY)

# ── 启动时初始化，之后复用 ──
print(f"[search_server] 初始化 client & db...")
_client = openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
_db = lancedb.connect(LANCE_DB_PATH)
_http_client = httpx.Client(timeout=10.0)

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

if RECALL_ENABLED:
    print(f"[search_server] Recall Agent 已启用，模型: {RECALL_MODEL}")
else:
    print(f"[search_server] ⚠ Recall Agent 未启用（缺少 OPENROUTER_API_KEY）")


# ══════════════════════════════════════════
#  BM25 索引（session_archive.md + session_index.jsonl）
# ══════════════════════════════════════════

ARCHIVE_FILE = Path("/root/.claude/session_archive.md")
JSONL_INDEX_FILE = Path("/root/.claude/session_index.jsonl")

_bm25_chunks: list[str] = []
_bm25_index: BM25Okapi | None = None
_bm25_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w一-鿿]+", text.lower())


def _build_bm25_index() -> None:
    global _bm25_chunks, _bm25_index
    chunks: list[str] = []

    # session_archive.md — 按 ### 开头切块
    if ARCHIVE_FILE.exists():
        text = ARCHIVE_FILE.read_text(encoding="utf-8")
        blocks = re.split(r"(?=^### )", text, flags=re.MULTILINE)
        for block in blocks:
            block = block.strip()
            if block:
                chunks.append(block)

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
                chunks.append(f"[jsonl] {date} {key}：{txt}")
            except json.JSONDecodeError:
                continue

    if not chunks:
        print("[bm25] 没有可索引的内容，跳过")
        return

    tokenized = [_tokenize(c) for c in chunks]
    with _bm25_lock:
        _bm25_chunks = chunks
        _bm25_index = BM25Okapi(tokenized)

    print(f"[bm25] 索引已建立，共 {len(chunks)} 个 chunk")


threading.Thread(target=_build_bm25_index, daemon=True).start()


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
            snippet = _bm25_chunks[i][:200].replace("\n", " ")
            lines.append(snippet)
        return "\n".join(lines)


def _bm25_search_raw(query: str, top_k: int = 20) -> list[str]:
    """返回 BM25 排序后的 chunk 文本列表（已过滤 score<=0）。"""
    with _bm25_lock:
        if _bm25_index is None or not _bm25_chunks:
            return []
        scores = _bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [_bm25_chunks[i] for i in ranked if scores[i] > 0]


def _vector_search_raw(query: str, top_k: int = 20, threshold: float = 0.3) -> list[str]:
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


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """RRF 混合检索：向量 + BM25，返回 [{text, score}, ...] 列表。"""
    vec_results = _vector_search_raw(query, top_k=20)
    bm25_results = _bm25_search_raw(query, top_k=20)

    # RRF: score = Σ 1/(60 + rank)，rank 从 1 开始
    rrf_scores: dict[str, float] = {}
    # 用文本前 80 字作为去重 key
    key_of: dict[str, str] = {}  # dedup_key → full text

    def add(results: list[str]) -> None:
        for rank, text in enumerate(results, start=1):
            dk = text[:80]
            if dk not in key_of:
                key_of[dk] = text
            rrf_scores[dk] = rrf_scores.get(dk, 0.0) + 1.0 / (60 + rank)

    add(vec_results)
    add(bm25_results)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": key_of[dk], "score": round(score, 6)} for dk, score in ranked]


# ══════════════════════════════════════════
#  原有搜索功能（不变）
# ══════════════════════════════════════════

def search(query: str, top_k: int = 3, threshold: float = 0.45) -> str:
    if _table is None:
        return ""

    # 向量语义搜索
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = resp.data[0].embedding
    vec_results = _table.search(vec).metric("cosine").limit(top_k).to_pandas()

    # FTS 关键词搜索
    try:
        fts_results = _table.search(query, query_type="fts").limit(top_k).to_pandas()
    except Exception:
        fts_results = None

    # 合并，按 chunk_id 去重
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

    fallback = _jsonl_fallback(query)
    if fallback:
        lines.append(fallback)

    return "\n".join(lines)


def _extract_content(obj: dict) -> str:
    """从 jsonl 行里提取可读文本。"""
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
    """最早命中行（带前后 context_window 条上下文）+ 最新命中行（只取命中行）。"""
    try:
        grep_result = subprocess.run(
            ["grep", "-F", "-r", "-n", "--include=*.jsonl", query, str(JSONL_DIR)],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        return ""

    if not grep_result.stdout.strip():
        return ""

    # 解析所有命中，去重，收集 (ts, filepath, lineno, text)
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
        # 跳过 tool_result（OB 召回等注入内容，时间戳不可信）
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

    # 最早命中：带上下文
    ctx = load_context(oldest[1], oldest[2])
    if ctx:
        parts_out.append("# 最早出现")
        parts_out.extend(ctx)

    # 最新命中：只取命中行（如果和最早是同一条就跳过）
    if newest[0] != oldest[0] or newest[1] != oldest[1] or newest[2] != oldest[2]:
        ts, _, _, text = newest
        role = ""
        parts_out.append("# 最近出现")
        parts_out.append(fmt_line(text, ts, role))

    if not parts_out:
        return ""

    return "[jsonl_fallback]\n" + "\n".join(parts_out)


# ══════════════════════════════════════════
#  Recall Agent（新增）
# ══════════════════════════════════════════

RECALL_PROMPT_TEMPLATE = """你是记忆挑选员。你的工作是从候选记忆中挑出真正和当前对话相关的，或者判断没有相关的。

规则：
1. 只挑真正和当下话题/情绪/语境相关的记忆，最多保留3条
2. "长得像"不等于"该出现"——话题相似但当下不需要的，扔掉
3. 已经在对话里提过的信息，不要重复给
4. 如果候选里没有任何真正相关的，返回空列表——宁可不给，绝不硬塞
5. 不要编造理由来让某条记忆显得相关

当前消息：
{prompt}

候选记忆（编号从0开始）：
{candidates}

返回JSON，不要其他内容：
{{"keep": [0, 2], "reason": "简短说明"}}
或
{{"keep": [], "reason": "没有相关记忆"}}"""


def recall_agent(prompt: str, candidates: list[str]) -> list[str]:
    """调用 OpenRouter Haiku 筛选候选记忆"""
    if not RECALL_ENABLED:
        return candidates  # 未配置则直接透传

    if not candidates:
        return []

    # 格式化候选
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

        # 解析 JSON（容错：去掉 markdown 围栏）
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        keep_indices = result.get("keep", [])

        if not keep_indices:
            print(f"[recall_agent] 返回空（{result.get('reason', '')}）")
            return []

        filtered = [candidates[i] for i in keep_indices if i < len(candidates)]
        print(f"[recall_agent] {len(candidates)}→{len(filtered)} ({result.get('reason', '')})")
        return filtered

    except Exception as e:
        print(f"[recall_agent] 调用失败，透传原结果: {e}")
        return candidates  # 失败时不阻塞，透传原结果


# ══════════════════════════════════════════
#  HTTP Handler
# ══════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]

        if parsed.path == "/bm25":
            if not query:
                self.send_response(400)
                self.end_headers()
                return
            top_k = int(params.get("top_k", [5])[0])
            try:
                result = bm25_search(query, top_k)
            except Exception as e:
                result = ""
                print(f"[bm25] 搜索出错: {e}")
            body = result.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/hybrid":
            if not query:
                self.send_response(400)
                self.end_headers()
                return
            top_k = int(params.get("top_k", [5])[0])
            try:
                results = hybrid_search(query, top_k)
            except Exception as e:
                results = []
                print(f"[hybrid] 出错: {e}")
            body = json.dumps({"results": results}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/reload_bm25":
            threading.Thread(target=_build_bm25_index, daemon=True).start()
            body = b"reloading"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path != "/search":
            self.send_response(404)
            self.end_headers()
            return

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
            print(f"[search_server] 搜索出错: {e}")

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

        # 读取 POST body
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

        # 调用 recall agent
        filtered = recall_agent(prompt, candidates)

        body = json.dumps({"results": filtered}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[search_server] 监听 127.0.0.1:{PORT}")
    server.serve_forever()
