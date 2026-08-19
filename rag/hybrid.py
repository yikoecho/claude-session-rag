"""
rag/hybrid.py — hybrid_search（RRF 混合检索）+ 原有 /search 逻辑
"""

import json
import subprocess
from pathlib import Path

from index.bm25 import bm25_search_raw
from index.vector import vector_search_raw, get_table, get_client
from utils.config import EMBEDDING_MODEL, JSONL_DIR


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """RRF 混合检索：向量 + BM25，返回 [{text, score}, ...] 列表。"""
    vec_results = vector_search_raw(query, top_k=20)
    bm25_results = bm25_search_raw(query, top_k=20)

    # RRF: score = Σ 1/(60 + rank)，rank 从 1 开始
    rrf_scores: dict[str, float] = {}
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
        parts_out.append("# 最早出现")
        parts_out.extend(ctx)

    if newest[0] != oldest[0] or newest[1] != oldest[1] or newest[2] != oldest[2]:
        ts, _, _, text = newest
        parts_out.append("# 最近出现")
        parts_out.append(fmt_line(text, ts))

    if not parts_out:
        return ""

    return "[jsonl_fallback]\n" + "\n".join(parts_out)


def search(query: str, top_k: int = 3, threshold: float = 0.45) -> str:
    table = get_table()
    client = get_client()
    if table is None:
        return ""

    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = resp.data[0].embedding
    vec_results = table.search(vec).metric("cosine").limit(top_k).to_pandas()

    try:
        fts_results = table.search(query, query_type="fts").limit(top_k).to_pandas()
    except Exception:
        fts_results = None

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
