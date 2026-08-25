"""
rag/hybrid.py — hybrid_search（三路检索 + RRF 融合）

检索架构：
  - vec 路：语义向量搜索（余弦相似度 0-1），支持传入改写后的 retro_query
  - bm25_archive 路：BM25 全文搜索 session 归档
  - bm25_entity 路：BM25 实体/JSONL 搜索，top-2 保底
  - kw 路：关键词分词后逐词 LIKE 精确匹配，命中的 pin 到最前
  - 融合：pool 内用 RRF（1/(60+rank) 求和），量纲统一
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from index.bm25 import bm25_search_raw, bm25_aliases_fallback
from index.vector import vector_search_raw, get_table, get_client
from utils.config import DEDUP_THRESHOLD, EMBEDDING_MODEL, JSONL_DIR, BM25_ENTITY_MIN_SCORE


def _escape_like(s: str) -> str:
    """Escape special LIKE characters so they are treated as literals."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")


def hybrid_search_split(query: str, top_k: int = 5, retro_query: str = "") -> dict:
    """分路检索，返回三路结果（vec / bm25_archive / bm25_entity）。

    vec 路使用 retro_query（语义改写后的查询）；BM25 两路用原 query（字面匹配）。
    retro_query 为空时退回 query。
    """
    # 向量路：使用 retro_query（语义改写），没有则退回原句
    vec_query = retro_query if retro_query else query
    vec_result_dicts = vector_search_raw(vec_query, top_k=top_k)
    vec_results = []
    vec_map: dict[str, list[float]] = {}
    for item in vec_result_dicts:
        text = item["text"]
        score = item.get("score", 0.0)
        if item.get("vector") is not None:
            vec_map[text[:80]] = item["vector"]
        vec_results.append({"text": text, "score": round(score, 6), "source": "vec"})

    # BM25路：按来源分开
    bm25_all = bm25_search_raw(query, top_k=top_k * 4)
    bm25_archive = []
    bm25_entity = []
    for item in bm25_all:
        src = item.get("source", "archive")
        if src == "aliases":
            continue  # aliases 不进主路
        entry = {"text": item["text"], "score": round(item["score"], 6),
                 "source": "bm25_archive" if src == "archive" else "bm25_entity"}
        if src == "archive":
            if len(bm25_archive) < top_k:
                bm25_archive.append(entry)
        else:
            if len(bm25_entity) < top_k:
                bm25_entity.append(entry)

    # aliases 兜底：仅在三路均无结果时启用，不能挤位
    aliases_fallback = []
    VEC_ZERO_THRESHOLD = 0.35
    BM25_ZERO_THRESHOLD = 3.0
    vec_top_score = vec_results[0]["score"] if vec_results else 0.0
    bm25_top_score = max(
        (bm25_archive[0]["score"] if bm25_archive else 0.0),
        (bm25_entity[0]["score"] if bm25_entity else 0.0),
    )
    if vec_top_score < VEC_ZERO_THRESHOLD and bm25_top_score < BM25_ZERO_THRESHOLD:
        candidates = bm25_aliases_fallback(query, top_k=top_k)
        if candidates:
            print(f"[hybrid] aliases 兜底触发 query='{query[:40]}' found={len(candidates)}")
            aliases_fallback = candidates

    return {"vec": vec_results, "bm25_archive": bm25_archive, "bm25_entity": bm25_entity,
            "aliases_fallback": aliases_fallback, "_vec_map": vec_map}


def hybrid_search(query: str, top_k: int = 5, retro_query: str = "") -> list[dict]:
    """混合检索：vec + bm25_archive + bm25_entity 三路，RRF 融合后返回。

    vec 路传 retro_query（语义改写），BM25 两路传原 query。
    kw 路：关键词逐词 LIKE，命中的 pin 到最前，不参与 RRF 排序。
    bm25_entity top-2 超阈值保底，不参与 RRF 排序。
    pool 内用 RRF（k=60）融合三路排名。
    """
    split = hybrid_search_split(query, top_k=top_k, retro_query=retro_query)
    vec_results = split["vec"]
    bm25_archive = split["bm25_archive"]
    bm25_entity = split["bm25_entity"]
    vec_map = split["_vec_map"]
    aliases_fallback = split.get("aliases_fallback", [])

    # 精确关键词匹配：按 jieba 分词后逐词 LIKE，补偿 BM25/FTS 对中文无效的问题
    # 用分词词语而非完整 query，否则子串匹配几乎不可能命中
    kw_items = []
    try:
        import jieba
        jieba.setLogLevel(20)
        table = get_table()
        if table is not None:
            kw_seen: set[str] = set()
            tokens = [w for w in jieba.lcut(query) if len(w) >= 2][:5]
            for token in tokens:
                if len(kw_items) >= 2:
                    break
                safe_tok = _escape_like(token)
                kw_rows = table.search().where(f"text LIKE '%{safe_tok}%'").limit(3).to_list()
                for row in kw_rows:
                    dk = row["text"][:80]
                    if dk in kw_seen:
                        continue
                    kw_seen.add(dk)
                    date = str(row["timestamp_start"])[:10]
                    snippet = row["text"][:200].replace("\n", " ")
                    text = f"{date}: {snippet}"
                    # pinned=True 标记来源；score=None（无天然分数），不污染分数字段
                    kw_items.append({"text": text, "score": None, "source": "keyword",
                                     "_exempt": True, "_key": dk, "pinned": True})
                    if len(kw_items) >= 2:
                        break
    except Exception as e:
        print(f"[WARN][B] kw_items 精确匹配失败，降级跳过: {e}", flush=True)

    # bm25_entity top-2 有条件保底：分数超过阈值才占坑，防止虚词低分命中被强塞进来
    BM25_ENTITY_RESERVED = 2
    guaranteed_entity = [r for r in bm25_entity[:BM25_ENTITY_RESERVED] if r["score"] >= BM25_ENTITY_MIN_SCORE]
    for r in guaranteed_entity:
        r["_exempt"] = True

    # Pool 内用 RRF 融合三路排名（k=60），统一量纲（vec 0-1 vs bm25 0-50+）
    guaranteed_keys = {r["text"][:80] for r in guaranteed_entity}
    kw_keys = {r["text"][:80] for r in kw_items}

    # 建三路各自的排名字典 dk→rank
    vec_rank  = {r["text"][:80]: i for i, r in enumerate(vec_results)}
    bm25a_rank = {r["text"][:80]: i for i, r in enumerate(bm25_archive)}
    bm25e_rank = {r["text"][:80]: i for i, r in enumerate(bm25_entity[BM25_ENTITY_RESERVED:])}

    RRF_K = 60
    pool_map: dict[str, dict] = {}  # dk → item（去重保留第一次见到的 dict）
    for r in vec_results + bm25_archive + bm25_entity[BM25_ENTITY_RESERVED:]:
        dk = r["text"][:80]
        if dk not in guaranteed_keys and dk not in kw_keys and dk not in pool_map:
            pool_map[dk] = r

    def _rrf(dk: str) -> float:
        score = 0.0
        if dk in vec_rank:  score += 1 / (RRF_K + vec_rank[dk])
        if dk in bm25a_rank: score += 1 / (RRF_K + bm25a_rank[dk])
        if dk in bm25e_rank: score += 1 / (RRF_K + bm25e_rank[dk])
        return score

    pool_sorted = sorted(pool_map.values(), key=lambda r: _rrf(r["text"][:80]), reverse=True)
    # 给 pool 结果附上 RRF 分供日志参考
    for r in pool_sorted:
        r["rrf_score"] = round(_rrf(r["text"][:80]), 6)

    remaining_slots = max(0, top_k - len(kw_items) - len(guaranteed_entity))
    merged = kw_items + guaranteed_entity + pool_sorted[:remaining_slots]

    # aliases 兜底：split 层已通过阈值判定（vec<0.35 且 bm25<3.0），aliases_fallback 非空即为触发
    # 追加到 merged 末尾，不挤位
    if aliases_fallback:
        merged = merged + aliases_fallback

    merged = _dedup_by_vector(merged, DEDUP_THRESHOLD, vec_map)
    return merged[:top_k]


def _dedup_by_vector(results: list[dict], threshold: float, vec_map: dict[str, list[float]] | None = None) -> list[dict]:
    """对hybrid搜索结果按语义相似度去冗余，保留最新的。贪心算法。"""
    if not results or threshold >= 1.0:
        return results

    if vec_map is None:
        vec_map = {}

    client = get_client()
    if client is None:
        # strip _exempt before returning
        for r in results:
            r.pop("_exempt", None)
        return results

    def _parse_date(r: dict) -> datetime:
        text = r.get("text", "")
        m = re.search(r'\d{4}-\d{2}-\d{2}', text[:40])
        if m:
            try:
                return datetime.fromisoformat(m.group())
            except ValueError:
                pass
        return datetime.min

    # collect texts that need embedding (not already in vec_map)
    needs_embed: list[str] = []
    for r in results:
        dk = r.get("_key") or r["text"][:80]
        if dk not in vec_map:
            needs_embed.append(r["text"])

    vecs_lookup: dict[str, np.ndarray] = {}

    # seed from vec_map
    for dk, v in vec_map.items():
        arr = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        vecs_lookup[dk] = arr

    # batch embed missing ones
    if needs_embed:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=needs_embed)
            # build a mapping from text → result dict to get the right _key
            text_to_r = {r["text"]: r for r in results}
            for text, emb_data in zip(needs_embed, resp.data):
                arr = np.array(emb_data.embedding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                r = text_to_r.get(text)
                dk = (r.get("_key") if r else None) or text[:80]
                vecs_lookup[dk] = arr
        except Exception as e:
            print(f"[WARN][B] _dedup_by_vector 向量化失败，跳过去重直接返回: {e}", flush=True)
            for r in results:
                r.pop("_exempt", None)
            return results

    kept_indices: list[int] = []
    for i, item in enumerate(results):
        dk_i = item.get("_key") or item["text"][:80]
        vec_i = vecs_lookup.get(dk_i)
        exempt_i = item.get("_exempt", False)

        dominated = False
        for j in kept_indices:
            kept_item = results[j]
            dk_j = kept_item.get("_key") or kept_item["text"][:80]
            vec_j = vecs_lookup.get(dk_j)
            exempt_j = kept_item.get("_exempt", False)

            if vec_i is None or vec_j is None:
                # can't compare, skip dedup for this pair
                continue

            sim = float(np.dot(vec_i, vec_j))
            if sim > threshold:
                # exempt always beats non-exempt
                # 修顺序 bug：用 replace-at-position 而非 remove+append，保持 pin 位置
                replace_pos = kept_indices.index(j)
                if exempt_i and not exempt_j:
                    kept_indices[replace_pos] = i  # i 继承 j 的位置
                elif not exempt_i and exempt_j:
                    pass  # keep existing exempt j
                else:
                    # both same exempt status: keep newer date
                    if _parse_date(item) > _parse_date(kept_item):
                        kept_indices[replace_pos] = i
                dominated = True
                break

        if not dominated:
            kept_indices.append(i)

    final = [results[i] for i in kept_indices]
    # strip internal markers
    for r in final:
        r.pop("_exempt", None)
        r.pop("_key", None)
    return final


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
            ["grep", "-F", "-r", "-n", "--include=*.jsonl", "--", query, str(JSONL_DIR)],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        return ""

    if not grep_result.stdout.strip():
        return ""

    seen_texts = set()
    hits = []
    for line in grep_result.stdout.strip().splitlines():
        parts = line.rsplit(":", 2)
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


