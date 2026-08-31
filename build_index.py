#!/usr/bin/env python3
"""
build_index.py — session_archive.md + JSONL → LanceDB 语义索引
用法:
  python build_index.py                   # 使用默认路径
  python build_index.py /path/to/archive.md

环境变量（从 <repo>/.env 读，回退 /root/.env）:
  EMBEDDING_API_KEY=sk-...
  LANCE_DB_PATH=<repo>/data/lancedb        (可选)
  EMBEDDING_MODEL=BAAI/bge-m3              (可选)
  SESSION_ARCHIVE_PATH=<repo>/data/session_archive.md  (可选)
  JSONL_DIR=~/.claude/projects/-root       (可选，JSONL数据源目录)
"""

import os
import re
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime

# Load .env (repo dir first, then legacy /root/.env for backward compat)
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
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
LANCE_DB_PATH = os.environ.get(
    "LANCE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "lancedb"),
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_DIM = 1024
BATCH_SIZE = 20

ARCHIVE_PATH = os.environ.get(
    "SESSION_ARCHIVE_PATH",
    os.path.join(os.path.dirname(__file__), "data", "session_archive.md"),
)
JSONL_DIR = os.environ.get("JSONL_DIR", os.path.expanduser("~/.claude/projects/-root"))
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
# assistant text块（如"已回复小九'在。'"）不索引：
# 这是 Claude Code v2.1.183+ 引入的"无可见输出"注入的副作用文本，
# 不是克实际发给小九的内容。实际发出的话在 tool_use 块（TG reply）里。
# 参见 2026-08-05 的对话记录。
def _extract_text_from_content(content) -> str:
    """从 assistant message.content 提取实际发给小九的文本。

    只提取 mcp__plugin_telegram_telegram__reply 的 tool_use 块中的
    input.text 字段——这才是克真正说出口的话。
    text 块是 Claude Code v2.1.183+ 的"无可见输出"注入副作用，不索引。
    """
    if isinstance(content, str):
        # 纯字符串：旧格式，可能是实际内容，保留
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "tool_use":
                    # 只提取 TG reply 的实际发送文本
                    if block.get("name") == "mcp__plugin_telegram_telegram__reply":
                        input_data = block.get("input", {})
                        tg_text = input_data.get("text", "")
                        if tg_text:
                            parts.append(tg_text)
                # text 块不索引（是 CC 注入的副作用描述，不是对话内容）
                # tool_result / thinking 同样跳过
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return ""


EXCLUDE_RANGES_PATH = os.path.join(os.path.dirname(__file__), "exclude_ranges.json")


def _load_exclude_ranges() -> list[tuple[str, str]]:
    """加载排除时间段配置，返回 [(start_iso, end_iso), ...] 列表。"""
    try:
        ranges = json.loads(Path(EXCLUDE_RANGES_PATH).read_text(encoding="utf-8"))
        result = []
        for r in ranges:
            start = r.get("start", "")
            end = r.get("end", "")
            if start and end:
                result.append((start, end))
                print(f"  🚫 排除区间: {start} — {end} ({r.get('reason', '')})")
        return result
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"  ⚠ 读取 exclude_ranges.json 失败: {e}")
        return []


def _normalize_ts(ts: str) -> str:
    """统一 ISO 8601 时间戳格式：去掉毫秒，统一用 'Z' 结尾，便于字符串比较。"""
    ts = ts.rstrip("Z").split(".")[0] + "Z"
    return ts


def _is_excluded(timestamp: str, exclude_ranges: list[tuple[str, str]]) -> bool:
    """判断给定时间戳是否落在任一排除区间内。两端均归一化到秒精度再比较。"""
    norm_ts = _normalize_ts(timestamp)
    for start, end in exclude_ranges:
        norm_start = _normalize_ts(start)
        norm_end = _normalize_ts(end)
        if norm_start <= norm_ts <= norm_end:
            return True
    return False


_CHANNEL_RE = re.compile(r'<channel[^>]*>(.*?)</channel>', re.DOTALL)


def _extract_user_channel_text(content) -> str:
    """从 user 轮次的 content 中提取 <channel ...>...</channel> 标签内的文本（小九发来的消息）。"""
    raw = ""
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        raw = "\n".join(parts)

    matches = _CHANNEL_RE.findall(raw)
    return "\n".join(m.strip() for m in matches if m.strip())


def parse_jsonl_dir(jsonl_dir: str) -> list[dict]:
    """
    读取目录下所有 .jsonl，提取两类对话内容：
    1. assistant 轮次：mcp__plugin_telegram_telegram__reply tool_use 块中的 input.text
    2. user 轮次：<channel ...>...</channel> 标签内的文本
    使用滑动窗口（size=5, step=3）将相邻消息合并为 chunk。
    断窗条件：跨 session、时间间隔 >30 分钟、或进入 exclude_ranges 区间。
    exclude_ranges 区间后的消息开新段，不与区间前的段合并。
    """
    from datetime import timezone

    dirpath = Path(jsonl_dir)
    if not dirpath.exists():
        print(f"  ⚠ JSONL目录不存在，跳过: {jsonl_dir}")
        return []

    jsonl_files = sorted(dirpath.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ⚠ 未找到 .jsonl 文件: {jsonl_dir}")
        return []

    exclude_ranges = _load_exclude_ranges()
    print(f"  📄 找到 {len(jsonl_files)} 个 JSONL 文件")

    # ── 1. 收集所有消息 ──
    all_msgs = []
    excluded_count = 0

    for jfile in jsonl_files:
        session_id = f"jsonl:{jfile.stem}"
        try:
            for line in jfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                obj_type = obj.get("type", "")
                ts_full = obj.get("timestamp", "")
                if not ts_full:
                    continue

                if obj_type == "assistant":
                    msg = obj.get("message", {})
                    if msg.get("role") != "assistant":
                        continue
                    excluded = exclude_ranges and _is_excluded(ts_full, exclude_ranges)
                    if excluded:
                        excluded_count += 1
                    text = _extract_text_from_content(msg.get("content", ""))
                    if text and len(text) >= 20:
                        all_msgs.append({
                            "ts": ts_full,
                            "text": text,
                            "session_id": session_id,
                            "excluded": excluded,
                        })

                elif obj_type == "user":
                    msg = obj.get("message", {})
                    if msg.get("role") != "user":
                        continue
                    excluded = exclude_ranges and _is_excluded(ts_full, exclude_ranges)
                    if excluded:
                        excluded_count += 1
                    text = _extract_user_channel_text(msg.get("content", ""))
                    if text and len(text) >= 5:
                        all_msgs.append({
                            "ts": ts_full,
                            "text": text,
                            "session_id": session_id,
                            "excluded": excluded,
                        })

        except Exception as e:
            print(f"  ⚠ 读取 {jfile.name} 失败: {e}")
            continue

    if excluded_count:
        print(f"  🚫 已跳过 {excluded_count} 条被排除区间覆盖的消息")

    # ── 2. 按时间戳排序 ──
    all_msgs.sort(key=lambda m: m["ts"])

    # ── 3. 切段 ──
    def _ts_diff_minutes(ts1: str, ts2: str) -> float:
        def _parse(ts: str) -> datetime:
            ts = ts.rstrip("Z").split(".")[0]
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        try:
            return abs((_parse(ts2) - _parse(ts1)).total_seconds()) / 60
        except Exception:
            return 9999.0

    segments: list[list[dict]] = []
    current_seg: list[dict] = []
    prev_was_excluded = False

    for msg in all_msgs:
        if msg["excluded"]:
            # 跳过此消息；标记后面要开新段
            if current_seg:
                segments.append(current_seg)
                current_seg = []
            prev_was_excluded = True
            continue

        if not current_seg:
            current_seg.append(msg)
            prev_was_excluded = False
            continue

        prev = current_seg[-1]
        # 断窗条件
        if (msg["session_id"] != prev["session_id"]
                or _ts_diff_minutes(prev["ts"], msg["ts"]) > 30):
            segments.append(current_seg)
            current_seg = [msg]
        else:
            current_seg.append(msg)
        prev_was_excluded = False

    if current_seg:
        segments.append(current_seg)

    print(f"  🔀 切成 {len(segments)} 个对话段")

    # ── 4. 滑动窗口生成 chunk ──
    WINDOW_SIZE = 3
    STEP_SIZE = 2

    seen_ids: set[str] = set()
    chunks: list[dict] = []

    for seg in segments:
        start = 0
        while start < len(seg):
            end = min(start + WINDOW_SIZE, len(seg))
            window = seg[start:end]
            combined_text = "\n".join(m["text"] for m in window)
            if len(combined_text.strip()) >= 20:
                first = window[0]
                chunk_id = hashlib.md5(
                    f"{first['session_id']}:{first['ts']}:{combined_text[:80]}".encode()
                ).hexdigest()
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "session_id": first["session_id"],
                        "text": combined_text,
                        "timestamp_start": first["ts"][:10],
                        "timestamp_end": window[-1]["ts"][:10],
                        "msg_count": len(window),
                        "uuids": "[]",
                    })
            if end == len(seg):
                break
            start += STEP_SIZE

    print(f"  📦 JSONL解析完成，共 {len(chunks)} 个 chunk（滑动窗口 size={WINDOW_SIZE} step={STEP_SIZE}）")
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
