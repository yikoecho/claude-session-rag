# Session Archive Example
#
# 格式说明：用 --- 分隔的 Markdown，每个 session 有时间戳和关键词摘要。
# 真实文件不要提交进版本库（包含对话内容）。
# 用 hooks/stop_archive.sh 可以在 Claude Code 退出时自动追加新 session。

---

## Session: 2026-01-10 to 2026-01-12

### 2026-01-10 22:15 CST
Set up the initial project structure. Created `config.py` and basic API client.
Connected to SiliconFlow embedding API, tested with a few sample texts.
Indexing pipeline returns ~300 chunks for a typical session log.

### 2026-01-11 14:30 CST
Debugged embedding batch size — was hitting rate limits at batch=64, dropped to batch=32.
BM25 index now co-located with LanceDB for consistency.
Added `enrich_entities.py` to extract named entities from session_index entries.

### 2026-01-12 09:00 CST
First full recall test: query "what did we decide about the API key rotation" correctly
returned the 2026-01-10 decision fragment. Partial verdict — the decision is there but
the rationale was in a follow-up message not captured in the same chunk.
Added chunk overlap (50 tokens) to reduce edge fragmentation.

---

## Session: 2026-01-15 to 2026-01-17

### 2026-01-15 20:00 CST
Integrated the UserPromptSubmit hook. First end-to-end test: asked about "the retry
backoff config" and the correct session was surfaced. Hook latency ~800ms (acceptable).
notice_filter.py blocking ~40% of messages — mostly single-word acks ("ok", "got it").

### 2026-01-16 11:45 CST
Tuned BM25_ENTITY_MIN_SCORE from 8.0 to 6.5 — entity path was missing low-frequency
proper nouns (project codenames, library names).
Recall agent (Haiku) consistent on multi-run: same verdict for same candidates.

### 2026-01-17 23:30 CST
Added session_index.jsonl for high-confidence direct injection.
Each entry: {"date": "2026-01-17", "key": "short title", "text": "one-line summary"}.
Grep path bypasses notice_filter and agent — always injected if keyword matches.
Useful for frequently-referenced facts (project decisions, key configs).
