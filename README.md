# claude-session-rag

Semantic memory recall for Claude Code sessions. Indexes your conversation history and injects relevant past context into each new prompt via a `UserPromptSubmit` hook.

This is not a general-purpose RAG framework. It's built specifically for the Claude Code CLI — where conversations are long-running sessions, the "documents" are conversation summaries, and the goal is helping Claude remember things from weeks ago without blowing up the context window.

> **Security note:** the search server has **no authentication** and binds to `127.0.0.1` only. Run it on a trusted local machine and never expose port `15200`. Never commit your `.env`.

## Architecture

```
session_archive.md (conversation summaries)  +  *.jsonl (raw sessions)
        ↓ build_index.py
   LanceDB (bge-m3 vectors)  +  in-memory BM25
        ↓ search_server.py (HTTP on 127.0.0.1:15200)
   UserPromptSubmit hook (memory_recall.sh)
        ↓
   additionalContext injected into the Claude Code prompt
```

Optional: `enrich_entities.py` extracts named entities from a `session_index.jsonl` file to improve keyword recall.

Optional: Haiku filter via OpenRouter — the `/recall` endpoint re-ranks and prunes candidates before injection.

## Requirements

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) in PATH for the hook
- An embedding API compatible with the OpenAI client (tested with [SiliconFlow](https://siliconflow.cn) using `BAAI/bge-m3`)
- OpenRouter API key (optional, for the Haiku recall filter and entity enrichment)

```
pip install -r requirements.txt
```

## Quick Start

```
git clone https://github.com/yikoecho/claude-session-rag.git
cd claude-session-rag
cp config.example.env .env
# edit .env: set EMBEDDING_API_KEY at minimum
```

Prepare your session archive (see `data/session_archive.example.md` for the format):

```
cp data/session_archive.example.md data/session_archive.md
# or copy your real session_archive.md from Claude Code
```

Build the index:

```
python build_index.py
# or: python build_index.py /path/to/session_archive.md [/path/to/jsonl_dir]
```

Start the search server:

```
python search_server.py
```

Test it:

```
curl "http://127.0.0.1:15200/hybrid?q=your+query&top_k=3"
```

Configure the Claude Code hook — add to `.claude/settings.json`:

```
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash /path/to/claude-session-rag/hooks/memory_recall.sh"
          }
        ]
      }
    ]
  }
}
```

## Endpoints

`search_server.py` serves:

| Method | Path            | Description                                                        |
| ------ | --------------- | ----------------------------------------------------------------- |
| GET    | `/hybrid`       | RRF hybrid search (vector + BM25 + exact keyword). Returns JSON.   |
| GET    | `/bm25`         | BM25-only search. Returns plain text.                             |
| GET    | `/reload_bm25`  | Rebuilds the in-memory BM25 index (e.g. after archive updates).   |
| POST   | `/recall`       | Optional Haiku filter that prunes candidates to relevant ones.    |
| GET    | `/search`       | **Deprecated** — returns HTTP 410. Use `/hybrid`.                 |

## How It Works

**`build_index.py`** — reads `session_archive.md` (and optionally a directory of raw `*.jsonl` sessions), splits into ~400-token chunks with 50-token overlap, embeds each with bge-m3, and stores vectors in LanceDB. Incremental: already-indexed chunks are skipped on re-runs.

**`search_server.py` / `server.py`** — HTTP server (see Endpoints above). The BM25 index is built in-memory at startup from `session_archive.md` + `session_index.jsonl`.

**`hooks/memory_recall.sh`** — the Claude Code hook. On each user message it: gates trivial messages via `notice_filter.py`, extracts keywords via `recall_keywords.py`, searches the server, optionally filters with Haiku, and returns results as `additionalContext`.

**`hooks/recall_keywords.py`** — tokenizes the prompt with `jieba`, strips stopwords, and emits keywords plus an optional retrospective query. (Requires `jieba`.)

**`hooks/breath_search.py`** — optional external "OB memory" source. A stub is provided as `breath_search.example.py`; copy and adapt it, or leave it out (the hook skips it if absent).

**`enrich_entities.py`** — optional post-processing. Reads `session_index.jsonl` (one JSON object per line with `key`, `text`, `date` fields), extracts named entities via an LLM, and writes them back as an `entities` field. Improves grep-based recall for technical terms.

## Configuration

All config via environment variables (or `.env` file). Variable names below match the code exactly:

| Variable               | Default                          | Description                                              |
| ---------------------- | -------------------------------- | ------------------------------------------------------- |
| `EMBEDDING_API_KEY`    | —                                | Required. Embedding API key                             |
| `EMBEDDING_BASE_URL`   | `https://api.siliconflow.cn/v1`  | Embedding API base URL                                  |
| `EMBEDDING_MODEL`      | `BAAI/bge-m3`                    | Embedding model name                                    |
| `LANCE_DB_PATH`        | `<repo>/data/lancedb`            | LanceDB storage path                                    |
| `SESSION_ARCHIVE_PATH` | `<repo>/data/session_archive.md` | Session archive file (build_index)                      |
| `ARCHIVE_FILE`         | `~/.claude/session_archive.md`   | Archive file used by the BM25 index (server)            |
| `JSONL_INDEX_FILE`     | `~/.claude/session_index.jsonl`  | session_index.jsonl used by the BM25 index (server)     |
| `SESSION_INDEX_PATH`   | `./data/session_index.jsonl`     | session_index.jsonl for `enrich_entities.py`            |
| `JSONL_DIR`            | `~/.claude/projects/-root`       | Directory of raw `*.jsonl` sessions (build + fallback)  |
| `SEARCH_PORT`          | `15200`                          | Port for the search server                              |
| `OPENROUTER_API_KEY`   | —                                | Optional: enables the Haiku recall filter (OpenRouter)  |
| `LLM_BACKEND`          | auto (`openrouter` or `none`)    | LLM backend: `none` \| `ollama` \| `openrouter`        |
| `OLLAMA_BASE_URL`      | `http://127.0.0.1:11434/v1`      | Ollama API base URL (only used when `LLM_BACKEND=ollama`) |
| `LLM_BASE_URL`         | derived from backend             | Override the LLM API base URL directly                  |
| `LLM_API_KEY`          | derived from backend             | Override the LLM API key directly                       |
| `RECALL_MODEL`         | `anthropic/claude-haiku-4-5`     | Model for the recall filter                             |
| `ENRICH_MODEL`         | `anthropic/claude-haiku-4-5`     | Model for entity extraction                             |

## LLM Backend

The recall filter and entity enrichment both use an LLM. Three modes are available:

| `LLM_BACKEND` | Cost | Setup |
|---|---|---|
| `none` | Free | Pure vector+BM25 retrieval, no filtering. Lower precision. |
| `ollama` | Free (local) | Run `ollama pull qwen2.5:3b` first, then start Ollama. Set `LLM_BACKEND=ollama`. |
| `openrouter` | Paid | Set `OPENROUTER_API_KEY`. Auto-selected when the key is present. |

The backend is auto-detected: if `OPENROUTER_API_KEY` is set, `openrouter` is used; otherwise `none`. Override with `LLM_BACKEND=ollama`.

Legacy variables `RECALL_API_KEY` and `RECALL_BASE_URL` are still recognized for backward compatibility.

> Note: `build_index.py` and the server currently read the archive from different defaults (`SESSION_ARCHIVE_PATH` vs `ARCHIVE_FILE`). Point both at the same file if you want the vector and BM25 indexes to cover identical content.

## Session Archive Format

Plain Markdown split by `---` separators. Each section should have a `### YYYY-MM-DD` timestamp line and optionally a `## Session: ...` header:

```
## Session: 2026-06-15 to 2026-06-17

### 2026-06-15 23:45 CST

Summary of what happened in this conversation...

---

### 2026-06-16 14:30 CST

Another day's summary...

---
```

See `data/session_archive.example.md` for a complete example.

## Hooks

### UserPromptSubmit — `hooks/recall_keywords.py`

Extracts search keywords from the user's message and optionally rewrites the query using an LLM.

**LLM query rewrite** (optional): set `QUERY_REWRITE_ENABLED=true` in `.env`. When enabled, the last 3 conversation turns (from `CONTEXT_FILE` if set) are sent to `RECALL_MODEL` to produce a single concise search query instead of raw jieba keywords. Useful when the user's message is ambiguous or short ("那次的事" → "狼山游览 蜜雪冰城 南通").

Requires `RECALL_API_KEY`. Without it, falls back to jieba silently.

### Stop — `hooks/stop_archive.sh`

Automatically appends a session summary to `session_archive.md` when Claude Code stops (session end or `/clear`).

Install in `.claude/settings.json`:
```json
"hooks": {
  "Stop": [{
    "matcher": "",
    "hooks": [{"type": "command",
      "command": "bash /path/to/claude-session-rag/hooks/stop_archive.sh"}]
  }]
}
```

Requires `RECALL_API_KEY` and `CLAUDE_TRANSCRIPT_PATH` (set automatically by Claude Code). Reads the last 200 lines of the session transcript, calls `RECALL_MODEL` for a 3-5 sentence summary, and appends it under a dated `---` header. No-ops silently if the API key is missing.

## Limitations / TODO

- **Date-range filtering**: the index stores timestamps but search doesn't filter by date yet — all history is searched equally.
- **Config unification**: `build_index.py` uses its own path constants; the server uses `utils/config.py`. Worth consolidating.
- ~~**Keyword extraction**: `recall_keywords.py` uses jieba + stopwords.~~ ✅ LLM-based query rewrite added (`QUERY_REWRITE_ENABLED=true`).
- **No tests / CI** yet.

## License

MIT
