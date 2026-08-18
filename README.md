# claude-session-rag

Semantic memory recall for Claude Code sessions. Indexes your conversation history and injects relevant past context into each new prompt via a `UserPromptSubmit` hook.

This is not a general-purpose RAG framework. It's built specifically for the Claude Code CLI — where conversations are long-running sessions, the "documents" are conversation summaries, and the goal is helping Claude remember things from weeks ago without blowing up the context window.

## Architecture

```
session_archive.md (conversation summaries)
        ↓ build_index.py
   LanceDB (bge-m3 vectors + FTS)
        ↓ search_server.py (HTTP on :15200)
   UserPromptSubmit hook (memory_recall.sh)
        ↓
   additionalContext injected into Claude Code prompt
```

Optional: `enrich_entities.py` extracts named entities from a `session_index.jsonl` file to improve keyword recall.

Optional: Haiku filter via OpenRouter — the `/recall` endpoint re-ranks and prunes candidates before injection.

## Requirements

- Python 3.10+
- [lancedb](https://lancedb.github.io/lancedb/)
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) in PATH for the hook
- An embedding API compatible with the OpenAI client (tested with [SiliconFlow](https://siliconflow.cn) using `BAAI/bge-m3`)
- OpenRouter API key (optional, for Haiku filter in entity enrichment)

```
pip install lancedb openai pyarrow httpx
```

## Quick Start

```bash
git clone https://github.com/yikoecho/claude-session-rag.git
cd claude-session-rag
cp config.example.env .env
# edit .env: set EMBEDDING_API_KEY at minimum
```

Prepare your session archive (see `data/session_archive.example.md` for format):
```bash
cp data/session_archive.example.md data/session_archive.md
# or copy your real session_archive.md from Claude Code
```

Build the index:
```bash
python build_index.py
# or: python build_index.py /path/to/session_archive.md
```

Start the search server:
```bash
python search_server.py
```

Test it:
```bash
curl "http://127.0.0.1:15200/search?q=your+query&top_k=3"
```

Configure the Claude Code hook — add to `.claude/settings.json`:
```json
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

## How It Works

**`build_index.py`** — reads `session_archive.md`, splits it into chunks at `---` separators, embeds each chunk with bge-m3, and stores vectors in LanceDB. Incremental: already-indexed chunks are skipped on re-runs.

**`search_server.py`** — serves two endpoints:
- `GET /search` — hybrid search (vector + FTS), returns ranked text snippets
- `POST /recall` — optional Haiku filter that prunes candidates to only genuinely relevant ones

**`hooks/memory_recall.sh`** — the Claude Code hook. On each user message it: extracts keywords, searches the server in parallel, optionally filters with Haiku, and returns results as `additionalContext`.

**`enrich_entities.py`** — optional post-processing step. Reads a `session_index.jsonl` (one JSON object per line with `key`, `text`, `date` fields), extracts named entities via LLM, and writes them back as an `entities` field. This improves grep-based recall for technical terms.

## Configuration

All config via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_API_KEY` | — | Required. Embedding API key |
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | Embedding API base URL |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `LANCEDB_PATH` | `./data/lancedb` | LanceDB storage path |
| `SESSION_ARCHIVE_PATH` | `./data/session_archive.md` | Session archive file |
| `SESSION_INDEX_PATH` | `./data/session_index.jsonl` | Session index (for enrich_entities) |
| `SEARCH_SERVER_PORT` | `15200` | Port for search_server.py |
| `JSONL_DIR` | — | Optional: path to raw `.jsonl` conversation files for grep fallback |
| `OPENROUTER_API_KEY` | — | Optional: enables Haiku recall filter |
| `RECALL_MODEL` | `anthropic/claude-haiku-4-5` | Model for recall filter |
| `ENRICH_MODEL` | `anthropic/claude-haiku-4-5` | Model for entity extraction |

## Session Archive Format

The archive is a plain Markdown file split by `---` separators. Each section should have a `### YYYY-MM-DD` timestamp line and optionally a `## Session: ...` header:

```markdown
## Session: 2026-06-15 to 2026-06-17

### 2026-06-15 23:45 CST

Summary of what happened in this conversation...

---

### 2026-06-16 14:30 CST

Another day's summary...

---
```

See `data/session_archive.example.md` for a complete example.

## Limitations / TODO

- **Chunk splitting**: currently splits on `---` only. Long sections (>2000 chars) are truncated rather than split semantically. Overlapping sliding-window chunking would improve recall on long summaries.
- **Date metadata filtering**: the index stores timestamps but `/search` doesn't support filtering by date range yet. All history is searched equally.
- **Haiku filter is optional but improves precision**: without it, the hook injects up to 8 candidates directly. With it, irrelevant-but-similar memories are pruned. Set `OPENROUTER_API_KEY` to enable.
- **`recall_keywords.py` is a stub**: the included version uses simple stopword filtering. Replace it with something smarter (e.g., extracting noun phrases, using spacy, or calling an LLM) for better keyword search results.
- **`session_index.jsonl` format**: `enrich_entities.py` expects entries with `key`, `text`, and `date` fields. This matches the format produced by Claude Code's archive scripts but you may need to adapt it for other sources.
