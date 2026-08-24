# claude-session-rag

Semantic memory recall for Claude Code sessions. Indexes your conversation history and injects relevant past context into each new prompt via a `UserPromptSubmit` hook.

This is not a general-purpose RAG framework. It's built specifically for the Claude Code CLI — where conversations are long-running sessions, the primary documents are raw jsonl conversation turns written automatically by Claude Code, with curated session summaries as an optional enhancement layer, and the goal is helping Claude remember things from weeks ago without blowing up the context window.

> **Security note:** the search server has **no authentication** and binds to `127.0.0.1` only. Never expose port `15200`. Never commit your `.env`. Also keep out of version control: `session_archive.md`, `*.jsonl` files, `memory_db/`, and `eval_baseline/` — all of these contain real conversation content. `eval_baseline/` is a new directory type that's easy to overlook; it stores query results used for regression comparison and will contain excerpts of your actual conversations. The `.gitignore` in this repo excludes all of the above.

## Architecture

```
*.jsonl (required)                session_archive.md (optional)
   │  Raw Claude Code sessions        Curated session summaries
   │  generated automatically         written by a Stop hook
   └──────────────┬───────────────────────────┘
                  ↓ build_index.py
     LanceDB (bge-m3 vectors)  +  in-memory BM25
                  ↓ search_server.py (HTTP on 127.0.0.1:15200)
     UserPromptSubmit hook (memory_recall.sh)
                  ↓
     additionalContext injected into the Claude Code prompt
```

**Primary data source:** `*.jsonl` files from `JSONL_DIR`. Claude Code writes these automatically — no manual steps needed. Point `JSONL_DIR` at your `~/.claude/projects/` subdirectory and you're done.

**Secondary (optional):** `session_archive.md` adds a curated human-readable summary layer to the BM25 index. Useful for long-running projects with hundreds of sessions. See the Stop hook section for auto-generation.

**Entity extraction:** `enrich_entities.py` reads `session_index.jsonl` (one entry per line with `key`, `text`, `date` fields), calls an LLM to extract named entities, and writes them back as an `entities` field. This powers the BM25 entity path and is what makes keyword recall work for proper nouns, technical terms, and names. With ~780 entity entries in the author's corpus, it's not optional post-processing — it's the main BM25 signal.

## Requirements

- Python 3.10+
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) in PATH for the hook
- An embedding API compatible with the OpenAI client (tested with [SiliconFlow](https://siliconflow.cn) using `BAAI/bge-m3`)
- OpenRouter API key (optional, for the Haiku recall filter and entity enrichment)

```
pip install -r requirements.txt
```

## Quick Start

```bash
git clone https://github.com/yikoecho/claude-session-rag.git
cd claude-session-rag
cp config.example.env .env
# edit .env — set EMBEDDING_API_KEY and JSONL_DIR at minimum
```

Build the index (point it at your Claude Code jsonl directory):

```bash
python build_index.py
# or: python build_index.py /path/to/session_archive.md /path/to/jsonl_dir
# session_archive.md is optional — omit or pass an empty path to skip it
# Note: if ARCHIVE_FILE is set in .env, it takes precedence over the positional argument
```

Start the search server:

```bash
python search_server.py
```

Test it:

```bash
curl "http://127.0.0.1:15200/hybrid?q=your+query&top_k=3"
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

## Endpoints

| Method | Path            | Description                                                        |
| ------ | --------------- | ----------------------------------------------------------------- |
| GET    | `/hybrid`       | RRF hybrid search (vector + BM25 + exact keyword). Returns JSON.   |
| GET    | `/bm25`         | BM25-only search. Returns plain text.                             |
| GET    | `/reload_bm25`  | Rebuilds the in-memory BM25 index (e.g. after archive updates).   |
| POST   | `/recall`       | Optional Haiku filter — classifies candidates and injects them with a three-state verdict (see below). |
| GET    | `/search`       | **Deprecated** — returns HTTP 410. Use `/hybrid`.                 |

### `/recall` verdict system

`/recall` is the most distinctive part of the pipeline. It sends hybrid-search candidates to a small LLM (default: `claude-haiku-4-5`) and returns one of four verdicts:

- **`sufficient`** — the retrieved passages clearly answer the query; injected directly into `additionalContext`.
- **`partial`** — something was found but confidence is low; injected with the prefix `以下内容与问题相关但可能未直接回答，仅供参考：` so Claude knows to treat it as suggestive, not definitive.
- **`none`** — nothing relevant was found; injects `[archive_status] 档案中未找到与该问题相关的记录。` so Claude knows the silence is intentional, not a pipeline failure.
- **`unfiltered`** — recall was disabled (`RECALL_ENABLED=False`) or the LLM call failed; candidates are injected as-is with the prefix `以下内容未经过滤，可能包含不相关的记录：`. This is distinct from `partial` so that downstream consumers can tell the difference between "LLM said low confidence" and "LLM was never asked".

The `none` verdict is what prevents hallucination: without it, an absent result looks identical to a skipped search, and Claude may confabulate. Requires `OPENROUTER_API_KEY` (or equivalent `LLM_BACKEND` config); falls back to `unfiltered` (returning raw candidates) if no LLM is available.

## Configuration

All config via environment variables (or `.env` file):

| Variable                | Default                           | Description                                                      |
| ----------------------- | --------------------------------- | ---------------------------------------------------------------- |
| `EMBEDDING_API_KEY`     | —                                 | **Required.** Embedding API key                                  |
| `EMBEDDING_BASE_URL`    | `https://api.siliconflow.cn/v1`   | Embedding API base URL                                           |
| `EMBEDDING_MODEL`       | `BAAI/bge-m3`                     | Embedding model name                                             |
| `LANCE_DB_PATH`         | `<repo>/memory_db`                | LanceDB storage path                                             |
| `JSONL_DIR`             | `~/.claude/projects/-root`        | **Required.** Directory of raw `*.jsonl` sessions. The `-root` suffix is an example — Claude Code names the subdirectory after your project path; check `~/.claude/projects/` for the actual name on your system. |
| `ARCHIVE_FILE`          | `~/.claude/session_archive.md`    | Optional archive file for BM25. Leave unset to skip.            |
| `JSONL_INDEX_FILE`      | `~/.claude/session_index.jsonl`   | session_index.jsonl used by BM25 entity path                    |
| `BM25_ENTITY_MIN_SCORE` | `7.0`                             | Min BM25 score for entity hits to claim a guaranteed result slot |
| `SEARCH_PORT`           | `15200`                           | Port for the search server                                       |
| `LLM_BACKEND`           | auto (`openrouter` or `none`)     | `none` \| `siliconflow` \| `ollama` \| `api` \| `openrouter`   |
| `OPENROUTER_API_KEY`    | —                                 | Optional: enables the Haiku recall filter                        |
| `RECALL_MODEL`          | `anthropic/claude-haiku-4-5`      | Model for the recall filter                                      |
| `ENRICH_MODEL`          | `anthropic/claude-haiku-4-5`      | Model for entity extraction                                      |

See `config.example.env` for the full list.

### `BM25_ENTITY_MIN_SCORE`

The entity BM25 path reserves 2 result slots for high-confidence entity hits. A hit only claims a slot if its BM25 score meets this threshold (default `7.0`). This value was tuned empirically on the author's corpus (~780 entity entries, Chinese-language conversations). You will likely need to retune it for your own data:

- If you see irrelevant entity results, raise the threshold.
- If known proper nouns fail to appear, lower it.
- Run `eval_rag.py` with `--diff` to measure the effect of changes.

## LLM Backend

| Mode          | Cost | Notes |
| ------------- | ---- | ----- |
| `none`        | Free | Zero config, pure vector+BM25, no LLM |
| `siliconflow` | Free | Remote API via siliconflow.cn; reuse your `EMBEDDING_API_KEY` |
| `ollama`      | Free | Local inference; run `ollama pull qwen2.5:3b` |
| `api`         | Paid | Any OpenAI-compatible endpoint |
| `openrouter`  | Paid | Alias for `api`, kept for backward compatibility |

Auto-detected: if `OPENROUTER_API_KEY` is set, `openrouter` is used; otherwise `none`.

Legacy variables `RECALL_API_KEY` and `RECALL_BASE_URL` are still recognized.

## Evaluation

`eval_rag.py` runs a fixed set of test queries and measures recall quality. Edit the query list in the script to match your own data:

```bash
python eval_rag.py
python eval_rag.py --diff   # compare against a saved baseline to detect regressions
```

Each query has an expected keyword that should appear in at least one of the top-3 results. Output is a pass/fail table with scores. Run this before and after tuning `BM25_ENTITY_MIN_SCORE` or changing the index to catch regressions.

**Writing good eval queries:** phrase them the way you'd actually ask during a session ("what did we decide about X"), not as verbatim excerpts from the archive. Split your query list into two groups:

- **Known-recorded** — events or facts you know are in the index. These should pass. They test retrieval quality.
- **Known-unrecorded** — things that were never logged. These should return a `none` verdict. They test hallucination suppression, which is often overlooked but catches the most dangerous failure mode: a system that confidently injects fabricated "memories."

## Known Pitfalls

**Chinese stopwords:** Without a stopword list, high-frequency function words (一起, 过, 然后) get near-zero IDF scores but still appear as BM25 tokens. This causes false positives where generic conversational fragments score above the entity threshold. The jieba stopword list in `hooks/recall_keywords.py` handles this for query-side tokenization; make sure your index-side tokenization uses the same list.

**Single-path RRF penalty:** RRF scoring is `1/(60 + rank)` per path. An item that appears in only one retrieval path (say BM25 but not vector) always loses to an item that appears in both paths, even if its single-path rank is high. This means low-frequency proper nouns with strong BM25 scores can disappear from the final top-k if they have no vector match. The entity guaranteed-slot mechanism exists to compensate for this — but only if `BM25_ENTITY_MIN_SCORE` is set appropriately.

**Cross-query score incompatibility:** BM25 scores are not comparable across queries of different lengths. A 1-token query will always produce lower absolute scores than a 3-token query for the same passage. Don't compare `BM25_ENTITY_MIN_SCORE` hits across different query types; tune the threshold against a representative sample of your actual queries.

## Session Archive Format

Plain Markdown split by `---` separators. Each section should have a `### YYYY-MM-DD` timestamp line:

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

## Hooks

**`hooks/memory_recall.sh`** — the Claude Code hook. On each user message it: gates trivial messages via `notice_filter.py`, extracts keywords via `recall_keywords.py`, searches the server, optionally filters with Haiku, and returns results as `additionalContext`.

**`hooks/recall_keywords.py`** — tokenizes the prompt with `jieba`, strips stopwords, emits keywords. Optional LLM query rewrite: set `QUERY_REWRITE_ENABLED=true` to send the last 3 turns to `RECALL_MODEL` for a condensed query ("那次的事" → "蜜雪冰城 夏日聚餐"). Note that query rewrite adds one LLM round-trip on every message through the synchronous hook, which increases per-prompt latency noticeably. It is off by default for this reason.

**`hooks/stop_archive.sh`** — automatically appends a session summary to `session_archive.md` at session end. Requires `RECALL_API_KEY`. Install as a `Stop` hook in `.claude/settings.json`.

## Limitations / TODO

- **Date-range filtering**: timestamps are indexed but search doesn't filter by date yet.
- **Config unification**: `build_index.py` uses its own path constants; the server uses `utils/config.py`. Worth consolidating.
- Use `eval_rag.py` as a regression check after tuning index parameters.

## License

MIT
