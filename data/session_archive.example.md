# Session Archive — conversation summaries

Each session's daily summaries are appended here before context clear.

---

## Session: 2026-06-15 to 2026-06-17

### 2026-06-15 23:45 CST

Fixed a bug in the scheduled message system where morning messages weren't firing due to stale session state. Reworked the logic to use file-based state instead of in-memory flags. Also added two late-night trigger segments (A: fires after 30 minutes of silence, B: fires 1 hour after a goodnight message).

---

## Session: 2026-06-18 to 2026-06-20

### 2026-06-18 14:30 CST

Implemented semantic memory recall using LanceDB + bge-m3 embeddings. Built a UserPromptSubmit hook that runs on every message, extracts keywords, searches the vector index, and injects relevant past context into the prompt via additionalContext. Added optional Haiku filter to reduce noise from loosely related matches.

### 2026-06-19 22:10 CST

Tuned the recall threshold from 0.5 to 0.45 to catch more partial matches. Added FTS (full-text search) as a secondary search method alongside vector search, with deduplication by chunk_id. The FTS results get a fixed score of 0.5 so they appear alongside semantic results without dominating.

---

## Session: 2026-06-21 to 2026-06-22

### 2026-06-21 16:20 CST

Added enrich_entities.py for extracting named entities (function names, file names, technical terms) from session index entries. Runs concurrently with ThreadPoolExecutor. Supports incremental mode (--new-only) for use in post-archive cron jobs. Entities are merged into an "aliases" field for improved keyword recall.
