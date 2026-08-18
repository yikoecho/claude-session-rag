#!/usr/bin/env python3
"""
enrich_entities.py — Extract named entities from session_index.jsonl entries.

Reads session_index.jsonl, calls an LLM (via OpenRouter) to extract entities
from each entry's key+text fields, and writes back an "entities" field.
Supports incremental updates (skips entries that already have entities).

Usage:
  python3 enrich_entities.py              # process all missing entries
  python3 enrich_entities.py --limit 50   # process first 50 missing entries
  python3 enrich_entities.py --new-only   # process only last 2 days (for post-archive runs)

Environment variables:
  OPENROUTER_API_KEY    Required
  SESSION_INDEX_PATH    Path to session_index.jsonl (default: ./data/session_index.jsonl)
"""
import sys
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Load .env if present
_env = Path(".env")
if not _env.exists():
    _env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SESSION_INDEX_PATH = os.environ.get("SESSION_INDEX_PATH", "./data/session_index.jsonl")
MODEL = os.environ.get("ENRICH_MODEL", "anthropic/claude-haiku-4-5")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

PROMPT_TEMPLATE = """Extract specific named entities from the text below. Return a flat JSON string array — no objects, no generic words, no explanation. Return only the array itself, e.g.: ["monitor.py", "save_data()", "deploy_flag"]

Text:
{key}
{text}"""


def extract_entities(key: str, text: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(key=key, text=text)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
        )
        output = resp.choices[0].message.content.strip()
        # Try to match a top-level array first
        m = re.search(r'\[([^\[\]]*)\]', output, re.DOTALL)
        if m:
            entities = json.loads(m.group())
            if isinstance(entities, list):
                return [e for e in entities if isinstance(e, str) and e.strip()]
        # Fallback: if model returned an object, collect all string values
        m2 = re.search(r'\{.*\}', output, re.DOTALL)
        if m2:
            obj = json.loads(m2.group())
            result = []
            for v in obj.values():
                if isinstance(v, list):
                    result.extend(e for e in v if isinstance(e, str))
                elif isinstance(v, str):
                    result.append(v)
            return result
    except Exception as e:
        print(f"  [warn] {e}", file=sys.stderr)
    return []


def enrich_entries(entries: list[dict], limit: int | None = None, new_only_days: int | None = None) -> tuple[list[dict], int]:
    to_process = []
    cutoff = None
    if new_only_days:
        cutoff = (datetime.now() - timedelta(days=new_only_days)).strftime("%Y-%m-%d")

    for entry in entries:
        if "entities" in entry:
            continue
        if cutoff and entry.get("date", "") < cutoff:
            continue
        to_process.append(entry)

    if limit:
        to_process = to_process[:limit]

    total = len(to_process)
    if total == 0:
        return entries, 0

    print(f"Entries to process: {total}, concurrency: 10")
    updated = 0
    done_count = 0

    def process(entry):
        key = entry.get("key", "")
        text = entry.get("text", "")
        return entry, extract_entities(key, text)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(process, e): e for e in to_process}
        for fut in as_completed(futures):
            entry, entities = fut.result()
            existing_aliases = entry.get("aliases", [])
            merged = list(dict.fromkeys(existing_aliases + entities))
            entry["aliases"] = merged
            entry["entities"] = entities
            done_count += 1
            updated += 1
            print(f"[{done_count}/{total}] {entry.get('key','')[:40]} → {entities}", flush=True)

    return entries, updated


def atomic_write(path: str, entries: list[dict]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    if not OPENROUTER_KEY:
        print("Error: OPENROUTER_API_KEY is not set")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--new-only", action="store_true", help="Only process entries from the last 2 days (for post-archive runs)")
    parser.add_argument("--file", default=SESSION_INDEX_PATH, help="Path to session_index.jsonl")
    args = parser.parse_args()

    index_file = args.file
    with open(index_file, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    already = sum(1 for e in entries if "entities" in e)
    print(f"Total entries: {len(entries)}, with entities: {already}, missing: {len(entries) - already}")

    new_only_days = 2 if args.new_only else None
    entries, updated = enrich_entries(entries, limit=args.limit, new_only_days=new_only_days)

    if updated > 0:
        atomic_write(index_file, entries)
        print(f"\nDone. Updated {updated} entries, written to {index_file}")
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
