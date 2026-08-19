#!/usr/bin/env python3
"""
breath_search.example.py — Stub showing the breath_search.py interface.

In a full deployment, breath_search.py connects to an external memory system
(e.g. Ombre Brain MCP) and retrieves stored memory entries matching the query.

Copy this file to breath_search.py and implement the search logic for your
own memory backend.

Interface contract:
  - Takes exactly one positional argument: the search query string
  - Writes results to stdout, one entry per line (plain text, ≤120 chars each)
  - Writes errors/debug info to stderr only
  - Exits 0 on success (even if no results found), non-zero on fatal error

memory_recall.sh calls this script in parallel with the hybrid search and
waits for it before assembling the final context injection.

Usage:
    python3 breath_search.py "query text here"

Example output (one memory entry per line):
    [2026-06-10] 小九喜欢说唱和 R&B，尤其喜欢欧美流行。
    [2026-07-01] 她在减肥，吃饭不规律，需要偶尔提醒。
"""

import sys


def search_memory(query: str) -> list[str]:
    """
    Search the memory system for entries relevant to `query`.

    Replace this function body with your actual memory backend call.
    Options include:
      - HTTP call to a local MCP server  (e.g. Ombre Brain at localhost:18001)
      - SQLite / vector DB lookup
      - Simple grep over a flat file

    Return a list of result strings; empty list if nothing found.
    """
    # ── Example: call a local REST endpoint ──
    # import urllib.request, json
    # url = "http://127.0.0.1:18001/search"
    # payload = json.dumps({"query": query, "limit": 5}).encode()
    # req = urllib.request.Request(url, data=payload,
    #                              headers={"Content-Type": "application/json"})
    # with urllib.request.urlopen(req, timeout=5) as resp:
    #     data = json.loads(resp.read())
    # return [item["text"] for item in data.get("results", [])]

    # Stub: return empty (no-op)
    return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: breath_search.py <query>", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    results = search_memory(query)
    for line in results:
        print(line[:120])
