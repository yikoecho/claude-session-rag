#!/usr/bin/env python3
"""RAG evaluation script for the hybrid search system."""

import argparse
import json
import os
import sys
from datetime import datetime

import urllib.request
import urllib.parse
import urllib.error

BASE_URL = "http://localhost:15200"
TOP_K = 5
OUTPUT_DIR = "/root/semantic/eval_baseline"

# 替换为你的真实查询（不要把私人内容提交进仓库）
QUERIES = [
    "某次 bug 修复记录",
    "某个技术方案的讨论",
    "某次系统配置变更",
]


def query_hybrid(query: str, top_k: int = TOP_K) -> list[dict]:
    encoded_q = urllib.parse.quote(query)
    url = f"{BASE_URL}/hybrid?q={encoded_q}&top_k={top_k}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("results", [])


def run_eval() -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    queries_out = []

    for q in QUERIES:
        try:
            raw = query_hybrid(q)
        except Exception as e:
            print(f"  [ERROR] query failed: {e}", file=sys.stderr)
            raw = []

        results = []
        for rank, item in enumerate(raw, start=1):
            results.append({
                "rank": rank,
                "source": item.get("source", "?"),
                "score": round(float(item.get("score", 0)), 2),
                "text_preview": item.get("text", "")[:80],
            })
        queries_out.append({"query": q, "results": results})

    return {"timestamp": timestamp, "queries": queries_out}


def print_summary(data: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  RAG Eval — {data['timestamp']}")
    print(f"{'='*70}")
    for entry in data["queries"]:
        print(f"\n▸ {entry['query']}")
        if not entry["results"]:
            print("    (no results)")
            continue
        for r in entry["results"]:
            src   = r["source"].ljust(14)
            score = f"{r['score']:.2f}"
            preview = r["text_preview"].replace("\n", " ")
            print(f"  [{r['rank']}] {src} {score}  {preview}")
    print(f"\n{'='*70}\n")


def diff_baselines(current: dict, previous: dict) -> None:
    prev_map = {e["query"]: e["results"] for e in previous.get("queries", [])}

    changed = []
    for entry in current["queries"]:
        q = entry["query"]
        cur_results = entry["results"]
        prev_results = prev_map.get(q, [])

        reasons = []

        # Different top result
        cur_top = cur_results[0]["source"] if cur_results else None
        prev_top = prev_results[0]["source"] if prev_results else None
        if cur_top != prev_top:
            reasons.append(f"top source changed: {prev_top} → {cur_top}")

        # Source list changed
        cur_sources = [r["source"] for r in cur_results]
        prev_sources = [r["source"] for r in prev_results]
        if cur_sources != prev_sources:
            reasons.append(f"sources: {prev_sources} → {cur_sources}")

        # Large score shifts
        for cr in cur_results:
            for pr in prev_results:
                if cr["rank"] == pr["rank"]:
                    delta = abs(cr["score"] - pr["score"])
                    if delta > 0.05:
                        reasons.append(
                            f"rank {cr['rank']} score {pr['score']:.2f}→{cr['score']:.2f} (Δ{delta:.2f})"
                        )

        if reasons:
            changed.append((q, reasons))

    print(f"\n{'='*70}")
    print(f"  DIFF  {previous['timestamp']}  →  {current['timestamp']}")
    print(f"{'='*70}")
    if not changed:
        print("  No significant changes detected.")
    else:
        for q, reasons in changed:
            print(f"\n▸ {q}")
            for r in reasons:
                print(f"    • {r}")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid RAG search.")
    parser.add_argument(
        "--diff",
        metavar="BASELINE_FILE",
        help="Compare current run against a previous baseline JSON file.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running evaluation queries…")
    data = run_eval()

    # Save
    fname = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")

    print_summary(data)

    if args.diff:
        with open(args.diff, encoding="utf-8") as f:
            previous = json.load(f)
        diff_baselines(data, previous)


if __name__ == "__main__":
    main()
