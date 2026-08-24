#!/usr/bin/env python3
"""
实体抽取增强：从 session_index.jsonl 的 key+text 中提取命名实体，补进 entities 字段。
支持断点续跑（已有 entities 字段的条目跳过）。
用法：
  python3 enrich_entities.py              # 处理全部缺失的条目
  python3 enrich_entities.py --limit 50   # 只处理前50条缺失的
  python3 enrich_entities.py --new-only   # 只处理最近N天新增的（归档时用）
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

# 把 utils/ 加到 path，确保能 import utils.config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import LLM_BASE_URL, LLM_API_KEY, ENRICH_MODEL, RECALL_ENABLED

INDEX_FILE = os.environ.get(
    "SESSION_INDEX_PATH",
    os.path.expanduser("~/.claude/session_index.jsonl"),
)

if not RECALL_ENABLED:
    print("[enrich_entities] LLM backend 未启用（LLM_BACKEND=none），跳过实体抽取。")
    print("设置 LLM_BACKEND=ollama 或配置 OPENROUTER_API_KEY 后重试。")
    sys.exit(0)

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
MODEL = ENRICH_MODEL

PROMPT_TEMPLATE = """从以下文本中提取具体的命名实体，合并成一个扁平JSON字符串数组返回，不要对象，不要通用词，不要解释。只返回数组本身，例如：["solitude_monitor", "save_data()", "flag"]

文本：
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
        # 优先匹配顶层数组
        m = re.search(r'\[([^\[\]]*)\]', output, re.DOTALL)
        if m:
            entities = json.loads(m.group())
            if isinstance(entities, list):
                return [e for e in entities if isinstance(e, str) and e.strip()]
        # fallback：如果返回对象，收集所有字符串值
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

    print(f"待处理条目: {total}，并发数: 10")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--new-only", action="store_true", help="只处理最近2天的新条目（归档时用）")
    args = parser.parse_args()

    with open(INDEX_FILE, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    already = sum(1 for e in entries if "entities" in e)
    print(f"总条目: {len(entries)}, 已有entities: {already}, 缺失: {len(entries) - already}")

    new_only_days = 2 if args.new_only else None
    entries, updated = enrich_entries(entries, limit=args.limit, new_only_days=new_only_days)

    if updated > 0:
        atomic_write(INDEX_FILE, entries)
        print(f"\n完成，更新 {updated} 条，已写回 {INDEX_FILE}")
    else:
        print("无需更新。")


if __name__ == "__main__":
    main()
