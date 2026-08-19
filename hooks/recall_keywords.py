#!/usr/bin/env python3
"""
recall_keywords.py — Extract search keywords from a user prompt.

Reads a prompt from stdin (up to 200 chars). Outputs two lines:
  Line 1: pipe-separated keywords suitable for a keyword/BM25 search
  Line 2: a retrospective query string (may be empty) for temporal lookups

Usage:
    echo "你还记得上次我们聊的那件事吗" | python3 recall_keywords.py

Requires: jieba  (pip install jieba)
"""

import sys
import jieba
jieba.setLogLevel(20)

STOPWORDS = {
    "好了", "我", "在", "的", "是", "了", "你", "他", "她", "它",
    "这", "那", "就", "都", "也", "还", "啊", "吧", "呢", "吗",
    "嗯", "哦", "然后", "但是", "因为", "所以", "可以", "一个",
    "什么", "怎么", "这个", "那个",
}

# Trigger words that indicate the user is asking about the past,
# mapped to a human-readable retrospective query sent to the memory store.
RETRO_MAP = {
    "记得":   "过去的事",
    "以前":   "以前",
    "当时":   "当时",
    "前任":   "前任",
    "妈妈":   "妈妈 家庭",
    "家庭":   "家庭",
    "童年":   "童年 小时候",
    "小时候": "小时候 童年",
    "爸爸":   "爸爸 家庭",
    "父亲":   "父亲 家庭",
    "母亲":   "母亲 妈妈",
    "上次":   "上次",
    "那时":   "那时候",
}

text = sys.stdin.read().strip()[:200]
words = [w for w in jieba.lcut(text) if w not in STOPWORDS and len(w) >= 2]
keywords = words[:5]

retro_query = ""
for trigger, ob_query in RETRO_MAP.items():
    if trigger in text:
        retro_query = ob_query
        break

print("|".join(keywords))
print(retro_query)
