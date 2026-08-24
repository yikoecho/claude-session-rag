#!/usr/bin/env python3
"""
notice_filter.py — Rule-based notice layer for memory recall gating.

Reads a prompt from stdin. Prints "SKIP" if the message is too trivial
to warrant a memory search (short acks, pure punctuation, simple greetings).
Prints nothing (exits 0) if recall should proceed.

Usage:
    echo "some user message" | python3 notice_filter.py
"""

import sys
import re

# ── Messages to skip outright (exact match, case-insensitive) ──
SKIP_EXACT = {
    # Acknowledgements
    "好", "好的", "嗯", "嗯嗯", "对", "对的", "是的", "是", "行", "行的",
    "ok", "OK", "Ok", "收到", "明白", "了解", "知道了", "好吧", "可以",
    "哦", "噢", "啊", "哈", "哈哈", "哈哈哈", "廻廻", "呜呜",
    # Simple greetings
    "你好", "早", "早安", "晚安", "午安", "嗨", "hi", "hello",
    # Simple farewells / thanks
    "谢谢", "感谢", "辛苦了", "再见", "拜拜", "bye",
    "谢谢你", "多谢", "thx", "thanks",
    # Continuation prompts
    "继续", "接着说", "然后呢", "下一个", "go on", "continue",
}

# ── Patterns that signal a skip (pure filler / ack sentences) ──
SKIP_PATTERNS = [
    r"^(好的?|嗯+|对+|是的?|行|哦+|啊+|哈+)[,.，。！!~？?]*$",
    r"^(没事|没关系|无所谓|随便|都行|算了)[,.，。！!]*$",
    r"^[.。，,!！?？~…]+$",                   # pure punctuation
    r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+$",  # pure emoji
]

# ── Patterns that warrant a memory search ──
SIGNAL_PATTERNS = [
    # Retrospective / reference signals
    r"(上次|之前|以前|那时|那天|那个|记得|还记得|提到过|说过|聊过)",
    r"(第一次|最早|最初|一开始|刚开始)",
    # Specific time references
    r"\d{1,2}月|\d{1,2}号|\d{4}年|昨天|前天|上周|上个月|去年",
    r"(周[一二三四五六日天]|星期[一二三四五六日天])",
    # Name / entity signals
    r"(叫|叫做|名字是|姓|认识).{1,6}",
    # Contrast / revelation markers
    r"(突然|忽然|其实|说实话|不过|但是|然而|可是).{4,}",
    # Questions that likely need context
    r"(什么时候|怎么回事|为什么|怎么了|发生了什么|哪里|哪个|谁说的)",
    # Technical identifiers / filenames
    r"[a-zA-Z_][a-zA-Z0-9_.-]{3,}",
    r"[一-鿿]{2,4}(项目|系统|功能|模块|文件|工具|脚本|服务)",
]

# ── Emotion signals (may indicate an important moment) ──
EMOTION_PATTERNS = [
    r"[！!]{2,}",
    r"(生气|难过|开心|高兴|伤心|害怕|担心|焦虑|烦|累|崩溃)",
    r"(喜欢|讨厌|爱|恨|想你|想念|思念|舍不得)",
    r"(对不起|抱歉|sorry|道歉|原谅)",
]


def should_recall(message: str) -> bool:
    """Return True if this message warrants a memory search."""
    text = message.strip()

    if not text:
        return False

    if text.lower() in {s.lower() for s in SKIP_EXACT}:
        return False

    if len(text) <= 6:
        return False

    for pat in SKIP_PATTERNS:
        if re.match(pat, text):
            return False

    # Short messages (7–15 chars): only search if a signal is present
    if len(text) <= 15:
        for pat in SIGNAL_PATTERNS + EMOTION_PATTERNS:
            if re.search(pat, text):
                return True
        return False

    # Long messages (30+ chars): almost always worth searching
    if len(text) > 30:
        return True

    # Medium messages (16–30 chars): search if any signal present;
    # otherwise still search (recall layer will filter irrelevant hits)
    for pat in SIGNAL_PATTERNS + EMOTION_PATTERNS:
        if re.search(pat, text):
            return True

    return True


if __name__ == "__main__":
    msg = sys.stdin.read().strip()
    if not should_recall(msg):
        print("SKIP")
