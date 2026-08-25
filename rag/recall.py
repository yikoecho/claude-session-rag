"""
rag/recall.py — /recall 端点的 Haiku 筛选逻辑（三档判定版）
"""

import json

import httpx

from utils.config import LLM_BASE_URL, LLM_API_KEY, RECALL_MODEL, RECALL_ENABLED

_http_client = httpx.Client(timeout=10.0)

RECALL_SYSTEM_PROMPT = """你的任务是判断一批档案片段能否回答用户的问题。

这不是相关性判断。话题相关但没有回答问题的片段，不算能回答。

三档判定：
- sufficient：至少有一条片段直接回答了问题
- partial：问题所指的那个具体事件本身在片段中出现过，只是细节不全
- none：片段与问题没有实质关联，或仅共享话题/人物/关键词但指向不同的事

**判断顺序（必须按这个顺序）：**
1. 先问「问题问的这件事，在片段里发生过吗？」
2. 答案是否 → 直接判 none，不再考虑相关度
3. 答案是是 → 再看细节是否完整，完整是 sufficient，不完整是 partial

**partial 的必要条件**：问题所指的那个具体事件本身必须在片段中出现过，只是细节不全。
如果片段里只有相关的人、工具、话题，而那个事件本身从未出现 → none，无论相关度多高。

例（partial）：用户问「某个功能怎么接入VPS的」，片段里有「VPS上接入该功能的记录」但缺少具体步骤——
这件事在档案里出现过，只是细节不全。判 partial。

例（none）：用户问「和某人视频通话」，片段里有「跟某人聊技术架构」——
同一个人出现在档案里，但视频通话这个事件本身从未出现。判 none。

例（none）：用户问「上次提过的那次争吵」，片段里有「家庭相关内容」——
相关话题出现了，但那次争吵这个具体事件没有出现。判 none。

关于 none：
档案里没有相关记录是常态，不是失败。输出 none 是完全可以接受的结果。
不要因为必须给出结果，就从不相关的片段里挑一条最接近的。
宁可判 none，也不要把弱相关的片段当成答案。

硬约束：
- 只做选择，不要合并、改写、补全或总结片段内容
- 片段之间的信息空白不要推测填补
- 严格输出 JSON，不要任何解释性文字

输出格式：
{"verdict": "sufficient" | "partial" | "none", "selected": [片段编号], "reason": "一句话说明为什么是这个判断"}

verdict 为 none 时，selected 为空数组。"""


def recall_agent(prompt: str, candidates: list[dict]) -> dict:
    """
    candidates: list of {"source": str, "text": str, "score": float}
    returns: {"verdict": "sufficient"|"partial"|"none", "selected": [dicts], "reason": str}
    """
    if not RECALL_ENABLED:
        return {"verdict": "unfiltered", "selected": candidates, "reason": "recall未启用"}

    if not candidates:
        return {"verdict": "none", "selected": [], "reason": "无候选片段"}

    # Build user message with source labels
    lines = []
    for i, c in enumerate(candidates):
        source_label = f"[{c.get('source', '')}]" if c.get('source') else ""
        text = c.get('text', '')
        lines.append(f"[{i + 1}] {source_label} {text}")
    candidates_text = "\n".join(lines)

    user_message = f"用户问题：{prompt}\n\n档案片段：\n\n{candidates_text}"

    try:
        resp = _http_client.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RECALL_MODEL,
                "messages": [
                    {"role": "system", "content": RECALL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 400,
                "temperature": 0,
            },
        )
        if resp.status_code != 200:
            print(f"[recall_agent] HTTP {resp.status_code}: {resp.text[:200]}")
            raise ValueError(f"LLM API returned {resp.status_code}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        # Haiku 偶尔在 reason 里插真换行，json.loads 会炸——先折叠
        text_clean = " ".join(text.splitlines())
        try:
            result = json.loads(text_clean)
        except json.JSONDecodeError:
            result = json.loads(text)  # fallback 原始文本

        verdict = result.get("verdict")
        if verdict not in ("sufficient", "partial", "none"):
            raise ValueError(f"unexpected verdict: {verdict!r}")
        selected_indices = result.get("selected", [])
        reason = result.get("reason", "")

        # verdict/reason 不一致检测：reason 说"没有记录"但 verdict 不是 none
        _no_record_signals = ("没有记录", "未找到", "不存在", "并无", "没有直接记录")
        if verdict != "none" and any(s in reason for s in _no_record_signals):
            print(f"[recall_agent] [WARN] verdict/reason不一致: verdict={verdict}, reason={reason}")

        if verdict == "none":
            print(f"[recall_agent] verdict=none ({reason})")
            return {"verdict": "none", "selected": [], "reason": reason}

        # Map 1-based indices back to candidate dicts
        selected = [
            candidates[i - 1]
            for i in selected_indices
            if isinstance(i, int) and 1 <= i <= len(candidates)
        ]

        print(f"[recall_agent] verdict={verdict} {len(candidates)}→{len(selected)} ({reason})")
        return {"verdict": verdict, "selected": selected, "reason": reason}

    except json.JSONDecodeError as e:
        print(f"[recall_agent] JSON解析失败: {e}")
        return {"verdict": "unfiltered", "selected": candidates, "reason": f"recall_agent调用失败: {e}"}
    except Exception as e:
        print(f"[recall_agent] 调用失败: {e}")
        return {"verdict": "unfiltered", "selected": candidates, "reason": f"recall_agent调用失败: {e}"}
