"""
rag/recall.py — /recall 端点的 Haiku 筛选逻辑
"""

import json

import httpx

from utils.config import OPENROUTER_API_KEY, RECALL_MODEL, RECALL_ENABLED

_http_client = httpx.Client(timeout=10.0)

RECALL_PROMPT_TEMPLATE = """你是记忆挑选员。你的工作是从候选记忆中挑出真正和当前对话相关的，或者判断没有相关的。

规则：
1. 只挑真正和当下话题/情绪/语境相关的记忆，最多保留3条
2. "长得像"不等于"该出现"——话题相似但当下不需要的，扔掉
3. 已经在对话里提过的信息，不要重复给
4. 如果候选里没有任何真正相关的，返回空列表——宁可不给，绝不硬塞
5. 不要编造理由来让某条记忆显得相关

当前消息：
{prompt}

候选记忆（编号从0开始）：
{candidates}

返回JSON，不要其他内容：
{{"keep": [0, 2], "reason": "简短说明"}}
或
{{"keep": [], "reason": "没有相关记忆"}}"""


def recall_agent(prompt: str, candidates: list[str]) -> list[str]:
    """调用 OpenRouter Haiku 筛选候选记忆"""
    if not RECALL_ENABLED:
        return candidates

    if not candidates:
        return []

    formatted = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    full_prompt = RECALL_PROMPT_TEMPLATE.format(prompt=prompt, candidates=formatted)

    try:
        resp = _http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RECALL_MODEL,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 150,
                "temperature": 0,
            },
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        keep_indices = result.get("keep", [])

        if not keep_indices:
            print(f"[recall_agent] 返回空（{result.get('reason', '')}）")
            return []

        filtered = [candidates[i] for i in keep_indices if i < len(candidates)]
        print(f"[recall_agent] {len(candidates)}→{len(filtered)} ({result.get('reason', '')})")
        return filtered

    except Exception as e:
        print(f"[recall_agent] 调用失败，透传原结果: {e}")
        return candidates
