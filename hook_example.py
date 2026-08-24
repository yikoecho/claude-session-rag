"""
hook_example.py — 接入示例

展示如何在你现有的 UserPromptSubmit hook 中集成 JSONL 语义检索。
这不是可直接运行的代码，是给你的 Claude Code session 看的集成参考。
根据你实际的 hook 框架调整。
"""

from search import should_trigger_search, semantic_search, format_context


def on_user_prompt_submit(user_message: str, context: dict) -> dict:
    """
    你现有 hook 的伪代码结构，加入语义检索层。
    
    三层检索:
      ① 关键词触发 — should_trigger_search 判断
      ② 双路检索   — breath 搜 OB + semantic_search 搜 JSONL
      ③ 注入上下文 — 合并结果塞 additionalContext
    """

    additional_context_parts = []

    # ─── 层① 关键词触发判断 ───
    if should_trigger_search(user_message):

        # ─── 层② 双路检索 ───
        
        # 路线A: 你现有的 OB breath 搜索（保持不变）
        ob_results = breath_search(user_message)  # 你已有的函数
        if ob_results:
            additional_context_parts.append(ob_results)

        # 路线B: JSONL 语义检索（新增）
        jsonl_results = semantic_search(user_message, top_k=3)
        if jsonl_results:
            formatted = format_context(jsonl_results)
            additional_context_parts.append(formatted)

    # ─── 层③ 注入上下文 ───
    if additional_context_parts:
        context["additionalContext"] = (
            context.get("additionalContext", "")
            + "\n"
            + "\n".join(additional_context_parts)
        )

    return context


# ─── 增量更新建议 ───
# 
# 方案1: cron 定时跑
#   在 VPS 上加一条 crontab:
#   0 4 * * * cd /path/to/semantic_memory && python build_index.py /path/to/logs/*.jsonl
#   每天凌晨4点增量更新，build_index.py 自动跳过已处理的 chunk
#
# 方案2: 对话结束时触发
#   在 session 结束 / compact 的 hook 里调用:
#   import subprocess
#   subprocess.Popen(["python", "build_index.py", latest_jsonl_path])
#
# 方案3: 实时追加（如果 JSONL 是追加写入的）
#   用 watchdog 监听文件变化，有新行就解析+embedding+写入
#   内存开销很小但需要额外一个常驻进程
