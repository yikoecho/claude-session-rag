#!/usr/bin/env python3
"""通过真实 Telegram 路径（memory_recall.sh hook）重跑 eval 查询。
   每条查询包成 Telegram channel XML 格式传给 hook，捕获注入结果。

用法：
  python eval_telegram_path.py                  # 正常路径（notice_filter 生效）
  python eval_telegram_path.py --force-recall   # 绕过 notice_filter，测检索层
"""
import json, subprocess, time, sys, os
from pathlib import Path
from datetime import datetime

HOOK = "/root/.claude/hooks/memory_recall.sh"
EVAL_DIR = Path("/root/semantic/eval_baseline")

FORCE_RECALL = "--force-recall" in sys.argv

suffix = "force" if FORCE_RECALL else "normal"
OUT_FILE = EVAL_DIR / f"telegram_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.json"

# 8 条标准查询 + 3 条自然语言查询
# 替换为你的真实查询（不要把私人内容提交进仓库）
# 建议从 eval_baseline/eval_queries.example.json 加载
QUERIES = [
    "某次 bug 修复记录",
    "某个技术方案的讨论",
    "某次系统配置变更",
]

def make_telegram_prompt(query: str, msg_id: int = 99999) -> str:
    return (
        f'<channel source="plugin:telegram:telegram" chat_id="8536346553" '
        f'message_id="{msg_id}" user="8536346553" user_id="8536346553" '
        f'ts="2026-08-24T14:00:00.000Z">{query}</channel>'
    )

def run_hook(query: str, msg_id: int, force_recall: bool = False) -> dict:
    prompt = make_telegram_prompt(query, msg_id)
    hook_input = json.dumps({"prompt": prompt})
    env = os.environ.copy()
    if force_recall:
        env["FORCE_RECALL"] = "1"
    t0 = time.time()
    try:
        result = subprocess.run(
            ["bash", HOOK],
            input=hook_input,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        elapsed = round(time.time() - t0, 2)
        stdout = result.stdout.strip()
        injected = ""
        if stdout:
            try:
                d = json.loads(stdout)
                injected = d.get("hookSpecificOutput", {}).get("additionalContext", "")
            except Exception:
                injected = stdout[:200]
        return {
            "query": query,
            "input_path": "telegram",
            "filter_bypassed": force_recall,
            "elapsed_s": elapsed,
            "injected": injected,
            "hook_exit": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"query": query, "input_path": "telegram", "filter_bypassed": force_recall,
                "error": "timeout", "elapsed_s": 30}
    except Exception as e:
        return {"query": query, "input_path": "telegram", "filter_bypassed": force_recall,
                "error": str(e)}

mode = "FORCE_RECALL（绕过 notice_filter）" if FORCE_RECALL else "正常路径（notice_filter 生效）"
print(f"跑 {len(QUERIES)} 条查询 — {mode}")
results = []
for i, q in enumerate(QUERIES):
    print(f"  [{i+1}/{len(QUERIES)}] {q[:40]}...", end=" ", flush=True)
    r = run_hook(q, 99000 + i, force_recall=FORCE_RECALL)
    results.append(r)
    injected_preview = (r.get("injected", "") or "")[:80].replace("\n", " ")
    status = "(空)" if not injected_preview else injected_preview
    print(f"{r.get('elapsed_s','?')}s | {status}")
    time.sleep(1)

out = {
    "ts": datetime.now().isoformat(),
    "input_path": "telegram",
    "filter_bypassed": FORCE_RECALL,
    "queries": results,
}
OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n已存: {OUT_FILE}")
