#!/usr/bin/env python3
"""
eval_recall.py — 直接调用 memory_recall.sh hook（FORCE_RECALL=1），
从 /tmp/recall_eval_result.json 读结构化 verdict/reason/scores。

用法：
  python eval_recall.py                    # recall + no_record 组，FORCE_RECALL
  python eval_recall.py --group=recall     # 只跑 recall 组
  python eval_recall.py --group=no_record  # 只跑 no_record 组
  python eval_recall.py --with-filter      # 走正常路径（filter 生效，SKIP 单独统计）
"""
import json, subprocess, time, sys, os
from pathlib import Path
from datetime import datetime

HOOK = "/root/.claude/hooks/memory_recall.sh"
EVAL_RESULT_FILE = Path("/tmp/recall_eval_result.json")
EVAL_DIR = Path("/root/semantic/eval_baseline")
QUERIES_FILE = EVAL_DIR / "eval_queries.json"

WITH_FILTER = "--with-filter" in sys.argv
GROUP_FILTER = None
for a in sys.argv[1:]:
    if a.startswith("--group="):
        GROUP_FILTER = a.split("=", 1)[1]

def make_xml(query: str) -> str:
    return (f'<channel source="plugin:telegram:telegram" chat_id="8536346553" '
            f'message_id="99999" user="8536346553" user_id="8536346553" '
            f'ts="2026-08-25T00:00:00.000Z">{query}</channel>')

def run_query(query: str) -> dict:
    t0 = time.time()
    prompt = make_xml(query)
    hook_input = json.dumps({"prompt": prompt})

    env = os.environ.copy()
    if not WITH_FILTER:
        env["FORCE_RECALL"] = "1"

    # 清除上一次的 eval result
    EVAL_RESULT_FILE.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            ["bash", HOOK],
            input=hook_input,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"query": query, "verdict": "timeout", "reason": "hook timeout",
                "candidates_count": 0, "top_scores": {}, "injected_lines": 0,
                "filter_result": "bypassed", "elapsed_s": 30}
    except Exception as e:
        return {"query": query, "verdict": "error", "reason": str(e),
                "candidates_count": 0, "top_scores": {}, "injected_lines": 0,
                "filter_result": "bypassed", "elapsed_s": round(time.time()-t0, 2)}

    elapsed = round(time.time() - t0, 2)

    # 解析 hook stdout（注入内容）
    stdout = result.stdout.strip()
    injected = ""
    filter_result = "bypassed" if not WITH_FILTER else "unknown"
    if stdout:
        try:
            d = json.loads(stdout)
            injected = d.get("hookSpecificOutput", {}).get("additionalContext", "")
            # hook 无输出 = SKIP（filter 模式下）
        except Exception:
            injected = stdout[:500]

    # 如果 hook 直接 exit 0 无输出，可能是 SKIP 或 no candidates
    if not stdout and WITH_FILTER:
        filter_result = "SKIP"

    injected_lines = len([l for l in injected.splitlines() if l.strip()]) if injected else 0

    # 读结构化 eval 结果（仅 FORCE_RECALL 模式下写）
    eval_data = {}
    if EVAL_RESULT_FILE.exists():
        try:
            eval_data = json.loads(EVAL_RESULT_FILE.read_text())
        except Exception:
            pass

    verdict = eval_data.get("verdict", "none" if not injected else "partial")
    reason = eval_data.get("reason", "(no eval result file)")
    candidates_count = eval_data.get("candidates_count", 0)
    top_scores = eval_data.get("top_scores", {})
    candidates_detail = eval_data.get("candidates_detail", [])

    # WITH_FILTER + no stdout = SKIP（filter 拦截了）
    if WITH_FILTER and not stdout and filter_result == "SKIP":
        verdict = "skipped"
        reason = "notice_filter SKIP"
        candidates_count = 0

    return {
        "query": query,
        "filter_result": filter_result,
        "verdict": verdict,
        "reason": reason,
        "candidates_count": candidates_count,
        "candidates_detail": candidates_detail,
        "top_scores": top_scores,
        "injected_lines": injected_lines,
        "elapsed_s": elapsed,
    }

# ── 加载 eval_queries.json ──
spec = json.loads(QUERIES_FILE.read_text())
groups = spec["groups"]

results_by_group = {}
for gname, gdata in groups.items():
    if GROUP_FILTER and gname != GROUP_FILTER:
        continue
    queries = gdata["queries"]
    mode = "正常路径（filter 生效）" if WITH_FILTER else "FORCE_RECALL（绕过 filter）"
    print(f"\n[{gname}组] {len(queries)}条 — {mode}")
    results = []
    for i, q in enumerate(queries):
        qtext = q["query"]
        print(f"  [{i+1}/{len(queries)}] {qtext[:40]}...", end=" ", flush=True)
        r = run_query(qtext)
        r["id"] = q.get("id", "")
        r["difficulty"] = q.get("difficulty", "")
        r["ground_truth"] = q.get("ground_truth", "")
        r["note"] = q.get("note", "")
        results.append(r)
        v = r["verdict"]
        print(f"{r['elapsed_s']}s | cands={r['candidates_count']} verdict={v}")
        time.sleep(0.3)
    results_by_group[gname] = results

# ── 统计 ──
print("\n── 指标 ──")
for gname, results in results_by_group.items():
    total = len(results)
    if gname == "recall":
        hits = sum(1 for r in results if r["verdict"] in ("sufficient", "partial"))
        skips = sum(1 for r in results if r["verdict"] == "skipped")
        denom = total - skips
        rate = f"{hits/(denom)*100:.0f}%" if denom else "N/A"
        print(f"recall组: 命中率 {hits}/{denom} (排除SKIP {skips}条) = {rate}")
    elif gname == "no_record":
        nones = sum(1 for r in results if r["verdict"] == "none")
        skips = sum(1 for r in results if r["verdict"] == "skipped")
        halluc = sum(1 for r in results if r["verdict"] in ("sufficient", "partial"))
        denom = total - skips
        print(f"no_record组: none率 {nones}/{denom} (排除SKIP {skips}条) | 编造 {halluc}条")
        if halluc > 0:
            print("  ⚠️  编造题：")
            for r in results:
                if r["verdict"] in ("sufficient", "partial"):
                    print(f"    [{r['id']}] {r['query'][:40]} → reason: {r['reason'][:80]}")

# ── 词汇鸿沟对照（recall组专属）──
# 词汇鸿沟对照组：(自然说法id, 精确说法id, 标签)
# 替换为你的真实对照组
vocab_gap_pairs: list[tuple[str, str, str]] = [
    # ("r01", "r12", "自然说法 vs 精确说法"),
]
vocab_gap_analysis = {}
if "recall" in results_by_group:
    r_map = {r["id"]: r for r in results_by_group["recall"]}
    verdict_rank = {"sufficient": 2, "partial": 1, "none": 0, "skipped": -1, "agent_failed": -1}
    for id_a, id_b, label in vocab_gap_pairs:
        ra, rb = r_map.get(id_a), r_map.get(id_b)
        if ra and rb:
            score_a = verdict_rank.get(ra["verdict"], 0)
            score_b = verdict_rank.get(rb["verdict"], 0)
            vocab_gap_analysis[label] = {
                "natural": {"id": id_a, "query": ra["query"], "verdict": ra["verdict"],
                            "injected_lines": ra["injected_lines"], "top_scores": ra.get("top_scores", {})},
                "precise": {"id": id_b, "query": rb["query"], "verdict": rb["verdict"],
                            "injected_lines": rb["injected_lines"], "top_scores": rb.get("top_scores", {})},
                "verdict_gap": score_a - score_b,
                "note": "gap=0 说明词汇鸿沟已消除；gap>0 自然说法更好；gap<0 精确说法更好"
            }
            print(f"\n词汇鸿沟 [{label}]:")
            print(f"  自然({id_a}): verdict={ra['verdict']} injected={ra['injected_lines']}行 top={ra.get('top_scores',{})}")
            print(f"  精确({id_b}): verdict={rb['verdict']} injected={rb['injected_lines']}行 top={rb.get('top_scores',{})}")
            print(f"  gap={score_a - score_b}")

# ── 存文件 ──
suffix = "filter" if WITH_FILTER else "force"
grp = f"_{GROUP_FILTER}" if GROUP_FILTER else ""
out_file = EVAL_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}{grp}.json"
out = {
    "ts": datetime.now().isoformat(),
    "mode": "with_filter" if WITH_FILTER else "force_recall",
    "filter_bypassed": not WITH_FILTER,
    "eval_method": "hook_direct",  # 标记这是真实 hook 路径
    "vocab_gap_analysis": vocab_gap_analysis,
    "results": results_by_group,
}
out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n已存: {out_file}")
