#!/bin/bash
# UserPromptSubmit hook: 搜索相关记忆并注入上下文
#
# 改动说明（对比原版）：
# 1. 原来的 10字过滤 → notice_filter.py 规则层（更精准的信号检测）
# 2. 新增 recall agent：搜索结果送 /recall 端点，由 Haiku 筛选
# 3. 其余逻辑不变

input=$(cat)
prompt=$(echo "$input" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt',''))" 2>/dev/null)

if [ -z "$prompt" ]; then
  exit 0
fi

# ── Configurable script paths (override via env vars or fall back to script dir) ──
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NOTICE_FILTER_SCRIPT="${NOTICE_FILTER_SCRIPT:-${SCRIPT_DIR}/notice_filter.py}"
RECALL_KEYWORDS_SCRIPT="${RECALL_KEYWORDS_SCRIPT:-${SCRIPT_DIR}/recall_keywords.py}"

# Fall back to legacy absolute paths if not found relative to script dir
[ -f "$NOTICE_FILTER_SCRIPT" ]   || NOTICE_FILTER_SCRIPT="/root/notice_filter.py"
[ -f "$RECALL_KEYWORDS_SCRIPT" ] || RECALL_KEYWORDS_SCRIPT="/root/recall_keywords.py"

# ── Layer 1: Notice 规则层（替代原来的 10字过滤）──
notice_result=$(echo "$prompt" | python3 "$NOTICE_FILTER_SCRIPT" 2>/dev/null)
if [ "$notice_result" = "SKIP" ]; then
  exit 0
fi

py_out=$(echo "$prompt" | python3 "$RECALL_KEYWORDS_SCRIPT" 2>/dev/null)
search_terms=$(echo "$py_out" | sed -n '1p')
retro_query=$(echo "$py_out" | sed -n '2p')

if [ -z "$search_terms" ]; then
  exit 0
fi

MEMORY_DIR="/root/.claude/projects/-root/memory"
SESSION_FILE="/root/.claude/last_session.md"
TECH_FILE="/root/.claude/tech_todo.md"
ARCHIVE_FILE="/root/.claude/session_archive.md"
ERROR_LOG="/root/.claude/memory_recall_error.log"

TMPDIR_RECALL=$(mktemp -d)
trap "rm -rf $TMPDIR_RECALL" EXIT

rg_search() {
  local pattern="$1"; shift
  local rg_output rg_exit
  rg_output=$(rg -i "$@" "$pattern" 2>>"$ERROR_LOG")
  rg_exit=$?
  [ $rg_exit -eq 0 ] && echo "$rg_output"
}

# ── Layer 2: 搜索层（不变）──
t_search_start=$(date +%s%3N)
results=""

if [ -f "$SESSION_FILE" ]; then
  hits=$(rg_search "$search_terms" -m 3 --no-filename "$SESSION_FILE" | head -3)
  [ -n "$hits" ] && results="$results$hits\n"
fi

if [ -f "$TECH_FILE" ]; then
  hits=$(rg_search "$search_terms" -m 2 --no-filename "$TECH_FILE" | head -2)
  [ -n "$hits" ] && results="$results$hits\n"
fi

if [ -d "$MEMORY_DIR" ]; then
  hits=$(rg_search "$search_terms" -m 5 --no-filename -g '!MEMORY.md' "$MEMORY_DIR" | head -5)
  [ -n "$hits" ] && results="$results$hits\n"
fi

# session_index.jsonl grep（高置信度，直接注入）
JSONL_FILE="/root/.claude/session_index.jsonl"
jsonl_hits=""
if [ -f "$JSONL_FILE" ]; then
  while IFS= read -r line; do
    date=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('date',''))" 2>/dev/null)
    key=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('key',''))" 2>/dev/null)
    text=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('text',''))" 2>/dev/null)
    jsonl_hits="$jsonl_hits[jsonl记忆] ${date} ${key}：${text}\n"
  done < <(grep -i "$search_terms" "$JSONL_FILE" 2>/dev/null | head -3)
fi

# session_archive.md + session_index.jsonl BM25 搜索（替代 grep）
archive_grep_hits=""
bm25_result=$(curl -sG "http://127.0.0.1:15200/bm25" \
  --data-urlencode "q=$search_terms" \
  --data-urlencode "top_k=5" 2>/dev/null)
if [ -n "$bm25_result" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && archive_grep_hits="$archive_grep_hits[archive_bm25] ${line:0:120}\n"
  done <<< "$bm25_result"
fi

# ── Hybrid 检索 + OB breath 并行 ──
ob_query="${retro_query:-$prompt}"

curl -s -G "http://127.0.0.1:15200/hybrid" \
  --data-urlencode "q=$prompt" \
  --data-urlencode "top_k=5" \
  > "$TMPDIR_RECALL/hybrid" 2>/dev/null &
pid_hybrid=$!

BREATH_SCRIPT="${BREATH_SEARCH_SCRIPT:-${SCRIPT_DIR}/breath_search.py}"
[ -f "$BREATH_SCRIPT" ] || BREATH_SCRIPT="/root/breath_search.py"

if [ -f "$BREATH_SCRIPT" ]; then
  python3 "$BREATH_SCRIPT" "$ob_query" > "$TMPDIR_RECALL/ob" 2>/dev/null &
  pid_ob=$!
  wait $pid_hybrid $pid_ob
  ob_hits=$(cat "$TMPDIR_RECALL/ob")
else
  wait $pid_hybrid
  ob_hits=""
fi

# 解析 hybrid JSON → 文本行
hybrid_hits=$(python3 -c "
import sys, json
try:
    data = json.load(open('$TMPDIR_RECALL/hybrid'))
    for item in data.get('results', []):
        print(item['text'][:120])
except Exception:
    pass
" 2>/dev/null)

[ -n "$hybrid_hits" ] && results="$results[archive_hybrid]\n$hybrid_hits\n"
[ -n "$ob_hits" ]     && results="$results[OB记忆]\n$ob_hits\n"
t_search_end=$(date +%s%3N)
echo "[recall_timing] search层: $((t_search_end - t_search_start))ms" >> "$ERROR_LOG"

if [ -z "$results" ]; then
  exit 0
fi

# 截断每行到100字，最多8条（普通候选，送 Haiku 过滤）
formatted=$(echo -e "$results" | grep -v '^$' | head -8 | while IFS= read -r line; do
  echo "${line:0:100}"
done)

# archive_bm25 命中条目已合并进 hybrid，archive_direct 统一清空
archive_direct=""

if [ -z "$formatted" ]; then
  exit 0
fi

# ── Layer 3: Recall Agent（新增）──
t_recall_start=$(date +%s%3N)
# 把候选列表发给 /recall，由 Haiku 判断哪些真正相关
recall_result=$(echo "$formatted" | PROMPT_TEXT="$prompt" python3 -c "
import sys, json, urllib.request, os

lines = [l.strip() for l in sys.stdin if l.strip()]
prompt = os.environ['PROMPT_TEXT']

# 构造请求
data = json.dumps({'prompt': prompt, 'candidates': lines}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:15200/recall',
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = json.loads(resp.read())
        filtered = body.get('results', lines)
        # 如果 agent 返回空列表 → 不注入任何记忆
        if not filtered:
            sys.exit(0)
        for line in filtered:
            print(line)
except Exception as e:
    # agent 调用失败 → 透传原结果
    import sys as s
    print(f'[recall_agent_error] {e}', file=s.stderr)
    for line in lines:
        print(line)
" 2>>"$ERROR_LOG")
t_recall_end=$(date +%s%3N)
echo "[recall_timing] recall_agent: $((t_recall_end - t_recall_start))ms | total: $((t_recall_end - t_search_start))ms" >> "$ERROR_LOG"

# 如果两路都空，不注入
if [ -z "$recall_result" ] && [ -z "$jsonl_hits" ]; then
  exit 0
fi

final_content=""
[ -n "$recall_result" ] && final_content="$recall_result"
[ -n "$jsonl_hits" ] && final_content="$final_content\n$(echo -e "$jsonl_hits" | grep -v '^$')"

content="[记忆召回]\n${final_content}"

python3 -c "
import sys, json
content = sys.stdin.read()
print(json.dumps({'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit', 'additionalContext': content}}))
" <<< "$content"
