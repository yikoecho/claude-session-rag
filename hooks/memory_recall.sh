#!/bin/bash
# Claude Code UserPromptSubmit hook — semantic memory recall
#
# Place this file at .claude/hooks/UserPromptSubmit/memory_recall.sh
# and register it in .claude/settings.json under hooks.UserPromptSubmit
#
# Architecture:
#   Layer 1: Quick keyword extraction (recall_keywords.py)
#   Layer 2: Parallel search — LanceDB semantic + optional grep fallback
#   Layer 3: Optional Haiku filter via /recall endpoint
#
# The hook injects relevant memories as additionalContext into the Claude Code
# prompt via the hookSpecificOutput mechanism.
#
# Configuration (adjust paths to match your setup):
SEARCH_SERVER="http://127.0.0.1:15200"
SESSION_FILE="${SESSION_FILE:-./data/last_session.md}"
MEMORY_DIR="${MEMORY_DIR:-}"   # optional: path to Claude Code memory .md files
ERROR_LOG="${ERROR_LOG:-/tmp/memory_recall_error.log}"

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('prompt', ''))
" 2>/dev/null)

if [ -z "$prompt" ]; then
  exit 0
fi

# ── Layer 1: Keyword extraction ──
# recall_keywords.py reads stdin (the prompt) and outputs:
#   line 1: search_terms (for grep)
#   line 2: retro_query (for semantic search, if different)
py_out=$(echo "$prompt" | python3 "$(dirname "$0")/recall_keywords.py" 2>/dev/null)
search_terms=$(echo "$py_out" | sed -n '1p')
retro_query=$(echo "$py_out" | sed -n '2p')

if [ -z "$search_terms" ]; then
  exit 0
fi

TMPDIR_RECALL=$(mktemp -d)
trap "rm -rf $TMPDIR_RECALL" EXIT

# ── Layer 2: Search ──
results=""

# Grep session file if available
if [ -n "$SESSION_FILE" ] && [ -f "$SESSION_FILE" ]; then
  hits=$(rg -i -m 3 --no-filename "$search_terms" "$SESSION_FILE" 2>/dev/null | head -3) || true
  [ -n "$hits" ] && results="$results$hits\n"
fi

# Grep memory directory if configured
if [ -n "$MEMORY_DIR" ] && [ -d "$MEMORY_DIR" ]; then
  hits=$(rg -i -m 5 --no-filename "$search_terms" "$MEMORY_DIR" 2>/dev/null | head -5) || true
  [ -n "$hits" ] && results="$results$hits\n"
fi

# Semantic search (LanceDB via search_server)
if [ -n "$retro_query" ]; then
  semantic_top_k=5
else
  semantic_top_k=3
fi

curl -s --max-time 8 -G "$SEARCH_SERVER/search" \
  --data-urlencode "q=$prompt" \
  --data-urlencode "top_k=$semantic_top_k" \
  --data-urlencode "threshold=0.45" \
  > "$TMPDIR_RECALL/semantic" 2>/dev/null

semantic_hits=$(cat "$TMPDIR_RECALL/semantic")
[ -n "$semantic_hits" ] && results="$results[session memory]\n$semantic_hits\n"

if [ -z "$results" ]; then
  exit 0
fi

# Truncate candidates: 100 chars per line, max 8 lines
formatted=$(echo -e "$results" | grep -v '^$' | head -8 | while IFS= read -r line; do
  echo "${line:0:100}"
done)

if [ -z "$formatted" ]; then
  exit 0
fi

# ── Layer 3: Haiku filter (optional) ──
# If search_server has OPENROUTER_API_KEY set, /recall filters with Haiku.
# If not, it passes candidates through unchanged.
recall_result=$(echo "$formatted" | PROMPT_TEXT="$prompt" python3 -c "
import sys, json, urllib.request, os

lines = [l.strip() for l in sys.stdin if l.strip()]
prompt = os.environ['PROMPT_TEXT']

data = json.dumps({'prompt': prompt, 'candidates': lines}).encode()
req = urllib.request.Request(
    '${SEARCH_SERVER}/recall',
    data=data,
    headers={'Content-Type': 'application/json'},
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = json.loads(resp.read())
        filtered = body.get('results', lines)
        if not filtered:
            sys.exit(0)
        for line in filtered:
            print(line)
except Exception as e:
    # On failure, pass through original results
    import sys as s
    print(f'[recall_error] {e}', file=s.stderr)
    for line in lines:
        print(line)
" 2>>"$ERROR_LOG")

if [ -z "$recall_result" ]; then
  exit 0
fi

content="[memory recall]\n${recall_result}"

python3 -c "
import sys, json
content = sys.stdin.read()
print(json.dumps({'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit', 'additionalContext': content}}))
" <<< "$content"
