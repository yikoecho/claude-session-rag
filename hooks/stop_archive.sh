#!/usr/bin/env bash
# stop_archive.sh — Stop hook: append today's conversation summary to session_archive.md
#
# Install in .claude/settings.json:
#   "hooks": {
#     "Stop": [{ "matcher": "", "hooks": [{"type": "command",
#       "command": "bash /path/to/claude-session-rag/hooks/stop_archive.sh"}] }]
#   }
#
# Required env vars (set in .env or shell):
#   ARCHIVE_FILE   — path to session_archive.md  (default: <repo>/../data/session_archive.md)
#   RECALL_API_KEY — LLM API key for summarization
#   RECALL_BASE_URL / RECALL_MODEL — optional overrides
#
# What it does:
#   1. Reads the last WINDOW_LINES lines from the Claude Code transcript (stdin is the
#      Stop hook payload, but transcript path is provided via CLAUDE_TRANSCRIPT_PATH).
#   2. Calls an LLM to produce a one-paragraph summary of the session so far.
#   3. Appends the summary under a dated header to ARCHIVE_FILE.
#
# If RECALL_API_KEY is not set, the hook exits silently (no-op).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env
for env_file in "$REPO_DIR/.env" "/root/.env"; do
    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        source <(grep -v '^\s*#' "$env_file" | grep '=') 2>/dev/null || true
        set +a
        break
    fi
done

ARCHIVE_FILE="${ARCHIVE_FILE:-$REPO_DIR/data/session_archive.md}"
RECALL_API_KEY="${RECALL_API_KEY:-}"
RECALL_BASE_URL="${RECALL_BASE_URL:-https://openrouter.ai/api/v1}"
RECALL_MODEL="${RECALL_MODEL:-anthropic/claude-haiku-4-5}"
WINDOW_LINES="${WINDOW_LINES:-200}"

[[ -z "$RECALL_API_KEY" ]] && exit 0
[[ ! -f "${CLAUDE_TRANSCRIPT_PATH:-}" ]] && exit 0

TRANSCRIPT=$(tail -n "$WINDOW_LINES" "$CLAUDE_TRANSCRIPT_PATH" 2>/dev/null || true)
[[ -z "$TRANSCRIPT" ]] && exit 0

DATE=$(date '+%Y-%m-%d %H:%M %Z')

SUMMARY=$(python3 - <<PYEOF
import os, json, sys
try:
    import openai
    client = openai.OpenAI(
        api_key="${RECALL_API_KEY}",
        base_url="${RECALL_BASE_URL}",
    )
    transcript = """${TRANSCRIPT}"""
    resp = client.chat.completions.create(
        model="${RECALL_MODEL}",
        messages=[
            {"role": "system", "content": (
                "You are a conversation archivist. "
                "Summarize the following Claude Code session in 3-5 sentences. "
                "Focus on: what was discussed, decisions made, technical changes, "
                "and any unresolved items. Be specific and concrete. "
                "Output only the summary paragraph."
            )},
            {"role": "user", "content": transcript[:8000]},
        ],
        max_tokens=300,
        temperature=0,
    )
    print(resp.choices[0].message.content.strip())
except Exception as e:
    print(f"[stop_archive] summarization failed: {e}", file=sys.stderr)
    sys.exit(0)
PYEOF
) || true

[[ -z "$SUMMARY" ]] && exit 0

mkdir -p "$(dirname "$ARCHIVE_FILE")"
{
    echo ""
    echo "---"
    echo "### $DATE"
    echo ""
    echo "$SUMMARY"
} >> "$ARCHIVE_FILE"

echo "[stop_archive] appended summary to $ARCHIVE_FILE" >&2
