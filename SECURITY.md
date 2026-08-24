# Security Policy

## Threat model

`claude-session-rag` is a **single-user local tool**. The search server (`search_server.py`)
has **no authentication** and is intended to bind to `127.0.0.1` only. Anything that can reach
port `15200` on the host can query your indexed conversation history. Do not expose the port,
and do not run this on a shared/multi-tenant machine without adding your own auth layer.

## Handling secrets

- Never commit your `.env`. It is already in `.gitignore`.
- `EMBEDDING_API_KEY` and `OPENROUTER_API_KEY` are read from the environment / `.env` only.

## Untrusted input

User prompts flow into the search query. The `text LIKE` filter escapes single quotes and the
`grep` fallback uses `--` to prevent option injection. If you extend the query path, keep
user input parameterized — never string-format it into a `.where()` clause or a shell command.

## Reporting

Open a GitHub issue (or private advisory) with reproduction steps.
