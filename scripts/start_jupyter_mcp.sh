#!/usr/bin/env bash
# Start JupyterLab from the project's .venv for the Datalayer Jupyter MCP server.
#
# The MCP server is registered in Claude Code (`claude mcp list` -> "jupyter") and
# connects to THIS server over Jupyter's real-time-collaboration (RTC) protocol to
# list / read / run / edit notebooks. Notebook code therefore executes in the same
# .venv used by the news-clustering project.
#
# Token is persisted in .jupyter_mcp_token so restarts reuse the same value the
# Claude Code MCP config was registered with. Re-run this script anytime to restart.
set -euo pipefail

PROJ="/Users/alex/Projcts/news-clustering"
cd "$PROJ"

TOKEN_FILE="$PROJ/.jupyter_mcp_token"
if [[ ! -f "$TOKEN_FILE" ]]; then
  .venv/bin/python -c "import secrets; print(secrets.token_hex(24))" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

PORT="${JUPYTER_PORT:-8888}"

exec .venv/bin/jupyter lab \
  --no-browser \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.port="$PORT" \
  --IdentityProvider.token="$TOKEN" \
  --ServerApp.root_dir="$PROJ"
