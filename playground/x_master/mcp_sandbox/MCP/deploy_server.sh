#!/usr/bin/env bash
set -euo pipefail

# Default: start EvoMaster-compatible MCP server (Streamable HTTP)
#   http://127.0.0.1:8000/mcp
#
# Optional: also start the legacy REST server (/execute) if you still need it:
#   START_LEGACY_EXECUTE_SERVER=1 bash deploy_server.sh

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [[ "${START_LEGACY_EXECUTE_SERVER:-0}" == "1" ]]; then
  LEGACY_PORT="${LEGACY_PORT:-30008}"
  uvicorn tool_server:app --host "${HOST}" --port "${LEGACY_PORT}" --lifespan on --workers 1 &
  echo "[deploy_server] legacy /execute server started on :${LEGACY_PORT}"
fi

echo "[deploy_server] MCP server starting on ${HOST}:${PORT}/mcp"
python evomaster_mcp_server.py

