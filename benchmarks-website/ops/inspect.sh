#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Run a read-only SQL query against the live DuckDB without stopping
# the server. Calls /api/admin/sql and prints the duckdb-cli-style
# table.
#
# Usage:
#   ./inspect.sh "SELECT COUNT(*) FROM commits;"
#   echo "PRAGMA table_info('commits');" | ./inspect.sh
#   ./inspect.sh -j "SELECT * FROM compression_sizes LIMIT 3"   # raw json
#
# The server allows SELECT, WITH, PRAGMA, SHOW, DESCRIBE, EXPLAIN.
# Anything else is rejected with 403 by the server (so a typo'd UPDATE
# can't run).

set -euo pipefail

ENV_FILE="${ENV_FILE:-/etc/vortex-bench.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: missing ${ENV_FILE}" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
: "${ADMIN_BEARER_TOKEN:?ADMIN_BEARER_TOKEN must be set in ${ENV_FILE}}"
# Admin SQL lives on the loopback-only admin listener; the public bind
# (`SERVER_URL`) does not match `/api/admin/*` at all.
: "${ADMIN_URL:=http://127.0.0.1:3001}"

format=table
if [ "${1:-}" = "-j" ] || [ "${1:-}" = "--json" ]; then
    format=json
    shift
fi

if [ -n "${1:-}" ]; then
    sql="$1"
else
    sql="$(cat)"
fi

# Build the JSON body with a here-doc so quoting in the SQL is a non-issue.
body=$(jq -nc --arg sql "$sql" '{sql: $sql}' 2>/dev/null) || {
    # Fallback when jq isn't installed: hand-escape with python or printf.
    if command -v python3 >/dev/null; then
        body=$(python3 -c 'import json,sys; print(json.dumps({"sql": sys.argv[1]}))' "$sql")
    else
        echo "ERROR: install jq or python3 to call /api/admin/sql safely" >&2
        exit 2
    fi
}

curl -fsS \
    -X POST \
    -H "Authorization: Bearer ${ADMIN_BEARER_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-binary "$body" \
    "${ADMIN_URL}/api/admin/sql?format=${format}"
echo
