#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Manual v2→v3 migration wrapper. The migration tool needs exclusive
# access to the DB file, so the server is stopped first, the current DB
# is snapshotted to prev-bench.duckdb.<ts> for instant rollback, the
# migrate binary runs, and the server is started back up.
#
# Run from any directory while SSH'd onto the EC2 host. The args are
# passed through verbatim to `cargo run -p vortex-bench-migrate --`, so
# the operator owns the migrator's CLI surface (which has been changing
# while v3 stabilises). The wrapper only handles stop / snapshot prev
# DB / restart.
#
# Examples:
#   /var/lib/vortex-bench/ops/migrate.sh run --output "$VORTEX_BENCH_DB"
#
# (Run as ec2-user is fine — we sudo only for systemctl.)

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
: "${REPO_DIR:?REPO_DIR must be set in ${ENV_FILE}}"
: "${VORTEX_BENCH_DB:?VORTEX_BENCH_DB must be set in ${ENV_FILE}}"
: "${SERVER_URL:=http://127.0.0.1:3000}"

log() { printf '[migrate %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "ERROR: REPO_DIR=${REPO_DIR} is not a git checkout" >&2
    exit 2
fi

# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true

log "stopping vortex-bench-server"
sudo /bin/systemctl stop vortex-bench-server

# Snapshot the current DB so a botched migration can be reverted with
# one mv. WAL is folded in by DuckDB on next clean shutdown; if it
# survives a stop, copy it too.
ts="$(date -u +%Y%m%dT%H%M%SZ)"
prev="${VORTEX_BENCH_DB%.duckdb}.prev-${ts}.duckdb"
if [ -f "$VORTEX_BENCH_DB" ]; then
    log "snapshotting ${VORTEX_BENCH_DB} → ${prev}"
    cp -p "$VORTEX_BENCH_DB" "$prev"
    [ -f "${VORTEX_BENCH_DB}.wal" ] && cp -p "${VORTEX_BENCH_DB}.wal" "${prev}.wal"
fi

log "running cargo run --release -p vortex-bench-migrate -- $*"
pushd "$REPO_DIR" >/dev/null
# Pass through whatever args the operator gave us. Don't inject a path
# flag — the migrator's CLI is owned by that crate.
if ! cargo run --release --quiet -p vortex-bench-migrate -- "$@"; then
    popd >/dev/null
    echo "ERROR: migration failed. Server is still stopped." >&2
    echo "  Restore previous DB with: mv \"$prev\" \"$VORTEX_BENCH_DB\"" >&2
    echo "  Then: sudo systemctl start vortex-bench-server" >&2
    exit 3
fi
popd >/dev/null

log "starting vortex-bench-server"
sudo /bin/systemctl start vortex-bench-server

# Give it a few seconds to come up.
deadline=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
        log "migrate ok — server is up"
        log "  prev DB kept at ${prev} (delete when you've verified data)"
        exit 0
    fi
    sleep 1
done
echo "ERROR: server did not respond on /health within 30s" >&2
echo "  prev DB kept at ${prev} for rollback" >&2
exit 1
