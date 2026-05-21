#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Manual v2 to v3 migration wrapper. The migration tool needs exclusive
# access to the DB file, so the server is stopped first, the current DB
# is snapshotted to bench.prev-<ts>.duckdb for instant rollback, the
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
# (Run as ec2-user is fine - we sudo only for systemctl.)

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

# Pause the autopilot for the duration of the migration. Stopping the
# timers alone is not enough - if deploy.service or backup.service is
# already mid-run, the active oneshot keeps going and can restart the
# server or call /api/admin/snapshot while the migrator owns the DB.
# Stop the services first (interrupting any active run, idempotent
# no-op if inactive), then the timers, then the server.
#
# The migration_succeeded flag is flipped to 1 only after the server
# comes back healthy. The trap restores the autopilot on success; on
# failure the autopilot stays paused so the operator can perform the
# documented mv-rollback without the deploy timer trying to re-fetch
# origin and run a fresh build on top of the half-rolled-back DB.
migration_succeeded=0
# The sudoers fragment install.sh writes lists each unit on its own
# Cmnd line, and sudoers requires argv match exactly: multi-unit
# `systemctl stop A B C D` would NOT be authorized by per-unit entries
# and must be split into N single-unit invocations. Same on the
# success-restore path. We deliberately do NOT use `|| true` so a real
# sudo failure surfaces in the journal instead of silently no-op'ing the
# autopilot pause. Install the trap BEFORE the stop calls so a partial
# stop (one of the four sudo calls failing under set -e) still triggers
# the restore path.
restore_autopilot() {
    if [ "$migration_succeeded" = "1" ]; then
        log "restoring autopilot timers (deploy + backup)"
        sudo /bin/systemctl start vortex-bench-deploy.timer
        sudo /bin/systemctl start vortex-bench-backup.timer
    else
        log "migration did not complete - leaving autopilot timers stopped"
        log "  inspect with: systemctl status vortex-bench-server \\"
        log "    vortex-bench-deploy.service vortex-bench-deploy.timer \\"
        log "    vortex-bench-backup.service vortex-bench-backup.timer"
        log "  after rollback and verification, restart timers with:"
        log "    sudo systemctl start vortex-bench-deploy.timer"
        log "    sudo systemctl start vortex-bench-backup.timer"
    fi
}
trap restore_autopilot EXIT

log "stopping autopilot services (deploy + backup) + timers for migration window"
sudo /bin/systemctl stop vortex-bench-deploy.timer
sudo /bin/systemctl stop vortex-bench-deploy.service
sudo /bin/systemctl stop vortex-bench-backup.timer
sudo /bin/systemctl stop vortex-bench-backup.service

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

log "running cargo run --release -p vortex-bench-migrate -- (${#} args; argv NOT logged so future flags carrying secrets do not leak to journal)"
pushd "$REPO_DIR" >/dev/null
# Pass through whatever args the operator gave us. Don't inject a path
# flag - the migrator's CLI is owned by that crate.
if ! cargo run --release --quiet -p vortex-bench-migrate -- "$@"; then
    popd >/dev/null
    echo "ERROR: migration failed. Server is still stopped." >&2
    echo "  Roll back:" >&2
    echo "    mv \"$prev\" \"$VORTEX_BENCH_DB\"" >&2
    echo "    [ -f \"${prev}.wal\" ] && mv \"${prev}.wal\" \"${VORTEX_BENCH_DB}.wal\" || true" >&2
    echo "  Then start the server and re-enable autopilot timers:" >&2
    echo "    sudo systemctl start vortex-bench-server" >&2
    echo "    sudo systemctl start vortex-bench-deploy.timer" >&2
    echo "    sudo systemctl start vortex-bench-backup.timer" >&2
    exit 3
fi
popd >/dev/null

log "starting vortex-bench-server"
sudo /bin/systemctl start vortex-bench-server

# Give it a few seconds to come up.
deadline=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
        migration_succeeded=1
        log "migrate ok - server is up"
        log "  prev DB kept at ${prev} (delete when you've verified data)"
        exit 0
    fi
    sleep 1
done
echo "ERROR: server did not respond on /health within 30s" >&2
# Stop the server BEFORE printing rollback instructions: the unit has
# Restart=on-failure RestartSec=2, so leaving it running would loop a
# broken/half-migrated binary against the new DB, and the rollback `mv`
# below races against the still-open file handle (on Linux the mv
# succeeds but the live server keeps writing to the unlinked inode).
# Do NOT swallow a `systemctl stop` failure with `|| true` here: if the
# stop fails (mid-procedure sudoers regression, systemd bus stuck), the
# rollback `mv` below races a still-running server and the operator
# corrupts the prev DB by following the printed instructions verbatim.
# Bail loudly so the operator fixes the stop path before any mv.
if ! sudo /bin/systemctl stop vortex-bench-server; then
    echo "CRITICAL: 'sudo systemctl stop vortex-bench-server' failed." >&2
    echo "  Do NOT execute the documented rollback 'mv' commands until the server is" >&2
    echo "  verifiably stopped (check 'systemctl status vortex-bench-server')." >&2
    echo "  The autopilot timers stay paused; debug the stop first." >&2
    exit 4
fi
echo "  server stopped. Roll back:" >&2
echo "    mv \"$prev\" \"$VORTEX_BENCH_DB\"" >&2
echo "    [ -f \"${prev}.wal\" ] && mv \"${prev}.wal\" \"${VORTEX_BENCH_DB}.wal\" || true" >&2
echo "    sudo systemctl start vortex-bench-server" >&2
echo "    sudo systemctl start vortex-bench-deploy.timer" >&2
echo "    sudo systemctl start vortex-bench-backup.timer" >&2
exit 1
