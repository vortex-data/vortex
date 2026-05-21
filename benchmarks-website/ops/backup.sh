#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Hourly snapshot to S3, called by vortex-bench-backup.timer.
#
# Asks the running server to write a per-table Vortex snapshot via
# /api/admin/snapshot (so the writer uses the same DuckDB process
# that owns the file - no stop required), `tar czf`s the resulting
# directory into a single archive, uploads it to
# $S3_BACKUP_PREFIX/<ts>.tar.gz, and deletes the local copies.
#
# Vortex compresses our shape (mostly BIGINT[] runtime arrays + short
# strings) far better than gzipped CSV; the additional gzip on the
# tarball is largely catching schema.sql and tar metadata, not the
# data files themselves.
#
# The instance IAM role must already permit s3:PutObject under
# $S3_BACKUP_PREFIX. The v3 bucket is vortex-benchmark-results-database
# (distinct from v2's vortex-ci-benchmark-results).

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
: "${VORTEX_BENCH_SNAPSHOT_DIR:?VORTEX_BENCH_SNAPSHOT_DIR must be set}"
: "${S3_BACKUP_PREFIX:?S3_BACKUP_PREFIX must be set in ${ENV_FILE}}"
# `ADMIN_URL` points at the loopback-only admin listener; `SERVER_URL`
# stays for /health checks on the public listener.
: "${ADMIN_URL:=http://127.0.0.1:3001}"
: "${STATE_DIR:=/var/lib/vortex-bench}"
: "${BACKUP_LOCK_FILE:=${STATE_DIR}/.backup.lock}"

log() { printf '[backup %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# Serialise against ourselves: a manual `bash backup.sh` racing the timer
# fires would otherwise both hit /api/admin/snapshot at the same ts (the
# server returns 409 to the loser), then both race on `rm -rf "$local_dir"`
# while the survivor is mid-tar. Quiet bail on contention so the timer
# journal stays clean.
mkdir -p "$(dirname "$BACKUP_LOCK_FILE")"
exec 200>"$BACKUP_LOCK_FILE"
if ! flock -n 200; then
    log "another backup is in progress; bailing"
    exit 0
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
local_dir="${VORTEX_BENCH_SNAPSHOT_DIR}/${ts}"
archive="${VORTEX_BENCH_SNAPSHOT_DIR}/${ts}.tar.gz"
remote="${S3_BACKUP_PREFIX}/${ts}.tar.gz"

# Per-PID scratch files so a manual `bash backup.sh` invocation running
# alongside the timer-driven invocation does not clobber each other's
# response capture or the curl auth header. Cleaned up on exit/trap.
scratch="$(mktemp -d "${TMPDIR:-/tmp}/vortex-bench-backup.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
response="${scratch}/snapshot.out"
auth_header="${scratch}/auth.hdr"

# Write the Authorization header to a 0600 file and pass via `-H @path`
# so the bearer token never appears in argv (visible to anyone reading
# `ps aux`). Same pattern in inspect.sh. Wrap in `set +x; ...; set -x`
# guard so an operator running `bash -x backup.sh` does not see the
# bearer in the trace output.
umask 077
{ _xtrace="$(set +o | grep xtrace)"; set +x; } 2>/dev/null
printf 'Authorization: Bearer %s\n' "${ADMIN_BEARER_TOKEN}" > "$auth_header"
eval "$_xtrace" 2>/dev/null || true

log "triggering /api/admin/snapshot?ts=${ts}"
http_status=$(curl -sS -o "$response" -w '%{http_code}' \
    -X POST \
    -H "@${auth_header}" \
    "${ADMIN_URL}/api/admin/snapshot?ts=${ts}" || echo "000")
if [ "$http_status" != "200" ]; then
    echo "ERROR: /api/admin/snapshot returned ${http_status}" >&2
    cat "$response" >&2 || true
    exit 3
fi

if [ ! -d "$local_dir" ]; then
    echo "ERROR: server reported success but ${local_dir} does not exist" >&2
    exit 4
fi

# Completeness check: the server writes schema.sql plus one .vortex file
# per fact + dim table. If a deploy-timer restart interrupted the snapshot
# write mid-stream, the directory may be partially populated; the only
# completeness signal otherwise would be the presence of the dir, which
# tar+s3 cp would happily pack and upload as a "valid" archive that
# fails restore (`INSERT INTO ... silently no-op'd` per BOOTSTRAP 8.5).
required_files=(
    schema.sql
    commits.vortex
    query_measurements.vortex
    compression_times.vortex
    compression_sizes.vortex
    random_access_times.vortex
    vector_search_runs.vortex
)
missing=()
for f in "${required_files[@]}"; do
    [ -e "${local_dir}/${f}" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: snapshot ${local_dir} is incomplete; missing: ${missing[*]}" >&2
    echo "  Most common cause: vortex-bench-server was restarted mid-snapshot." >&2
    echo "  Leaving the partial directory in place for inspection." >&2
    exit 4
fi

# Compress the snapshot directory into a single tar.gz. `tar -C` so paths
# inside the archive are relative to the snapshot id (i.e. `<ts>/schema.sql`
# and `<ts>/<table>.vortex`), which matches the layout expected by the
# restore docs.
log "compressing ${local_dir} → ${archive}"
if ! tar -C "$VORTEX_BENCH_SNAPSHOT_DIR" -czf "$archive" "$ts"; then
    echo "ERROR: tar czf failed" >&2
    rm -f "$archive"
    exit 5
fi

orig_bytes=$(du -sb "$local_dir" | awk '{print $1}')
gz_bytes=$(stat -c %s "$archive")
log "compressed ${orig_bytes} → ${gz_bytes} bytes ($(( orig_bytes / (gz_bytes > 0 ? gz_bytes : 1) ))x)"

log "uploading ${archive} → s3://${remote#s3://}"
# Retry transient `aws s3 cp` failures (rate limit / ELB blip / IAM
# role refresh hiccup) before giving up. Backoff 2s, 8s, 30s.
upload_ok=0
for delay in 0 2 8 30; do
    [ "$delay" -gt 0 ] && sleep "$delay"
    if aws s3 cp --quiet "${archive}" "${remote}"; then
        upload_ok=1
        break
    fi
    log "aws s3 cp failed; retrying after ${delay:-0}s (next attempt)"
done
if [ "$upload_ok" != "1" ]; then
    echo "ERROR: aws s3 cp failed after retries; keeping ${archive} and ${local_dir} for manual recovery" >&2
    exit 6
fi

log "deleting local copies (${archive}, ${local_dir})"
rm -f "$archive"
rm -rf "$local_dir"

log "snapshot ${ts} ok → ${remote}"
