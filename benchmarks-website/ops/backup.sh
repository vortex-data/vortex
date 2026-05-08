#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Hourly snapshot to S3, called by vortex-bench-backup.timer.
#
# Asks the running server to EXPORT DATABASE via /api/admin/snapshot
# (so the export uses the same DuckDB process that owns the file — no
# stop required), `tar czf`s the resulting CSV dir into a single
# archive, uploads it to $S3_BACKUP_PREFIX/<ts>.tar.gz, and deletes
# the local copies.
#
# We gzip rather than uploading raw CSVs because DuckDB's CSV EXPORT
# is verbose for our shape (most data lands in BIGINT[] runtime
# columns that bloat 2–3× as text); gzip typically reclaims 5–7× on
# this kind of payload.
#
# The instance IAM role must already permit s3:PutObject under
# $S3_BACKUP_PREFIX. (Same bucket the v2 backup script used.)

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
: "${SERVER_URL:=http://127.0.0.1:3000}"

log() { printf '[backup %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

ts="$(date -u +%Y%m%dT%H%M%SZ)"
local_dir="${VORTEX_BENCH_SNAPSHOT_DIR}/${ts}"
archive="${VORTEX_BENCH_SNAPSHOT_DIR}/${ts}.tar.gz"
remote="${S3_BACKUP_PREFIX}/${ts}.tar.gz"

log "triggering /api/admin/snapshot?ts=${ts}"
http_status=$(curl -sS -o /tmp/snapshot.out -w '%{http_code}' \
    -X POST \
    -H "Authorization: Bearer ${ADMIN_BEARER_TOKEN}" \
    "${SERVER_URL}/api/admin/snapshot?ts=${ts}" || echo "000")
if [ "$http_status" != "200" ]; then
    echo "ERROR: /api/admin/snapshot returned ${http_status}" >&2
    cat /tmp/snapshot.out >&2 || true
    exit 3
fi
rm -f /tmp/snapshot.out

if [ ! -d "$local_dir" ]; then
    echo "ERROR: server reported success but ${local_dir} does not exist" >&2
    exit 4
fi

# Compress the export directory into a single tar.gz. `tar -C` so paths
# inside the archive are relative to the snapshot id (i.e. `<ts>/foo.csv`),
# which matches the layout expected by the restore docs.
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
if ! aws s3 cp --quiet "${archive}" "${remote}"; then
    echo "ERROR: aws s3 cp failed; keeping ${archive} and ${local_dir} for manual recovery" >&2
    exit 6
fi

log "deleting local copies (${archive}, ${local_dir})"
rm -f "$archive"
rm -rf "$local_dir"

log "snapshot ${ts} ok → ${remote}"
