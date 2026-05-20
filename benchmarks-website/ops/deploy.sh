#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Idempotent rebuild + restart, called by vortex-bench-deploy.timer
# every 60s. Cheap and silent on the common path (no new commits).
#
# Flow:
#   1. flock on a state file (concurrent runs bail).
#   2. git fetch origin $DEPLOY_BRANCH.
#   3. If origin SHA == last-deployed SHA → exit 0.
#   4. Else: git diff against a path filter. If nothing in the filter
#      changed, sync the working tree (destructive checkout) to the
#      new SHA, update the stamp, exit 0. (Skips a build for monorepo
#      changes that don't touch the server.)
#   5. Else: sync working tree + cargo build --release -p vortex-bench-server.
#   6. Compare new binary's sha256 to the currently-running symlink target.
#      If unchanged (cargo did no real work), update stamp + exit 0.
#   7. Else: copy to bin/vortex-bench-server.<ts>.<pid> (PID suffix
#      breaks same-second deploy collisions), atomically swap the
#      symlink (staging symlink + `mv -Tf` so the swap is rename(2)),
#      sudo systemctl restart vortex-bench-server.
#   8. Wait for /health. On failure: revert symlink, restart the prior
#      binary, re-probe /health (so a rollback to an also-broken
#      binary is loud), do NOT update the stamp — next tick retries.
#   9. On success: update stamp, prune binary versions older than $KEEP_BINARIES.
#
# The working-tree sync is `git checkout --force --detach <sha>`, not
# `git pull --ff-only`, so the script survives force-pushes on the
# tracked branch.
#
# Exit codes:
#   0  success (either a real deploy or a clean no-op)
#   1  another deploy is in progress (lock held)
#   2  config error (missing env file, REPO_DIR, etc.)
#   3  git fetch failed
#   4  cargo build failed
#   5  systemctl restart failed
#   6  /health check failed; rolled back to previous binary successfully
#   7  /health check failed AND rollback to the previous binary ALSO
#      failed /health — server is down, manual intervention required.
#      This is the worst-case path; the timer will retry on next tick
#      but the prior binary is itself broken, so the retry will not
#      heal the host.

set -euo pipefail

ENV_FILE="${ENV_FILE:-/etc/vortex-bench.env}"
STATE_DIR="${STATE_DIR:-/var/lib/vortex-bench}"
LOCK_FILE="${LOCK_FILE:-${STATE_DIR}/.deploy.lock}"
STAMP_FILE="${STAMP_FILE:-${STATE_DIR}/last-deployed-sha}"
BIN_DIR="${BIN_DIR:-${STATE_DIR}/bin}"
BIN_SYMLINK="${BIN_DIR}/vortex-bench-server"
KEEP_BINARIES="${KEEP_BINARIES:-3}"

log() { printf '[deploy %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
err() { printf '[deploy %s] ERROR: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

# --- Load env ---
if [ ! -f "$ENV_FILE" ]; then
    err "missing ${ENV_FILE}"
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
: "${REPO_DIR:?REPO_DIR must be set in ${ENV_FILE}}"
: "${DEPLOY_BRANCH:=develop}"
: "${SERVER_URL:=http://127.0.0.1:3000}"

if [ ! -d "${REPO_DIR}/.git" ]; then
    err "${REPO_DIR} is not a git checkout"
    exit 2
fi

# --- Lock ---
mkdir -p "$(dirname "$LOCK_FILE")"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    log "another deploy is in progress; bailing"
    exit 1
fi

# Pick up cargo from the user's profile if not on PATH already.
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true

cd "$REPO_DIR"

last_sha=""
[ -f "$STAMP_FILE" ] && last_sha="$(cat "$STAMP_FILE")"

# --- Fetch ---
if ! git fetch --quiet origin "$DEPLOY_BRANCH"; then
    err "git fetch origin ${DEPLOY_BRANCH} failed"
    exit 3
fi
new_sha="$(git rev-parse "origin/${DEPLOY_BRANCH}")"

# --- Force mode ---
# A `FORCE=1` env var (or a `.force-rebuild` sentinel under the state
# dir) bypasses the SHA-unchanged fast path and treats this run as if
# nothing was ever deployed. Used by `force-rebuild.sh` for the
# "redeploy current branch right now" path. The sentinel is consumed
# so the next ordinary timer fire is a no-op again.
force=0
if [ "${FORCE:-0}" = "1" ]; then
    force=1
fi
if [ -f "${STATE_DIR}/.force-rebuild" ]; then
    rm -f "${STATE_DIR}/.force-rebuild"
    force=1
fi
if [ "$force" = "1" ]; then
    log "force mode: ignoring stamp comparison and path filter"
    last_sha=""
fi

if [ "$force" = "0" ] && [ "$new_sha" = "$last_sha" ]; then
    # Common case: nothing new since last fire. Silent on stdout to
    # keep the journal clean.
    exit 0
fi

# --- Path filter ---
# Rebuild + restart only when commits in the range touch website code,
# the workspace lockfile, or workspace Cargo manifests. Other changes
# (e.g. vortex-array fixes) update the working tree but don't restart.
filter_paths=(
    benchmarks-website/server
    benchmarks-website/migrate
    benchmarks-website/Cargo.toml
    Cargo.lock
    Cargo.toml
)

if [ -z "$last_sha" ] || ! git cat-file -e "${last_sha}^{commit}" 2>/dev/null; then
    # First run, or stamp points at a commit we no longer have. Treat
    # as "must rebuild" so we don't silently skip a real change. Clear
    # last_sha in the vanished-commit branch too so the hash-equal
    # restart-skip fast path below sees an empty last_sha and forces a
    # full restart + /health verify (the prior stamp commit being gone
    # means we can't trust whatever the symlink currently points at).
    log "first run / unknown stamp '${last_sha:-<empty>}'; full rebuild"
    last_sha=""
    relevant_changed=1
else
    if git diff --name-only "${last_sha}" "${new_sha}" -- "${filter_paths[@]}" | grep -q .; then
        relevant_changed=1
    else
        relevant_changed=0
    fi
fi

# --- Sync the working tree to origin/$DEPLOY_BRANCH ---
# `git pull --ff-only` breaks the moment the tracked branch is
# force-pushed (typical during PR iteration). The deploy worker's
# checkout is build-only — no human edits live here — so a destructive
# `git checkout --force --detach $new_sha` is the right semantics.
# Detached HEAD avoids any local-branch ref drift.
if ! git checkout --quiet --force --detach "$new_sha"; then
    err "git checkout --force --detach ${new_sha} failed"
    exit 3
fi

if [ "$relevant_changed" = "0" ]; then
    log "no website-relevant paths changed in ${last_sha:0:7}..${new_sha:0:7}; skipping rebuild"
    echo "$new_sha" > "$STAMP_FILE"
    exit 0
fi

# --- Build ---
prev_short="${last_sha:0:7}"
log "building ${new_sha:0:7} (was ${prev_short:-<empty>})"
if ! cargo build --release --quiet -p vortex-bench-server; then
    err "cargo build -p vortex-bench-server failed"
    exit 4
fi
new_binary="${REPO_DIR}/target/release/vortex-bench-server"
if [ ! -x "$new_binary" ]; then
    err "expected binary not found at ${new_binary}"
    exit 4
fi

# --- Compare hashes; skip restart if cargo produced byte-identical output ---
# Force mode (FORCE=1 / .force-rebuild) explicitly opts out of this fast
# path: the operator asked for "redeploy and reverify", not "skip if the
# byte image matches", so we must still restart + /health-poll.
#
# An empty last_sha (no stamp file, OR a stamp file we just rejected as
# pointing at a vanished commit at line 130) ALSO opts out: the only
# guarantee that the symlink's current target was health-verified comes
# from the stamp file being present. Without it the live binary could
# be a partially-applied prior deploy that crashed between symlink swap
# and the restart, in which case the byte-equal path would stamp
# success without ever restarting the server.
new_hash="$(sha256sum "$new_binary" | awk '{print $1}')"
current_hash=""
if [ -L "$BIN_SYMLINK" ] && [ -e "$BIN_SYMLINK" ]; then
    current_hash="$(sha256sum "$BIN_SYMLINK" | awk '{print $1}')"
fi
if [ "$force" = "0" ] && [ -n "$last_sha" ] && [ "$new_hash" = "$current_hash" ]; then
    log "binary unchanged (sha256 ${new_hash:0:12}); skipping restart"
    echo "$new_sha" > "$STAMP_FILE"
    exit 0
fi

# --- Install + atomic symlink swap ---
# `ln -sfnT` is unlink+create — there is a brief window where $BIN_SYMLINK
# does not exist, and a concurrent reader (e.g. systemctl restart firing
# from another timer fire) would see ENOENT. Do the swap in two steps so
# the final transition is `rename(2)`, which IS atomic on POSIX: create
# the new symlink under a sibling name, then `mv -Tf` it onto $BIN_SYMLINK.
# Same pattern is used in both rollback paths below.
# Include a per-process suffix so two deploys within the same UTC
# second (e.g. timer fire racing with a manual force-rebuild) cannot
# collide on the versioned filename. Without it, the second `install`
# would overwrite the first's binary and `prev_target` could end up
# pointing at a path whose contents are not the prior binary anymore.
ts="$(date -u +%Y%m%dT%H%M%SZ)"
versioned="${BIN_DIR}/vortex-bench-server.${ts}.$$"
install -m 0755 "$new_binary" "$versioned"
prev_target=""
if [ -L "$BIN_SYMLINK" ]; then
    prev_target="$(readlink "$BIN_SYMLINK")"
fi

atomic_symlink() {
    # $1 = symlink target, $2 = symlink path
    local target="$1" path="$2" staging
    staging="${path}.new.$$"
    ln -snT -- "$target" "$staging"
    mv -Tf -- "$staging" "$path"
}

atomic_symlink "$versioned" "$BIN_SYMLINK"
log "swapped symlink → ${versioned}"

# --- Restart + verify ---
if ! sudo /bin/systemctl restart vortex-bench-server; then
    err "systemctl restart failed"
    if [ -n "$prev_target" ]; then
        atomic_symlink "$prev_target" "$BIN_SYMLINK"
        sudo /bin/systemctl restart vortex-bench-server || true
    fi
    exit 5
fi

# Give it a moment to come up, then poll /health.
deadline=$(( $(date +%s) + 30 ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 1
done
if [ "$healthy" != "1" ]; then
    err "/health did not respond within 30s — rolling back"
    if [ -n "$prev_target" ]; then
        atomic_symlink "$prev_target" "$BIN_SYMLINK"
        sudo /bin/systemctl restart vortex-bench-server || true
        # Verify the rolled-back binary is itself healthy before claiming
        # clean rollback. A "previous binary" that's also broken (e.g. a
        # prior failed deploy nobody caught) needs a louder signal.
        roll_deadline=$(( $(date +%s) + 30 ))
        roll_healthy=0
        while [ "$(date +%s)" -lt "$roll_deadline" ]; do
            if curl -fsS --max-time 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
                roll_healthy=1
                break
            fi
            sleep 1
        done
        if [ "$roll_healthy" = "1" ]; then
            log "rolled back symlink to ${prev_target} (verified healthy)"
            exit 6
        fi
        err "rollback to ${prev_target} ALSO failed /health — server is down; manual intervention required"
        exit 7
    else
        err "no previous binary to roll back to"
    fi
    exit 6
fi

# --- Success: update stamp, prune old binaries ---
echo "$new_sha" > "$STAMP_FILE"
log "deploy ok: ${new_sha:0:7} → live (binary ${ts})"

# Keep the most recent $KEEP_BINARIES versioned binaries, drop the rest.
# Sort by name (timestamp prefix is sortable), keep the tail.
mapfile -t binaries < <(ls -1 "${BIN_DIR}"/vortex-bench-server.* 2>/dev/null | sort)
if [ "${#binaries[@]}" -gt "$KEEP_BINARIES" ]; then
    drop_count=$(( ${#binaries[@]} - KEEP_BINARIES ))
    for b in "${binaries[@]:0:$drop_count}"; do
        # Never delete what the symlink currently points at.
        if [ "$b" != "$(readlink -f "$BIN_SYMLINK")" ]; then
            rm -f "$b"
            log "pruned ${b}"
        fi
    done
fi
