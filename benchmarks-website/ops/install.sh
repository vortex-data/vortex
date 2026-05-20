#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# One-time bootstrap of vortex-bench-server on a fresh EC2 host.
# Idempotent — safe to re-run after editing units or to recover from
# partial state. See ops/README.md for the full operator runbook.
#
# Run as a user with sudo (typically ec2-user). The script will:
#   1. Create state and log directories under /var/lib/vortex-bench
#      and /var/log/vortex-bench, owned by $RUN_USER.
#   2. Drop a sudoers fragment that lets $RUN_USER restart the server
#      service without a password (so the deploy timer can run as the
#      service user).
#   3. Copy /etc/vortex-bench.env from the template if missing (mode 0600).
#   4. Install the systemd units and reload systemd.
#   5. Symlink the ops/ directory into /var/lib/vortex-bench so the
#      systemd units have a stable path (the repo can move).
#   6. Enable + start the server, deploy timer, and backup timer.
#
# Usage:
#   ./benchmarks-website/ops/install.sh
#   REPO_DIR=$HOME/vortex ./benchmarks-website/ops/install.sh
#
# Only REPO_DIR is honored as an env override; the run user, state-dir,
# env-file, systemd-dir, and sudoers-file paths are pinned (they have
# to match the shipped systemd units, which hard-code these values).

set -euo pipefail

# The installed systemd units hard-code `User=ec2-user`,
# `EnvironmentFile=/etc/vortex-bench.env`, and the
# `/var/lib/vortex-bench` state-dir. Keep these values aligned with
# the units in `systemd/` and the runbook in `README.md`; the script
# does NOT template the units at install time. Anyone running on a
# different user / state-dir / env-file path needs to hand-edit the
# units before this script copies them into /etc/systemd/system.
RUN_USER="ec2-user"
RUN_GROUP="${RUN_USER}"
REPO_DIR="${REPO_DIR:-$HOME/vortex}"
STATE_DIR="/var/lib/vortex-bench"
LOG_DIR="/var/log/vortex-bench"
ENV_FILE="/etc/vortex-bench.env"
SYSTEMD_DIR="/etc/systemd/system"
SUDOERS_FILE="/etc/sudoers.d/vortex-bench"

ops_dir="${REPO_DIR}/benchmarks-website/ops"
if [ ! -d "$ops_dir" ]; then
    echo "ERROR: ${ops_dir} not found. Set REPO_DIR=<repo path>." >&2
    exit 2
fi

# The deploy timer runs as ${RUN_USER} with no SSH agent, so an SSH
# remote fails with "Permission denied (publickey)" on every fire.
# Public-repo HTTPS reads need no auth — warn early so this is not the
# first surprise out of the gate.
if [ -d "${REPO_DIR}/.git" ]; then
    origin_url="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
    case "$origin_url" in
        git@*|ssh://*)
            echo "WARNING: ${REPO_DIR}'s origin is ${origin_url}." >&2
            echo "  The deploy timer cannot fetch over SSH (no agent). Fix with:" >&2
            echo "    git -C ${REPO_DIR} remote set-url origin https://github.com/vortex-data/vortex.git" >&2
            ;;
    esac
fi

log() { printf '[install] %s\n' "$*"; }

# --- 1. State + log directories ---
log "creating ${STATE_DIR} and ${LOG_DIR} (owner ${RUN_USER}:${RUN_GROUP})"
sudo install -d -m 0755 -o "$RUN_USER" -g "$RUN_GROUP" \
    "$STATE_DIR" \
    "${STATE_DIR}/bin" \
    "${STATE_DIR}/snapshots" \
    "${STATE_DIR}/duckdb-extensions" \
    "$LOG_DIR"

# --- 2. Sudoers fragment ---
# Let RUN_USER restart/start/stop only vortex-bench-server, no password.
# The script that uses this is ops/deploy.sh (atomic restart after build).
log "writing sudoers fragment to ${SUDOERS_FILE}"
sudo tee "$SUDOERS_FILE" >/dev/null <<EOF
# Auto-deploy + manual migration helpers run as ${RUN_USER}; only the
# systemctl call into the server unit needs root.
${RUN_USER} ALL=(root) NOPASSWD: /bin/systemctl restart vortex-bench-server, /bin/systemctl start vortex-bench-server, /bin/systemctl stop vortex-bench-server, /bin/systemctl status vortex-bench-server, /usr/bin/systemctl restart vortex-bench-server, /usr/bin/systemctl start vortex-bench-server, /usr/bin/systemctl stop vortex-bench-server, /usr/bin/systemctl status vortex-bench-server
EOF
sudo chmod 0440 "$SUDOERS_FILE"
sudo visudo -cf "$SUDOERS_FILE" >/dev/null

# --- 3. Env file ---
if [ ! -f "$ENV_FILE" ]; then
    log "creating ${ENV_FILE} from template (mode 0600 owned by ${RUN_USER})"
    sudo install -m 0600 -o "$RUN_USER" -g "$RUN_GROUP" \
        "${ops_dir}/config/vortex-bench.env.example" \
        "$ENV_FILE"
    log "EDIT ${ENV_FILE} to set INGEST_BEARER_TOKEN, ADMIN_BEARER_TOKEN, REPO_DIR"
else
    log "${ENV_FILE} already present — leaving alone"
fi

# --- 4. Symlink ops/ into the state dir ---
# Gives systemd units a stable path that doesn't depend on the repo
# checkout location moving.
log "symlinking ${ops_dir} -> ${STATE_DIR}/ops"
sudo ln -sfnT "$ops_dir" "${STATE_DIR}/ops"

# --- 5. systemd units ---
log "installing systemd units to ${SYSTEMD_DIR}"
for unit in \
    vortex-bench-server.service \
    vortex-bench-deploy.service \
    vortex-bench-deploy.timer \
    vortex-bench-backup.service \
    vortex-bench-backup.timer
do
    sudo install -m 0644 -o root -g root \
        "${ops_dir}/systemd/${unit}" \
        "${SYSTEMD_DIR}/${unit}"
done
sudo systemctl daemon-reload

# --- 6. Enable (and start, if tokens are set) ---
# The server unit needs a binary at /var/lib/vortex-bench/bin/vortex-bench-server
# before it can start. If the symlink isn't there yet, the deploy timer
# will lay one down on its first run; until then the server will fail.
if [ ! -e "${STATE_DIR}/bin/vortex-bench-server" ]; then
    log "no binary at ${STATE_DIR}/bin/vortex-bench-server yet"
    log "  → the first deploy-timer fire (after start) will build + install one."
    log "  → tail it with: journalctl -fu vortex-bench-deploy.service"
fi

# Detect whether the operator has filled in the bearer tokens. An empty
# INGEST_BEARER_TOKEN makes the server fail startup; an empty
# ADMIN_BEARER_TOKEN leaves the admin listener unbound. Both cases mean
# starting the units now would just produce noisy failures — enable but
# defer the start instead.
ingest_set=$(grep -E '^INGEST_BEARER_TOKEN=.+' "$ENV_FILE" || true)
admin_set=$(grep -E '^ADMIN_BEARER_TOKEN=.+' "$ENV_FILE" || true)

if [ -n "$ingest_set" ] && [ -n "$admin_set" ]; then
    log "tokens present in ${ENV_FILE} — enabling + starting timers and server"
    sudo systemctl enable --now vortex-bench-deploy.timer
    sudo systemctl enable --now vortex-bench-backup.timer
    sudo systemctl enable vortex-bench-server.service
    sudo systemctl start vortex-bench-server.service || \
        log "  server didn't start — likely no binary yet; deploy timer will handle it"
else
    log "tokens not set in ${ENV_FILE} — timers and server enabled but not started"
    sudo systemctl enable vortex-bench-deploy.timer
    sudo systemctl enable vortex-bench-backup.timer
    sudo systemctl enable vortex-bench-server.service
    log "after editing ${ENV_FILE}, run:"
    log "  sudo systemctl start vortex-bench-server vortex-bench-deploy.timer vortex-bench-backup.timer"
fi

log ""
log "install complete. Next steps:"
log "  1. Edit ${ENV_FILE} (chmod 0600, owned by ${RUN_USER})"
log "     - INGEST_BEARER_TOKEN=$(openssl rand -hex 32)"
log "     - ADMIN_BEARER_TOKEN=$(openssl rand -hex 32)"
log "     - confirm REPO_DIR points at the actual checkout"
log "  2. After starting the timers, watch the first deploy fire build the"
log "     binary and bring the server up with an empty DuckDB:"
log "       journalctl -fu vortex-bench-deploy.service"
log "       curl http://127.0.0.1:3000/health"
log "  3. Populate the DB with the v2→v3 migration (server is stopped"
log "     and restarted automatically):"
log "       ${STATE_DIR}/ops/migrate.sh run --output \"${STATE_DIR}/bench.duckdb\""
log "  4. (If preserving an existing \$HOME/bench.duckdb instead of"
log "     re-migrating, copy it into place before step 3:"
log "       sudo systemctl stop vortex-bench-server"
log "       sudo -u ${RUN_USER} mv \$HOME/bench.duckdb ${STATE_DIR}/bench.duckdb"
log "       sudo systemctl start vortex-bench-server"
log "     and skip step 3.)"
