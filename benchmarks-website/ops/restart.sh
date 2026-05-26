#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Restart the vortex-bench-server binary in place (no rebuild), with
# visible before/after state so you don't have to wonder if it worked.
#
# Prints, in order:
#   - the running pid + start time + binary path the symlink points at,
#     before the restart
#   - the systemctl exit code
#   - the running pid + start time + /health response after the restart
#   - "RESTART OK" / "RESTART FAILED" + non-zero exit on failure
#
# Use this instead of `sudo systemctl restart vortex-bench-server`
# when you want any sign that it actually happened.

set -euo pipefail

ENV_FILE="${ENV_FILE:-/etc/vortex-bench.env}"
# Source the env file first so any SERVER_URL in /etc/vortex-bench.env is
# picked up, THEN apply the local default if both env-file and caller env
# left it unset. (Matches the sibling scripts; replaces the prior
# `default-then-source-then-no-op-default` shape that was misleading.)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi
SERVER_URL="${SERVER_URL:-http://127.0.0.1:3000}"

snap() {
    # Use systemd as the source of truth for the running pid (matches
    # whatever it would restart). Falls back to "?" if the unit is
    # already inactive.
    local pid started binary health
    pid="$(systemctl show -p MainPID --value vortex-bench-server 2>/dev/null || echo 0)"
    started="$(systemctl show -p ActiveEnterTimestamp --value vortex-bench-server 2>/dev/null || echo '?')"
    if [ -L /var/lib/vortex-bench/bin/vortex-bench-server ]; then
        binary="$(readlink /var/lib/vortex-bench/bin/vortex-bench-server)"
    else
        binary="?"
    fi
    health="$(curl -fsS --max-time 2 "${SERVER_URL}/health" 2>/dev/null \
        | (command -v jq >/dev/null && jq -c . || cat) \
        || echo '(unreachable)')"
    printf '  pid:        %s\n  started:    %s\n  binary:     %s\n  /health:    %s\n' \
        "$pid" "$started" "$binary" "$health"
}

echo "BEFORE:"
snap

echo
echo "running: sudo systemctl restart vortex-bench-server"
if ! sudo /bin/systemctl restart vortex-bench-server; then
    echo "ERROR: systemctl restart returned non-zero" >&2
    echo
    echo "AFTER (restart did not complete):"
    snap
    exit 1
fi

# Wait up to 30s for the new process to take requests.
deadline=$(( $(date +%s) + 30 ))
ok=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "${SERVER_URL}/health" >/dev/null 2>&1; then
        ok=1
        break
    fi
    sleep 0.5
done

echo
echo "AFTER:"
snap

if [ "$ok" = "1" ]; then
    echo
    echo "RESTART OK"
    exit 0
else
    echo
    echo "RESTART FAILED - /health did not respond within 30s" >&2
    echo "Inspect with: journalctl -u vortex-bench-server --since '1 min ago' --no-pager" >&2
    exit 1
fi
