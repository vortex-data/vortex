#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Force a rebuild + restart of origin/$DEPLOY_BRANCH right now, even
# if origin hasn't moved since the last successful deploy. Drops a
# sentinel that the next deploy.sh fire consumes, then triggers it.
#
# Use cases:
#   - You changed /etc/vortex-bench.env and want a fresh binary build
#     (e.g. a feature flag baked into config) rather than just a
#     `systemctl restart` of the existing one.
#   - You flipped DEPLOY_BRANCH and want the new tip in <60s rather
#     than waiting for the timer.
#   - Build artefacts got wedged and you want a clean rebuild.
#
# For "build whatever I have locally checked out" rather than fetching
# origin, edit /etc/vortex-bench.env to point DEPLOY_BRANCH at a
# branch the local tip is already on, then run this. The deploy
# script always builds origin's tip — there is no "use local HEAD"
# mode by design; push to a branch first.

set -euo pipefail

STATE_DIR="${STATE_DIR:-/var/lib/vortex-bench}"

if [ ! -d "$STATE_DIR" ]; then
    echo "ERROR: ${STATE_DIR} not found — has install.sh run?" >&2
    exit 2
fi

# The sentinel file needs to be writable by the user the deploy
# service runs as. install.sh chowns STATE_DIR to that user, so this
# works without sudo. If you're running as a different user, sudo.
if ! touch "${STATE_DIR}/.force-rebuild" 2>/dev/null; then
    echo "ERROR: cannot write ${STATE_DIR}/.force-rebuild — run as the install user or sudo" >&2
    exit 2
fi

echo "[force-rebuild] sentinel dropped; firing deploy service"
sudo /bin/systemctl start vortex-bench-deploy.service
echo "[force-rebuild] tail with: journalctl -fu vortex-bench-deploy.service"
