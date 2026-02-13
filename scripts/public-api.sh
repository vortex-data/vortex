#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail

# Thin wrapper around `cargo xtask public-api` for CI compatibility.
exec cargo xtask public-api
