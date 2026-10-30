#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Prints `-p <name>` for every publishable workspace package, skipping any names passed as arguments.
# Used in place of `cargo hack --ignore-private` when checking packages in a single Cargo invocation.

set -Eeu -o pipefail

cargo metadata --no-deps --format-version 1 \
  | jq -r --args '.packages[] | select(.publish != []) | .name | select(IN($ARGS.positional[]) | not) | "-p " + .' "$@" \
  | tr '\n' ' '
