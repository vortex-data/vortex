#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run cargo with headstart (https://github.com/PowderworksCode/headstart), an experimental
# patched rustc and cargo that start dependent crates as soon as a dependency's interface is
# checked, instead of waiting for its whole metadata.
#
#   scripts/headstart.sh setup               # once: fetch and build the patched toolchain
#   scripts/headstart.sh check -p vortex     # any cargo command, with headstart on
#   HEADSTART=0 scripts/headstart.sh check   # same toolchain, headstart off (a fair baseline)
#
# The patched rustc is newer than rust-toolchain.toml, so its artifacts go to their own target
# directory (target/headstart unless CARGO_TARGET_DIR is set).
#
# A stock sccache would hide rustc's early-metadata notifications until rustc exits, so nothing
# would start early. When RUSTC_WRAPPER is sccache, this runs a patched sccache instead
# (scripts/headstart/sccache.patch), in client-side mode with rustc's output streamed. It can
# share the stock sccache server, which serves as its gateway to the cache storage.
set -euo pipefail

HEADSTART_REPO=https://github.com/PowderworksCode/headstart
HEADSTART_REV=ff81bfecda2116d54776a7fec538fc2f4e349d1f
SCCACHE_REPO=https://github.com/mozilla/sccache
SCCACHE_REV=01c35e69b97f516c2ebfbdb418b5cd6277bde280 # v0.18.0
HEADSTART_DIR=${HEADSTART_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/vortex-headstart}

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
rustc_bin=$HEADSTART_DIR/rustc/build/host/stage1/bin/rustc
cargo_bin=$HEADSTART_DIR/cargo/target/release/cargo
sccache_bin=$HEADSTART_DIR/sccache/target/release/sccache

checkout() { # <repo url> <rev> <dir>
  if [ ! -d "$3/.git" ]; then
    git init --quiet "$3"
  fi
  git -C "$3" fetch --quiet --depth 1 "$1" "$2"
  git -C "$3" checkout --quiet --force FETCH_HEAD
}

if [ "${1:-}" = setup ]; then
  checkout "$HEADSTART_REPO" "$HEADSTART_REV" "$HEADSTART_DIR"
  cp "$repo_root/scripts/headstart/bootstrap.toml" "$HEADSTART_DIR/config/bootstrap.toml"
  # Builds a stage 1 rustc from source: allow over an hour on 4 cores, and 20 GB of disk.
  # Under GITHUB_ACTIONS, rustc's bootstrap assumes rust-lang's own CI checkout and runs
  # `git rev-parse HEAD^1`, which fails in the shallow submodule.
  (unset RUSTC_WRAPPER GITHUB_ACTIONS && "$HEADSTART_DIR/scripts/setup.sh")

  checkout "$SCCACHE_REPO" "$SCCACHE_REV" "$HEADSTART_DIR/sccache"
  git -C "$HEADSTART_DIR/sccache" apply "$repo_root/scripts/headstart/sccache.patch"
  (cd "$HEADSTART_DIR/sccache" && unset RUSTC_WRAPPER && cargo +1.98.0 build --release --locked)
  exit
fi

if [ $# -eq 0 ]; then
  echo "usage: $0 setup | <cargo arguments>" >&2
  exit 2
fi
if [ ! -x "$rustc_bin" ] || [ ! -x "$cargo_bin" ]; then
  echo "headstart toolchain not found in $HEADSTART_DIR; run: $0 setup" >&2
  exit 1
fi

if [ "$(basename "${RUSTC_WRAPPER:-}")" = sccache ]; then
  if [ ! -x "$sccache_bin" ]; then
    echo "patched sccache not found in $HEADSTART_DIR; run: $0 setup" >&2
    exit 1
  fi
  export RUSTC_WRAPPER=$sccache_bin SCCACHE_CLIENT_SIDE=1 SCCACHE_STREAM_STDERR=1
else
  # Any other wrapper that buffers rustc's output would hide the early-metadata notifications.
  unset RUSTC_WRAPPER
fi
export RUSTC=$rustc_bin
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$repo_root/target/headstart}
if [ "${HEADSTART:-1}" = 0 ]; then
  export CARGO_UNSTABLE_HEADSTART=false
else
  export CARGO_UNSTABLE_HEADSTART=true
fi
exec "$cargo_bin" "$@"
