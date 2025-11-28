# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

#!/bin/bash
# Build the vortex-fuzz crate for wasmfuzz
#
# This script builds the fuzzer binary for the wasm32-wasip1 target,
# which can then be used with wasmfuzz for coverage-guided fuzzing.
#
# Prerequisites:
#   - Nightly Rust toolchain (for -Z flags)
#   - wasm32-wasip1 target: rustup +nightly target add wasm32-wasip1
#   - wasmfuzz: cargo install --git https://github.com/CISPA-SysSec/wasmfuzz
#
# Usage:
#   ./build-wasmfuzz.sh
#
# After building, run with wasmfuzz:
#   wasmfuzz fuzz --timeout=1h --cores 8 --dir corpus/ \
#       target/wasm32-wasip1/release/array_ops_wasm.wasm

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building vortex-fuzz for wasm32-wasip1..."

# Build the WASM binary with nightly for -Z flags (build-std, embed-source, etc.)
rustup run nightly cargo build \
    --manifest-path fuzz/Cargo.toml \
    --target wasm32-wasip1 \
    --no-default-features \
    --features wasmfuzz \
    --release \
    --bin array_ops_wasm

WASM_OUTPUT="target/wasm32-wasip1/release/array_ops_wasm.wasm"

if [ -f "$WASM_OUTPUT" ]; then
    echo ""
    echo "Build successful!"
    echo "Output: $WASM_OUTPUT"
    echo ""
    echo "To run with wasmfuzz:"
    echo "  wasmfuzz fuzz --timeout=1h --cores 8 --dir corpus/ $WASM_OUTPUT"
    echo ""
    echo "See: https://github.com/CISPA-SysSec/wasmfuzz"
else
    echo "Build completed but .wasm output not found at expected location."
    echo "Check target/wasm32-wasip1/release/ for outputs:"
    ls -la target/wasm32-wasip1/release/ 2>/dev/null || echo "Directory not found"
fi
