#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper around `cargo bench -p fastlanes-kernel-bench` that compiles the
# bitpacking kernels using the *widest* SIMD width the host supports.
#
# By default we add `target-feature=-prefer-256-bit` on top of
# `target-cpu=native`. On Skylake-X / Sapphire-Rapids / Emerald-Rapids and
# similar AVX-512 cores LLVM ordinarily defaults to 256-bit `ymm` vectors
# even though `zmm` is available -- a leftover guard against the AVX-512
# downclock penalty on older Xeons. Disabling that hint pushes the codegen
# to 512-bit `zmm`, which on Emerald Rapids is the right choice for *most*
# compressed bit widths but is **not** universally faster:
#   * AVX-512 wins compute-bound narrow-W cases (u32 W<24, u64 W<33 etc.)
#     by 1.2-1.6x vs AVX2.
#   * AVX-512 loses memory-bound full-width-identity cases (W == T,
#     e.g. u64 W=64) by 1.2-1.6x vs AVX2.
# See the matrix in README.md. Set PREFER=256 to compare against AVX2.
#
# Usage:
#   ./bench.sh                              # run all 360 cases
#   ./bench.sh __u32__w10                   # filter
#   ./bench.sh bare_unpack --sample-count 500
#
# To compare against the 256-bit (AVX2 only) build, set:
#   PREFER=256 ./bench.sh
# To pin a portable baseline (e.g. for cross-machine numbers):
#   RUSTFLAGS_NATIVE='-C target-cpu=x86-64-v3' ./bench.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

# Build the rustflags: host CPU + (optionally) force 512-bit vectors.
RUSTFLAGS_BASE="${RUSTFLAGS_NATIVE:--C target-cpu=native}"
case "${PREFER:-512}" in
    512) EXTRA="-C target-feature=-prefer-256-bit" ;;
    256) EXTRA="" ;;
    *)   echo "PREFER must be 256 or 512" >&2; exit 1 ;;
esac
export RUSTFLAGS="${RUSTFLAGS_BASE} ${EXTRA} ${RUSTFLAGS:-}"

# codegen-units=1 keeps every (T, W) monomorphisation in one TU so LLVM can
# inline + lay out the unpack body contiguously, helping icache for the
# back-to-back per-bit-width benchmarks.
exec cargo bench \
    -p fastlanes-kernel-bench \
    --bench unpack_vs_fused \
    --config 'profile.bench.codegen-units=1' \
    -- "$@"
