#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper around `cargo bench -p fastlanes-kernel-bench` that compiles the
# bitpacking kernels with the host's full SIMD feature set and a single codegen
# unit. Default cargo builds the workspace at the `x86-64-v1` baseline (SSE2
# only) which leaves a large speedup on the table -- e.g. on Sapphire Rapids
# the AVX2 ymm path is ~15-30% faster than SSE2 for `BitPacking::unpack`, and
# more for fused FoR variants.
#
# Usage:
#   ./bench.sh                              # run all 360 cases
#   ./bench.sh __u32__w10                   # filter
#   ./bench.sh bare_unpack --sample-count 500
set -euo pipefail

cd "$(dirname "$0")/../.."

# `target-cpu=native` enables every ISA extension the host supports
# (AVX2/AVX-512/BMI2/...). Override RUSTFLAGS_NATIVE in the environment if you
# need to publish reproducible numbers from a portable baseline, e.g.
#   RUSTFLAGS_NATIVE='-C target-cpu=x86-64-v3' ./bench.sh
export RUSTFLAGS="${RUSTFLAGS_NATIVE:--C target-cpu=native} ${RUSTFLAGS:-}"

# codegen-units=1 keeps every (T, W) monomorphisation in one TU so LLVM can
# inline + lay out the unpack body contiguously, helping icache for the
# back-to-back per-bit-width benchmarks.
exec cargo bench \
    -p fastlanes-kernel-bench \
    --bench unpack_vs_fused \
    --config 'profile.bench.codegen-units=1' \
    -- "$@"
