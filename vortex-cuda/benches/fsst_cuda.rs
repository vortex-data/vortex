// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CPU baseline for FSST decompression.
//!
//! Sits in `vortex-cuda` so subsequent commits can swap the body of
//! `cuda/fsst/decompress` from CPU to GPU under the same criterion id and
//! the user can compare against this baseline with `--save-baseline`.
//!
//! The CPU path shards the array across all available cores via `thread::scope`
//! so the baseline reflects best-case CPU throughput, not a single-threaded
//! straw man.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

#[allow(dead_code)]
mod bench_config;

use std::thread;
use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use mimalloc::MiMalloc;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::error::VortexExpect;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

// MiMalloc reduces cross-thread allocator contention for the multi-threaded
// CPU shards. Removed once the bench dispatches to the GPU.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Bench-local size instead of the workspace 100M default: each input is a
// clickbench URL, much heavier per-element than the fixed-width primitives
// other kernels benchmark.
const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

fn benchmark_fsst_cuda_decompress(c: &mut Criterion) {
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        let mut setup_ctx = LEGACY_SESSION.create_execution_ctx();
        let fsst = make_fsst_clickbench_urls(n, &mut setup_ctx);

        let lens = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(&mut setup_ctx)
            .vortex_expect("canonicalize uncompressed_lengths");
        let total_size: usize = match_each_integer_ptype!(lens.ptype(), |P| {
            lens.as_slice::<P>().iter().map(|x| *x as usize).sum()
        });
        let uncompressed_size = total_size as u64;

        let fsst_array: ArrayRef = fsst.into_array();
        let shard_size = n.div_ceil(num_threads);

        group.throughput(Throughput::Bytes(uncompressed_size));
        group.bench_with_input(
            BenchmarkId::new("cuda/fsst/decompress", len_str),
            &fsst_array,
            |b, fsst_array: &ArrayRef| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        thread::scope(|s| {
                            for i in 0..num_threads {
                                let lo = i * shard_size;
                                let hi = (lo + shard_size).min(n);
                                if lo >= hi {
                                    continue;
                                }
                                let shard = fsst_array
                                    .slice(lo..hi)
                                    .vortex_expect("slice failed");
                                s.spawn(move || {
                                    let mut ctx = LEGACY_SESSION.create_execution_ctx();
                                    drop(
                                        shard
                                            .execute::<Canonical>(&mut ctx)
                                            .vortex_expect("CPU FSST decompression failed"),
                                    );
                                });
                            }
                        });
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_fsst_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
