// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FSST decompression.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

#[allow(dead_code)]
mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

use crate::timed_launch_strategy::TimedLaunchStrategy;

// Bench-local size instead of the workspace 100M default: each input is a
// clickbench URL, much heavier per-element than the fixed-width primitives
// other kernels benchmark.
const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

fn benchmark_fsst_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        let mut setup_ctx = CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
            .vortex_expect("failed to create execution context");
        let fsst = make_fsst_clickbench_urls(n, setup_ctx.execution_ctx());

        let lens = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(setup_ctx.execution_ctx())
            .vortex_expect("canonicalize uncompressed_lengths");
        let total_size: usize = match_each_integer_ptype!(lens.ptype(), |P| {
            lens.as_slice::<P>().iter().map(|x| *x as usize).sum()
        });
        let uncompressed_size = total_size as u64;

        let fsst_array = fsst.into_array();

        group.throughput(Throughput::Bytes(uncompressed_size));
        group.bench_with_input(
            BenchmarkId::new("cuda/fsst/decompress", len_str),
            &fsst_array,
            |b, fsst_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                        .vortex_expect("failed to create execution context")
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(fsst_array.clone().execute_cuda(&mut cuda_ctx)).unwrap();
                    }
                    Duration::from_nanos(timer.load(Ordering::Relaxed))
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
