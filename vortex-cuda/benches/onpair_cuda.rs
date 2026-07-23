// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for OnPair decompression.

#![expect(clippy::unwrap_used)]

mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::VarBinArray;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexExpect;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::onpair_compress;

use crate::timed_launch_strategy::TimedLaunchStrategy;

// Bench-local size instead of the workspace 100M default: each input is a
// URL-shaped string, much heavier per-element than the fixed-width primitives
// other kernels benchmark.
const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

struct OnPairBenchFixture {
    array: ArrayRef,
    uncompressed_size: u64,
}

fn make_fixture(n: usize) -> OnPairBenchFixture {
    let mut setup_ctx = CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
        .vortex_expect("failed to create execution context");

    let strings: Vec<String> = (0..n)
        .map(|i| format!("https://www.example.com/path/{i}/segment?q={}", i % 97))
        .collect();
    let uncompressed_size = strings.iter().map(|s| s.len() as u64).sum();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let array = onpair_compress(&varbin, DEFAULT_DICT12_CONFIG, setup_ctx.execution_ctx())
        .vortex_expect("OnPair compression failed");

    OnPairBenchFixture {
        array,
        uncompressed_size,
    }
}

fn benchmark_onpair_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        let fixture = make_fixture(n);

        group.throughput(Throughput::Bytes(fixture.uncompressed_size));
        group.bench_with_input(
            BenchmarkId::new("cuda/onpair/decompress_to_varbinview", len_str),
            &fixture.array,
            |b, onpair_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx =
                        CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                            .vortex_expect("failed to create execution context")
                            .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                            .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(onpair_array.clone().execute_cuda(&mut cuda_ctx)).unwrap();
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
    targets = benchmark_onpair_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
