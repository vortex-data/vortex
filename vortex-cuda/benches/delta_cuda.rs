// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FastLanes delta decompression.
//!
//! Each element width is its own case: the lane count is `1024 / bit-width`, so the width sets
//! both how a chunk is split (128 lanes of 8 rows for `u8` through 16 lanes of 64 rows for
//! `u64`) and how much of a block is busy during the scan.

#![expect(clippy::unwrap_used)]

mod bench_config;
mod timed_launch_strategy;

use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::array_session;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaArray;
use vortex::error::VortexExpect;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Builds a delta-encoded array of `len` values that stay inside `T`.
fn make_delta_array<T>(len: usize) -> DeltaArray
where
    T: NativePType + From<u8>,
{
    // A small repeating step keeps every delta narrow, which is the shape delta is chosen for.
    let data: Vec<T> = (0..len)
        .map(|i| <T as From<u8>>::from(u8::try_from(i % 251).vortex_expect("modulo fits u8")))
        .collect();
    let primitive = PrimitiveArray::new(Buffer::from(data), Validity::NonNullable);

    let mut ctx = array_session().create_execution_ctx();
    Delta::try_from_primitive_array(&primitive, &mut ctx).vortex_expect("failed to delta encode")
}

fn benchmark_delta_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8>,
{
    let mut group = c.benchmark_group("cuda");

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let delta = make_delta_array::<T>(len);

        group.bench_with_input(
            BenchmarkId::new(format!("cuda/delta/{type_name}"), len_str),
            &delta,
            |b, delta| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx =
                        CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                            .vortex_expect("failed to create execution context")
                            .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                            .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(delta.clone().into_array().execute_cuda(&mut cuda_ctx)).unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_delta(c: &mut Criterion) {
    benchmark_delta_typed::<u8>(c, "u8");
    benchmark_delta_typed::<u16>(c, "u16");
    benchmark_delta_typed::<u32>(c, "u32");
    benchmark_delta_typed::<u64>(c, "u64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_delta
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
