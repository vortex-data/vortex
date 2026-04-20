// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for run-end decoding.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod common;

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
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArray;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::common::TimedLaunchStrategy;

/// Creates a run-end encoded array with the specified output length and average run length.
fn make_runend_array_typed<T>(
    output_len: usize,
    avg_run_len: usize,
    ctx: &mut vortex::array::ExecutionCtx,
) -> RunEndArray
where
    T: NativePType + From<u8>,
{
    let num_runs = output_len.div_ceil(avg_run_len);
    let mut ends: Vec<u64> = Vec::with_capacity(num_runs);
    let mut values: Vec<T> = Vec::with_capacity(num_runs);

    let mut pos: usize = 0;
    for i in 0..num_runs {
        pos += avg_run_len;
        if pos > output_len {
            pos = output_len;
        }
        ends.push(pos as u64);
        values.push(<T as From<u8>>::from((i % 256) as u8));
    }

    let ends_array = PrimitiveArray::new(Buffer::from(ends), Validity::NonNullable).into_array();
    let values_array =
        PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array();
    RunEnd::new(ends_array, values_array, ctx)
}

/// Benchmark run-end decoding for a specific type with varying run lengths
fn benchmark_runend_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8>,
{
    let mut group = c.benchmark_group("runend_cuda");

    for (len, len_str) in [(10_000_000usize, "10M"), (100_000_000usize, "100M")] {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        for run_len in [10, 1000, 100000] {
            let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();
            let runend_array = make_runend_array_typed::<T>(len, run_len, cuda_ctx.execution_ctx());

            group.bench_with_input(
                BenchmarkId::new("runend", format!("{len_str}_{type_name}_runlen_{run_len}")),
                &runend_array,
                |b, runend_array| {
                    b.iter_custom(|iters| {
                        let timed = TimedLaunchStrategy::default();
                        let timer = Arc::clone(&timed.total_time_ns);

                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .unwrap()
                                .with_launch_strategy(Arc::new(timed));

                        for _ in 0..iters {
                            block_on(
                                runend_array
                                    .clone()
                                    .into_array()
                                    .execute_cuda(&mut cuda_ctx),
                            )
                            .unwrap();
                        }

                        Duration::from_nanos(timer.load(Ordering::Relaxed))
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark run-end decoding with varying run lengths for all types
fn benchmark_runend(c: &mut Criterion) {
    benchmark_runend_typed::<i32>(c, "i32");
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default().without_plots()
        .sample_size(10)
        .warm_up_time(Duration::from_nanos(1))
        .measurement_time(Duration::from_nanos(1))
        .nresamples(10);
    targets = benchmark_runend
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
