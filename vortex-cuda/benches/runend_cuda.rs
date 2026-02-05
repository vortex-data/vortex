// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for run-end decoding.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_runend::RunEndArray;
use vortex_session::VortexSession;

/// Creates a run-end encoded array with the specified output length and average run length.
fn make_runend_array_typed<T>(output_len: usize, avg_run_len: usize) -> RunEndArray
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
    RunEndArray::new(ends_array, values_array)
}

/// Launches runend decode kernel and returns elapsed GPU time.
fn launch_runend_kernel_timed_typed<T>(
    runend_array: &RunEndArray,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration>
where
    T: NativePType + DeviceRepr,
{
    let ends_prim = runend_array.ends().to_primitive();
    let values_prim = runend_array.values().to_primitive();

    let output_len = runend_array.len();
    let num_runs = ends_prim.len();
    let offset = runend_array.offset();

    let ends_device = block_on(
        cuda_ctx
            .copy_to_device(ends_prim.as_slice::<u64>().to_vec())
            .unwrap(),
    )
    .vortex_expect("failed to copy ends to device");

    let values_device = block_on(
        cuda_ctx
            .copy_to_device(values_prim.as_slice::<T>().to_vec())
            .unwrap(),
    )
    .vortex_expect("failed to copy values to device");

    let output_device = block_on(
        cuda_ctx
            .copy_to_device(vec![T::default(); output_len])
            .unwrap(),
    )
    .vortex_expect("failed to allocate output buffer");

    let ends_view = ends_device
        .cuda_view::<u64>()
        .vortex_expect("failed to get ends view");
    let values_view = values_device
        .cuda_view::<T>()
        .vortex_expect("failed to get values view");
    let output_view = output_device
        .cuda_view::<T>()
        .vortex_expect("failed to get output view");

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "runend",
        ptypes: &[T::PTYPE, PType::U64],
        launch_args: [ends_view, num_runs, values_view, offset, output_len, output_view],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: output_len
    );

    events.duration()
}

/// Benchmark run-end decoding for a specific type with varying run lengths
fn benchmark_runend_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8>,
{
    let mut group = c.benchmark_group("runend_cuda");
    group.sample_size(10);

    for (len, len_str) in [
        (1_000_000usize, "1M"),
        (10_000_000usize, "10M"),
        (100_000_000usize, "100M"),
    ] {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        for run_len in [10, 100, 1000, 10000, 100000] {
            let runend_array = make_runend_array_typed::<T>(len, run_len);

            group.bench_with_input(
                BenchmarkId::new("runend", format!("{len_str}_{type_name}_runlen_{run_len}")),
                &runend_array,
                |b, runend_array| {
                    b.iter_custom(|iters| {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let mut total_time = Duration::ZERO;

                        for _ in 0..iters {
                            let kernel_time =
                                launch_runend_kernel_timed_typed::<T>(runend_array, &mut cuda_ctx)
                                    .vortex_expect("kernel launch failed");
                            total_time += kernel_time;
                        }

                        total_time
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

criterion::criterion_group!(benches, benchmark_runend);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
