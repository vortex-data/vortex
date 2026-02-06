// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for DateTimeParts decoding.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_datetime_parts::DateTimePartsArray;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

fn make_datetimeparts_array(len: usize, time_unit: TimeUnit) -> DateTimePartsArray {
    let days: Vec<i16> = (0..len).map(|i| (i / 1000) as i16).collect();
    let days_arr = PrimitiveArray::new(Buffer::from(days), Validity::NonNullable).into_array();
    let seconds_arr = ConstantArray::new(0i8, len).into_array();
    let subseconds_arr = ConstantArray::new(0i8, len).into_array();

    let dtype = DType::Extension(Timestamp::new(time_unit, Nullability::NonNullable).erased());

    DateTimePartsArray::try_new(dtype, days_arr, seconds_arr, subseconds_arr)
        .vortex_expect("Failed to create DateTimePartsArray")
}

/// Launches DateTimeParts decode kernel and returns elapsed GPU time.
fn launch_datetimeparts_kernel_timed(
    dtp_array: &DateTimePartsArray,
    time_unit: TimeUnit,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let days_prim = dtp_array.days().to_primitive();

    // TODO(0ax1): figure out how to represent constant array in CUDA kernels
    let seconds_prim = dtp_array.seconds().to_primitive();
    let subseconds_prim = dtp_array.subseconds().to_primitive();

    let output_len = dtp_array.len();

    let divisor: i64 = match time_unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => unreachable!("Days not supported for DateTimeParts"),
    };

    let days_device = block_on(
        cuda_ctx
            .copy_to_device(days_prim.as_slice::<i16>().to_vec())
            .unwrap(),
    )
    .vortex_expect("failed to copy days to device");

    let seconds_device = block_on(
        cuda_ctx
            .copy_to_device(seconds_prim.as_slice::<i8>().to_vec())
            .unwrap(),
    )
    .vortex_expect("failed to copy seconds to device");

    let subseconds_device = block_on(
        cuda_ctx
            .copy_to_device(subseconds_prim.as_slice::<i8>().to_vec())
            .unwrap(),
    )
    .vortex_expect("failed to copy subseconds to device");

    // Allocate output buffer
    let output_device = block_on(cuda_ctx.copy_to_device(vec![0i64; output_len]).unwrap())
        .vortex_expect("failed to allocate output buffer");

    let days_view = days_device
        .cuda_view::<i16>()
        .vortex_expect("failed to get days view");
    let seconds_view = seconds_device
        .cuda_view::<i8>()
        .vortex_expect("failed to get seconds view");
    let subseconds_view = subseconds_device
        .cuda_view::<i8>()
        .vortex_expect("failed to get subseconds view");
    let output_view = output_device
        .cuda_view::<i64>()
        .vortex_expect("failed to get output view");

    let array_len_u64 = output_len as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "date_time_parts",
        ptypes: &[PType::I16, PType::I8, PType::I8],
        launch_args: [days_view, seconds_view, subseconds_view, divisor, output_view, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: output_len
    );

    events.duration()
}

fn benchmark_datetimeparts(c: &mut Criterion) {
    let mut group = c.benchmark_group("datetimeparts_cuda");
    group.sample_size(10);

    for (len, len_str) in [
        (1_000_000usize, "1M"),
        (10_000_000usize, "10M"),
        (100_000_000usize, "100M"),
    ] {
        group.throughput(Throughput::Bytes((len * size_of::<i64>()) as u64));

        let (time_unit, unit_str) = (TimeUnit::Milliseconds, "ms");
        let dtp_array = make_datetimeparts_array(len, time_unit);

        group.bench_with_input(
            BenchmarkId::new("datetimeparts", format!("{len_str}_{unit_str}")),
            &dtp_array,
            |b, dtp_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time =
                            launch_datetimeparts_kernel_timed(dtp_array, time_unit, &mut cuda_ctx)
                                .vortex_expect("kernel launch failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group!(benches, benchmark_datetimeparts);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
