// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::has_nvcc;
use vortex_error::VortexExpect;
use vortex_fastlanes::FoRArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[
    (1_000, "1K"),
    (10_000, "10K"),
    (100_000, "100K"),
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

/// Creates a FoR array for the given size.
fn make_for_array(len: usize) -> FoRArray {
    let primitive_array = PrimitiveArray::new(
        Buffer::from((0u32..len as u32).collect::<Vec<u32>>()),
        vortex_array::validity::Validity::NonNullable,
    )
    .into_array();

    let for_offset = 10u32;

    FoRArray::try_new(primitive_array, for_offset.into())
        .vortex_expect("failed to create FoR array")
}

/// Launches FoR decompression kernel and returns elapsed GPU time in seconds.
fn launch_for_kernel_timed(
    for_array: &FoRArray,
    reference: u32,
    device_data: cudarc::driver::CudaSlice<u32>,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let array_len = for_array.len() as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype()],
        launch_args: [device_data, reference, array_len],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    let elapsed_ms = events
        .before_launch
        .elapsed_ms(&events.after_launch) // synchronizes
        .map_err(|e| vortex_error::vortex_err!("failed to get elapsed time: {}", e))?;

    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

fn benchmark_for_cuda(c: &mut Criterion) {
    if !has_nvcc() {
        eprintln!("nvcc not found, skipping CUDA benchmarks");
        return;
    }

    let mut group = c.benchmark_group("FoR_cuda");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let for_array = make_for_array(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u32_FoR", label),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let encoded = for_array.encoded();
                    let unpacked_array = encoded.to_primitive();
                    let unpacked_slice = unpacked_array.as_slice::<u32>();

                    let reference = 10u32;
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let device_data = cuda_ctx
                            .to_device(unpacked_slice)
                            .vortex_expect("failed to copy to device");

                        let kernel_time = launch_for_kernel_timed(
                            for_array,
                            reference,
                            device_data,
                            &mut cuda_ctx,
                        )
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

criterion_group!(benches, benchmark_for_cuda);
criterion_main!(benches);
