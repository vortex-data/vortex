// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FoR decompression.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::ops::Add;
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
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::FoRArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];
const REFERENCE_VALUE: u8 = 10;

/// Creates a FoR array with the specified type and length.
fn make_for_array_typed<T>(len: usize) -> FoRArray
where
    T: NativePType + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let reference = <T as From<u8>>::from(REFERENCE_VALUE);
    let data: Vec<T> = (0..len)
        .map(|i| <T as From<u8>>::from((i % 256) as u8) + reference)
        .collect();

    let primitive_array =
        PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

    FoRArray::try_new(primitive_array, reference.into()).vortex_expect("failed to create FoR array")
}

/// Launches FoR decompression kernel and returns elapsed GPU time.
fn launch_for_kernel_timed_typed<T>(
    for_array: &FoRArray,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration>
where
    T: NativePType + DeviceRepr + From<u8>,
{
    let encoded = for_array.encoded();
    let unpacked_array = encoded.to_primitive();
    let unpacked_slice = unpacked_array.as_slice::<T>();

    let device_data = block_on(cuda_ctx.copy_to_device(unpacked_slice.to_vec()).unwrap())
        .vortex_expect("failed to copy to device");

    let reference = <T as From<u8>>::from(REFERENCE_VALUE);
    let array_len_u64 = for_array.len() as u64;

    let device_view = device_data
        .cuda_view::<T>()
        .vortex_expect("failed to get device view");

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype()],
        launch_args: [device_view, reference, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    events.duration()
}

/// Benchmark FoR decompression for a specific type.
fn benchmark_for_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let mut group = c.benchmark_group("for_cuda");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let for_array = make_for_array_typed::<T>(*len);

        group.bench_with_input(
            BenchmarkId::new("for", format!("{len_str}_{type_name}")),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time =
                            launch_for_kernel_timed_typed::<T>(for_array, &mut cuda_ctx)
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

/// Benchmark FoR decompression for all types.
fn benchmark_for(c: &mut Criterion) {
    benchmark_for_typed::<u8>(c, "u8");
    benchmark_for_typed::<u16>(c, "u16");
    benchmark_for_typed::<u32>(c, "u32");
    benchmark_for_typed::<u64>(c, "u64");
}

criterion::criterion_group!(benches, benchmark_for);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
