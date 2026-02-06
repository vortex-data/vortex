// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for dictionary decoding.

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
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];

/// Configuration for a dictionary benchmark specifying value and code types along with dictionary size.
struct DictBenchConfig {
    dict_size: usize,
    value_type_name: &'static str,
    code_type_name: &'static str,
}

/// Creates a Dict array with parameterized value type V and code type C.
fn make_dict_array_typed<V, C>(len: usize, dict_size: usize) -> (DictArray, Vec<V>, Vec<C>)
where
    V: NativePType + From<u32>,
    C: NativePType + TryFrom<usize>,
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    // Dictionary values
    let values: Vec<V> = (0..dict_size)
        .map(|i| <V as From<u32>>::from((i * 1000) as u32))
        .collect();
    let values_array = PrimitiveArray::new(Buffer::from(values.clone()), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<C> = (0..len)
        .map(|i| C::try_from(i % dict_size).unwrap())
        .collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);

    let dict_array = DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array");

    (dict_array, values, codes)
}

/// Launches Dict decompression kernel and returns elapsed GPU time.
fn launch_dict_kernel_timed_typed<V, C>(
    values: &[V],
    codes: &[C],
    output_len: usize,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration>
where
    V: NativePType + DeviceRepr,
    C: NativePType + DeviceRepr,
{
    let values_device = block_on(cuda_ctx.copy_to_device(values.to_vec()).unwrap())
        .vortex_expect("failed to copy values to device");

    let codes_device = block_on(cuda_ctx.copy_to_device(codes.to_vec()).unwrap())
        .vortex_expect("failed to copy codes to device");

    let output_slice = cuda_ctx
        .device_alloc::<V>(output_len)
        .vortex_expect("failed to allocate output");
    let output_device = CudaDeviceBuffer::new(output_slice);

    let codes_view = codes_device
        .cuda_view::<C>()
        .vortex_expect("failed to get codes view");
    let values_view = values_device
        .cuda_view::<V>()
        .vortex_expect("failed to get values view");
    let output_view = output_device.as_view::<V>();

    let codes_len_u64 = output_len as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "dict",
        ptypes: &[V::PTYPE, C::PTYPE],
        launch_args: [codes_view, codes_len_u64, values_view, output_view],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: output_len
    );

    events.duration()
}

/// Benchmark Dict decompression for specific value and code types.
fn benchmark_dict_typed<V, C>(c: &mut Criterion, config: &DictBenchConfig)
where
    V: NativePType + DeviceRepr + From<u32>,
    C: NativePType + DeviceRepr + TryFrom<usize>,
    <C as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut group = c.benchmark_group("dict_cuda");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        // Throughput is based on output size (values read from dictionary)
        group.throughput(Throughput::Bytes((len * size_of::<V>()) as u64));

        let (dict_array, values, codes) = make_dict_array_typed::<V, C>(*len, config.dict_size);

        group.bench_with_input(
            BenchmarkId::new(
                "dict",
                format!(
                    "{len_str}_{}_values_{}_codes",
                    config.value_type_name, config.code_type_name
                ),
            ),
            &(dict_array, values, codes),
            |b, (dict_array, values, codes)| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time = launch_dict_kernel_timed_typed::<V, C>(
                            values,
                            codes,
                            dict_array.len(),
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

/// Benchmark Dict decompression for all type combinations.
fn benchmark_dict(c: &mut Criterion) {
    // u32 values with u8 codes
    benchmark_dict_typed::<u32, u8>(
        c,
        &DictBenchConfig {
            dict_size: 256,
            value_type_name: "u32",
            code_type_name: "u8",
        },
    );

    // u32 values with u16 codes
    benchmark_dict_typed::<u32, u16>(
        c,
        &DictBenchConfig {
            dict_size: 4096,
            value_type_name: "u32",
            code_type_name: "u16",
        },
    );

    // u64 values with u8 codes
    benchmark_dict_typed::<u64, u8>(
        c,
        &DictBenchConfig {
            dict_size: 256,
            value_type_name: "u64",
            code_type_name: "u8",
        },
    );

    // u64 values with u32 codes
    benchmark_dict_typed::<u64, u32>(
        c,
        &DictBenchConfig {
            dict_size: 65536,
            value_type_name: "u64",
            code_type_name: "u32",
        },
    );
}

criterion::criterion_group!(benches, benchmark_dict);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
